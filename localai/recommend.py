"""Подбор модели под железо и режим.

Быстрый режим берёт маленькую модель без рассуждений.
Умный — самую сильную, которая ещё нормально встаёт (не «половина весов в swap»).
Середина — на ступень выше быстрого.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from localai.catalog import CATALOG, ModelSpec, quant_rank, smallest_model
from localai.hardware import HardwareProfile


CPU_OVERHEAD_GB = 1.5
GPU_RESERVE_GB = 0.7
APPLE_OVERHEAD_GB = 2.0
HYBRID_RAM_OVERHEAD_GB = 1.8
MIN_CTX = 1024

TIERS = (0.6, 1.7, 4.0, 8.0, 14.0, 32.0)


@dataclass(frozen=True)
class Placement:
    device: str  # gpu, hybrid, metal, cpu
    n_ctx: int
    n_gpu_layers: int
    gpu_fraction: float
    kv_gb: float
    gpu_name: str | None = None
    reasonable: bool = True

    @property
    def device_rank(self) -> int:
        return {"gpu": 3, "metal": 3, "hybrid": 2, "cpu": 1}.get(self.device, 0)


@dataclass
class Recommendation:
    mode: str
    model: ModelSpec
    placement: Placement | None
    fits: bool
    notes: list[str] = field(default_factory=list)
    alternatives: list[tuple[ModelSpec, Placement]] = field(default_factory=list)


def kv_gb(params_b: float, n_ctx: int) -> float:
    return (n_ctx / 4096) * max(0.20, params_b * 0.11)


def effective_size(model: ModelSpec) -> float:
    if model.size_estimated:
        return model.size_gb * 1.08
    return model.size_gb


def fast_param_cap(hw: HardwareProfile) -> float:
    if hw.apple_silicon:
        if hw.ram_total_gb >= 64:
            return 14.0
        if hw.ram_total_gb >= 32:
            return 8.0
        if hw.ram_total_gb >= 16:
            return 4.0
        if hw.ram_total_gb >= 8:
            return 1.7
        return 0.6
    vram = hw.vram_gb
    if vram >= 24:
        return 14.0
    if vram >= 16:
        return 8.0
    if vram >= 6:
        return 4.0
    if vram >= 4:
        return 1.7
    if hw.ram_total_gb >= 32 and hw.cpu_physical >= 8:
        return 4.0
    if hw.ram_total_gb >= 16:
        return 1.7
    return 0.6


def next_tier(cap: float) -> float:
    for tier in TIERS:
        if tier > cap + 0.15:
            # 14B → 32B — слишком большой прыжок, середина остаётся на текущей ступени.
            if tier > cap * 2.2 and tier > cap + 6:
                return cap
            return tier
    return cap


def cpu_reasonable_cap(hw: HardwareProfile) -> float:
    if hw.ram_total_gb >= 32 and hw.cpu_physical >= 8:
        return 14.0
    if hw.ram_total_gb >= 16:
        return 8.0
    if hw.ram_total_gb >= 8:
        return 4.0
    return 1.7


def desired_ctx(mode: str) -> int:
    return {"fast": 4096, "balance": 4096, "smart": 8192}.get(mode, 4096)


def ctx_options(mode: str) -> list[int]:
    desired = desired_ctx(mode)
    options = [desired, 4096, 2048, MIN_CTX]
    seen: list[int] = []
    for ctx in options:
        if ctx > desired:
            continue
        if ctx not in seen:
            seen.append(ctx)
    return seen


def recommend(hw: HardwareProfile, mode: str, *, gpu_offload: bool = True) -> Recommendation:
    mode = mode if mode in {"fast", "smart", "balance"} else "balance"
    cap = {
        "fast": fast_param_cap(hw),
        "balance": next_tier(fast_param_cap(hw)),
        "smart": 32.0,
    }[mode]
    ranked = _ranked(hw, mode, gpu_offload)
    fitting = [(model, placement) for model, placement in ranked if placement is not None]
    reasonable = [(model, placement) for model, placement in fitting if placement.reasonable]
    pool = reasonable or fitting
    under_cap = [(model, placement) for model, placement in pool if model.params_b <= cap + 0.15]
    chosen_pool = under_cap or pool

    notes: list[str] = []
    if not chosen_pool:
        model = smallest_model()
        notes.append(
            "Свободной памяти мало даже для самой маленькой модели. "
            "Можно попробовать, но будет swap и очень медленно."
        )
        if hw.ram_available_gb < hw.ram_total_gb * 0.35 and hw.ram_total_gb >= 8:
            notes.append("Свободно мало RAM — закрой браузер и другие тяжёлые программы и пересканируй.")
        return Recommendation(
            mode=mode,
            model=model,
            placement=None,
            fits=False,
            notes=notes,
            alternatives=[],
        )

    model, placement = _pick(chosen_pool, mode)
    alternatives = _alternatives(fitting, model)
    notes.extend(_notes(hw, mode, model, placement, fitting))
    return Recommendation(
        mode=mode,
        model=model,
        placement=placement,
        fits=True,
        notes=notes,
        alternatives=alternatives,
    )


def recommend_all(hw: HardwareProfile, *, gpu_offload: bool = True) -> dict[str, Recommendation]:
    return {mode: recommend(hw, mode, gpu_offload=gpu_offload) for mode in ("fast", "balance", "smart")}


def explain(rec: Recommendation) -> str:
    mode_text = {
        "fast": "Режим «тупой, но быстрый»: маленькая модель, без долгих рассуждений.",
        "smart": "Режим «умный, но медленный»: самая сильная модель, которая ещё нормально встаёт.",
        "balance": "Середина: не самая тупая и не самая тяжёлая.",
    }[rec.mode]
    if rec.placement is None:
        return mode_text + " В память уверенно не влезает."
    where = _where(rec.placement)
    speed = speed_hint(rec.model, rec.placement)
    return (
        f"{mode_text}\n"
        f"Беру {rec.model.display_name} — {where}.\n"
        f"Контекст {rec.placement.n_ctx}. {speed}."
    )


def speed_hint(model: ModelSpec, placement: Placement) -> str:
    if placement.device in {"gpu", "metal"}:
        if model.params_b <= 4:
            return "Должно отвечать быстро"
        if model.params_b <= 14:
            return "Скорость нормальная, не мгновенная"
        return "Будет ощутимо думать даже на видеокарте"
    if placement.device == "hybrid":
        return "Часть на видеокарте, часть на процессоре — быстрее чистого CPU, но не летает"
    if model.params_b <= 1.7:
        return "На процессоре ещё терпимо"
    if model.params_b <= 4:
        return "На процессоре уже неспешно"
    return "На процессоре будет медленно — это плата за ум"


def mode_label(mode: str) -> str:
    return {
        "fast": "тупой, но быстрый",
        "smart": "умный, но медленный",
        "balance": "середина",
    }.get(mode, mode)


def thinking_for(mode: str, model: ModelSpec) -> bool:
    return mode == "smart" and model.thinking


def default_max_tokens(mode: str) -> int:
    return {"fast": 768, "balance": 1024, "smart": 2048}[mode]


def _ranked(
    hw: HardwareProfile,
    mode: str,
    gpu_offload: bool,
) -> list[tuple[ModelSpec, Placement | None]]:
    rows: list[tuple[ModelSpec, Placement | None]] = []
    for model in CATALOG:
        rows.append((model, choose_placement(hw, model, mode, gpu_offload)))
    return rows


def choose_placement(
    hw: HardwareProfile,
    model: ModelSpec,
    mode: str,
    gpu_offload: bool,
) -> Placement | None:
    candidates: list[Placement] = []
    for ctx in ctx_options(mode):
        candidates.extend(placements(hw, model, ctx, gpu_offload))
    if not candidates:
        return None
    reasonable = [item for item in candidates if item.reasonable]
    pool = reasonable or candidates
    return max(pool, key=lambda item: (item.device_rank, item.n_ctx, item.gpu_fraction))


def placements(
    hw: HardwareProfile,
    model: ModelSpec,
    n_ctx: int,
    gpu_offload: bool,
) -> list[Placement]:
    size = effective_size(model)
    kv = kv_gb(model.params_b, n_ctx)
    found: list[Placement] = []
    if hw.apple_silicon and gpu_offload:
        if hw.ram_available_gb >= size + kv + APPLE_OVERHEAD_GB:
            found.append(
                Placement(
                    device="metal",
                    n_ctx=n_ctx,
                    n_gpu_layers=-1,
                    gpu_fraction=1.0,
                    kv_gb=kv,
                    gpu_name=hw.best_gpu.name if hw.best_gpu else "Apple Silicon",
                    reasonable=True,
                )
            )
    elif gpu_offload and hw.vram_gb > 0:
        gpu_name = hw.best_gpu.name if hw.best_gpu else "GPU"
        if hw.vram_gb >= size + kv + GPU_RESERVE_GB and hw.ram_available_gb >= 1.6:
            found.append(
                Placement(
                    device="gpu",
                    n_ctx=n_ctx,
                    n_gpu_layers=-1,
                    gpu_fraction=1.0,
                    kv_gb=kv,
                    gpu_name=gpu_name,
                    reasonable=True,
                )
            )
        else:
            layers = _estimate_layers(model, hw.vram_gb, kv)
            if layers >= 4 and model.n_layers:
                fraction = layers / model.n_layers
                cpu_side = size * (1 - fraction) + kv * 0.5
                if hw.ram_available_gb >= cpu_side + HYBRID_RAM_OVERHEAD_GB:
                    found.append(
                        Placement(
                            device="hybrid",
                            n_ctx=n_ctx,
                            n_gpu_layers=layers,
                            gpu_fraction=fraction,
                            kv_gb=kv,
                            gpu_name=gpu_name,
                            reasonable=fraction >= 0.85,
                        )
                    )
    if hw.ram_available_gb >= size + kv + CPU_OVERHEAD_GB:
        found.append(
            Placement(
                device="cpu",
                n_ctx=n_ctx,
                n_gpu_layers=0,
                gpu_fraction=0.0,
                kv_gb=kv,
                gpu_name=None,
                reasonable=model.params_b <= cpu_reasonable_cap(hw) + 0.15,
            )
        )
    found.sort(key=lambda item: (item.device_rank, item.gpu_fraction), reverse=True)
    return found


def _estimate_layers(model: ModelSpec, vram_gb: float, kv: float) -> int:
    if model.n_layers <= 0 or vram_gb <= 0:
        return 0
    per_layer = (effective_size(model) + kv) / model.n_layers
    if per_layer <= 0:
        return 0
    usable = max(0.0, vram_gb - 0.5)
    return max(0, min(model.n_layers, int(usable / per_layer)))


def _pick(pool: list[tuple[ModelSpec, Placement]], mode: str) -> tuple[ModelSpec, Placement]:
    if mode == "fast":
        # Самый крупный размер в пуле, но квант Q4 — чтобы было быстро.
        best_params = max(model.params_b for model, _ in pool)
        group = [(model, placement) for model, placement in pool if abs(model.params_b - best_params) < 0.25]
        q4 = [(model, placement) for model, placement in group if model.quant.upper().startswith("Q4")]
        target = q4 or group
        return min(target, key=lambda row: (quant_rank(row[0].quant), row[0].size_gb))
    if mode == "balance":
        best_params = max(model.params_b for model, _ in pool)
        group = [(model, placement) for model, placement in pool if abs(model.params_b - best_params) < 0.25]
        q5 = [(model, placement) for model, placement in group if model.quant.upper().startswith("Q5")]
        q4 = [(model, placement) for model, placement in group if model.quant.upper().startswith("Q4")]
        target = q5 or q4 or group
        return min(target, key=lambda row: (row[0].size_gb, -row[1].device_rank))
    # Сначала то, что целиком сидит на GPU/Metal. Гибрид «половина на CPU»
    # не должен обгонять модель, которая нормально влезает в видеокарту.
    full = [(model, placement) for model, placement in pool if placement.device in {"gpu", "metal"}]
    cpu = [(model, placement) for model, placement in pool if placement.device == "cpu"]
    hybrid = [(model, placement) for model, placement in pool if placement.device == "hybrid"]
    target = full or cpu or hybrid or pool
    return max(target, key=_smart_key)


def _smart_key(row: tuple[ModelSpec, Placement]) -> tuple:
    model, placement = row
    # Контекст 1024 ради чуть более жирного кванта — плохой обмен.
    usable = 1 if placement.n_ctx >= 2048 else 0
    return (model.params_b, usable, quant_rank(model.quant), placement.device_rank, placement.n_ctx)


def _alternatives(
    fitting: list[tuple[ModelSpec, Placement]],
    chosen: ModelSpec,
) -> list[tuple[ModelSpec, Placement]]:
    # По одному лучшему кванту на размер, плюс сам выбранный не дублируем.
    by_size: dict[float, tuple[ModelSpec, Placement]] = {}
    for model, placement in fitting:
        key = model.params_b
        current = by_size.get(key)
        if current is None or _alt_score(model, placement) > _alt_score(*current):
            by_size[key] = (model, placement)
    rows = [row for row in by_size.values() if row[0].id != chosen.id]
    rows.sort(key=lambda row: row[0].params_b)
    return rows


def _alt_score(model: ModelSpec, placement: Placement) -> tuple[int, int, int]:
    prefer_q4 = 1 if model.quant.upper().startswith("Q4") else 0
    return (placement.device_rank, prefer_q4, quant_rank(model.quant))


def _notes(
    hw: HardwareProfile,
    mode: str,
    model: ModelSpec,
    placement: Placement,
    fitting: list[tuple[ModelSpec, Placement]],
) -> list[str]:
    notes: list[str] = []
    if not placement.reasonable:
        notes.append("Эта модель влезет только впритык и будет очень медленной.")
    if model.size_estimated:
        notes.append("Размер файла оценочный, перед скачиванием сверю с Hugging Face.")
    if hw.ram_available_gb < hw.ram_total_gb * 0.35 and hw.ram_total_gb >= 8:
        notes.append(
            f"Свободно только {hw.ram_available_gb:.1f} ГБ из {hw.ram_total_gb:.0f}. "
            "Закрой лишнее, если хочешь модель побольше."
        )
    if hw.swap_gb < 0.5 and placement.device == "cpu" and model.params_b >= 8:
        notes.append("Swap почти нет. Если память кончится, процесс просто упадёт.")
    same = [other for other, _ in fitting if abs(other.params_b - model.params_b) < 0.15]
    if mode == "smart" and len({item.params_b for item, _ in fitting}) == 1 and len(same) <= 2:
        notes.append(
            "Железо узкое: умный и быстрый режим могут сесть на близкие модели. "
            "Разница тогда в рассуждениях, контексте и кванте."
        )
    return notes


def _where(placement: Placement) -> str:
    if placement.device == "gpu":
        name = placement.gpu_name or "видеокарте"
        return f"целиком на {name}"
    if placement.device == "metal":
        return "на Apple Silicon, общая память"
    if placement.device == "hybrid":
        return (
            f"частично на GPU ({placement.n_gpu_layers} слоёв, "
            f"{placement.gpu_fraction * 100:.0f}%), остальное на CPU"
        )
    return "на процессоре"
