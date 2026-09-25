"""Скорость и ум — два отдельных вопроса, не одна кнопка."""

from __future__ import annotations

from dataclasses import dataclass

from localai.hardware import HardwareProfile
from localai.recommend import fast_param_cap, next_tier


SPEEDS = ("fast", "normal", "slow")
SMARTS = ("simple", "normal", "max")


def speed_label(speed: str) -> str:
    return {
        "fast": "очень быстро",
        "normal": "нормально",
        "slow": "можно медленно",
    }.get(speed, speed)


def smart_label(smart: str) -> str:
    return {
        "simple": "попроще",
        "normal": "нормальный ум",
        "max": "максимально умный",
    }.get(smart, smart)


def mode_from_prefs(speed: str, smart: str) -> str:
    """Режим для контекста и рассуждений. Умный режим думает, быстрый — нет."""
    if smart == "max" and speed != "fast":
        return "smart"
    if speed == "fast":
        return "fast"
    if speed == "slow" and smart != "simple":
        return "smart"
    return "balance"


def prefs_from_mode(mode: str | None) -> tuple[str, str]:
    if mode == "fast":
        return "fast", "simple"
    if mode == "smart":
        return "slow", "max"
    return "normal", "normal"


def memory_param_ceiling(hw: HardwareProfile, gpu_offload: bool) -> float:
    """Грубая верхняя граница размера в миллиардах параметров."""
    if hw.apple_silicon:
        return max(0.6, (hw.ram_available_gb - 2.2) / 0.7)
    if gpu_offload and hw.vram_gb >= 4:
        return max(0.6, (hw.vram_gb - 1.2) / 0.62)
    return max(0.6, (hw.ram_available_gb - 1.8) / 0.75)


def comfort_text(hw: HardwareProfile, gpu_offload: bool) -> str:
    ceiling = memory_param_ceiling(hw, gpu_offload)
    sizes = (0.6, 1.7, 4.0, 8.0, 14.0, 32.0)
    fit = [size for size in sizes if size <= ceiling + 0.35]
    top = fit[-1] if fit else 0.6
    where = "видеокарта" if gpu_offload and (hw.vram_gb >= 4 or hw.apple_silicon) else "процессор"
    return f"по свободной памяти ориентир до {top:g}B, считать будет {where}"


@dataclass(frozen=True)
class Band:
    speed: str
    smart: str
    lo: float
    hi: float
    quants: tuple[str, ...]
    mode: str

    @property
    def label(self) -> str:
        return f"скорость «{speed_label(self.speed)}», ум «{smart_label(self.smart)}»"


def band_for(hw: HardwareProfile, speed: str, smart: str, gpu_offload: bool = True) -> Band:
    speed = speed if speed in SPEEDS else "normal"
    smart = smart if smart in SMARTS else "normal"
    ceiling = memory_param_ceiling(hw, gpu_offload)
    fast_cap = min(fast_param_cap(hw), ceiling)
    if speed == "fast":
        hi = fast_cap
    elif speed == "normal":
        hi = min(next_tier(fast_cap), ceiling)
    else:
        hi = ceiling
    hi = max(0.6, hi)
    if smart == "simple":
        lo = 0.4
        hi = min(hi, max(1.7, fast_cap))
        quants = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q3_K_M")
    elif smart == "normal":
        lo = 0.4 if hi <= 2 else hi * 0.35
        quants = ("Q5_K_M", "Q4_K_M", "Q6_K", "Q8_0")
    else:
        lo = 0.4 if hi <= 2 else hi * 0.55
        quants = ("Q6_K", "Q5_K_M", "Q8_0", "Q4_K_M")
    return Band(
        speed=speed,
        smart=smart,
        lo=lo,
        hi=hi,
        quants=quants,
        mode=mode_from_prefs(speed, smart),
    )


def search_queries(band: Band) -> list[str]:
    """Дополнительные запросы по размеру. Основной обход — весь каталог GGUF, без бренда."""
    sizes = [size for size in (0.6, 1.7, 4.0, 8.0, 14.0, 32.0) if band.lo - 0.4 <= size <= band.hi + 0.6]
    if not sizes:
        sizes = [round(band.hi, 1)]
    if band.smart == "max":
        sizes = sorted(sizes, reverse=True)
    elif band.smart == "simple":
        sizes = sorted(sizes)
    else:
        mid = (band.lo + band.hi) / 2
        sizes = sorted(sizes, key=lambda size: abs(size - mid))
    queries: list[str] = []
    for size in sizes[:3]:
        label = f"{size:g}B"
        queries.append(f"{label} GGUF")
        queries.append(f"{label} Instruct GGUF")
    queries.append("Instruct GGUF")
    unique: list[str] = []
    for query in queries:
        if query not in unique:
            unique.append(query)
    return unique[:8]
