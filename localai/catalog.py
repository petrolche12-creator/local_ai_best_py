"""Каталог GGUF-моделей с Hugging Face.

Размеры проверенных файлов записаны точно. У части квантов размер оценочный:
перед скачиванием приложение сверяет его с Hugging Face.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def _gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    params_b: float
    quant: str
    repo_id: str
    filename: str
    size_gb: float
    n_layers: int
    blurb: str
    thinking: bool = True
    size_estimated: bool = False
    license: str = "apache-2.0"

    @property
    def display_name(self) -> str:
        return f"{self.name} {self.quant}"

    @property
    def hub_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    def resized(self, size_gb: float, *, estimated: bool = False) -> ModelSpec:
        return replace(self, size_gb=size_gb, size_estimated=estimated)


def quant_rank(quant: str) -> int:
    head = quant.upper().split("_", 1)[0]
    order = {"Q2": 2, "Q3": 3, "Q4": 4, "Q5": 5, "Q6": 6, "Q8": 8}
    for key, rank in order.items():
        if head.startswith(key):
            return rank
    return 4


# Точные размеры — с дерева файлов Hugging Face (сентябрь 2026).
_Q4_06 = _gb(484_220_320)
_Q8_06 = _gb(639_446_688)
_Q8_17 = _gb(1_834_426_016)
_Q4_4 = _gb(2_497_280_256)
_Q5_4 = _gb(2_889_513_184)
_Q6_4 = _gb(3_306_260_704)
_Q8_4 = _gb(4_280_404_704)
_Q4_8 = _gb(5_027_783_488)
_Q5_8 = _gb(5_851_112_224)
_Q6_8 = _gb(6_725_899_040)
_Q8_8 = _gb(8_709_518_112)
_Q4_14 = _gb(9_001_752_960)
_Q4_32 = _gb(19_762_149_024)

# Оценки старших квантов 14B/32B — по соотношению квантов 8B. Уточняются по сети.
_RATIO_Q5 = _Q5_8 / _Q4_8
_RATIO_Q6 = _Q6_8 / _Q4_8
_RATIO_Q8 = _Q8_8 / _Q4_8


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="qwen3-0.6b-q4",
        name="Qwen3 0.6B",
        params_b=0.6,
        quant="Q4_K_M",
        repo_id="bartowski/Qwen_Qwen3-0.6B-GGUF",
        filename="Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        size_gb=_Q4_06,
        n_layers=28,
        blurb="еле болтает, зато летает",
    ),
    ModelSpec(
        id="qwen3-0.6b-q8",
        name="Qwen3 0.6B",
        params_b=0.6,
        quant="Q8_0",
        repo_id="Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
        size_gb=_Q8_06,
        n_layers=28,
        blurb="та же кроха, чуть аккуратнее",
    ),
    ModelSpec(
        id="qwen3-1.7b-q4",
        name="Qwen3 1.7B",
        params_b=1.7,
        quant="Q4_K_M",
        repo_id="bartowski/Qwen_Qwen3-1.7B-GGUF",
        filename="Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        size_gb=1.20,
        n_layers=28,
        blurb="простой чат, факты иногда путает",
        size_estimated=True,
    ),
    ModelSpec(
        id="qwen3-1.7b-q8",
        name="Qwen3 1.7B",
        params_b=1.7,
        quant="Q8_0",
        repo_id="Qwen/Qwen3-1.7B-GGUF",
        filename="Qwen3-1.7B-Q8_0.gguf",
        size_gb=_Q8_17,
        n_layers=28,
        blurb="1.7B без сильного квантования",
    ),
    ModelSpec(
        id="qwen3-4b-q4",
        name="Qwen3 4B",
        params_b=4.0,
        quant="Q4_K_M",
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        size_gb=_Q4_4,
        n_layers=36,
        blurb="уже нормальный собеседник",
    ),
    ModelSpec(
        id="qwen3-4b-q5",
        name="Qwen3 4B",
        params_b=4.0,
        quant="Q5_K_M",
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q5_K_M.gguf",
        size_gb=_Q5_4,
        n_layers=36,
        blurb="4B, чуть умнее Q4",
    ),
    ModelSpec(
        id="qwen3-4b-q6",
        name="Qwen3 4B",
        params_b=4.0,
        quant="Q6_K",
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q6_K.gguf",
        size_gb=_Q6_4,
        n_layers=36,
        blurb="4B почти без потерь",
    ),
    ModelSpec(
        id="qwen3-4b-q8",
        name="Qwen3 4B",
        params_b=4.0,
        quant="Q8_0",
        repo_id="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q8_0.gguf",
        size_gb=_Q8_4,
        n_layers=36,
        blurb="4B максимального качества",
    ),
    ModelSpec(
        id="qwen3-8b-q4",
        name="Qwen3 8B",
        params_b=8.0,
        quant="Q4_K_M",
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        size_gb=_Q4_8,
        n_layers=36,
        blurb="уверенный помощник",
    ),
    ModelSpec(
        id="qwen3-8b-q5",
        name="Qwen3 8B",
        params_b=8.0,
        quant="Q5_K_M",
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q5_K_M.gguf",
        size_gb=_Q5_8,
        n_layers=36,
        blurb="8B, хороший баланс качества",
    ),
    ModelSpec(
        id="qwen3-8b-q6",
        name="Qwen3 8B",
        params_b=8.0,
        quant="Q6_K",
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q6_K.gguf",
        size_gb=_Q6_8,
        n_layers=36,
        blurb="8B, почти полный вес",
    ),
    ModelSpec(
        id="qwen3-8b-q8",
        name="Qwen3 8B",
        params_b=8.0,
        quant="Q8_0",
        repo_id="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q8_0.gguf",
        size_gb=_Q8_8,
        n_layers=36,
        blurb="8B максимального качества",
    ),
    ModelSpec(
        id="qwen3-14b-q4",
        name="Qwen3 14B",
        params_b=14.0,
        quant="Q4_K_M",
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q4_K_M.gguf",
        size_gb=_Q4_14,
        n_layers=40,
        blurb="заметно умнее 8B и заметно медленнее",
    ),
    ModelSpec(
        id="qwen3-14b-q5",
        name="Qwen3 14B",
        params_b=14.0,
        quant="Q5_K_M",
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q5_K_M.gguf",
        size_gb=_Q4_14 * _RATIO_Q5,
        n_layers=40,
        blurb="14B, квант поплотнее",
        size_estimated=True,
    ),
    ModelSpec(
        id="qwen3-14b-q6",
        name="Qwen3 14B",
        params_b=14.0,
        quant="Q6_K",
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q6_K.gguf",
        size_gb=_Q4_14 * _RATIO_Q6,
        n_layers=40,
        blurb="14B высокого качества",
        size_estimated=True,
    ),
    ModelSpec(
        id="qwen3-14b-q8",
        name="Qwen3 14B",
        params_b=14.0,
        quant="Q8_0",
        repo_id="Qwen/Qwen3-14B-GGUF",
        filename="Qwen3-14B-Q8_0.gguf",
        size_gb=_Q4_14 * _RATIO_Q8,
        n_layers=40,
        blurb="14B почти без квантования",
        size_estimated=True,
    ),
    ModelSpec(
        id="qwen3-32b-q4",
        name="Qwen3 32B",
        params_b=32.0,
        quant="Q4_K_M",
        repo_id="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q4_K_M.gguf",
        size_gb=_Q4_32,
        n_layers=64,
        blurb="серьёзная модель, нужно много памяти",
    ),
    ModelSpec(
        id="qwen3-32b-q5",
        name="Qwen3 32B",
        params_b=32.0,
        quant="Q5_K_M",
        repo_id="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q5_K_M.gguf",
        size_gb=_Q4_32 * _RATIO_Q5,
        n_layers=64,
        blurb="32B плотнее, ещё тяжелее",
        size_estimated=True,
    ),
    ModelSpec(
        id="qwen3-32b-q6",
        name="Qwen3 32B",
        params_b=32.0,
        quant="Q6_K",
        repo_id="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q6_K.gguf",
        size_gb=_Q4_32 * _RATIO_Q6,
        n_layers=64,
        blurb="32B высокого качества",
        size_estimated=True,
    ),
    ModelSpec(
        id="qwen3-32b-q8",
        name="Qwen3 32B",
        params_b=32.0,
        quant="Q8_0",
        repo_id="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q8_0.gguf",
        size_gb=_Q4_32 * _RATIO_Q8,
        n_layers=64,
        blurb="32B максимального качества, очень большой файл",
        size_estimated=True,
    ),
)


def get_model(model_id: str) -> ModelSpec | None:
    for model in CATALOG:
        if model.id == model_id:
            return model
    return None


def smallest_model() -> ModelSpec:
    return min(CATALOG, key=lambda model: (model.size_gb, model.params_b))
