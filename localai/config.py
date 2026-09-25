"""Настройки и пути. Всё лежит в ./data, пока не задан LOCALAI_HOME."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from localai.prompts import DEFAULT_PROMPT, normalize_prompt


SCHEMA = 1


@dataclass
class Settings:
    mode: str | None = None
    speed: str | None = None
    smart: str | None = None
    agent: bool = True
    agent_auto: bool = False
    workspace: str | None = None
    model_id: str | None = None
    custom_repo: str | None = None
    custom_filename: str | None = None
    custom_size_gb: float | None = None
    system_prompt: str = DEFAULT_PROMPT
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1024
    n_ctx: int | None = None
    n_threads: int | None = None
    n_gpu_layers: str = "auto"
    show_thinking: bool | None = None
    models_dir: str | None = None
    last_chat_id: str | None = None
    hf_token: str | None = None
    rag_max_gb: float = 1.0
    rag_minutes: int = 20
    rag_speed: str = "normal"
    rag_depth: int = 2

    def thinking_enabled(self, mode: str | None = None) -> bool:
        if self.show_thinking is not None:
            return self.show_thinking
        return (mode or self.mode or "balance") == "smart"


def data_root() -> Path:
    env = os.environ.get("LOCALAI_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve() / "data"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    config: Path
    chats: Path
    models: Path
    exports: Path
    input_history: Path
    rag: Path


def resolve_paths(settings: Settings | None = None, root: Path | None = None) -> AppPaths:
    base = (root or data_root()).resolve()
    models = base / "models"
    if settings and settings.models_dir:
        models = Path(settings.models_dir).expanduser().resolve()
    return AppPaths(
        root=base,
        config=base / "config.json",
        chats=base / "chats",
        models=models,
        exports=base / "exports",
        input_history=base / "input_history",
        rag=base / "RAG",
    )


def ensure_dirs(paths: AppPaths) -> None:
    from localai.rag import ensure_rag

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.chats.mkdir(parents=True, exist_ok=True)
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.exports.mkdir(parents=True, exist_ok=True)
    ensure_rag(paths.rag)


def load_settings(paths: AppPaths) -> Settings:
    if not paths.config.exists():
        return Settings()
    try:
        raw = json.loads(paths.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    known = {field for field in Settings.__dataclass_fields__}
    data = {key: value for key, value in raw.items() if key in known}
    settings = Settings(**data)
    settings.system_prompt = normalize_prompt(settings.system_prompt or DEFAULT_PROMPT)
    if settings.mode not in {None, "fast", "smart", "balance"}:
        settings.mode = None
    if settings.speed not in {None, "fast", "normal", "slow"}:
        settings.speed = None
    if settings.smart not in {None, "simple", "normal", "max"}:
        settings.smart = None
    settings.agent = bool(settings.agent)
    settings.agent_auto = bool(settings.agent_auto)
    try:
        settings.rag_max_gb = min(100.0, max(0.05, float(settings.rag_max_gb)))
    except (TypeError, ValueError):
        settings.rag_max_gb = 1.0
    try:
        settings.rag_minutes = min(240, max(1, int(settings.rag_minutes)))
    except (TypeError, ValueError):
        settings.rag_minutes = 20
    if settings.rag_speed not in {"fast", "normal", "slow"}:
        settings.rag_speed = "normal"
    try:
        settings.rag_depth = min(3, max(1, int(settings.rag_depth)))
    except (TypeError, ValueError):
        settings.rag_depth = 2
    if not isinstance(settings.n_gpu_layers, str) or not settings.n_gpu_layers.strip():
        settings.n_gpu_layers = "auto"
    settings.n_gpu_layers = settings.n_gpu_layers.strip().lower()
    return settings


def save_settings(paths: AppPaths, settings: Settings) -> None:
    ensure_dirs(paths)
    payload = asdict(settings)
    payload["version"] = SCHEMA
    paths.config.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hf_token(settings: Settings) -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or settings.hf_token
        or None
    )
