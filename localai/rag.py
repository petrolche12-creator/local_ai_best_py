"""Папка RAG: память обучения. Модель складывает сюда план и источники."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


README = """Папка обучения.

Сюда модель пишет план, ответы на свои вопросы и найденные страницы.
Можно создавать вложенные папки. Лимит размера и времени задаётся при обучении.
Это не дообучение весов: чат потом читает эти заметки и опирается на них.
"""


@dataclass(frozen=True)
class LearnLimits:
    max_gb: float = 1.0
    minutes: int = 20
    speed: str = "normal"
    depth: int = 2

    @property
    def max_bytes(self) -> int:
        return int(max(0.05, self.max_gb) * (1024**3))

    def label(self) -> str:
        return (
            f"{self.max_gb:g} ГБ, {self.minutes} мин, "
            f"скорость {speed_name(self.speed)}, глубина {self.depth}"
        )


def speed_name(speed: str) -> str:
    return {"fast": "быстро", "normal": "нормально", "slow": "медленно"}.get(speed, speed)


def depth_name(depth: int) -> str:
    return {1: "поверхностно", 2: "обычно", 3: "глубоко"}.get(depth, str(depth))


def normalize_limits(max_gb: float, minutes: int, speed: str, depth: int) -> LearnLimits:
    speed = speed if speed in {"fast", "normal", "slow"} else "normal"
    try:
        depth_n = int(depth)
    except (TypeError, ValueError):
        depth_n = 2
    depth_n = min(3, max(1, depth_n))
    try:
        gb = float(max_gb)
    except (TypeError, ValueError):
        gb = 1.0
    gb = min(100.0, max(0.05, gb))
    try:
        mins = int(minutes)
    except (TypeError, ValueError):
        mins = 20
    mins = min(240, max(1, mins))
    return LearnLimits(max_gb=gb, minutes=mins, speed=speed, depth=depth_n)


def ensure_rag(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    readme = root / "README.txt"
    if not readme.exists():
        readme.write_text(README, encoding="utf-8")
    return root


def used_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def used_gb(root: Path) -> float:
    return used_bytes(root) / (1024**3)


def slug(text: str, limit: int = 48) -> str:
    raw = text.strip().lower()
    raw = re.sub(r"[^\w]+", "-", raw, flags=re.UNICODE)
    raw = raw.strip("-")
    return (raw or "tema")[:limit]


def topic_dir(root: Path, topic: str) -> Path:
    path = root / slug(topic)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_relative(root: Path, relative: str) -> Path | None:
    raw = relative.replace("\\", "/").strip()
    if not raw or raw.startswith("/"):
        return None
    parts = [part for part in raw.split("/") if part and part not in {".", ".."}]
    if not parts:
        return None
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def write_text(root: Path, relative: str, content: str, limits: LearnLimits) -> str:
    path = safe_relative(root, relative)
    if path is None:
        return "путь вне папки RAG"
    data = content.encode("utf-8")
    if used_bytes(root) + len(data) > limits.max_bytes:
        return f"лимит {limits.max_gb:g} ГБ исчерпан, не записал {relative}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return f"записал {path} ({len(data)} байт)"


def retrieve(root: Path, query: str, limit_chars: int = 1600) -> str:
    if not root.exists() or not query.strip():
        return ""
    words = [word for word in re.findall(r"\w{3,}", query.lower(), flags=re.UNICODE) if word not in _STOP]
    if not words:
        return ""
    scored: list[tuple[int, Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.name == "README.txt":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hay = (path.name + "\n" + text).lower()
        score = sum(hay.count(word) for word in words)
        if score:
            scored.append((score, path, text))
    scored.sort(key=lambda item: item[0], reverse=True)
    chunks: list[str] = []
    used = 0
    for _score, path, text in scored[:4]:
        piece = text.strip()
        if len(piece) > 500:
            piece = piece[:500] + "…"
        block = f"[{path.relative_to(root)}]\n{piece}"
        if used + len(block) > limit_chars:
            break
        chunks.append(block)
        used += len(block)
    if not chunks:
        return ""
    return "Заметки из RAG, опирайся на них, если они по делу:\n\n" + "\n\n".join(chunks)


def context_for(root: Path, query: str) -> str:
    return retrieve(root, query)


_STOP = {
    "что",
    "как",
    "это",
    "для",
    "или",
    "the",
    "and",
    "что",
    "мне",
    "про",
}
