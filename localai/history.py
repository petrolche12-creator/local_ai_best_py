"""Истории чатов: JSON в data/chats, экспорт в Markdown."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from localai.think import strip_think


SCHEMA = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slug(text: str, limit: int = 24) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"[^0-9a-zа-яё\-]+", "", cleaned)
    cleaned = cleaned.strip("-")
    return (cleaned[:limit] or "chat")


@dataclass
class Chat:
    id: str
    title: str
    created_at: str
    updated_at: str
    model_id: str
    model_name: str
    mode: str
    system_prompt: str
    messages: list[dict[str, str]] = field(default_factory=list)
    thinking_model: bool = True
    repo_id: str = ""
    filename: str = ""

    @property
    def user_turns(self) -> int:
        return sum(1 for item in self.messages if item.get("role") == "user")


def new_chat(
    *,
    model_id: str,
    model_name: str,
    mode: str,
    system_prompt: str,
    thinking_model: bool,
    repo_id: str = "",
    filename: str = "",
    title: str = "Новый чат",
) -> Chat:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Chat(
        id=f"{stamp}-{slug(title)}",
        title=title,
        created_at=now_iso(),
        updated_at=now_iso(),
        model_id=model_id,
        model_name=model_name,
        mode=mode,
        system_prompt=system_prompt,
        thinking_model=thinking_model,
        repo_id=repo_id,
        filename=filename,
    )


def retitle(chat: Chat, text: str) -> None:
    title = re.sub(r"\s+", " ", text).strip()
    if not title:
        return
    chat.title = title[:60]
    match = re.match(r"^(\d{8}-\d{6})", chat.id)
    prefix = match.group(1) if match else chat.id[:15]
    chat.id = f"{prefix}-{slug(chat.title)}"


def chat_path(folder: Path, chat: Chat) -> Path:
    return folder / f"{chat.id}.json"


def save_chat(folder: Path, chat: Chat) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    chat.updated_at = now_iso()
    path = chat_path(folder, chat)
    payload = {
        "version": SCHEMA,
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at,
        "updated_at": chat.updated_at,
        "model_id": chat.model_id,
        "model_name": chat.model_name,
        "mode": chat.mode,
        "system_prompt": chat.system_prompt,
        "thinking_model": chat.thinking_model,
        "repo_id": chat.repo_id,
        "filename": chat.filename,
        "messages": chat.messages,
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    _drop_stale_files(folder, chat)
    return path


def _drop_stale_files(folder: Path, chat: Chat) -> None:
    """Если чат переименовали, старый файл с тем же timestamp больше не нужен."""
    match = re.match(r"^(\d{8}-\d{6})", chat.id)
    if not match:
        return
    prefix = match.group(1)
    for path in folder.glob(f"{prefix}-*.json"):
        if path.stem != chat.id:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(raw, dict) and raw.get("created_at") == chat.created_at:
                path.unlink(missing_ok=True)


def load_chat(path: Path) -> Chat:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"битый чат: {path.name}")
    messages = raw.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    return Chat(
        id=str(raw.get("id") or path.stem),
        title=str(raw.get("title") or path.stem),
        created_at=str(raw.get("created_at") or ""),
        updated_at=str(raw.get("updated_at") or ""),
        model_id=str(raw.get("model_id") or ""),
        model_name=str(raw.get("model_name") or ""),
        mode=str(raw.get("mode") or "balance"),
        system_prompt=str(raw.get("system_prompt") or ""),
        messages=[item for item in messages if isinstance(item, dict)],
        thinking_model=bool(raw.get("thinking_model", True)),
        repo_id=str(raw.get("repo_id") or ""),
        filename=str(raw.get("filename") or ""),
    )


def list_chats(folder: Path) -> list[Chat]:
    if not folder.exists():
        return []
    chats: list[Chat] = []
    for path in folder.glob("*.json"):
        try:
            chats.append(load_chat(path))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    chats.sort(key=lambda chat: chat.updated_at, reverse=True)
    return chats


def find_chat(folder: Path, chat_id: str) -> Chat | None:
    direct = folder / f"{chat_id}.json"
    if direct.exists():
        return load_chat(direct)
    for chat in list_chats(folder):
        if chat.id == chat_id or chat.id.startswith(chat_id):
            return chat
    return None


def delete_chat(folder: Path, chat: Chat) -> None:
    path = chat_path(folder, chat)
    path.unlink(missing_ok=True)


def export_markdown(chat: Chat, dest: Path, *, include_thinking: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {chat.title}",
        "",
        f"- модель: {chat.model_name} (`{chat.model_id}`)",
        f"- режим: {chat.mode}",
        f"- создан: {chat.created_at}",
        f"- обновлён: {chat.updated_at}",
        "",
        "## системный промпт",
        "",
        chat.system_prompt.strip() or "_(пусто)_",
        "",
    ]
    for item in chat.messages:
        role = item.get("role", "user")
        label = "ты" if role == "user" else "ассистент"
        lines.append(f"## {label}")
        lines.append("")
        reasoning = (item.get("reasoning") or "").strip()
        if include_thinking and reasoning:
            lines.append("рассуждения:")
            lines.append("")
            lines.extend(f"> {line}" if line else ">" for line in reasoning.splitlines())
            lines.append("")
        content = item.get("content", "")
        if not include_thinking:
            content, _hidden = strip_think(content)
        lines.append(content.strip())
        lines.append("")
    dest.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return dest
