"""Сборка промпта и обрезка истории под контекст."""

from __future__ import annotations

import json
from collections.abc import Callable

from jinja2 import ChainableUndefined, Environment, TemplateError


def render_chat_prompt(
    template: str,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> str | None:
    """Рендерит Jinja-шаблон из GGUF. None, если шаблон битый."""
    if not template.strip():
        return None
    env = Environment(undefined=ChainableUndefined, autoescape=False)
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    try:
        rendered = env.from_string(template).render(
            messages=messages,
            add_generation_prompt=True,
            tools=None,
            enable_thinking=enable_thinking,
        )
    except (TemplateError, TypeError, ValueError):
        return None
    text = rendered.strip("\n")
    return text or None


def model_messages(
    system_prompt: str,
    history: list[dict[str, str]],
    *,
    thinking_model: bool,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "tool":
            name = item.get("name") or "tool"
            messages.append({"role": "user", "content": f"<tool_result name=\"{name}\">\n{content}\n</tool_result>"})
            continue
        reasoning = (item.get("reasoning") or "").strip()
        if role == "assistant" and reasoning and thinking_model:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        messages.append({"role": role, "content": content})
    return messages


def drop_old_messages(
    messages: list[dict[str, str]],
    fits: Callable[[list[dict[str, str]]], bool],
) -> tuple[list[dict[str, str]], int]:
    """Оставляет system (если он первый) и самые свежие реплики, пока fits() истинно.

    Возвращает (сообщения, сколько выкинули).
    """
    if fits(messages):
        return messages, 0
    system: list[dict[str, str]] = []
    rest = list(messages)
    if rest and rest[0].get("role") == "system":
        system = [rest[0]]
        rest = rest[1:]
    dropped = 0
    while rest and not fits(system + rest):
        rest.pop(0)
        dropped += 1
    if fits(system + rest):
        return system + rest, dropped
    # Даже последнее сообщение не влезает — отдаём как есть, вызывающий обрежет текст.
    if rest:
        return system + rest[-1:], dropped
    return system, dropped
