"""Прячет или показывает блоки <think> у Qwen3 и похожих моделей."""

from __future__ import annotations

from dataclasses import dataclass, field


OPEN = "<think>"
CLOSE = "</think>"


def strip_think(text: str) -> tuple[str, str]:
    """Возвращает (ответ без рассуждений, сами рассуждения)."""
    filt = ThinkFilter(show_thinking=True)
    filt.feed(text)
    filt.finish()
    return filt.answer.strip(), filt.thinking.strip()


@dataclass
class ThinkFilter:
    show_thinking: bool = False
    pending: str = ""
    in_think: bool = False
    thinking: str = ""
    answer: str = ""
    events: list[tuple[str, str]] = field(default_factory=list)

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        self.pending += text
        emitted: list[tuple[str, str]] = []
        while True:
            tag = CLOSE if self.in_think else OPEN
            idx = self.pending.find(tag)
            if idx == -1:
                keep = _partial_suffix(self.pending, tag)
                chunk = self.pending[:-keep] if keep else self.pending
                self.pending = self.pending[-keep:] if keep else ""
                event = self._emit(chunk)
                if event:
                    emitted.append(event)
                break
            before = self.pending[:idx]
            event = self._emit(before)
            if event:
                emitted.append(event)
            self.pending = self.pending[idx + len(tag) :]
            self.in_think = not self.in_think
        return emitted

    def finish(self) -> list[tuple[str, str]]:
        if not self.pending:
            return []
        event = self._emit(self.pending)
        self.pending = ""
        return [event] if event else []

    def _emit(self, text: str) -> tuple[str, str] | None:
        if not text:
            return None
        if self.in_think:
            self.thinking += text
            return ("think", text)
        self.answer += text
        return ("answer", text)


def _partial_suffix(text: str, tag: str) -> int:
    limit = min(len(text), len(tag) - 1)
    for size in range(limit, 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0
