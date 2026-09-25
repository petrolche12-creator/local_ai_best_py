"""Загрузка GGUF и потоковая генерация через llama-cpp-python."""

from __future__ import annotations

import gc
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path

from localai.catalog import ModelSpec
from localai.prompting import drop_old_messages, model_messages, render_chat_prompt
from localai.recommend import Placement
from localai.think import ThinkFilter


class EngineError(RuntimeError):
    pass


@dataclass
class Generation:
    answer: str
    reasoning: str
    tokens: int
    seconds: float
    stopped_early: bool = False
    trimmed: int = 0
    hit_limit: bool = False

    @property
    def tokens_per_sec(self) -> float:
        if self.seconds <= 0:
            return 0.0
        return self.tokens / self.seconds


@dataclass
class Engine:
    model: ModelSpec
    path: Path
    placement: Placement
    n_threads: int
    verbose: bool = False
    llm: object | None = None
    template: str = ""
    _bos: bool = False
    _eos: str = "<|im_end|>"
    loaded_gpu_layers: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def n_ctx(self) -> int:
        return self.placement.n_ctx

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise EngineError(
                "Нет llama-cpp-python. Поставь зависимости:\n"
                "  pip install -r requirements.txt\n"
                "Для NVIDIA смотри README, раздел GPU."
            ) from exc
        if not self.path.is_file():
            raise EngineError(f"Файл модели не найден: {self.path}")
        layers = self.placement.n_gpu_layers
        kwargs = {
            "model_path": str(self.path),
            "n_ctx": self.placement.n_ctx,
            "n_gpu_layers": layers,
            "n_threads": self.n_threads,
            "n_threads_batch": self.n_threads,
            "n_batch": 128 if self.model.size_gb >= 3 or self.placement.device == "cpu" else 512,
            "use_mmap": True,
            "use_mlock": False,
            "verbose": self.verbose,
        }
        sig = inspect.signature(Llama.__init__)
        if "flash_attn" in sig.parameters and self.placement.device in {"gpu", "metal", "hybrid"}:
            kwargs["flash_attn"] = True
        filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
        try:
            self.llm = Llama(**filtered)
        except Exception as exc:
            if "flash_attn" not in filtered:
                raise EngineError(_human_load_error(exc)) from exc
            filtered.pop("flash_attn", None)
            try:
                self.llm = Llama(**filtered)
            except Exception as second:
                raise EngineError(_human_load_error(second)) from second
        self.loaded_gpu_layers = layers
        self.template = _metadata(self.llm, "tokenizer.chat_template")
        eos = _metadata(self.llm, "tokenizer.ggml.eos_token") or _metadata(self.llm, "tokenizer.eos_token")
        if eos:
            self._eos = eos

    def close(self) -> None:
        llm = self.llm
        self.llm = None
        if llm is not None:
            closer = getattr(llm, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        gc.collect()

    def generate(
        self,
        *,
        system_prompt: str,
        history: list[dict[str, str]],
        enable_thinking: bool,
        temperature: float,
        top_p: float,
        max_tokens: int,
        on_token,
    ) -> Generation:
        if self.llm is None:
            raise EngineError("Модель не загружена.")
        thinking = enable_thinking and self.model.thinking
        messages = model_messages(system_prompt, history, thinking_model=self.model.thinking)
        messages, trimmed = self._fit(messages, max_tokens, thinking)
        started = time.perf_counter()
        raw_parts: list[str] = []
        stopped = False
        hit_limit = False
        try:
            for piece, reason in self._stream(messages, thinking, temperature, top_p, max_tokens):
                if piece:
                    raw_parts.append(piece)
                    on_token(piece)
                if reason == "length":
                    hit_limit = True
                if reason == "stop":
                    break
        except KeyboardInterrupt:
            stopped = True
        seconds = max(time.perf_counter() - started, 1e-6)
        raw = "".join(raw_parts)
        filt = ThinkFilter(show_thinking=True)
        filt.feed(raw)
        filt.finish()
        answer = filt.answer.strip()
        reasoning = filt.thinking.strip()
        tokens = self._count(raw) or max(1, len(raw_parts))
        return Generation(
            answer=answer,
            reasoning=reasoning,
            tokens=tokens,
            seconds=seconds,
            stopped_early=stopped,
            trimmed=trimmed,
            hit_limit=hit_limit,
        )

    def _fit(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        thinking: bool,
    ) -> tuple[list[dict[str, str]], int]:
        budget = max(128, self.n_ctx - max_tokens - 32)

        def fits(batch: list[dict[str, str]]) -> bool:
            prompt = self._render(batch, thinking)
            if prompt is None:
                # Грубая оценка, если шаблон не разобрался.
                chars = sum(len(item.get("content", "")) for item in batch)
                return chars / 2 <= budget
            return self._count(prompt) <= budget

        fitted, dropped = drop_old_messages(messages, fits)
        if fitted and not fits(fitted) and fitted[-1].get("role") != "system":
            last = dict(fitted[-1])
            content = last.get("content", "")
            # Режем длинный последний запрос, system не трогаем.
            keep = max(200, int(len(content) * 0.5))
            while keep > 200 and not fits(fitted[:-1] + [{**last, "content": content[:keep]}]):
                keep = int(keep * 0.7)
            last["content"] = content[:keep]
            fitted = fitted[:-1] + [last]
        return fitted, dropped

    def _render(self, messages: list[dict[str, str]], thinking: bool) -> str | None:
        if not self.template:
            return None
        return render_chat_prompt(self.template, messages, enable_thinking=thinking)

    def _stream(self, messages, thinking: bool, temperature: float, top_p: float, max_tokens: int):
        prompt = self._render(messages, thinking)
        if prompt:
            yield from self._stream_completion(prompt, temperature, top_p, max_tokens)
            return
        yield from self._stream_chat(messages, thinking, temperature, top_p, max_tokens)

    def _stream_completion(self, prompt: str, temperature: float, top_p: float, max_tokens: int):
        assert self.llm is not None
        kwargs = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stream": True,
            "stop": [self._eos, "<|im_end|>", "<|endoftext|>"],
        }
        fn = self.llm.create_completion
        kwargs = _accepted(fn, kwargs)
        for chunk in fn(**kwargs):
            choice = (chunk.get("choices") or [{}])[0]
            text = choice.get("text") or ""
            reason = choice.get("finish_reason")
            yield text, reason

    def _stream_chat(self, messages, thinking: bool, temperature: float, top_p: float, max_tokens: int):
        assert self.llm is not None
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "stream": True,
        }
        if thinking is not None:
            kwargs["chat_template_kwargs"] = {"enable_thinking": thinking}
        fn = self.llm.create_chat_completion
        try:
            iterator = fn(**_accepted(fn, kwargs))
        except TypeError:
            kwargs.pop("chat_template_kwargs", None)
            iterator = fn(**_accepted(fn, kwargs))
        for chunk in iterator:
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content") or choice.get("text") or ""
            reason = choice.get("finish_reason")
            yield text, reason

    def _count(self, text: str) -> int:
        if not text or self.llm is None:
            return 0
        tokenize = getattr(self.llm, "tokenize", None)
        if not callable(tokenize):
            return 0
        try:
            return len(tokenize(text.encode("utf-8"), add_bos=False))
        except TypeError:
            try:
                return len(tokenize(text.encode("utf-8")))
            except Exception:
                return 0
        except Exception:
            return 0


def llama_gpu_offload() -> bool | None:
    try:
        import llama_cpp
    except ImportError:
        return None
    fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if not callable(fn):
        return None
    try:
        return bool(fn())
    except Exception:
        return None


def install_hint(vendor: str | None) -> str:
    base = "pip install -r requirements.txt"
    if vendor == "nvidia":
        return (
            "Сборка llama-cpp-python без GPU. Модель уйдёт на процессор.\n"
            "Чтобы использовать видеокарту NVIDIA:\n"
            "  pip install llama-cpp-python --force-reinstall --no-cache-dir "
            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124\n"
            "Если колесо не встало — смотри README."
        )
    if vendor == "apple":
        return (
            "На Mac обычно хватает обычной установки, Metal подхватывается сам:\n"
            f"  {base}\n"
            "Если GPU не виден, пересобери с CMAKE_ARGS=\"-DGGML_METAL=on\"."
        )
    return f"Ставится так:\n  {base}"


def _metadata(llm: object, key: str) -> str:
    meta = getattr(llm, "metadata", None)
    if not isinstance(meta, dict):
        return ""
    value = meta.get(key)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _accepted(fn, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in sig.parameters}


def _human_load_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "unknown model architecture" in low or "architecture" in low and "not supported" in low:
        return (
            "Эта сборка llama.cpp не знает архитектуру модели. Обнови пакет:\n"
            "  pip install -U llama-cpp-python"
        )
    if "out of memory" in low or "cuda" in low and "alloc" in low or "failed to load" in low:
        return (
            "Не хватило памяти, чтобы поднять модель. Возьми модель меньше "
            "или уменьши контекст в настройках.\n"
            f"({text})"
        )
    return f"Модель не загрузилась: {text}"
