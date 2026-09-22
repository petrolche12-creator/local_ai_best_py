"""Цикл одного чата."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from localai.config import AppPaths, Settings, save_settings
from localai.engine import Engine, Generation
from localai.history import Chat, export_markdown, retitle, save_chat
from localai.prompts import PRESETS, normalize_prompt, preset_by_id
from localai.recommend import mode_label
from localai.think import ThinkFilter
from localai.ui import HELP, emit_status, fmt_speed, write_out


def run_chat(
    console: Console,
    engine: Engine,
    chat: Chat,
    settings: Settings,
    paths: AppPaths,
) -> str:
    """Возвращает menu или exit."""
    path = save_chat(paths.chats, chat)
    settings.last_chat_id = chat.id
    save_settings(paths, settings)
    console.print()
    console.rule(f"[bold]{chat.title}[/]")
    emit_status(
        console,
        f"{engine.model.display_name} · {mode_label(chat.mode)} · ctx {engine.n_ctx} · "
        f"{'рассуждения вкл' if _thinking(settings, chat) else 'без рассуждений'} · "
        f"чат {path.name}",
    )
    emit_status(console, "пиши сообщение. /help — команды. Ctrl+C на ответе — оборвать.")
    _replay(console, chat, settings)
    while True:
        try:
            text = _read(console, engine, chat, settings)
        except EOFError:
            console.print()
            save_chat(paths.chats, chat)
            return "menu"
        except KeyboardInterrupt:
            console.print()
            save_chat(paths.chats, chat)
            return "menu"
        if text is None:
            continue
        stripped = text.strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            action = _command(console, stripped, engine, chat, settings, paths)
            if action in {"menu", "exit", "new"}:
                save_chat(paths.chats, chat)
                return action
            continue
        _turn(console, engine, chat, settings, paths, stripped)


def _thinking(settings: Settings, chat: Chat) -> bool:
    if settings.show_thinking is not None:
        return settings.show_thinking and chat.thinking_model
    return chat.mode == "smart" and chat.thinking_model


def _show_thinking(settings: Settings) -> bool:
    return bool(settings.show_thinking)


def _replay(console: Console, chat: Chat, settings: Settings) -> None:
    if not chat.messages:
        return
    console.print("[dim]— прошлые реплики —[/]")
    for item in chat.messages[-8:]:
        role = item.get("role")
        if role == "user":
            console.print(f"[bold cyan]ты[/] › {item.get('content', '')}")
        elif role == "assistant":
            console.print(f"[bold green]ассистент[/] › {item.get('content', '')}")
            if _show_thinking(settings) and item.get("reasoning"):
                console.print(f"[dim]{item['reasoning']}[/]")
    console.print("[dim]— дальше —[/]")


def _read(console: Console, engine: Engine, chat: Chat, settings: Settings) -> str | None:
    toolbar = (
        f"{engine.model.name} · {mode_label(chat.mode)} · "
        f"t={settings.temperature:g} · /help"
    )
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return console.input("[bold cyan]ты[/] › ")
    try:
        session = PromptSession(history=FileHistory(str(paths_history())))
        return session.prompt("ты › ", bottom_toolbar=toolbar)
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception:
        return console.input("[bold cyan]ты[/] › ")


def paths_history() -> Path:
    from localai.config import data_root

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "input_history"


def _turn(
    console: Console,
    engine: Engine,
    chat: Chat,
    settings: Settings,
    paths: AppPaths,
    text: str,
) -> None:
    if chat.user_turns == 0 and chat.title == "Новый чат":
        retitle(chat, text)
    chat.messages.append({"role": "user", "content": text})
    show = _show_thinking(settings)
    thinking = _thinking(settings, chat)
    filt = ThinkFilter(show_thinking=show)
    started_answer = False
    announced = False

    def on_token(piece: str) -> None:
        nonlocal started_answer, announced
        for kind, chunk in filt.feed(piece):
            if kind == "think":
                if not announced and thinking:
                    console.print("[dim]думаю…[/]")
                    announced = True
                if show:
                    write_out(console, chunk, dim=True)
            else:
                if not started_answer:
                    if announced or show:
                        console.print()
                    console.print("[bold green]ассистент[/] › ", end="")
                    started_answer = True
                    chunk = chunk.lstrip("\n")
                    if not chunk:
                        continue
                write_out(console, chunk)

    console.print()
    try:
        result = engine.generate(
            system_prompt=chat.system_prompt,
            history=chat.messages,
            enable_thinking=thinking,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=settings.max_tokens,
            on_token=on_token,
        )
    except Exception as exc:
        console.print(f"\n[red]генерация сломалась:[/] {exc}")
        chat.messages.pop()
        return
    for kind, chunk in filt.finish():
        if kind == "answer" and chunk:
            if not started_answer:
                console.print("[bold green]ассистент[/] › ", end="")
                started_answer = True
            write_out(console, chunk)
    console.print()
    _finish_turn(console, chat, result, paths)


def _finish_turn(console: Console, chat: Chat, result: Generation, paths: AppPaths) -> None:
    answer = result.answer.strip()
    if result.stopped_early and not answer and not result.reasoning:
        chat.messages.pop()
        emit_status(console, "оборвал, реплику не сохранил")
        return
    if not answer and result.reasoning:
        answer = "(модель так и не вышла из рассуждений — увеличь /tokens)"
        console.print(f"[bold green]ассистент[/] › {answer}")
    chat.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "reasoning": result.reasoning,
        }
    )
    save_chat(paths.chats, chat)
    bits = [fmt_speed(result.tokens_per_sec), f"{result.tokens} ток", f"{result.seconds:.1f} с"]
    if result.trimmed:
        bits.append(f"обрезал {result.trimmed} старых реплик")
    if result.hit_limit:
        bits.append("упёрся в лимит токенов")
    if result.stopped_early:
        bits.append("оборвано")
    emit_status(console, " · ".join(bits))


def _command(
    console: Console,
    raw: str,
    engine: Engine,
    chat: Chat,
    settings: Settings,
    paths: AppPaths,
) -> str | None:
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    if cmd in {"/exit", "/quit", "/q"}:
        return "menu"
    if cmd == "/help":
        console.print(HELP)
        return None
    if cmd == "/new":
        return "new"
    if cmd == "/clear":
        chat.messages.clear()
        emit_status(console, "историю реплик стёр, промпт на месте")
        return None
    if cmd == "/save":
        path = save_chat(paths.chats, chat)
        emit_status(console, f"сохранил {path}")
        return None
    if cmd == "/title":
        if not arg.strip():
            emit_status(console, "напиши: /title новое имя")
            return None
        retitle(chat, arg)
        emit_status(console, f"теперь это «{chat.title}»")
        return None
    if cmd == "/model":
        console.print(
            f"{engine.model.display_name}\n"
            f"{engine.model.repo_id}\n"
            f"{engine.path}\n"
            f"устройство: {engine.placement.device}, слои GPU: {engine.placement.n_gpu_layers}, "
            f"контекст: {engine.n_ctx}"
        )
        return None
    if cmd == "/think":
        settings.show_thinking = not _show_thinking(settings)
        save_settings(paths, settings)
        emit_status(console, "показ рассуждений " + ("вкл" if settings.show_thinking else "выкл"))
        return None
    if cmd == "/temp":
        try:
            settings.temperature = min(2.0, max(0.0, float(arg.replace(",", "."))))
        except ValueError:
            emit_status(console, "напиши число, например /temp 0.7")
            return None
        save_settings(paths, settings)
        emit_status(console, f"температура {settings.temperature:g}")
        return None
    if cmd == "/tokens":
        try:
            settings.max_tokens = min(8192, max(32, int(arg)))
        except ValueError:
            emit_status(console, "напиши число, например /tokens 1024")
            return None
        save_settings(paths, settings)
        emit_status(console, f"лимит ответа {settings.max_tokens} токенов")
        return None
    if cmd == "/export":
        dest = paths.exports / f"{chat.id}.md"
        export_markdown(chat, dest, include_thinking=bool(arg.strip() == "full" or _show_thinking(settings)))
        emit_status(console, f"экспорт {dest}")
        return None
    if cmd == "/prompt":
        _edit_prompt(console, chat, settings, paths, arg)
        return None
    if cmd == "/prompts":
        _pick_preset(console, chat, settings, paths)
        return None
    if cmd == "/history":
        from localai.history import list_chats

        chats = list_chats(paths.chats)
        if not chats:
            emit_status(console, "чатов пока нет")
            return None
        for index, item in enumerate(chats[:15], start=1):
            mark = " ←" if item.id == chat.id else ""
            console.print(f"  {index:>2}  {item.title}  [dim]{item.updated_at} · {item.user_turns} репл.{mark}[/]")
        return None
    if cmd == "/hw":
        emit_status(console, "железо и смена модели — в главном меню, /exit")
        return None
    if cmd == "/paste":
        console.print("[dim]вставляй текст, в конце строка EOF[/]")
        lines: list[str] = []
        while True:
            try:
                line = console.input()
            except EOFError:
                break
            if line.strip() == "EOF":
                break
            lines.append(line)
        pasted = "\n".join(lines).strip()
        if pasted:
            _turn(console, engine, chat, settings, paths, pasted)
        return None
    emit_status(console, "не знаю такую команду. /help")
    return None


def _edit_prompt(console: Console, chat: Chat, settings: Settings, paths: AppPaths, arg: str) -> None:
    console.print(PanelText(chat.system_prompt))
    if arg.strip():
        _apply_prompt(console, chat, settings, paths, arg.strip())
        return
    console.print("[dim]1 оставить   2 пресет   3 свой текст (пустая строка — конец)   4 сброс[/]")
    choice = console.input("промпт › ").strip()
    if choice in {"", "1"}:
        return
    if choice == "2":
        _pick_preset(console, chat, settings, paths)
        return
    if choice == "4":
        from localai.prompts import DEFAULT_PROMPT

        _apply_prompt(console, chat, settings, paths, DEFAULT_PROMPT)
        return
    if choice == "3":
        console.print("[dim]пиши промпт, закончи пустой строкой[/]")
        lines: list[str] = []
        while True:
            line = console.input()
            if not line.strip():
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if text:
            _apply_prompt(console, chat, settings, paths, text)
        return
    emit_status(console, "не понял. оставил как было")


def _pick_preset(console: Console, chat: Chat, settings: Settings, paths: AppPaths) -> None:
    for index, preset in enumerate(PRESETS, start=1):
        console.print(f"  {index}  {preset.title}")
    raw = console.input("пресет › ").strip()
    preset = None
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(PRESETS):
            preset = PRESETS[idx]
    else:
        preset = preset_by_id(raw)
    if preset is None:
        emit_status(console, "такого пресета нет")
        return
    _apply_prompt(console, chat, settings, paths, preset.text)
    emit_status(console, f"пресет «{preset.title}»")


def _apply_prompt(console: Console, chat: Chat, settings: Settings, paths: AppPaths, text: str) -> None:
    chat.system_prompt = normalize_prompt(text)
    save_chat(paths.chats, chat)
    also = console.input("сделать промптом по умолчанию для новых чатов? [Y/n] ").strip().lower()
    if also in {"", "y", "yes", "д", "да"}:
        settings.system_prompt = chat.system_prompt
        save_settings(paths, settings)
    emit_status(console, "промпт обновил. старые реплики на месте, дальше модель видит новый.")


class PanelText:
    """Чтобы не тянуть rich.panel в каждую печать промпта через console.print(str)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __rich__(self):
        from rich.panel import Panel

        return Panel(self.text or "[dim]пусто[/]", title="системный промпт", border_style="yellow")
