"""Цикл одного чата."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from localai.agent import with_tools
from localai.config import AppPaths, Settings, save_settings
from localai.rag import context_for, used_gb
from localai.engine import Engine, Generation
from localai.history import Chat, export_markdown, retitle, save_chat
from localai.prefs import smart_label, speed_label
from localai.prompts import PRESETS, normalize_prompt, preset_by_id
from localai.recommend import mode_label
from localai.think import ThinkFilter
from localai.tools import is_dangerous, parse_tool_calls, run_tool, strip_tool_calls
from localai.ui import HELP, ask_yes, emit_status, fmt_speed, write_out


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
    agent = "агент авто" if settings.agent and settings.agent_auto else "агент вкл" if settings.agent else "агент выкл"
    prefs = ""
    if settings.speed and settings.smart:
        prefs = f" · {speed_label(settings.speed)} · {smart_label(settings.smart)}"
    emit_status(
        console,
        f"{engine.model.display_name} · {mode_label(chat.mode)}{prefs} · ctx {engine.n_ctx} · "
        f"{'рассуждения вкл' if _thinking(settings, chat) else 'без рассуждений'} · {agent}",
    )
    emit_status(console, f"чат {path.name}. /help — команды. Ctrl+C на ответе — оборвать.")
    if settings.agent:
        emit_status(console, "умеет ходить по интернету и ПК, запускать программы и писать файлы. команды спрашивает.")
    emit_status(console, f"обучение: {paths.rag} · занято {used_gb(paths.rag):.2f} ГБ")
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
        from localai.study import looks_like_assignment, run_study

        if looks_like_assignment(stripped) and ask_yes(
            console,
            "это задание на обучение. сначала план и вопросы, потом поиск в RAG?",
            default=True,
        ):
            run_study(console, engine, settings, paths, topic=stripped)
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
            content = item.get("content", "")
            if str(content).startswith("<tool_result"):
                continue
            console.print(f"[bold cyan]ты[/] › {content}")
        elif role == "tool":
            preview = (item.get("content") or "").strip().splitlines()
            head = preview[0][:80] if preview else ""
            console.print(f"[dim]инструмент {item.get('name', '')}: {head}[/]")
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
    if settings.agent:
        _agent_loop(console, engine, chat, settings, paths)
        return
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
            system_prompt=_with_rag(chat.system_prompt, paths, text),
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


def _agent_loop(console: Console, engine: Engine, chat: Chat, settings: Settings, paths: AppPaths) -> None:
    last_user = next((item.get("content", "") for item in reversed(chat.messages) if item.get("role") == "user"), "")
    prompt = _with_rag(with_tools(chat.system_prompt, True), paths, str(last_user))
    workspace = Path(settings.workspace).expanduser() if settings.workspace else Path.cwd()
    for step in range(1, 9):
        result = _generate(console, engine, chat, settings, system_prompt=prompt)
        if result is None:
            if step == 1 and chat.messages and chat.messages[-1].get("role") == "user":
                chat.messages.pop()
            save_chat(paths.chats, chat)
            return
        calls = parse_tool_calls(result.answer) or parse_tool_calls(result.reasoning)
        visible = strip_tool_calls(result.answer)
        if not calls:
            if not visible and result.reasoning:
                visible = "(модель так и не вышла из рассуждений — увеличь /tokens)"
                console.print(f"[bold green]ассистент[/] › {visible}")
            chat.messages.append(
                {"role": "assistant", "content": visible, "reasoning": result.reasoning}
            )
            save_chat(paths.chats, chat)
            _stats(console, result)
            return
        chat.messages.append(
            {
                "role": "assistant",
                "content": visible or "(вызываю инструменты)",
                "reasoning": result.reasoning,
            }
        )
        for call in calls:
            console.print(f"[bold yellow]инструмент[/] {call.name} {_short_args(call.arguments)}")
            output = run_tool(
                call,
                cwd=workspace,
                confirm=lambda kind, detail, dangerous: _confirm_tool(
                    console, settings, kind, detail, dangerous
                ),
                rag_root=paths.rag,
                rag_max_gb=settings.rag_max_gb,
            )
            preview = output.strip().splitlines()
            shown = "\n".join(preview[:12])
            if len(preview) > 12:
                shown += "\n…"
            console.print(f"[dim]{shown}[/]")
            chat.messages.append({"role": "tool", "name": call.name, "content": output})
        save_chat(paths.chats, chat)
        emit_status(console, f"шаг {step}, смотрю результат и продолжаю")
    emit_status(console, "лимит шагов агента. напиши, что делать дальше")
    save_chat(paths.chats, chat)


def _generate(console, engine, chat, settings, system_prompt: str) -> Generation | None:
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
            system_prompt=system_prompt,
            history=chat.messages,
            enable_thinking=thinking,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=max(settings.max_tokens, 512),
            on_token=on_token,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]оборвал[/]")
        return None
    except Exception as exc:
        console.print(f"\n[red]генерация сломалась:[/] {exc}")
        return None
    for kind, chunk in filt.finish():
        if kind == "answer" and chunk and not started_answer:
            console.print("[bold green]ассистент[/] › ", end="")
            write_out(console, chunk)
    console.print()
    return result


def _confirm_tool(console: Console, settings: Settings, kind: str, detail: str, dangerous: bool) -> bool:
    if dangerous:
        console.print("[red]это опасная команда[/]")
        return ask_yes(console, "точно выполнить?", default=False)
    if settings.agent_auto and kind in {"shell", "launch"} and not is_dangerous(detail):
        return True
    if settings.agent_auto and kind == "write":
        return True
    if kind == "launch":
        return ask_yes(console, "запустить программу?", default=False)
    question = "выполнить?" if kind == "shell" else "записать?"
    return ask_yes(console, question, default=False)


def _short_args(arguments: dict) -> str:
    if not arguments:
        return ""
    if "command" in arguments:
        return str(arguments["command"])[:120]
    if "query" in arguments:
        return str(arguments["query"])[:120]
    if "url" in arguments:
        return str(arguments["url"])[:120]
    if "path" in arguments:
        return str(arguments["path"])[:120]
    return ""


def _stats(console: Console, result: Generation) -> None:
    bits = [fmt_speed(result.tokens_per_sec), f"{result.tokens} ток", f"{result.seconds:.1f} с"]
    if result.trimmed:
        bits.append(f"обрезал {result.trimmed} старых реплик")
    if result.hit_limit:
        bits.append("упёрся в лимит токенов")
    if result.stopped_early:
        bits.append("оборвано")
    emit_status(console, " · ".join(bits))


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
    if cmd == "/agent":
        settings.agent = not settings.agent
        save_settings(paths, settings)
        emit_status(console, "агент " + ("вкл" if settings.agent else "выкл"))
        return None
    if cmd == "/auto":
        settings.agent_auto = not settings.agent_auto
        settings.agent = True
        save_settings(paths, settings)
        emit_status(
            console,
            "автоподтверждение " + ("вкл" if settings.agent_auto else "выкл") + ". опасные команды всё равно спрошу",
        )
        return None
    if cmd == "/learn":
        from localai.study import run_study

        run_study(console, engine, settings, paths, topic=arg.strip() or None)
        return None
    if cmd == "/rag":
        emit_status(console, f"{paths.rag} · {used_gb(paths.rag):.2f} ГБ из {settings.rag_max_gb:g}")
        return None
    if cmd == "/limits":
        _edit_limits(console, settings, paths, arg)
        return None
    if cmd == "/cd":
        if not arg.strip():
            emit_status(console, f"папка агента: {settings.workspace or Path.cwd()}")
            return None
        target = Path(arg).expanduser()
        if not target.is_dir():
            emit_status(console, f"нет такой папки: {target}")
            return None
        settings.workspace = str(target.resolve())
        save_settings(paths, settings)
        emit_status(console, f"теперь работаю в {settings.workspace}")
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


def _with_rag(prompt: str, paths: AppPaths, query: str) -> str:
    extra = context_for(paths.rag, query)
    if not extra:
        return prompt
    return prompt.rstrip() + "\n\n" + extra


def _edit_limits(console: Console, settings: Settings, paths: AppPaths, arg: str) -> None:
    from localai.rag import normalize_limits

    if arg.strip():
        parts = arg.replace(",", " ").split()
        if len(parts) < 4:
            emit_status(console, "формат: /limits 1 20 normal 2  — ГБ, минуты, скорость, глубина")
            return None
        speed = {"1": "fast", "2": "normal", "3": "slow"}.get(parts[2], parts[2])
        try:
            limits = normalize_limits(float(parts[0]), int(parts[1]), speed, int(parts[3]))
        except ValueError:
            emit_status(console, "не разобрал числа")
            return None
    else:
        from localai.study import _ask_limits

        limits = _ask_limits(console, settings)
    settings.rag_max_gb = limits.max_gb
    settings.rag_minutes = limits.minutes
    settings.rag_speed = limits.speed
    settings.rag_depth = limits.depth
    save_settings(paths, settings)
    emit_status(console, f"лимиты обучения: {limits.label()}")
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
