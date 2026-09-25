"""Меню, мастер выбора и запуск."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from localai import __version__
from localai.catalog import CATALOG, ModelSpec, get_model
from localai.chat import run_chat
from localai.config import (
    Settings,
    ensure_dirs,
    hf_token,
    load_settings,
    resolve_paths,
    save_settings,
)
from localai.downloader import (
    DownloadError,
    download_model,
    downloaded_files,
    is_downloaded,
    list_repo_gguf,
    local_model_path,
    refresh_size,
    spec_from_repo,
)
from localai.engine import Engine, EngineError, install_hint, llama_gpu_offload
from localai.hardware import HardwareProfile, detect
from localai.hfsearch import SearchResult, search_huggingface
from localai.history import Chat, delete_chat, export_markdown, find_chat, list_chats, new_chat
from localai.prefs import comfort_text, mode_from_prefs, prefs_from_mode, smart_label, speed_label
from localai.prompts import DEFAULT_PROMPT, PRESETS, normalize_prompt
from localai.recommend import (
    Recommendation,
    choose_placement,
    default_max_tokens,
    mode_label,
)
from localai.ui import (
    ask_choice,
    ask_line,
    ask_yes,
    banner,
    emit_status,
    fmt_gb,
    hardware_panel,
    make_console,
    model_table,
)


MODES = {
    "1": "fast",
    "2": "smart",
    "3": "balance",
    "fast": "fast",
    "smart": "smart",
    "balance": "balance",
    "быстрый": "fast",
    "тупой": "fast",
    "умный": "smart",
    "середина": "balance",
}


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.console = make_console(args.no_color)
        self.settings = Settings()
        self.paths = resolve_paths(self.settings)
        self.hw: HardwareProfile | None = None
        self.engine: Engine | None = None
        self.offload = False
        self.model: ModelSpec | None = None
        self.placement = None

    def run(self) -> int:
        ensure_dirs(self.paths)
        self.settings = load_settings(self.paths)
        self._apply_cli_overrides()
        self.paths = resolve_paths(self.settings)
        ensure_dirs(self.paths)
        banner(self.console)
        self.console.print(f"[dim]папка обучения RAG: {self.paths.rag}[/]")
        self.scan()
        if self.hw is None:
            return 1
        if self.args.scan:
            self._scan_search()
            return 0
        if not self.prepare_model():
            return 1
        if self.args.chat:
            chat = find_chat(self.paths.chats, self.args.chat)
            if chat is None:
                self.console.print(f"[red]чат не найден:[/] {self.args.chat}")
                return 1
            self._chat_loop(chat)
            return 0
        if self.args.yes and (self.args.mode or self.args.speed or self.args.smart):
            self._new_chat()
            return 0
        self.menu()
        return 0

    def scan(self) -> None:
        self.console.print("[dim]смотрю процессор, память, видео, диск и питание…[/]")
        disk = str(self.paths.models)
        self.hw = detect(disk)
        hardware_panel(self.console, self.hw)
        offload = self._offload_assumed()
        if llama_gpu_offload() is False and (self.hw.vram_gb > 0 or self.hw.apple_silicon):
            vendor = "nvidia" if self.hw.vram_gb > 0 else "apple"
            self.console.print(PanelHint(install_hint(vendor)))
            offload = False
        elif llama_gpu_offload() is None and (self.hw.vram_gb > 0 or self.hw.apple_silicon):
            self.console.print(
                "[dim]llama-cpp ещё не стоит. Считаю, что видеокарту получится использовать. "
                "Если сборка окажется без GPU — при запуске пересчитаю на процессор.[/]"
            )
        self.offload = offload
        self.console.print(f"[dim]{comfort_text(self.hw, offload)}. модель пока не выбираю.[/]")

    def prepare_model(self) -> bool:
        assert self.hw is not None
        if self.args.model:
            model = get_model(self.args.model)
            if model is None:
                self.console.print(f"[red]нет такой модели в каталоге:[/] {self.args.model}")
                self.console.print("id смотри в каталоге или укажи репозиторий в мастере")
                return False
            mode = self.settings.mode or "balance"
            return self._commit(model, mode)
        if self.settings.model_id and not self.args.fresh and not (self.args.speed or self.args.smart):
            previous = self._saved_model()
            if previous is not None:
                reuse = self.args.yes or ask_yes(
                    self.console,
                    f"прошлый раз: {previous.display_name}, {self._prefs_text()}. запустить его?",
                    default=True,
                )
                if reuse:
                    return self._commit(previous, self.settings.mode or "balance")
        speed, smart = self._prefs_from_args()
        cli_prefs = bool(self.args.speed or self.args.smart or self.args.mode)
        if cli_prefs and speed and smart:
            self._remember_prefs(speed, smart)
            return self._search_and_commit(speed, smart)
        return self.wizard()

    def wizard(self) -> bool:
        speed = self._ask_speed()
        if not speed:
            return False
        smart = self._ask_smart()
        if not smart:
            return False
        self._remember_prefs(speed, smart)
        return self._search_and_commit(speed, smart)

    def _ask_speed(self) -> str | None:
        self.console.print()
        self.console.print("[bold]насколько быстро должно отвечать?[/]")
        self.console.print("  [bold]1[/]  очень быстро     [dim]даже если модель проще[/]")
        self.console.print("  [bold]2[/]  нормально        [dim]можно подождать пару секунд[/]")
        self.console.print("  [bold]3[/]  можно медленно   [dim]скорость не важна[/]")
        while True:
            raw = ask_choice(self.console, "скорость ›")
            if raw is None or raw in {"0", "q", "выход"}:
                return None
            speed = _parse_speed(raw)
            if speed:
                return speed
            self.console.print("[yellow]жми 1, 2 или 3[/]")

    def _ask_smart(self) -> str | None:
        self.console.print()
        self.console.print("[bold]насколько умным должен быть?[/]")
        self.console.print("  [bold]1[/]  попроще             [dim]лишь бы по делу отвечал[/]")
        self.console.print("  [bold]2[/]  нормальный ум      [dim]обычный помощник[/]")
        self.console.print("  [bold]3[/]  максимально умный  [dim]пусть думает дольше, но глубже[/]")
        while True:
            raw = ask_choice(self.console, "ум ›")
            if raw is None or raw in {"0", "q", "выход"}:
                return None
            smart = _parse_smart(raw)
            if smart:
                return smart
            self.console.print("[yellow]жми 1, 2 или 3[/]")

    def _search_and_commit(self, speed: str, smart: str) -> bool:
        assert self.hw is not None
        self.console.print()
        self.console.print(
            f"[bold]ищу по всему Hugging Face[/] под {speed_label(speed)} и {smart_label(smart)}…"
        )
        result = search_huggingface(
            self.hw,
            speed,
            smart,
            gpu_offload=self.offload,
            token=hf_token(self.settings),
            progress=lambda text: self.console.print(f"[dim]  {text}[/]"),
        )
        self._show_search(result)
        if result.best is None:
            self.console.print("[yellow]не нашёл модель, которую можно запустить.[/]")
            return False
        if self.args.yes or ask_yes(self.console, "скачать и запустить лучший вариант?", default=True):
            if self._commit(result.best.model, result.band.mode):
                return True
        return self._pick_other(result)

    def _show_search(self, result: SearchResult) -> None:
        if result.queries:
            self.console.print("[dim]запросы: " + " · ".join(result.queries) + "[/]")
        if result.note:
            self.console.print(f"[dim]{result.note}[/]")
        if result.best is None:
            return
        best = result.best
        where = best.placement.device if best.placement else "не влезла"
        ctx = best.placement.n_ctx if best.placement else 0
        self.console.print()
        self.console.print(f"[bold green]лучший:[/] {best.model.display_name}")
        self.console.print(f"  {best.model.repo_id}")
        self.console.print(f"  {best.model.filename}")
        self.console.print(f"  {fmt_gb(best.model.size_gb)} · {where} · ctx {ctx} · {best.model.hub_url}")
        self.console.print(f"  {best.reason}")
        if result.alternatives:
            self.console.print("[dim]ещё рядом:[/]")
            for index, item in enumerate(result.alternatives, start=2):
                self.console.print(f"  {index}  {item.model.display_name}  {fmt_gb(item.model.size_gb)}  {item.model.repo_id}")

    def _pick_other(self, result: SearchResult) -> bool:
        self.console.print("  [bold]2–4[/]  другой из списка, если есть")
        self.console.print("  [bold]m[/]    каталог")
        self.console.print("  [bold]r[/]    свой репозиторий Hugging Face")
        self.console.print("  [bold]0[/]    отмена")
        raw = ask_choice(self.console, "вместо лучшего ›")
        if not raw or raw in {"0", "q"}:
            return False
        if raw in {"m", "каталог"}:
            picked = self.pick_from_catalog()
            return bool(picked and self._commit(picked, result.band.mode))
        if raw in {"r", "5", "repo", "hf"}:
            picked = self.pick_custom()
            return bool(picked and self._commit(picked, result.band.mode))
        if raw.isdigit():
            index = int(raw) - 2
            if 0 <= index < len(result.alternatives):
                return self._commit(result.alternatives[index].model, result.band.mode)
        self.console.print("[yellow]не понял[/]")
        return False

    def _scan_search(self) -> None:
        speed, smart = self._prefs_from_args()
        explicit = bool(self.args.speed or self.args.smart or self.args.mode)
        if not explicit or not speed or not smart:
            self.console.print("[dim]модель не выбирал. без --scan спрошу скорость, потом ум, и только тогда поищу.[/]")
            return
        assert self.hw is not None
        result = search_huggingface(
            self.hw,
            speed,
            smart,
            gpu_offload=self.offload,
            token=hf_token(self.settings),
            progress=lambda text: self.console.print(f"[dim]  {text}[/]"),
        )
        self._show_search(result)

    def _prefs_from_args(self) -> tuple[str | None, str | None]:
        speed = _parse_speed(self.args.speed) if self.args.speed else self.settings.speed
        smart = _parse_smart(self.args.smart) if self.args.smart else self.settings.smart
        if self.args.mode and not (self.args.speed and self.args.smart):
            mapped_speed, mapped_smart = prefs_from_mode(self.args.mode)
            speed = speed or mapped_speed
            smart = smart or mapped_smart
        return speed, smart

    def _remember_prefs(self, speed: str, smart: str) -> None:
        self.settings.speed = speed
        self.settings.smart = smart
        self.settings.mode = mode_from_prefs(speed, smart)
        save_settings(self.paths, self.settings)

    def _prefs_text(self) -> str:
        if self.settings.speed and self.settings.smart:
            return f"{speed_label(self.settings.speed)}, {smart_label(self.settings.smart)}"
        return mode_label(self.settings.mode or "balance")

    def menu(self) -> None:
        while True:
            self.console.print()
            self.console.print("[bold]что делаем?[/]")
            self.console.print("  [bold]1[/]  новый чат")
            self.console.print("  [bold]2[/]  продолжить последний")
            self.console.print("  [bold]3[/]  история чатов")
            self.console.print("  [bold]4[/]  сменить модель / режим")
            self.console.print("  [bold]5[/]  системный промпт")
            self.console.print("  [bold]6[/]  железо")
            self.console.print("  [bold]7[/]  настройки")
            self.console.print("  [bold]8[/]  скачанные модели")
            self.console.print("  [bold]9[/]  обучение в RAG")
            self.console.print("  [bold]0[/]  выход")
            raw = ask_choice(self.console, "меню ›")
            if raw is None or raw in {"0", "q", "exit", "выход"}:
                self.shutdown()
                self.console.print("пока.")
                return
            if raw == "1":
                self._new_chat()
            elif raw == "2":
                self._continue_last()
            elif raw == "3":
                self._history_menu()
            elif raw == "4":
                self.shutdown()
                if not self.wizard():
                    self.console.print("[yellow]модель не сменил[/]")
            elif raw == "5":
                self._prompt_menu()
            elif raw == "6":
                self.scan()
            elif raw == "7":
                self._settings_menu()
            elif raw == "8":
                self._files_menu()
            elif raw == "9":
                self._learn_menu()
            else:
                self.console.print("[yellow]нет такого пункта[/]")

    def pick_from_catalog(self) -> ModelSpec | None:
        model_table(self.console)
        raw = ask_line(self.console, "номер или id (пусто — назад)")
        if not raw:
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(CATALOG):
                return CATALOG[index]
        found = get_model(raw)
        if found:
            return found
        self.console.print("[yellow]не нашёл[/]")
        return None

    def pick_custom(self) -> ModelSpec | None:
        repo = ask_line(self.console, "repo_id, например Qwen/Qwen3-4B-GGUF")
        if not repo or "/" not in repo:
            self.console.print("[yellow]нужен вид автор/имя[/]")
            return None
        token = hf_token(self.settings)
        try:
            files = list_repo_gguf(repo.strip(), token)
        except DownloadError as exc:
            self.console.print(f"[red]{exc}[/]")
            return None
        show = files[:20]
        for index, name in enumerate(show, start=1):
            self.console.print(f"  {index:>2}  {name}")
        if len(files) > len(show):
            self.console.print(f"[dim]ещё {len(files) - len(show)} файлов, введи имя целиком[/]")
        raw = ask_line(self.console, "номер файла", default="1")
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(show):
            filename = show[int(raw) - 1]
        elif raw in files:
            filename = raw
        else:
            self.console.print("[yellow]нет такого файла[/]")
            return None
        stub = spec_from_repo(repo.strip(), filename, size_gb=1.0)
        sized = refresh_size(stub, token)
        self.console.print(f"{sized.display_name} · {fmt_gb(sized.size_gb)} · {sized.repo_id}")
        return sized

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None

    def _commit(self, model: ModelSpec, mode: str, rec: Recommendation | None = None) -> bool:
        assert self.hw is not None
        token = hf_token(self.settings)
        self.console.print("[dim]сверяю размер на Hugging Face…[/]")
        model = refresh_size(model, token)
        offload = self._offload_assumed()
        placement = choose_placement(self.hw, model, mode, offload)
        if rec is not None and rec.model.size_estimated and not model.size_estimated:
            if abs(model.size_gb - rec.model.size_gb) > 0.4:
                self.console.print(
                    f"[dim]уточнил размер: {fmt_gb(rec.model.size_gb)} → {fmt_gb(model.size_gb)}[/]"
                )
        if placement is None:
            self.console.print(
                "[yellow]по прикидке модель не влезает в свободную память. "
                "можно всё равно попробовать — если упадёт, бери меньше.[/]"
            )
            if not ask_yes(self.console, "пробовать всё равно?", default=False):
                return False
            from localai.recommend import Placement

            placement = Placement(
                device="cpu",
                n_ctx=self.settings.n_ctx or 1024,
                n_gpu_layers=0,
                gpu_fraction=0.0,
                kv_gb=0.3,
                reasonable=False,
            )
        placement = self._apply_runtime_placement(placement)
        path = local_model_path(self.paths.models, model)
        if is_downloaded(self.paths.models, model):
            self.console.print(f"[green]уже скачана[/] {path}")
        else:
            need = model.size_gb + 0.4
            if self.hw.disk_free_gb < need:
                self.console.print(
                    f"[red]на диске мало места:[/] свободно {fmt_gb(self.hw.disk_free_gb)}, "
                    f"нужно около {fmt_gb(need)}"
                )
                return False
            self.console.print(
                f"скачаю [bold]{fmt_gb(model.size_gb)}[/] из {model.hub_url}"
            )
            if not self.args.yes and not ask_yes(self.console, "качать и запускать?", default=True):
                return False
            try:
                path = download_model(self.paths.models, model, token)
            except DownloadError as exc:
                self.console.print(f"[red]{exc}[/]")
                return False
            self.console.print(f"[green]скачал[/] {path}")
        return self._load(model, mode, path, placement)

    def _load(self, model: ModelSpec, mode: str, path: Path, placement) -> bool:
        assert self.hw is not None
        runtime = llama_gpu_offload()
        if runtime is False and placement.device in {"gpu", "hybrid", "metal"}:
            self.console.print("[yellow]сборка llama.cpp без GPU, пересчитываю на процессор[/]")
            cpu_place = choose_placement(self.hw, model, mode, gpu_offload=False)
            if cpu_place is None:
                self.console.print("[red]на CPU эта модель не влезает. возьми меньше или поставь GPU-сборку.[/]")
                self.console.print(install_hint("nvidia" if self.hw.vram_gb else "apple" if self.hw.apple_silicon else None))
                return False
            placement = cpu_place
        threads = self.settings.n_threads or _auto_threads(self.hw.cpu_physical)
        self.shutdown()
        self.console.print(
            f"[dim]гружу {model.display_name} в память. "
            "на большой модели это может занять минуту, это не зависон.[/]"
        )
        engine = Engine(
            model=model,
            path=path,
            placement=placement,
            n_threads=threads,
            verbose=self.args.verbose,
        )
        try:
            engine.load()
        except EngineError as exc:
            self.console.print(f"[red]{exc}[/]")
            return False
        self.engine = engine
        self.model = model
        self.placement = placement
        self.settings.mode = mode
        self.settings.model_id = model.id
        if get_model(model.id) is None:
            self.settings.custom_repo = model.repo_id
            self.settings.custom_filename = model.filename
            self.settings.custom_size_gb = model.size_gb
        if self.settings.max_tokens == 1024 and mode == "smart":
            self.settings.max_tokens = default_max_tokens(mode)
        if self.settings.max_tokens == 1024 and mode == "fast":
            self.settings.max_tokens = default_max_tokens(mode)
        save_settings(self.paths, self.settings)
        where = placement.device
        self.console.print(
            f"[green]готова[/] {model.display_name} · {where} · ctx {placement.n_ctx} · потоки {threads}"
        )
        return True

    def _new_chat(self) -> None:
        if self.engine is None or self.model is None or self.placement is None:
            self.console.print("[yellow]сначала выбери модель[/]")
            if not self.wizard():
                return
        assert self.engine is not None and self.model is not None
        self._chat_loop(self._make_chat())

    def _continue_last(self) -> None:
        chat = None
        if self.settings.last_chat_id:
            chat = find_chat(self.paths.chats, self.settings.last_chat_id)
        if chat is None:
            chats = list_chats(self.paths.chats)
            chat = chats[0] if chats else None
        if chat is None:
            self.console.print("[yellow]чатов ещё нет[/]")
            return
        self._ensure_chat_model(chat)
        self._chat_loop(chat)

    def _history_menu(self) -> None:
        chats = list_chats(self.paths.chats)
        if not chats:
            self.console.print("[yellow]чатов ещё нет[/]")
            return
        for index, chat in enumerate(chats[:30], start=1):
            self.console.print(
                f"  {index:>2}  {chat.title}  [dim]{chat.updated_at} · {chat.model_name} · {chat.user_turns} репл.[/]"
            )
        raw = ask_line(self.console, "номер, d номер — удалить, e номер — экспорт, пусто — назад")
        if not raw:
            return
        parts = raw.split()
        if parts[0] in {"d", "e"} and len(parts) == 2 and parts[1].isdigit():
            index = int(parts[1]) - 1
            if not 0 <= index < len(chats):
                self.console.print("[yellow]нет такого номера[/]")
                return
            target = chats[index]
            if parts[0] == "d":
                if ask_yes(self.console, f"удалить «{target.title}»?", default=False):
                    delete_chat(self.paths.chats, target)
                    emit_status(self.console, "удалил")
                return
            dest = self.paths.exports / f"{target.id}.md"
            export_markdown(target, dest)
            emit_status(self.console, f"экспорт {dest}")
            return
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(chats):
                chat = chats[index]
                self._ensure_chat_model(chat)
                self._chat_loop(chat)
                return
        self.console.print("[yellow]не понял[/]")

    def _make_chat(self) -> Chat:
        assert self.model is not None
        return new_chat(
            model_id=self.model.id,
            model_name=self.model.display_name,
            mode=self.settings.mode or "balance",
            system_prompt=self.settings.system_prompt,
            thinking_model=self.model.thinking,
            repo_id=self.model.repo_id,
            filename=self.model.filename,
        )

    def _chat_loop(self, chat: Chat) -> None:
        if self.engine is None or self.model is None:
            self.console.print("[yellow]модель не загружена[/]")
            return
        while True:
            action = run_chat(self.console, self.engine, chat, self.settings, self.paths)
            if action == "new":
                chat = self._make_chat()
                continue
            if action == "exit":
                self.shutdown()
                raise SystemExit(0)
            return

    def _ensure_chat_model(self, chat: Chat) -> None:
        if self.engine is not None and self.model is not None and self.model.id == chat.model_id:
            return
        model = get_model(chat.model_id)
        if model is None and chat.repo_id and chat.filename:
            model = spec_from_repo(chat.repo_id, chat.filename, size_gb=1.0)
            model = refresh_size(model, hf_token(self.settings))
        if model is None:
            self.console.print("[yellow]модель этого чата не в каталоге, остаюсь на текущей[/]")
            return
        if self.engine is not None and not ask_yes(
            self.console,
            f"чат был на {chat.model_name}. переключить модель?",
            default=True,
        ):
            return
        self._commit(model, chat.mode or self.settings.mode or "balance")

    def _prompt_menu(self) -> None:
        from rich.panel import Panel

        self.console.print(Panel(self.settings.system_prompt, title="промпт по умолчанию", border_style="yellow"))
        self.console.print("  1 оставить   2 пресет   3 свой текст   4 сброс")
        raw = ask_choice(self.console, "промпт ›") or ""
        if raw in {"", "1"}:
            return
        if raw == "4":
            self.settings.system_prompt = DEFAULT_PROMPT
        elif raw == "2":
            for index, preset in enumerate(PRESETS, start=1):
                self.console.print(f"  {index}  {preset.title}")
            pick = ask_line(self.console, "номер")
            if pick and pick.isdigit() and 1 <= int(pick) <= len(PRESETS):
                self.settings.system_prompt = PRESETS[int(pick) - 1].text
        elif raw == "3":
            self.console.print("[dim]пиши промпт, закончи пустой строкой[/]")
            lines: list[str] = []
            while True:
                try:
                    line = self.console.input()
                except EOFError:
                    break
                if not line.strip():
                    break
                lines.append(line)
            text = "\n".join(lines).strip()
            if text:
                self.settings.system_prompt = normalize_prompt(text)
        save_settings(self.paths, self.settings)
        emit_status(self.console, "сохранил. новые чаты пойдут с этим промптом.")

    def _settings_menu(self) -> None:
        s = self.settings
        self.console.print(
            f"температура {s.temperature:g} · top_p {s.top_p:g} · токены {s.max_tokens}\n"
            f"контекст {s.n_ctx or 'авто'} · потоки {s.n_threads or 'авто'} · "
            f"GPU слои {s.n_gpu_layers} · рассуждения {s.show_thinking}\n"
            f"папка моделей {self.paths.models}\n"
            f"обучение {s.rag_max_gb:g} ГБ · {s.rag_minutes} мин · {s.rag_speed} · глубина {s.rag_depth}"
        )
        self.console.print(
            "  1 температура   2 top_p   3 токены   4 контекст\n"
            "  5 потоки   6 GPU слои (auto/all/none/число)   7 рассуждения\n"
            "  8 папка моделей   9 сброс   10 лимиты обучения   0 назад"
        )
        raw = ask_choice(self.console, "настройка ›") or "0"
        try:
            self._apply_setting(raw)
        except ValueError:
            self.console.print("[yellow]это не число[/]")
            return

    def _apply_setting(self, raw: str) -> None:
        s = self.settings
        if raw == "1":
            value = ask_line(self.console, "температура 0..2", default=str(s.temperature))
            if value:
                s.temperature = min(2.0, max(0.0, float(value.replace(",", "."))))
        elif raw == "2":
            value = ask_line(self.console, "top_p 0..1", default=str(s.top_p))
            if value:
                s.top_p = min(1.0, max(0.0, float(value.replace(",", "."))))
        elif raw == "3":
            value = ask_line(self.console, "макс. токены", default=str(s.max_tokens))
            if value and value.isdigit():
                s.max_tokens = min(8192, max(32, int(value)))
        elif raw == "4":
            value = ask_line(self.console, "контекст, 0 = авто", default=str(s.n_ctx or 0))
            if value and value.isdigit():
                s.n_ctx = None if int(value) == 0 else max(512, int(value))
        elif raw == "5":
            value = ask_line(self.console, "потоки, 0 = авто", default=str(s.n_threads or 0))
            if value and value.isdigit():
                s.n_threads = None if int(value) == 0 else max(1, int(value))
        elif raw == "6":
            value = ask_line(self.console, "auto / all / none / число", default=s.n_gpu_layers)
            if value:
                s.n_gpu_layers = value.strip().lower()
        elif raw == "7":
            self.console.print("  1 авто (только в умном режиме)   2 всегда показывать   3 прятать")
            pick = ask_choice(self.console, "рассуждения ›")
            s.show_thinking = {"1": None, "2": True, "3": False}.get(pick or "1", s.show_thinking)
        elif raw == "8":
            value = ask_line(self.console, "папка", default=str(self.paths.models))
            if value:
                s.models_dir = value
                self.paths = resolve_paths(s)
                ensure_dirs(self.paths)
        elif raw == "10":
            from localai.study import _ask_limits

            limits = _ask_limits(self.console, self.settings)
            self.settings.rag_max_gb = limits.max_gb
            self.settings.rag_minutes = limits.minutes
            self.settings.rag_speed = limits.speed
            self.settings.rag_depth = limits.depth
        elif raw == "9":
            self.settings = Settings(
                system_prompt=s.system_prompt,
                last_chat_id=s.last_chat_id,
                mode=s.mode,
                model_id=s.model_id,
                custom_repo=s.custom_repo,
                custom_filename=s.custom_filename,
                custom_size_gb=s.custom_size_gb,
            )
        else:
            return
        save_settings(self.paths, self.settings)
        emit_status(self.console, "сохранил. контекст и GPU применятся при следующей загрузке модели.")

    def _learn_menu(self) -> None:
        from localai.study import run_study

        if self.engine is None or self.model is None:
            self.console.print("[dim]сначала загружу модель: она составит план. скорость и ум спрошу, если ещё не заданы.[/]")
            if not self.prepare_model():
                return
        run_study(self.console, self.engine, self.settings, self.paths)

    def _files_menu(self) -> None:
        files = downloaded_files(self.paths.models)
        if not files:
            self.console.print("[yellow]в папке моделей пусто[/]")
            self.console.print(f"[dim]{self.paths.models}[/]")
            return
        for index, path in enumerate(files, start=1):
            size = path.stat().st_size / (1024**3)
            self.console.print(f"  {index:>2}  {fmt_gb(size):>8}  {path.name}")
        raw = ask_line(self.console, "d номер — удалить файл, пусто — назад")
        if not raw:
            return
        parts = raw.split()
        if len(parts) == 2 and parts[0] == "d" and parts[1].isdigit():
            index = int(parts[1]) - 1
            if 0 <= index < len(files):
                target = files[index]
                if self.engine is not None and Path(self.engine.path).resolve() == target.resolve():
                    self.console.print("[yellow]эта модель сейчас загружена, сначала смени её[/]")
                    return
                if ask_yes(self.console, f"удалить {target.name}?", default=False):
                    target.unlink(missing_ok=True)
                    emit_status(self.console, "удалил")

    def _print_alternatives(self, rec: Recommendation) -> None:
        for index, (model, placement) in enumerate(rec.alternatives, start=1):
            slow = "" if placement.reasonable else " · очень медленно"
            self.console.print(
                f"  {index}  {model.display_name}  {fmt_gb(model.size_gb)}  {placement.device}{slow}"
            )

    def _pick_alternative(self, rec: Recommendation) -> ModelSpec | None:
        raw = ask_line(self.console, "номер альтернативы, пусто — оставить рекомендацию")
        if not raw:
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(rec.alternatives):
                return rec.alternatives[index][0]
        return None

    def _saved_model(self) -> ModelSpec | None:
        if not self.settings.model_id:
            return None
        known = get_model(self.settings.model_id)
        if known is not None:
            return known
        if self.settings.custom_repo and self.settings.custom_filename:
            return spec_from_repo(
                self.settings.custom_repo,
                self.settings.custom_filename,
                self.settings.custom_size_gb or 1.0,
            )
        return None

    def _offload_assumed(self) -> bool:
        status = llama_gpu_offload()
        if status is not None:
            return status
        assert self.hw is not None
        return self.hw.apple_silicon or self.hw.vram_gb > 0

    def _apply_runtime_placement(self, placement):
        ctx = self.settings.n_ctx
        layers = self.settings.n_gpu_layers
        updates = {}
        if ctx:
            updates["n_ctx"] = ctx
        if layers == "all":
            updates["n_gpu_layers"] = -1
            updates["gpu_fraction"] = 1.0
        elif layers == "none":
            updates["n_gpu_layers"] = 0
            updates["device"] = "cpu"
            updates["gpu_fraction"] = 0.0
        elif layers.isdigit():
            updates["n_gpu_layers"] = int(layers)
            updates["device"] = "hybrid"
        if not updates:
            return placement
        return replace(placement, **updates)

    def _apply_cli_overrides(self) -> None:
        if self.args.system:
            self.settings.system_prompt = normalize_prompt(self.args.system)
        if self.args.mode:
            self.settings.mode = self.args.mode


class PanelHint:
    def __init__(self, text: str) -> None:
        self.text = text

    def __rich__(self):
        from rich.panel import Panel

        return Panel(self.text, title="gpu", border_style="yellow")


def _auto_threads(physical: int) -> int:
    if physical > 2:
        return physical - 1
    return max(1, physical)


def _parse_speed(raw: str | None) -> str | None:
    if not raw:
        return None
    return {
        "1": "fast",
        "2": "normal",
        "3": "slow",
        "fast": "fast",
        "normal": "normal",
        "slow": "slow",
        "быстро": "fast",
        "нормально": "normal",
        "медленно": "slow",
    }.get(raw.strip().lower())


def _parse_smart(raw: str | None) -> str | None:
    if not raw:
        return None
    return {
        "1": "simple",
        "2": "normal",
        "3": "max",
        "simple": "simple",
        "normal": "normal",
        "max": "max",
        "проще": "simple",
        "умный": "max",
    }.get(raw.strip().lower())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localai",
        description="Локальный чат: сам смотрит ПК, качает GGUF с Hugging Face и запускает модель.",
    )
    parser.add_argument("--scan", action="store_true", help="только железо; модель ищет, если заданы скорость и ум")
    parser.add_argument("--speed", choices=["fast", "normal", "slow", "1", "2", "3"], help="скорость: fast, normal или slow")
    parser.add_argument("--smart", choices=["simple", "normal", "max", "1", "2", "3"], help="ум: simple, normal или max")
    parser.add_argument("--mode", choices=["fast", "smart", "balance"], help="старый режим, если не заданы скорость и ум")
    parser.add_argument("--model", help="id модели из каталога, например qwen3-4b-q4")
    parser.add_argument("--chat", help="открыть сохранённый чат по id")
    parser.add_argument("--system", help="системный промпт")
    parser.add_argument("--yes", action="store_true", help="не спрашивать подтверждение скачивания")
    parser.add_argument("--fresh", action="store_true", help="не предлагать прошлую модель")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="лог llama.cpp")
    parser.add_argument("--version", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return
    try:
        code = App(args).run()
    except KeyboardInterrupt:
        print()
        print("пока.")
        code = 0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"упало: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
