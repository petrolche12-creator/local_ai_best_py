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
from localai.history import Chat, delete_chat, export_markdown, find_chat, list_chats, new_chat
from localai.prompts import DEFAULT_PROMPT, PRESETS, normalize_prompt
from localai.recommend import (
    Recommendation,
    choose_placement,
    default_max_tokens,
    mode_label,
    recommend_all,
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
    recommendations_table,
    say_recommendation,
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
        self.recs: dict[str, Recommendation] = {}
        self.model: ModelSpec | None = None
        self.placement = None

    def run(self) -> int:
        ensure_dirs(self.paths)
        self.settings = load_settings(self.paths)
        self._apply_cli_overrides()
        self.paths = resolve_paths(self.settings)
        ensure_dirs(self.paths)
        banner(self.console)
        self.scan()
        if self.hw is None:
            return 1
        if self.args.scan:
            for rec in self.recs.values():
                say_recommendation(self.console, rec)
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
        if self.args.yes and self.args.mode:
            self._new_chat()
            return 0
        self.menu()
        return 0

    def scan(self) -> None:
        self.console.print("[dim]смотрю процессор, память, видео и диск…[/]")
        disk = str(self.paths.models)
        self.hw = detect(disk)
        hardware_panel(self.console, self.hw)
        offload = self._offload_assumed()
        if llama_gpu_offload() is False and (self.hw.vram_gb > 0 or self.hw.apple_silicon):
            vendor = "nvidia" if self.hw.vram_gb > 0 else "apple"
            self.console.print(PanelHint(install_hint(vendor)))
            offload = False
        self.recs = recommend_all(self.hw, gpu_offload=offload)
        recommendations_table(self.console, self.recs)

    def prepare_model(self) -> bool:
        assert self.hw is not None
        if self.args.model:
            model = get_model(self.args.model)
            if model is None:
                self.console.print(f"[red]нет такой модели в каталоге:[/] {self.args.model}")
                self.console.print("id смотри через python main.py --scan или в меню каталога")
                return False
            mode = self.args.mode or self.settings.mode or "balance"
            return self._commit(model, mode)
        if self.args.mode:
            rec = self.recs[self.args.mode]
            say_recommendation(self.console, rec)
            return self._commit(rec.model, rec.mode, rec=rec)
        if self.settings.model_id and not self.args.fresh:
            previous = self._saved_model()
            if previous is not None:
                label = mode_label(self.settings.mode or "balance")
                reuse = self.args.yes or ask_yes(
                    self.console,
                    f"прошлый раз: {previous.display_name}, режим «{label}». запустить его?",
                    default=True,
                )
                if reuse:
                    return self._commit(previous, self.settings.mode or "balance")
        return self.wizard()

    def wizard(self) -> bool:
        self.console.print()
        self.console.print("[bold]как отвечать?[/]")
        fast = self.recs["fast"].model.display_name
        smart = self.recs["smart"].model.display_name
        mid = self.recs["balance"].model.display_name
        self.console.print(f"  [bold]1[/]  тупой, но быстрый     [dim]{fast}[/]")
        self.console.print(f"  [bold]2[/]  умный, но медленный   [dim]{smart}[/]")
        self.console.print(f"  [bold]3[/]  середина              [dim]{mid}[/]")
        self.console.print("  [bold]4[/]  выбрать модель самому")
        self.console.print("  [bold]5[/]  свой репозиторий Hugging Face")
        while True:
            raw = ask_choice(self.console, "выбор ›")
            if raw is None:
                return False
            if raw in {"4", "каталог", "m"}:
                picked = self.pick_from_catalog()
                if picked is None:
                    continue
                if self._commit(picked, self.settings.mode or "balance"):
                    return True
                self.console.print("[yellow]не запустилось. выбери другое или 0 чтобы выйти.[/]")
                continue
            if raw in {"5", "hf", "repo"}:
                picked = self.pick_custom()
                if picked is None:
                    continue
                if self._commit(picked, self.settings.mode or "balance"):
                    return True
                self.console.print("[yellow]не запустилось. выбери другое или 0 чтобы выйти.[/]")
                continue
            if raw in {"0", "q", "выход"}:
                return False
            mode = MODES.get(raw.lower())
            if mode is None:
                self.console.print("[yellow]жми 1, 2, 3, 4 или 5[/]")
                continue
            rec = self.recs[mode]
            say_recommendation(self.console, rec)
            if rec.alternatives and ask_yes(self.console, "показать другие размеры?", default=False):
                self._print_alternatives(rec)
                alt = self._pick_alternative(rec)
                if alt is not None and self._commit(alt, mode):
                    return True
                if alt is not None:
                    self.console.print("[yellow]не запустилось.[/]")
                    continue
            if self._commit(rec.model, mode, rec=rec):
                return True
            self.console.print("[yellow]не запустилось. можно выбрать другой режим.[/]")

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
            f"папка моделей {self.paths.models}"
        )
        self.console.print(
            "  1 температура   2 top_p   3 токены   4 контекст\n"
            "  5 потоки   6 GPU слои (auto/all/none/число)   7 рассуждения\n"
            "  8 папка моделей   9 сброс   0 назад"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localai",
        description="Локальный чат: сам смотрит ПК, качает GGUF с Hugging Face и запускает модель.",
    )
    parser.add_argument("--scan", action="store_true", help="только железо и рекомендации")
    parser.add_argument("--mode", choices=["fast", "smart", "balance"], help="режим без вопроса")
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
