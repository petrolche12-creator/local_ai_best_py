"""Консольный интерфейс."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localai import __version__
from localai.catalog import CATALOG, ModelSpec
from localai.hardware import HardwareProfile, missing_avx2, power_is_saving
from localai.recommend import Recommendation, mode_label


HELP = """
команды в чате:
  /help            это
  /exit            выйти в меню
  /new             новый чат
  /clear           забыть реплики, промпт оставить
  /prompt          показать или сменить системный промпт
  /prompts         пресеты промпта
  /save            сохранить сейчас
  /title текст     переименовать чат
  /history         список чатов
  /model           какая модель и где сидит
  /hw              железо
  /think           вкл/выкл показ рассуждений
  /temp 0.7        температура
  /tokens 1024     лимит токенов ответа
  /export          сохранить чат в markdown
  /paste           длинный ввод, конец — строка EOF
  /agent           вкл/выкл инструменты (интернет, ПК, запуск программ)
  /auto            не спрашивать подтверждение обычных команд
  /cd путь         папка, в которой агент работает
  /learn тема      обучение: план, вопросы, потом поиск в RAG
  /rag             где папка обучения и сколько занято
  /limits          лимит ГБ, время, скорость и глубина обучения
  /stop            то же, что Ctrl+C на генерации
""".strip()


def make_console(no_color: bool = False) -> Console:
    return Console(no_color=no_color, highlight=False)


def fmt_gb(value: float) -> str:
    if value < 1:
        return f"{value * 1024:.0f} МБ"
    return f"{value:.1f} ГБ"


def fmt_speed(value: float) -> str:
    if value <= 0:
        return "—"
    return f"{value:.1f} ток/с"


def banner(console: Console) -> None:
    body = Text()
    body.append("local ai\n", style="bold cyan")
    body.append("смотрит ПК, ищет модель на Hugging Face,\n", style="dim")
    body.append("ходит по интернету и ПК, учится в папку RAG", style="dim")
    console.print(Panel(body, subtitle=f"v{__version__}", border_style="cyan", padding=(1, 2)))


def hardware_panel(console: Console, hw: HardwareProfile) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", width=16)
    table.add_column()
    table.add_row("система", hw.os_line)
    mhz = ""
    if hw.cpu_mhz and "@" not in hw.cpu_name and "GHz" not in hw.cpu_name and "МГц" not in hw.cpu_name:
        mhz = f" @ {hw.cpu_mhz:.0f} МГц"
    table.add_row("процессор", f"{hw.cpu_name}{mhz}")
    table.add_row("ядра", f"{hw.cpu_physical} физ. / {hw.cpu_logical} лог.")
    table.add_row(
        "память",
        f"{fmt_gb(hw.ram_total_gb)} всего, свободно {fmt_gb(hw.ram_available_gb)}, swap {fmt_gb(hw.swap_gb)}",
    )
    if hw.gpus:
        for index, gpu in enumerate(hw.gpus, start=1):
            label = "видео" if index == 1 else ""
            vram = "общая с RAM" if gpu.vendor == "apple" else (fmt_gb(gpu.vram_gb) if gpu.vram_gb else "VRAM не видна")
            table.add_row(label, f"{gpu.name} · {vram}")
    else:
        table.add_row("видео", "не нашёл, будет CPU")
    table.add_row("диск", f"свободно {fmt_gb(hw.disk_free_gb)} из {fmt_gb(hw.disk_total_gb)}")
    if hw.power_plan:
        table.add_row("питание", hw.power_plan)
    table.add_row("инструкции", _feature_line(hw.cpu_features))
    console.print(Panel(table, title="железо", border_style="cyan"))
    if power_is_saving(hw.power_plan):
        console.print("[yellow]Включена экономия питания — модель будет медленнее. Поставь режим «производительность».[/]")
    if missing_avx2(hw):
        console.print("[yellow]Нет AVX2. На этом CPU локальные модели будут еле ползти.[/]")
    if hw.python_bits < 64:
        console.print("[yellow]Python 32-битный. Нужен 64-битный, иначе большие модели не встанут.[/]")


def recommendations_table(console: Console, recs: dict[str, Recommendation]) -> None:
    table = Table(title="что влезет", expand=True)
    table.add_column("режим", style="bold", no_wrap=True)
    table.add_column("модель", no_wrap=True)
    table.add_column("файл", justify="right", no_wrap=True)
    table.add_column("где", no_wrap=True)
    table.add_column("заметка")
    short = {"fast": "быстрый", "balance": "середина", "smart": "умный"}
    for mode in ("fast", "balance", "smart"):
        rec = recs[mode]
        where = "не влезает"
        if rec.placement is not None:
            where = {
                "gpu": "GPU",
                "metal": "Apple",
                "hybrid": "GPU+CPU",
                "cpu": "CPU",
            }.get(rec.placement.device, rec.placement.device)
            where = f"{where}, ctx {rec.placement.n_ctx}"
        table.add_row(
            short[mode],
            rec.model.display_name,
            fmt_gb(rec.model.size_gb) + (" ~" if rec.model.size_estimated else ""),
            where,
            rec.model.blurb,
        )
    console.print(table)


def model_table(console: Console, models: list[ModelSpec] | None = None) -> None:
    rows = list(models or CATALOG)
    table = Table(title="каталог", expand=True)
    table.add_column("#", justify="right")
    table.add_column("id", style="cyan")
    table.add_column("модель")
    table.add_column("размер", justify="right")
    table.add_column("")
    for index, model in enumerate(rows, start=1):
        size = fmt_gb(model.size_gb) + (" ~" if model.size_estimated else "")
        table.add_row(str(index), model.id, model.display_name, size, model.blurb)
    console.print(table)


def say_recommendation(console: Console, rec: Recommendation) -> None:
    from localai.recommend import explain

    style = {"fast": "green", "smart": "magenta", "balance": "cyan"}.get(rec.mode, "cyan")
    console.print(Panel(explain(rec), title=mode_label(rec.mode), border_style=style))
    if rec.placement is not None:
        approx = " (оценка)" if rec.model.size_estimated else ""
        console.print(f"[dim]{rec.model.repo_id} / {rec.model.filename} · {fmt_gb(rec.model.size_gb)}{approx}[/]")
    for note in rec.notes:
        console.print(f"[yellow]· {note}[/]")


def ask_line(console: Console, prompt: str, default: str | None = None) -> str | None:
    hint = f" [{default}]" if default else ""
    try:
        raw = console.input(f"[bold]{prompt}[/]{hint}: ").strip()
    except EOFError:
        console.print()
        return None
    if not raw and default is not None:
        return default
    return raw


def ask_yes(console: Console, prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        raw = console.input(f"[bold]{prompt}[/] [{hint}] ").strip().lower()
    except EOFError:
        console.print()
        return False
    if not raw:
        return default
    return raw in {"y", "yes", "д", "да", "у"}


def ask_choice(console: Console, prompt: str) -> str | None:
    try:
        return console.input(f"[bold]{prompt}[/] ").strip()
    except EOFError:
        console.print()
        return None


def pause(console: Console) -> None:
    try:
        console.input("[dim]enter — дальше[/] ")
    except EOFError:
        console.print()


def write_out(console: Console, text: str, *, dim: bool = False) -> None:
    style = "dim italic" if dim else None
    console.print(text, end="", style=style, highlight=False, soft_wrap=True)


def emit_status(console: Console, text: str) -> None:
    console.print(f"[dim]{text}[/]")


def _feature_line(features: frozenset[str]) -> str:
    picked: list[str] = []
    if "avx2" in features:
        picked.append("avx2")
    elif "avx" in features:
        picked.append("avx")
    if any(flag.startswith("avx512") for flag in features):
        picked.append("avx512")
    if "neon" in features or "asimd" in features:
        picked.append("neon")
    return ", ".join(picked) or "не проверял"


def supports_utf8() -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or ""
    return "utf" in encoding.lower()
