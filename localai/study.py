"""Обучение в RAG: задание, план, вопросы, потом поиск и запись."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from localai.config import AppPaths, Settings, save_settings
from localai.rag import (
    LearnLimits,
    depth_name,
    normalize_limits,
    slug,
    speed_name,
    topic_dir,
    used_gb,
    write_text,
)
from localai.tools import fetch_url, web_search
from localai.ui import ask_line, ask_yes, emit_status


@dataclass
class StudyPlan:
    topic: str
    goal: str
    questions: list[str]
    queries: list[str]
    folders: list[str] = field(default_factory=list)


@dataclass
class StudyReport:
    saved: int
    stopped: str
    folder: Path


def budget(limits: LearnLimits) -> tuple[int, int, int]:
    """Сколько запросов, страниц на запрос и символов хранить."""
    queries, pages, chars = {
        "fast": (2, 1, 2500),
        "normal": (4, 2, 5000),
        "slow": (7, 3, 8000),
    }[limits.speed]
    if limits.depth <= 1:
        return min(queries, 3), 1, chars
    if limits.depth >= 3:
        return queries + 2, pages + 1, chars + 2000
    return queries, pages, chars


def fallback_plan(topic: str, limits: LearnLimits) -> StudyPlan:
    questions = [
        "что в этой теме важнее всего?",
        "на каком языке писать заметки?",
    ]
    if limits.depth >= 2:
        questions.append("нужны примеры и команды или только суть?")
    if limits.depth >= 3:
        questions.append("какие источники не брать?")
    queries = [topic, f"{topic} руководство", f"{topic} примеры"]
    if limits.depth >= 3:
        queries.append(f"{topic} частые ошибки")
    return StudyPlan(
        topic=topic,
        goal=f"Собрать по теме «{topic}» только то, что пригодится в ответах.",
        questions=questions,
        queries=queries,
        folders=["sources", "notes"],
    )


def parse_plan(text: str, topic: str, limits: LearnLimits) -> StudyPlan:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL)
    sections = _sections(cleaned)
    goal = sections.get("план") or sections.get("plan") or ""
    questions = _bullets(sections.get("вопросы") or sections.get("questions") or "")
    queries = _bullets(sections.get("запросы") or sections.get("queries") or "")
    folders = _bullets(sections.get("папки") or sections.get("folders") or "")
    base = fallback_plan(topic, limits)
    return StudyPlan(
        topic=topic,
        goal=(goal or base.goal).strip(),
        questions=questions or base.questions,
        queries=queries or base.queries,
        folders=folders or base.folders,
    )


def query_list(plan: StudyPlan, answers: dict[str, str], limits: LearnLimits) -> list[str]:
    count, _pages, _chars = budget(limits)
    queries = list(plan.queries)
    if limits.depth >= 2:
        for answer in answers.values():
            text = answer.strip()
            if text and text.lower() not in {"не важно", "нет", "-"}:
                queries.append(f"{plan.topic} {text[:80]}")
                break
    if limits.depth >= 3:
        queries.append(f"{plan.topic} подробно")
    unique: list[str] = []
    for query in queries:
        item = " ".join(query.split())
        if item and item not in unique:
            unique.append(item)
    return unique[:count]


def collect_sources(
    plan: StudyPlan,
    answers: dict[str, str],
    limits: LearnLimits,
    root: Path,
    *,
    search,
    fetch,
    clock=time.monotonic,
    started: float | None = None,
) -> StudyReport:
    folder = topic_dir(root, plan.topic)
    for name in plan.folders[:6]:
        relative = f"{slug(plan.topic)}/{slug(name, 32)}"
        write_text(root, relative + "/.keep", "папка обучения\n", limits)
    queries = query_list(plan, answers, limits)
    _queries, pages, chars = budget(limits)
    deadline = (started if started is not None else clock()) + limits.minutes * 60
    saved = 0
    stopped = "готово"
    index_lines = [f"# {plan.topic}", "", plan.goal, ""]
    for number, query in enumerate(queries, start=1):
        if clock() >= deadline:
            stopped = "время"
            break
        try:
            raw = search(query)
        except Exception as exc:
            write_text(root, f"{slug(plan.topic)}/errors.md", f"{query}: {exc}\n", limits)
            continue
        hits = _parse_search(raw)[:pages]
        if not hits:
            continue
        for hit_index, (title, url, snippet) in enumerate(hits, start=1):
            if clock() >= deadline:
                stopped = "время"
                break
            body = snippet
            if url.startswith("http"):
                try:
                    opened = fetch(url)
                except Exception as exc:
                    opened = f"не открылось: {exc}"
                if opened and "не открылось" not in opened[:40]:
                    body = opened
            body = (body or "").strip()
            if len(body) > chars:
                body = body[:chars] + "\n…обрезано"
            name = f"{slug(plan.topic)}/sources/{number:02d}-{hit_index}-{slug(title, 36)}.md"
            text = f"# {title}\n\nзапрос: {query}\nurl: {url}\n\n{body}\n"
            result = write_text(root, name, text, limits)
            if result.startswith("лимит"):
                stopped = "место"
                break
            if result.startswith("записал"):
                saved += 1
                index_lines.append(f"- {title} — {url}")
        if stopped == "место":
            break
    if saved == 0 and stopped == "готово":
        stopped = "нечего сохранять"
    index_lines.extend(["", f"сохранено файлов: {saved}", f"остановка: {stopped}"])
    write_text(root, f"{slug(plan.topic)}/index.md", "\n".join(index_lines) + "\n", limits)
    _refresh_root_index(root, limits)
    return StudyReport(saved=saved, stopped=stopped, folder=folder)


def run_study(console, engine, settings: Settings, paths: AppPaths, topic: str | None = None) -> None:
    root = paths.rag
    root.mkdir(parents=True, exist_ok=True)
    if not topic:
        topic = ask_line(console, "что изучить")
    topic = (topic or "").strip()
    if not topic:
        emit_status(console, "без задания не учусь")
        return
    limits = _ask_limits(console, settings)
    settings.rag_max_gb = limits.max_gb
    settings.rag_minutes = limits.minutes
    settings.rag_speed = limits.speed
    settings.rag_depth = limits.depth
    save_settings(paths, settings)
    emit_status(console, f"задание: {topic}")
    emit_status(console, f"лимиты: {limits.label()}")
    emit_status(console, "сначала план, потом вопросы, и только потом поиск")
    draft = _draft(engine, settings, topic, limits, console)
    plan = parse_plan(draft, topic, limits)
    plan_text = _plan_markdown(plan, limits)
    write_text(root, f"{slug(topic)}/plan.md", plan_text, limits)
    emit_status(console, f"план: {topic_dir(root, topic) / 'plan.md'}")
    console.print(plan.goal)
    answers: dict[str, str] = {}
    for index, question in enumerate(plan.questions, start=1):
        console.print(f"\n[bold]вопрос {index}/{len(plan.questions)}[/] {question}")
        answer = ask_line(console, "ответ", default="не важно")
        if answer is None or answer.strip().lower() in {"стоп", "stop", "q"}:
            emit_status(console, "остановлено до поиска, план уже лежит в RAG")
            return
        answers[question] = answer.strip() or "не важно"
    answers_text = "\n".join(f"- {q}\n  {a}" for q, a in answers.items())
    write_text(root, f"{slug(topic)}/answers.md", "# Ответы\n\n" + answers_text + "\n", limits)
    emit_status(console, "ответы записал, ищу именно это")
    started = time.monotonic()

    def search(query: str) -> str:
        emit_status(console, f"поиск: {query}")
        return web_search(query, limit=budget(limits)[1])

    def fetch(url: str) -> str:
        return fetch_url(url, limit=budget(limits)[2])

    report = collect_sources(
        plan,
        answers,
        limits,
        root,
        search=search,
        fetch=fetch,
        started=started,
    )
    summary = _summarize(engine, settings, topic, report.folder, console)
    if summary:
        write_text(root, f"{slug(topic)}/notes/выжимка.md", summary + "\n", limits)
    emit_status(
        console,
        f"{report.stopped}: файлов {report.saved}, папка {report.folder}, "
        f"занято {used_gb(root):.2f} ГБ из {limits.max_gb:g}",
    )


def looks_like_assignment(text: str) -> bool:
    low = text.strip().lower()
    return low.startswith(("изучи", "обучись", "выучи", "научись", "собери знания", "learn "))


def _ask_limits(console, settings: Settings) -> LearnLimits:
    current = normalize_limits(
        settings.rag_max_gb,
        settings.rag_minutes,
        settings.rag_speed,
        settings.rag_depth,
    )
    console.print(
        f"лимит {current.label()}. глубина: 1 {depth_name(1)}, 2 {depth_name(2)}, 3 {depth_name(3)}"
    )
    if not ask_yes(console, "менять лимиты?", default=False):
        return current
    gb = ask_line(console, "лимит, ГБ", default=str(current.max_gb))
    minutes = ask_line(console, "время, минуты", default=str(current.minutes))
    console.print("  1 быстро   2 нормально   3 медленно")
    speed_raw = ask_line(console, "скорость", default={"fast": "1", "normal": "2", "slow": "3"}[current.speed])
    depth_raw = ask_line(console, "глубина 1-3", default=str(current.depth))
    speed = {"1": "fast", "2": "normal", "3": "slow", "быстро": "fast", "нормально": "normal", "медленно": "slow"}.get(
        (speed_raw or "").strip().lower(),
        current.speed,
    )
    try:
        gb_n = float((gb or str(current.max_gb)).replace(",", "."))
        min_n = int(minutes or current.minutes)
        depth_n = int(depth_raw or current.depth)
    except ValueError:
        emit_status(console, "не разобрал числа, оставил прежние лимиты")
        return current
    return normalize_limits(gb_n, min_n, speed, depth_n)


def _draft(engine, settings: Settings, topic: str, limits: LearnLimits, console) -> str:
    if engine is None:
        return ""
    prompt = (
        f"Тема: {topic}\n"
        f"Глубина {limits.depth} ({depth_name(limits.depth)}), скорость {speed_name(limits.speed)}.\n"
        "Составь план учёбы. Не ищи сам. Ответь строго так:\n"
        "ПЛАН:\n"
        "одна-три строки\n"
        "ВОПРОСЫ:\n"
        "- вопрос\n"
        "ЗАПРОСЫ:\n"
        "- поисковый запрос\n"
        "ПАПКИ:\n"
        "- sources\n"
    )
    emit_status(console, "модель пишет план…")
    try:
        result = engine.generate(
            system_prompt="Составляешь план обучения. Коротко, без инструментов, без выдуманных ссылок.",
            history=[{"role": "user", "content": prompt}],
            enable_thinking=False,
            temperature=min(0.4, settings.temperature),
            top_p=settings.top_p,
            max_tokens=min(500, settings.max_tokens),
            on_token=lambda _piece: None,
        )
    except Exception as exc:
        emit_status(console, f"план от модели не вышел ({exc}), беру простой")
        return ""
    return (result.answer or result.reasoning or "").strip()


def _summarize(engine, settings: Settings, topic: str, folder: Path, console) -> str:
    if engine is None or not folder.exists():
        return ""
    bits: list[str] = []
    for path in sorted(folder.rglob("*.md"))[:6]:
        try:
            bits.append(path.read_text(encoding="utf-8", errors="replace")[:800])
        except OSError:
            continue
    if not bits:
        return ""
    emit_status(console, "сжимаю заметки…")
    try:
        result = engine.generate(
            system_prompt="Сжимаешь заметки в короткую выжимку на русском. Не выдумывай факты, которых нет в тексте.",
            history=[{"role": "user", "content": f"Тема: {topic}\n\n" + "\n\n".join(bits)}],
            enable_thinking=False,
            temperature=0.3,
            top_p=settings.top_p,
            max_tokens=min(400, settings.max_tokens),
            on_token=lambda _piece: None,
        )
    except Exception:
        return ""
    return (result.answer or "").strip()


def _plan_markdown(plan: StudyPlan, limits: LearnLimits) -> str:
    lines = [
        f"# Задание\n\n{plan.topic}",
        f"\n# Лимиты\n\n{limits.label()}",
        f"\n# План\n\n{plan.goal}",
        "\n# Вопросы\n",
    ]
    lines.extend(f"- {item}" for item in plan.questions)
    lines.append("\n# Что искать\n")
    lines.extend(f"- {item}" for item in plan.queries)
    lines.append("\n# Папки\n")
    lines.extend(f"- {item}" for item in plan.folders)
    lines.append(
        "\nСначала эти вопросы человеку. Поиск только после ответов. "
        "Сохранять только то, что отвечает на задание.\n"
    )
    return "\n".join(lines)


def _sections(text: str) -> dict[str, str]:
    found: dict[str, list[str]] = {}
    current = ""
    for line in text.splitlines():
        header = line.strip().strip("#").strip().rstrip(":").lower()
        if header in {"план", "plan", "вопросы", "questions", "запросы", "queries", "папки", "folders"}:
            current = header
            found.setdefault(current, [])
            continue
        if current:
            found[current].append(line)
    return {key: "\n".join(value).strip() for key, value in found.items()}


def _bullets(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        raw = re.sub(r"^[-*•\d.)]+\s*", "", raw).strip()
        if raw:
            items.append(raw)
    return items[:8]


def _parse_search(raw: str) -> list[tuple[str, str, str]]:
    if not raw or raw.startswith("поиск ничего не дал") or raw.startswith("пустой"):
        return []
    hits: list[tuple[str, str, str]] = []
    blocks = re.split(r"\n(?=\d+\. )", raw)
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or not re.match(r"\d+\. ", lines[0]):
            continue
        title = re.sub(r"^\d+\. ", "", lines[0])
        url = ""
        snippet = ""
        for line in lines[1:]:
            if line.startswith("http"):
                url = line
            elif not snippet:
                snippet = line
        hits.append((title, url, snippet))
    return hits


def _refresh_root_index(root: Path, limits: LearnLimits) -> None:
    lines = ["# RAG", "", f"лимит {limits.label()}", ""]
    for path in sorted(root.iterdir() if root.exists() else []):
        if not path.is_dir():
            continue
        files = [item for item in path.rglob("*") if item.is_file() and item.name != ".keep"]
        lines.append(f"- {path.name}: {len(files)} файлов")
    write_text(root, "INDEX.md", "\n".join(lines) + "\n", limits)
