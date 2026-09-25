from localai.rag import LearnLimits, normalize_limits, retrieve, slug, used_bytes, write_text
from localai.study import collect_sources, fallback_plan, parse_plan, query_list


def test_rag_respects_size_limit(tmp_path):
    limits = LearnLimits(max_gb=0.05, minutes=5, speed="fast", depth=1)
    assert write_text(tmp_path, "tema/a.txt", "привет", limits).startswith("записал")
    huge = "x" * (limits.max_bytes + 10)
    assert "лимит" in write_text(tmp_path, "tema/big.txt", huge, limits)
    assert used_bytes(tmp_path) < limits.max_bytes
    assert slug("Как устроен RAG?") == "как-устроен-rag"


def test_retrieve_finds_saved_note(tmp_path):
    limits = normalize_limits(1, 10, "normal", 2)
    write_text(tmp_path, "python/notes.md", "декоратор оборачивает функцию", limits)
    found = retrieve(tmp_path, "что такое декоратор")
    assert "декоратор" in found
    assert retrieve(tmp_path, "квантовая") == ""


def test_plan_parser_and_query_order():
    limits = normalize_limits(1, 15, "slow", 3)
    text = """
ПЛАН:
собрать факты про RAG
ВОПРОСЫ:
- какой язык?
- нужны ли примеры?
ЗАПРОСЫ:
- retrieval augmented generation
ПАПКИ:
- sources
"""
    plan = parse_plan(text, "RAG", limits)
    assert "факты" in plan.goal
    assert plan.questions[0] == "какой язык?"
    assert plan.queries == ["retrieval augmented generation"]
    queries = query_list(plan, {"какой язык?": "русский"}, limits)
    assert queries[0] == "retrieval augmented generation"
    assert any("русский" in item for item in queries)
    empty = parse_plan("мусор", "RAG", limits)
    assert empty.questions


def test_collect_stops_on_time_and_saves_sources(tmp_path):
    limits = normalize_limits(1, 1, "normal", 2)
    plan = fallback_plan("python", limits)
    calls = {"n": 0}

    def search(query: str) -> str:
        calls["n"] += 1
        return f"запрос: {query}\n1. Документ\n   https://example.com/doc\n   коротко про {query}"

    def fetch(url: str) -> str:
        return f"страница {url}\nполезная выдержка"

    clock = {"t": 0.0}

    def now() -> float:
        clock["t"] += 10
        return clock["t"]

    report = collect_sources(
        plan,
        {"что в этой теме важнее всего?": "синтаксис"},
        limits,
        tmp_path,
        search=search,
        fetch=fetch,
        clock=now,
        started=0.0,
    )
    assert report.saved >= 1
    assert report.stopped == "время"
    assert (report.folder / "sources").exists()
    assert calls["n"] < 7
