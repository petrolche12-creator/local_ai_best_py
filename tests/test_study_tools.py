from pathlib import Path

from localai.hfsearch import RepoHit, next_page_url, prioritize_repos
from localai.prefs import Band
from localai.tools import ToolCall, run_tool


def _band() -> Band:
    return Band(speed="slow", smart="max", lo=1.0, hi=2.4, quants=("Q6_K",), mode="smart")


def test_prioritize_keeps_fitting_sizes_and_drops_giants():
    repos = [
        RepoHit("someone/Llama-70B-GGUF", 9_000_000, "text-generation"),
        RepoHit("org/nice-1.7B-instruct-GGUF", 1000, "text-generation"),
        RepoHit("handy/Qwen3-ASR-1.7B-gguf", 5000, "automatic-speech-recognition"),
        RepoHit("mystery/chat-gguf", 800, "text-generation"),
    ]
    picked = prioritize_repos(repos, _band(), unknown_cap=2)
    ids = [item.repo_id for item in picked]
    assert "org/nice-1.7B-instruct-GGUF" in ids
    assert "mystery/chat-gguf" in ids
    assert all("70B" not in item and "ASR" not in item for item in ids)


def test_next_page_url():
    header = '<https://huggingface.co/api/models?cursor=abc>; rel="next", <https://huggingface.co/api/models?cursor=0>; rel="first"'
    assert next_page_url(header) == "https://huggingface.co/api/models?cursor=abc"
    assert next_page_url("") is None


def test_launch_and_search_files(tmp_path: Path):
    (tmp_path / "notes.txt").write_text("секретная пометка", encoding="utf-8")
    denied = run_tool(
        ToolCall("launch", {"command": "echo hi"}),
        cwd=tmp_path,
        confirm=lambda kind, detail, dangerous: False,
    )
    assert "запретил" in denied
    found = run_tool(
        ToolCall("search_files", {"path": str(tmp_path), "query": "пометка"}),
        cwd=tmp_path,
        confirm=lambda kind, detail, dangerous: False,
    )
    assert "notes.txt" in found
    made = run_tool(
        ToolCall("make_dir", {"path": "RAG/sources"}),
        cwd=tmp_path,
        confirm=lambda kind, detail, dangerous: True,
    )
    assert (tmp_path / "RAG" / "sources").is_dir()
    assert "создал" in made
