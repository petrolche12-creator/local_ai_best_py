from pathlib import Path

from localai.prompting import model_messages
from localai.tools import (
    fetch_url,
    html_to_text,
    is_dangerous,
    is_readonly,
    parse_duckduckgo_html,
    parse_tool_calls,
    run_tool,
    strip_tool_calls,
)


def test_parse_json_and_line_tool_calls():
    text = """
смотрю.
<tool_call>
{"name": "web_search", "arguments": {"query": "qwen3 gguf"}}
</tool_call>
<tool_call>
name: write_file
path: notes.txt
content:
первая строка
вторая
</tool_call>
<tool_call>
{"name": "shell", "arguments": {"command": "ls"}}
</tool_call>
"""
    calls = parse_tool_calls(text)
    assert len(calls) == 3
    assert calls[0].name == "web_search"
    assert calls[0].arguments["query"] == "qwen3 gguf"
    assert calls[1].name == "write_file"
    assert "вторая" in calls[1].arguments["content"]
    assert "tool_call" not in strip_tool_calls(text)


def test_dangerous_and_readonly():
    assert is_dangerous("rm -rf /tmp/x")
    assert is_dangerous("sudo dd if=/dev/zero of=/dev/sda")
    assert not is_dangerous("ls -la")
    assert is_readonly("git status")
    assert is_readonly("python --version")
    assert not is_readonly("rm notes.txt")
    assert not is_readonly("curl https://example.com")


def test_shell_confirm_and_write(tmp_path: Path):
    asked: list[tuple[str, bool]] = []

    def confirm(kind, detail, dangerous):
        asked.append((kind, dangerous))
        return kind == "shell"

    read = run_tool(
        parse_tool_calls('<tool_call>{"name":"shell","arguments":{"command":"pwd"}}</tool_call>')[0],
        cwd=tmp_path,
        confirm=confirm,
    )
    assert "код 0" in read
    assert asked == []

    denied = run_tool(
        parse_tool_calls('<tool_call>{"name":"write_file","arguments":{"path":"a.txt","content":"hi"}}</tool_call>')[0],
        cwd=tmp_path,
        confirm=confirm,
    )
    assert "запретил" in denied
    assert not (tmp_path / "a.txt").exists()
    assert asked == [("write", False)]

    blocked = run_tool(
        parse_tool_calls('<tool_call>{"name":"shell","arguments":{"command":"rm -rf /"}}</tool_call>')[0],
        cwd=tmp_path,
        confirm=lambda kind, detail, dangerous: False,
    )
    assert "запретил" in blocked


def test_fetch_rejects_non_http_and_html_parser():
    assert "http" in fetch_url("file:///etc/passwd")
    html = (
        '<a class="result__a" href="https://example.com/x">Qwen</a>'
        '<a class="result__snippet">коротко</a>'
    )
    rows = parse_duckduckgo_html(html)
    assert rows[0][0] == "Qwen"
    assert rows[0][1] == "https://example.com/x"
    assert "скрипт" not in html_to_text("<script>alert(1)</script><p>текст</p>")
    assert "текст" in html_to_text("<script>alert(1)</script><p>текст</p>")


def test_tool_role_becomes_user_result():
    messages = model_messages(
        "sys",
        [{"role": "tool", "name": "shell", "content": "код 0"}],
        thinking_model=False,
    )
    assert messages[-1]["role"] == "user"
    assert "tool_result" in messages[-1]["content"]
    assert "код 0" in messages[-1]["content"]
