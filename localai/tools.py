"""Инструменты агента: терминал, файлы, интернет.

Команды, которые меняют ПК, по умолчанию спрашивают подтверждение.
Совсем разрушительные тоже спрашивают, даже в авторежиме.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import Request, urlopen


MAX_TOOL_CALLS = 4
MAX_READ_CHARS = 20_000
MAX_WRITE_CHARS = 200_000
MAX_SHELL_CHARS = 12_000
SHELL_TIMEOUT = 60


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.IGNORECASE | re.DOTALL):
        block = match.group(1).strip()
        call = _parse_block(block)
        if call is not None:
            calls.append(ToolCall(call.name, call.arguments, match.group(0)))
        if len(calls) >= MAX_TOOL_CALLS:
            break
    return calls


def strip_tool_calls(text: str) -> str:
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def is_dangerous(command: str) -> bool:
    text = command.lower()
    patterns = (
        r"\brm\s+-[^\n]*\brf?\b",
        r"\brm\s+-rf\b",
        r"\bmkfs\b",
        r"\bdd\b[\s\S]{0,80}\bof=/dev/",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bformat\b",
        r"\bdiskpart\b",
        r":\(\)\s*\{",
        r"\bchmod\s+-r\s+777\s+/",
        r"\breg\s+delete\b",
        r"\bdel\s+/[fsq]",
        r"\brmdir\s+/s\b",
        r"\bremove-item\b[\s\S]{0,40}-recurse",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def is_readonly(command: str) -> bool:
    text = command.strip()
    if not text or is_dangerous(text):
        return False
    if re.search(r"\b(rm|del|mv|cp|chmod|chown|sudo|kill|mkfs|dd|shutdown|reboot|format|wget|curl)\b", text, re.I):
        return False
    if ">" in text:
        return False
    head = text.lower()
    prefixes = (
        "ls",
        "dir",
        "pwd",
        "whoami",
        "date",
        "uname",
        "echo",
        "type",
        "cat",
        "head",
        "tail",
        "wc",
        "find",
        "rg",
        "grep",
        "git status",
        "git diff",
        "git log",
        "git show",
        "python --version",
        "python3 --version",
        "py --version",
        "pip show",
        "pip list",
        "nvidia-smi",
    )
    return any(head == prefix or head.startswith(prefix + " ") for prefix in prefixes)


def run_tool(
    call: ToolCall,
    *,
    cwd: Path,
    confirm,
    rag_root: Path | None = None,
    rag_max_gb: float = 1.0,
) -> str:
    name = call.name.strip().lower().replace("-", "_")
    args = call.arguments or {}
    try:
        if name == "shell":
            return _shell(str(args.get("command") or args.get("cmd") or ""), cwd, confirm)
        if name == "launch":
            return _launch(str(args.get("command") or args.get("cmd") or ""), cwd, confirm)
        if name == "read_file":
            return _read(str(args.get("path") or ""), cwd)
        if name == "write_file":
            return _write(str(args.get("path") or ""), str(args.get("content") or ""), cwd, confirm)
        if name == "make_dir":
            return _mkdir(str(args.get("path") or ""), cwd, confirm)
        if name == "list_dir":
            return _list_dir(str(args.get("path") or "."), cwd)
        if name == "search_files":
            return search_files(str(args.get("path") or "."), str(args.get("query") or args.get("q") or ""), cwd)
        if name == "web_search":
            return web_search(str(args.get("query") or args.get("q") or ""))
        if name == "fetch_url":
            return fetch_url(str(args.get("url") or ""))
        if name == "save_rag":
            return _save_rag(
                str(args.get("path") or ""),
                str(args.get("content") or ""),
                rag_root,
                rag_max_gb,
            )
    except Exception as exc:
        return f"инструмент сломался: {exc}"
    return (
        "нет инструмента "
        + name
        + ". доступны: shell, launch, read_file, write_file, make_dir, list_dir, "
        "search_files, web_search, fetch_url, save_rag"
    )


def web_search(query: str, limit: int = 5) -> str:
    query = query.strip()
    if not query:
        return "пустой запрос"
    results: list[tuple[str, str, str]] = []
    errors: list[str] = []
    try:
        results.extend(_duckduckgo(query, limit))
    except Exception as exc:
        errors.append(f"duckduckgo: {exc}")
    if len(results) < 3:
        try:
            results.extend(_wikipedia(query, limit, "ru"))
        except Exception as exc:
            errors.append(f"wikipedia: {exc}")
    if len(results) < 3:
        try:
            results.extend(_wikipedia(query, limit, "en"))
        except Exception as exc:
            errors.append(f"wikipedia-en: {exc}")
    if len(results) < 3:
        try:
            results.extend(_github(query, limit))
        except Exception as exc:
            errors.append(f"github: {exc}")
    if not results:
        detail = "; ".join(errors) or "пусто"
        return f"поиск ничего не дал ({detail})"
    lines = [f"запрос: {query}"]
    for index, (title, url, snippet) in enumerate(results[:limit], start=1):
        lines.append(f"{index}. {title}\n   {url}\n   {snippet}".rstrip())
    return "\n".join(lines)


def fetch_url(url: str, limit: int = 8_000) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return "нужен http или https адрес"
    try:
        raw, final = _http_get(url, limit_bytes=1_500_000)
    except Exception as exc:
        return f"не открылось: {exc}"
    text = html_to_text(raw)
    if len(text) > limit:
        text = text[:limit] + "\n…обрезано"
    return f"url: {final}\n\n{text}" if text else f"url: {final}\n\n(пусто)"


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_duckduckgo_html(html: str, limit: int = 5) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    for match in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        url = _ddg_url(match.group(1))
        title = html_to_text(match.group(2))
        if url and title:
            results.append((title, url, ""))
        if len(results) >= limit:
            break
    snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|td|span)>', html, flags=re.I | re.S)
    merged: list[tuple[str, str, str]] = []
    for index, (title, url, _) in enumerate(results):
        snippet = html_to_text(snippets[index]) if index < len(snippets) else ""
        merged.append((title, url, snippet))
    return merged


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip += 1
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _parse_block(block: str) -> ToolCall | None:
    if block.startswith("{"):
        data = _load_jsonish(block)
        if not isinstance(data, dict):
            return None
        name = str(data.get("name") or "").strip()
        args = data.get("arguments") if isinstance(data.get("arguments"), dict) else data.get("args")
        if isinstance(args, str):
            args = {"command": args} if name == "shell" else {"query": args}
        if not isinstance(args, dict):
            args = {key: value for key, value in data.items() if key != "name"}
        if not name:
            return None
        return ToolCall(name, args)
    name = ""
    args: dict[str, str] = {}
    content_at = None
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith("content:"):
            content_at = index
            break
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "name":
            name = value.strip()
        elif key:
            args[key] = value.strip()
    if content_at is not None:
        first = lines[content_at].split(":", 1)[1]
        rest = "\n".join(lines[content_at + 1 :])
        args["content"] = (first + "\n" + rest).strip("\n")
    if not name:
        return None
    return ToolCall(name, args)


def _load_jsonish(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return None


def _shell(command: str, cwd: Path, confirm) -> str:
    command = command.strip()
    if not command:
        return "пустая команда"
    if command.lower().startswith("cd ") or command.lower() == "cd":
        target = command[2:].strip() or str(Path.home())
        path = _resolve(target, cwd)
        if not path.is_dir():
            return f"нет такой папки: {path}"
        return f"cd сам по себе в команде не запоминается. папка агента: {cwd}. сменить: /cd {path}"
    dangerous = is_dangerous(command)
    readonly = is_readonly(command)
    if dangerous or not readonly:
        if not confirm("shell", command, dangerous):
            return "пользователь запретил команду"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"команда дольше {SHELL_TIMEOUT} с, оборвал"
    except OSError as exc:
        return f"не запустилось: {exc}"
    out = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
    out = out.strip() or "(пустой вывод)"
    if len(out) > MAX_SHELL_CHARS:
        out = out[:MAX_SHELL_CHARS] + "\n…обрезано"
    return f"код {completed.returncode}\n{out}"


def _read(path: str, cwd: Path) -> str:
    if not path.strip():
        return "не указан path"
    target = _resolve(path, cwd)
    if not target.is_file():
        return f"нет файла: {target}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"не прочитался: {exc}"
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n…обрезано"
    return f"{target}\n\n{text}"


def _write(path: str, content: str, cwd: Path, confirm) -> str:
    if not path.strip():
        return "не указан path"
    if len(content) > MAX_WRITE_CHARS:
        return "слишком большой текст, режь на части"
    target = _resolve(path, cwd)
    preview = content[:400] + ("…" if len(content) > 400 else "")
    if not confirm("write", f"{target}\n{preview}", False):
        return "пользователь запретил запись"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"не записался: {exc}"
    return f"записал {target} ({len(content)} символов)"


def _launch(command: str, cwd: Path, confirm) -> str:
    command = command.strip()
    if not command:
        return "пустая команда"
    dangerous = is_dangerous(command)
    if not confirm("launch", command, dangerous):
        return "пользователь запретил запуск"
    kwargs = {
        "shell": True,
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(command, **kwargs)  # noqa: S603
    except OSError as exc:
        return f"не запустилось: {exc}"
    return f"запустил, pid {proc.pid}: {command}"


def _mkdir(path: str, cwd: Path, confirm) -> str:
    if not path.strip():
        return "не указан path"
    target = _resolve(path, cwd)
    if not confirm("write", f"папка {target}", False):
        return "пользователь запретил создание папки"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"не создалась: {exc}"
    return f"создал {target}"


def search_files(path: str, query: str, cwd: Path, limit: int = 30) -> str:
    query = query.strip()
    if not query:
        return "пустой запрос"
    target = _resolve(path or ".", cwd)
    if not target.exists():
        return f"нет пути: {target}"
    needle = query.lower()
    skip = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "models"}
    found: list[str] = []
    seen = 0
    deadline = time.monotonic() + 8
    walker = target.rglob("*") if target.is_dir() else [target]
    for entry in walker:
        if time.monotonic() > deadline or seen > 4000 or len(found) >= limit:
            break
        seen += 1
        rel = entry.parts if entry == target else entry.relative_to(target).parts[:-1]
        if any(part in skip for part in rel):
            continue
        name_hit = needle in entry.name.lower()
        content_hit = False
        try:
            size = entry.stat().st_size if entry.is_file() else 0
        except OSError:
            continue
        if entry.is_file() and not name_hit and size < 1_000_000:
            try:
                content_hit = needle in entry.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                content_hit = False
        if name_hit or content_hit:
            kind = "dir" if entry.is_dir() else "file"
            found.append(f"{kind}  {entry}")
    if not found:
        return f"ничего не нашёл по «{query}» в {target}"
    return "\n".join(found)


def _save_rag(path: str, content: str, rag_root: Path | None, rag_max_gb: float) -> str:
    if rag_root is None:
        return "папка RAG не задана"
    if not path.strip() or not content.strip():
        return "нужны path и content"
    from localai.rag import LearnLimits, write_text

    limits = LearnLimits(max_gb=rag_max_gb, minutes=1, speed="normal", depth=1)
    return write_text(rag_root, path, content, limits)


def _list_dir(path: str, cwd: Path) -> str:
    target = _resolve(path or ".", cwd)
    if not target.exists():
        return f"нет пути: {target}"
    if target.is_file():
        return f"это файл: {target}"
    lines = [str(target)]
    try:
        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as exc:
        return f"не открылась: {exc}"
    for entry in entries[:200]:
        kind = "dir" if entry.is_dir() else "file"
        lines.append(f"  {kind:4}  {entry.name}")
    if len(entries) > 200:
        lines.append(f"…ещё {len(entries) - 200}")
    return "\n".join(lines)


def _resolve(path: str, cwd: Path) -> Path:
    raw = Path(os.path.expanduser(path))
    if not raw.is_absolute():
        raw = cwd / raw
    return raw.resolve()


def _duckduckgo(query: str, limit: int) -> list[tuple[str, str, str]]:
    url = "https://html.duckduckgo.com/html/?q=" + quote(query)
    html, _final = _http_get(url, limit_bytes=800_000, timeout=8, attempts=1)
    return parse_duckduckgo_html(html, limit)


def _wikipedia(query: str, limit: int, lang: str = "ru") -> list[tuple[str, str, str]]:
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=opensearch&format=json&limit="
        + str(limit)
        + "&search="
        + quote(query)
    )
    raw, _final = _http_get(url, limit_bytes=400_000, timeout=8, attempts=1)
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) < 4:
        return []
    titles, snippets, urls = data[1], data[2], data[3]
    rows = []
    for title, snippet, link in zip(titles, snippets, urls):
        rows.append((str(title), str(link), str(snippet)))
    return rows


def _github(query: str, limit: int) -> list[tuple[str, str, str]]:
    url = (
        "https://api.github.com/search/repositories?q="
        + quote(query)
        + "&per_page="
        + str(max(1, min(limit, 8)))
    )
    raw, _final = _http_get(
        url,
        limit_bytes=400_000,
        headers={"Accept": "application/vnd.github+json"},
        timeout=8,
        attempts=1,
    )
    data = json.loads(raw)
    rows = []
    for item in data.get("items") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("full_name") or "")
        link = str(item.get("html_url") or "")
        snippet = str(item.get("description") or "")
        if title and link:
            rows.append((title, link, snippet))
    return rows


def _http_get(
    url: str,
    limit_bytes: int,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    attempts: int = 2,
) -> tuple[str, str]:
    last: Exception | None = None
    for _attempt in range(max(1, attempts)):
        request = Request(
            url,
            headers={"User-Agent": "localai/0.1", **(headers or {})},
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                final = response.geturl()
                payload = response.read(limit_bytes + 1)
            text = payload[:limit_bytes].decode("utf-8", errors="replace")
            return text, final
        except URLError as exc:
            last = exc
    raise RuntimeError(str(last)) from last


def _ddg_url(href: str) -> str:
    href = unquote(href.replace("&amp;", "&"))
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    if href.startswith("//"):
        return "https:" + href
    return href
