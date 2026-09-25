"""Живой поиск GGUF на Hugging Face под скорость, ум и железо."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from localai.catalog import ModelSpec, get_model, quant_rank
from localai.downloader import _guess_params, spec_from_repo
from localai.hardware import HardwareProfile
from localai.prefs import Band, band_for, search_queries
from localai.recommend import Placement, choose_placement, recommend


JUNK_NAME = (
    "vision",
    "vl-",
    "-vl",
    "whisper",
    "embed",
    "rerank",
    "tts",
    "ocr",
    "mmproj",
    "clip",
    "image",
    "audio",
    "stable-diffusion",
    "flux",
)
BAD_PIPELINE = {
    "image-text-to-text",
    "text-to-image",
    "automatic-speech-recognition",
    "feature-extraction",
    "text-to-speech",
    "text-to-audio",
}
_WORKING_ENDPOINT: str | None = None
TRUSTED = {
    "qwen": 10,
    "unsloth": 8,
    "bartowski": 8,
    "lmstudio-community": 6,
    "ggml-org": 6,
    "microsoft": 5,
    "google": 5,
    "huggingfacetb": 5,
}


@dataclass
class RepoHit:
    repo_id: str
    downloads: int = 0
    pipeline: str = ""
    likes: int = 0
    gated: bool = False
    tags: tuple[str, ...] = ()


@dataclass
class Found:
    model: ModelSpec
    placement: Placement | None
    downloads: int
    score: float
    source: str
    reason: str


@dataclass
class SearchResult:
    band: Band
    best: Found | None
    alternatives: list[Found] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    looked: int = 0
    online: bool = True
    note: str = ""


def search_huggingface(
    hw: HardwareProfile,
    speed: str,
    smart: str,
    *,
    gpu_offload: bool = True,
    token: str | None = None,
    progress=None,
    search_fn=None,
    files_fn=None,
    size_fn=None,
) -> SearchResult:
    band = band_for(hw, speed, smart, gpu_offload)
    queries = search_queries(band)
    planned = list(queries)
    injected = search_fn is not None
    search_fn = search_fn or (lambda query: _hf_search(query, token))
    files_fn = files_fn or (lambda repo: _hf_files(repo, token))
    size_fn = size_fn or (lambda repo, filename: _hf_size(repo, filename, token))
    if progress:
        progress(f"диапазон {band.lo:.1f}–{band.hi:.1f}B, кванты {', '.join(band.quants[:3])}")
    repos: list[RepoHit] = []
    successes = 0
    shown_error = False
    if not injected:
        if progress:
            progress("листаю весь каталог GGUF на Hugging Face, не один бренд")
        try:
            repos.extend(_hf_sweep(token, progress))
            successes += 1
        except Exception as exc:
            if progress and not shown_error:
                progress(f"запрос не удался: {exc}")
                shown_error = True
            if _is_connect(exc):
                queries = []
    for query in queries:
        if progress:
            progress(f"Hugging Face: {query}")
        try:
            repos.extend(search_fn(query))
            successes += 1
        except Exception as exc:
            if progress and not shown_error:
                progress(f"запрос не удался: {exc}")
                shown_error = True
            # Обрыв сети повторится на каждом запросе, дальше только теряем время.
            if _is_connect(exc):
                break
    online = successes > 0
    ranked = prioritize_repos(_dedupe(repos), band)
    found: list[Found] = []
    looked = 0
    for repo in ranked:
        if looked >= 16:
            break
        if not _repo_ok(repo):
            continue
        looked += 1
        try:
            files = files_fn(repo.repo_id)
        except Exception:
            continue
        best_found: Found | None = None
        for filename in candidate_filenames(files, band.quants):
            try:
                size = size_fn(repo.repo_id, filename)
            except Exception:
                size = None
            spec = _spec(repo.repo_id, filename, size)
            placement = choose_placement(hw, spec, band.mode, gpu_offload)
            score = score_found(spec, placement, repo.downloads, band)
            item = Found(
                model=spec,
                placement=placement,
                downloads=repo.downloads,
                score=score,
                source="huggingface",
                reason=_reason(spec, placement, band, repo.downloads),
            )
            if best_found is None or item.score > best_found.score:
                best_found = item
        if best_found is not None:
            found.append(best_found)
    found = [item for item in found if item.score > -1_000]
    found.sort(key=lambda item: item.score, reverse=True)
    if found:
        return SearchResult(
            band=band,
            best=found[0],
            alternatives=found[1:4],
            queries=planned,
            looked=looked,
            online=online,
            note="лучший из каталога Hugging Face под этот ПК, скорость и ум",
        )
    fallback = _catalog_fallback(hw, band, gpu_offload)
    note = "Hugging Face ничего подходящего не отдал, беру из запасного списка"
    if not online:
        note = "Hugging Face не ответил, беру из запасного списка"
    return SearchResult(
        band=band,
        best=fallback,
        alternatives=[],
        queries=planned,
        looked=looked,
        online=online,
        note=note,
    )


def pick_filename(files: list[str], quants: tuple[str, ...]) -> str | None:
    names = candidate_filenames(files, quants)
    return names[0] if names else None


def candidate_filenames(files: list[str], quants: tuple[str, ...]) -> list[str]:
    """Несколько квантов: если плотный не влезает, остаётся более лёгкий из того же репозитория."""
    gguf = [name for name in files if _file_ok(name)]
    if not gguf:
        return []
    picked: list[str] = []
    for quant in quants:
        hits = [name for name in gguf if quant.lower() in name.lower()]
        if not hits:
            continue
        choice = sorted(hits, key=len)[0]
        if choice not in picked:
            picked.append(choice)
        if len(picked) >= 3:
            break
    if not picked:
        q4 = [name for name in gguf if "q4" in name.lower()]
        picked.append(sorted(q4 or gguf, key=len)[0])
    lighter = [
        name
        for name in gguf
        if any(bit in name.lower() for bit in ("q4_k_m", "q4_k_s", "q4_0", "q3_k_m"))
    ]
    if lighter:
        choice = sorted(lighter, key=len)[0]
        if choice not in picked:
            picked.append(choice)
    return picked[:4]


def score_found(model: ModelSpec, placement: Placement | None, downloads: int, band: Band) -> float:
    if placement is None:
        return -1_000_000
    if model.params_b > band.hi + 1.0:
        return -1_000_000
    score = 0.0
    if not placement.reasonable:
        score -= 25
    span = max(band.hi - band.lo, 0.2)
    pos = min(1.0, max(0.0, (model.params_b - band.lo) / span))
    if band.smart == "max":
        score += pos * 14
    elif band.smart == "simple":
        score += (1 - pos) * 12
    else:
        score += 8 - abs(pos - 0.55) * 10
    score += _quant_bonus(model.quant, band.quants)
    score += repo_trust(model.repo_id)
    if _instructish(model.repo_id + " " + model.filename):
        score += 4
    score += min(6.0, math.log10(max(downloads, 1)))
    speed_weight = 3.5 if band.speed == "fast" else 1.2
    score += placement.device_rank * speed_weight
    if model.params_b + 0.2 < band.lo and band.smart == "max":
        score -= 6
    return score


def repo_trust(repo_id: str) -> float:
    author = repo_id.split("/", 1)[0].lower()
    return float(TRUSTED.get(author, 0))


def junk_repo(repo_id: str, pipeline: str = "", tags: tuple[str, ...] | list[str] = ()) -> bool:
    if pipeline in BAD_PIPELINE:
        return True
    for tag in tags:
        low = str(tag).lower()
        if low in BAD_PIPELINE or low.startswith("image-"):
            return True
    text = f"{repo_id} {pipeline}".lower()
    if any(bit in text for bit in JUNK_NAME):
        return True
    return bool(re.search(r"(^|[^a-z])asr([^a-z]|$)", text) or "speech" in text)


def _repo_ok(repo: RepoHit) -> bool:
    if repo.gated:
        return False
    return not junk_repo(repo.repo_id, repo.pipeline, repo.tags)


def _file_ok(name: str) -> bool:
    low = name.lower()
    if not low.endswith(".gguf"):
        return False
    if "mmproj" in low or "vision" in low:
        return False
    if re.search(r"\d{5}-of-\d{5}", low):
        return False
    return True


def _instructish(text: str) -> bool:
    low = text.lower()
    return any(bit in low for bit in ("instruct", "chat", "-it", "it-gguf"))


def _quant_bonus(quant: str, preferred: tuple[str, ...]) -> float:
    low = quant.lower()
    for index, name in enumerate(preferred):
        if name.lower() in low or low.startswith(name.lower().split("_")[0]):
            if name.lower() in low:
                return 8 - index
    return quant_rank(quant) * 0.2


def _spec(repo_id: str, filename: str, size: float | None) -> ModelSpec:
    guessed = spec_from_repo(repo_id, filename, size or 1.0)
    params = guessed.params_b
    if size is None:
        factor = {"Q8": 1.05, "Q6": 0.75, "Q5": 0.65, "Q4": 0.55, "Q3": 0.45}.get(guessed.quant[:2], 0.6)
        size = max(0.3, params * factor)
        guessed = guessed.resized(size, estimated=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{repo_id}-{guessed.quant}".lower()).strip("-")[:56]
    return replace(guessed, id=slug or guessed.id)


def _reason(model: ModelSpec, placement: Placement | None, band: Band, downloads: int) -> str:
    where = "не влезла"
    if placement is not None:
        where = {
            "gpu": "целиком на видеокарте",
            "metal": "на Apple Silicon",
            "hybrid": "частично на GPU",
            "cpu": "на процессоре",
        }.get(placement.device, placement.device)
    loads = f"{downloads:,}".replace(",", " ") if downloads else "мало скачиваний"
    return (
        f"{model.display_name} · {where}. "
        f"Под запрос «{band.label}». На Hugging Face {loads}."
    )


def prioritize_repos(repos: list[RepoHit], band: Band, unknown_cap: int = 8) -> list[RepoHit]:
    """Слишком большие отсекаем по имени. Размер в имени неизвестен — смотрим немного, не весь хвост."""
    fitting: list[RepoHit] = []
    unknown: list[RepoHit] = []
    for repo in repos:
        if repo.gated or junk_repo(repo.repo_id, repo.pipeline, repo.tags):
            continue
        params = _guess_params(repo.repo_id)
        if params is None:
            unknown.append(repo)
        elif params <= band.hi + 1.2:
            fitting.append(repo)
    return fitting + unknown[:unknown_cap]


def next_page_url(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(","):
        if "rel=\"next\"" not in part and "rel=next" not in part:
            continue
        match = re.search(r"<([^>]+)>", part)
        if match:
            return match.group(1)
    return None


def _dedupe(repos: list[RepoHit]) -> list[RepoHit]:
    best: dict[str, RepoHit] = {}
    for repo in repos:
        current = best.get(repo.repo_id)
        if current is None or repo.downloads > current.downloads:
            best[repo.repo_id] = repo
    return sorted(best.values(), key=lambda item: item.downloads, reverse=True)


def _catalog_fallback(hw: HardwareProfile, band: Band, gpu_offload: bool) -> Found | None:
    rec = recommend(hw, band.mode, gpu_offload=gpu_offload)
    model = rec.model
    known = get_model(model.id)
    if known is not None:
        model = known
    return Found(
        model=model,
        placement=rec.placement,
        downloads=0,
        score=0,
        source="catalog",
        reason=rec.notes[0] if rec.notes else "запасной вариант из локального списка",
    )


def parse_hub_models(payload) -> list[RepoHit]:
    if isinstance(payload, dict):
        payload = payload.get("models") or payload.get("items") or []
    if not isinstance(payload, list):
        return []
    hits: list[RepoHit] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("private") or item.get("disabled"):
            continue
        repo_id = str(item.get("id") or item.get("modelId") or "")
        if not repo_id:
            continue
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        hits.append(
            RepoHit(
                repo_id=repo_id,
                downloads=int(item.get("downloads") or 0),
                pipeline=str(item.get("pipeline_tag") or ""),
                likes=int(item.get("likes") or 0),
                gated=_is_gated(item.get("gated")),
                tags=tuple(str(tag) for tag in tags),
            )
        )
    return hits


def _hf_sweep(token: str | None, progress=None) -> list[RepoHit]:
    hits = _walk_models(
        "/api/models?filter=gguf&filter=text-generation&sort=downloads&limit=40",
        token,
        pages=3,
        progress=progress,
        label="каталог text-generation",
    )
    try:
        hits.extend(
            _walk_models(
                "/api/models?filter=gguf&sort=downloads&limit=30",
                token,
                pages=1,
                progress=progress,
                label="каталог всех GGUF",
            )
        )
    except Exception:
        if not hits:
            raise
    return hits


def _walk_models(path: str, token: str | None, pages: int, progress, label: str) -> list[RepoHit]:
    hits: list[RepoHit] = []
    url = ""
    for page in range(1, pages + 1):
        if progress:
            progress(f"{label}, страница {page}")
        if url:
            payload, link = _fetch_url(url, token)
        else:
            payload, link = _fetch_path(path, token)
        hits.extend(parse_hub_models(payload))
        nxt = next_page_url(link)
        if not nxt:
            break
        url = nxt
    return hits


def _hf_search(query: str, token: str | None) -> list[RepoHit]:
    payload = _get_json(
        "/api/models",
        {"filter": "gguf", "search": query, "sort": "downloads", "limit": "8"},
        token,
    )
    return parse_hub_models(payload)


def _hf_files(repo_id: str, token: str | None) -> list[str]:
    data = _get_json(f"/api/models/{quote(repo_id, safe='/')}", {}, token)
    siblings = data.get("siblings") if isinstance(data, dict) else None
    names = [
        str(item.get("rfilename"))
        for item in siblings or []
        if isinstance(item, dict) and item.get("rfilename")
    ]
    if names:
        return names
    tree = _get_json(f"/api/models/{quote(repo_id, safe='/')}/tree/main", {}, token)
    if not isinstance(tree, list):
        return []
    return [str(item.get("path") or item.get("rfilename")) for item in tree if isinstance(item, dict)]


def _hf_size(repo_id: str, filename: str, token: str | None) -> float:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        meta = get_hf_file_metadata(hf_hub_url(repo_id, filename), token=token)
        if meta.size:
            return meta.size / (1024**3)
    except Exception:
        pass
    return _head_size(repo_id, filename, token)


def _endpoints() -> list[str]:
    custom = os.environ.get("HF_ENDPOINT", "").strip().rstrip("/")
    if custom:
        return [custom]
    if _WORKING_ENDPOINT:
        return [_WORKING_ENDPOINT]
    return ["https://huggingface.co", "https://hf-mirror.com"]


def _get_json(path: str, params: dict[str, str], token: str | None):
    global _WORKING_ENDPOINT
    last: Exception | None = None
    for endpoint in _endpoints():
        url = endpoint + path
        if params:
            url += "?" + urlencode(params)
        for _attempt in range(2):
            try:
                payload = _read_json(url, token)
            except Exception as exc:
                last = exc
                if not _is_connect(exc):
                    raise
                continue
            if not os.environ.get("HF_ENDPOINT"):
                _WORKING_ENDPOINT = endpoint
            return payload
    raise ConnectionError(_short_net_error(last)) from last


def _fetch_path(path: str, token: str | None) -> tuple[object, str]:
    last: Exception | None = None
    for endpoint in _endpoints():
        url = endpoint + path
        for _attempt in range(2):
            try:
                return _fetch_url(url, token)
            except Exception as exc:
                last = exc
                if not _is_connect(exc):
                    raise
    raise ConnectionError(_short_net_error(last)) from last


def _fetch_url(url: str, token: str | None) -> tuple[object, str]:
    raw, headers = _read_raw(url, token)
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    return payload, headers.get("Link") or headers.get("link") or ""


def _read_json(url: str, token: str | None):
    raw, _headers = _read_raw(url, token)
    return json.loads(raw.decode("utf-8", errors="replace"))


def _read_raw(url: str, token: str | None) -> tuple[bytes, dict]:
    headers = {"User-Agent": "localai/0.1", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read()
            got = {key: value for key, value in response.headers.items()}
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise ConnectionError(str(exc.reason or exc)) from exc
    return raw, got


def _head_size(repo_id: str, filename: str, token: str | None) -> float:
    last: Exception | None = None
    quoted = quote(filename)
    for endpoint in _endpoints():
        url = f"{endpoint}/{quote(repo_id, safe='/')}/resolve/main/{quoted}"
        headers = {"User-Agent": "localai/0.1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, headers=headers, method="HEAD")
        try:
            with urlopen(request, timeout=20) as response:
                length = response.headers.get("Content-Length")
        except Exception as exc:
            last = exc
            continue
        if length and length.isdigit() and int(length) > 0:
            return int(length) / (1024**3)
    raise RuntimeError(_short_net_error(last) if last else "нет размера")


def _is_gated(value) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str) and value.strip().lower() in {"", "false", "none", "0"}:
        return False
    return bool(value)


def _is_connect(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) and not isinstance(exc, HTTPError):
        return True
    text = str(exc).lower()
    return any(bit in text for bit in ("ssl", "tls", "eof", "timed out", "timeout", "connection"))


def _short_net_error(exc: Exception | None) -> str:
    text = str(exc or "").lower()
    tried = "huggingface.co"
    if not os.environ.get("HF_ENDPOINT"):
        tried = "huggingface.co и hf-mirror.com"
    if "ssl" in text or "tls" in text or "eof" in text:
        return f"Hugging Face не открылся: сеть оборвала TLS ({tried})"
    if "timed out" in text or "timeout" in text:
        return "Hugging Face не ответил вовремя"
    if exc:
        return f"Hugging Face не открылся: {exc}"
    return "Hugging Face не открылся"
