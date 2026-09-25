"""Скачивание GGUF с Hugging Face и разбор своего репозитория."""

from __future__ import annotations

import re
from pathlib import Path

from localai.catalog import ModelSpec, quant_rank


class DownloadError(RuntimeError):
    pass


QUANT_PREFERENCE = ("Q4_K_M", "Q5_K_M", "Q4_K_S", "Q6_K", "Q8_0", "Q5_0", "Q4_0", "Q3_K_M")


def local_model_path(models_dir: Path, model: ModelSpec) -> Path:
    return models_dir / _safe_repo(model.repo_id) / model.filename


def is_downloaded(models_dir: Path, model: ModelSpec) -> bool:
    path = local_model_path(models_dir, model)
    return path.is_file() and path.stat().st_size > 1_000_000


def refresh_size(model: ModelSpec, token: str | None = None) -> ModelSpec:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
    except ImportError:
        return model
    try:
        meta = get_hf_file_metadata(hf_hub_url(model.repo_id, model.filename), token=token)
    except Exception:
        return model
    if not meta.size:
        return model
    return model.resized(meta.size / (1024**3), estimated=False)


def download_model(models_dir: Path, model: ModelSpec, token: str | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DownloadError("Нет huggingface_hub. Установи: pip install -r requirements.txt") from exc
    dest = models_dir / _safe_repo(model.repo_id)
    dest.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": model.repo_id,
        "filename": model.filename,
        "local_dir": str(dest),
        "token": token,
    }
    tqdm_class = _rich_tqdm()
    if tqdm_class is not None:
        kwargs["tqdm_class"] = tqdm_class
    try:
        path = hf_hub_download(**kwargs)
    except TypeError:
        kwargs.pop("tqdm_class", None)
        try:
            path = hf_hub_download(**kwargs)
        except Exception as exc:
            raise DownloadError(_human_download_error(exc, model)) from exc
    except Exception as exc:
        raise DownloadError(_human_download_error(exc, model)) from exc
    resolved = Path(path)
    if not resolved.is_file():
        raise DownloadError(f"Файл не появился после скачивания: {resolved}")
    return resolved


def list_repo_gguf(repo_id: str, token: str | None = None) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise DownloadError("Нет huggingface_hub. Установи: pip install -r requirements.txt") from exc
    try:
        files = HfApi(token=token).list_repo_files(repo_id)
    except Exception as exc:
        raise DownloadError(f"Не вижу репозиторий {repo_id}: {exc}") from exc
    gguf = [
        name
        for name in files
        if name.lower().endswith(".gguf")
        and "mmproj" not in name.lower()
        and "vision" not in name.lower()
    ]
    if not gguf:
        raise DownloadError(f"В {repo_id} нет GGUF-файлов. Нужен репозиторий с *.gguf.")
    return sorted(gguf, key=_file_sort_key)


def spec_from_repo(
    repo_id: str,
    filename: str,
    size_gb: float,
) -> ModelSpec:
    params = _guess_params(filename) or _guess_params(repo_id) or 4.0
    quant = _guess_quant(filename)
    thinking = "qwen3" in repo_id.lower() or "qwen3" in filename.lower()
    safe_id = "custom-" + re.sub(r"[^a-z0-9]+", "-", f"{repo_id}-{quant}".lower()).strip("-")[:48]
    return ModelSpec(
        id=safe_id,
        name=_guess_name(repo_id, filename, params),
        params_b=params,
        quant=quant,
        repo_id=repo_id,
        filename=filename,
        size_gb=size_gb,
        n_layers=_guess_layers(params),
        blurb="своя модель с Hugging Face",
        thinking=thinking,
        size_estimated=False,
        license="смотри карточку на HF",
    )


def downloaded_files(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    files = [path for path in models_dir.rglob("*.gguf") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_size, reverse=True)
    return files


def _safe_repo(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def _file_sort_key(name: str) -> tuple[int, int, str]:
    upper = name.upper()
    pref = 99
    for index, quant in enumerate(QUANT_PREFERENCE):
        if quant in upper:
            pref = index
            break
    return (pref, quant_rank(_guess_quant(name)), name.lower())


def _guess_quant(filename: str) -> str:
    match = re.search(r"(Q\d(?:_K(?:_[A-Z]+)?)?|Q\d_0|Q\d_1|IQ\d_[A-Z0-9]+|BF16|F16)", filename, re.I)
    if not match:
        return "Q4_K_M"
    return match.group(1).upper()


def _guess_params(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", text, re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _guess_name(repo_id: str, filename: str, params: float) -> str:
    tail = repo_id.split("/")[-1]
    tail = re.sub(r"-gguf$", "", tail, flags=re.I)
    tail = tail.replace("_", " ")
    if str(params) not in tail and f"{params:g}" not in tail:
        return f"{tail} {params:g}B"
    return tail


def _guess_layers(params: float) -> int:
    if params <= 0.8:
        return 28
    if params <= 2:
        return 28
    if params <= 4:
        return 36
    if params <= 9:
        return 36
    if params <= 16:
        return 40
    return 64


def _human_download_error(exc: Exception, model: ModelSpec) -> str:
    text = str(exc)
    low = text.lower()
    if "401" in text or "403" in text or "gated" in low or "unauthorized" in low:
        return (
            f"Репозиторий {model.repo_id} закрыт. "
            "Зайди на его страницу, прими лицензию и задай HF_TOKEN."
        )
    if "404" in text or "not found" in low:
        return (
            f"Файл {model.filename} не найден в {model.repo_id}. "
            "Открой репозиторий и выбери файл вручную."
        )
    return f"Не скачалось {model.display_name}: {text}"


def _rich_tqdm():
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    try:
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )
    except ImportError:
        return None

    class RichTqdm(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", False)
            super().__init__(*args, **kwargs)
            self._progress = Progress(
                TextColumn("[bold cyan]качаю[/]"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
            )
            total = self.total if self.total else None
            self._task = self._progress.add_task("model", total=total)
            self._progress.start()

        def update(self, n=1):
            displayed = super().update(n)
            try:
                advance = n if isinstance(n, (int, float)) else 0
                if self._progress.tasks and self.total and self._progress.tasks[0].total != self.total:
                    self._progress.update(self._task, total=self.total)
                self._progress.update(self._task, advance=advance)
            except Exception:
                pass
            return displayed

        def close(self):
            try:
                self._progress.stop()
            except Exception:
                pass
            super().close()

    return RichTqdm
