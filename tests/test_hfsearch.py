from localai.hfsearch import RepoHit, junk_repo, parse_hub_models, pick_filename, search_huggingface
from localai.hardware import GpuInfo, HardwareProfile


def _pc() -> HardwareProfile:
    return HardwareProfile(
        os_name="Linux",
        os_version="6",
        arch="x86_64",
        cpu_name="cpu",
        cpu_physical=8,
        cpu_logical=16,
        cpu_mhz=3600,
        ram_total_gb=32,
        ram_available_gb=24,
        swap_gb=4,
        disk_free_gb=80,
        disk_total_gb=200,
        disk_path=".",
        gpus=(GpuInfo("RTX 3060", 12, "nvidia"),),
    )


FILES = {
    "unsloth/Qwen3-VL-4B-Instruct-GGUF": ["mmproj-f16.gguf", "Qwen3-VL-4B-Q4_K_M.gguf"],
    "Qwen/Qwen3-1.7B-GGUF": ["Qwen3-1.7B-Q4_K_M.gguf", "Qwen3-1.7B-Q8_0.gguf"],
    "Qwen/Qwen3-8B-GGUF": ["Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q6_K.gguf"],
    "someone/whisper-tiny-gguf": ["whisper-tiny-q4.gguf"],
}


def _search(_query: str) -> list[RepoHit]:
    return [
        RepoHit("unsloth/Qwen3-VL-4B-Instruct-GGUF", 900_000, "image-text-to-text"),
        RepoHit("someone/whisper-tiny-gguf", 800_000, "automatic-speech-recognition"),
        RepoHit("Qwen/Qwen3-1.7B-GGUF", 50_000, "text-generation"),
        RepoHit("Qwen/Qwen3-8B-GGUF", 40_000, "text-generation"),
    ]


def test_junk_filter_drops_vision_and_whisper():
    assert junk_repo("unsloth/Qwen3-VL-4B-Instruct-GGUF", "image-text-to-text")
    assert junk_repo("org/whisper-small-gguf", "")
    assert junk_repo("handy-computer/Qwen3-ASR-1.7B-gguf", "")
    assert junk_repo("lmstudio-community/Qwen3.5-9B-GGUF", "", ("image-text-to-text",))
    assert not junk_repo("Qwen/Qwen3-4B-GGUF", "text-generation")


def test_pick_filename_prefers_requested_quant_and_skips_mmproj():
    files = FILES["unsloth/Qwen3-VL-4B-Instruct-GGUF"] + FILES["Qwen/Qwen3-8B-GGUF"]
    assert pick_filename(files, ("Q6_K", "Q4_K_M")) == "Qwen3-8B-Q6_K.gguf"


def test_ranking_prefers_smarter_model_that_fits_and_skips_vl():
    result = search_huggingface(
        _pc(),
        "slow",
        "max",
        gpu_offload=True,
        search_fn=_search,
        files_fn=lambda repo: FILES.get(repo, []),
        size_fn=lambda _repo, filename: 1.2 if "1.7B" in filename else 5.1,
    )
    assert result.best is not None
    assert "VL" not in result.best.model.repo_id
    assert "whisper" not in result.best.model.repo_id.lower()
    assert result.best.model.params_b >= 8
    assert result.online


def test_fast_simple_does_not_pick_the_max_model():
    smart = search_huggingface(
        _pc(),
        "slow",
        "max",
        gpu_offload=True,
        search_fn=_search,
        files_fn=lambda repo: FILES.get(repo, []),
        size_fn=lambda _repo, filename: 1.2 if "1.7B" in filename else 5.1,
    )
    fast = search_huggingface(
        _pc(),
        "fast",
        "simple",
        gpu_offload=True,
        search_fn=_search,
        files_fn=lambda repo: FILES.get(repo, []),
        size_fn=lambda _repo, filename: 1.2 if "1.7B" in filename else 5.1,
    )
    assert smart.best is not None and fast.best is not None
    assert fast.best.model.params_b <= smart.best.model.params_b


def test_parse_hub_payload_keeps_text_and_marks_gated():
    payload = [
        {"id": "Qwen/Qwen3-1.7B-GGUF", "downloads": 10, "pipeline_tag": "text-generation", "gated": False},
        {"id": "hidden/secret", "downloads": 9, "pipeline_tag": "text-generation", "gated": "auto"},
        {"id": "org/embed-gguf", "downloads": 8, "pipeline_tag": "feature-extraction", "tags": ["feature-extraction"]},
    ]
    hits = parse_hub_models(payload)
    assert [item.repo_id for item in hits] == [
        "Qwen/Qwen3-1.7B-GGUF",
        "hidden/secret",
        "org/embed-gguf",
    ]
    assert hits[1].gated
    assert not hits[0].gated


def test_offline_falls_back_to_catalog():
    def boom(_query: str):
        raise OSError("нет сети")

    result = search_huggingface(
        _pc(),
        "normal",
        "normal",
        gpu_offload=True,
        search_fn=boom,
        files_fn=lambda _repo: [],
        size_fn=lambda _repo, _name: 1.0,
    )
    assert result.online is False
    assert result.best is not None
    assert result.best.source == "catalog"
