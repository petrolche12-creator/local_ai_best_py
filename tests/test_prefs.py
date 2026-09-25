from localai.hardware import GpuInfo, HardwareProfile
from localai.prefs import band_for, mode_from_prefs, search_queries


def _pc(**kwargs) -> HardwareProfile:
    base = dict(
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
    base.update(kwargs)
    return HardwareProfile(**base)


def test_mode_from_two_questions():
    assert mode_from_prefs("fast", "max") == "fast"
    assert mode_from_prefs("slow", "simple") == "balance"
    assert mode_from_prefs("slow", "max") == "smart"
    assert mode_from_prefs("normal", "normal") == "balance"


def test_fast_band_stays_under_slow_band():
    hw = _pc()
    fast = band_for(hw, "fast", "simple", gpu_offload=True)
    slow = band_for(hw, "slow", "max", gpu_offload=True)
    assert fast.hi <= slow.hi
    assert fast.mode == "fast"
    assert slow.mode == "smart"
    assert fast.quants[0].startswith("Q4")
    assert slow.quants[0] in {"Q6_K", "Q5_K_M", "Q8_0"}


def test_queries_are_size_specific_and_short():
    hw = _pc(ram_available_gb=6, ram_total_gb=8, gpus=())
    band = band_for(hw, "fast", "simple", gpu_offload=False)
    queries = search_queries(band)
    assert queries
    assert len(queries) <= 8
    assert any("GGUF" in query for query in queries)
    assert not any(query.startswith("Qwen") for query in queries)
