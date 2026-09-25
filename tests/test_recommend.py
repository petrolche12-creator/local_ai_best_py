from localai.hardware import GpuInfo, HardwareProfile
from localai.recommend import recommend, recommend_all


def hw(**kwargs) -> HardwareProfile:
    data = dict(
        os_name="Linux",
        os_version="6",
        arch="x86_64",
        cpu_name="Test CPU",
        cpu_physical=8,
        cpu_logical=16,
        cpu_mhz=3600,
        ram_total_gb=16,
        ram_available_gb=11,
        swap_gb=4,
        disk_free_gb=100,
        disk_total_gb=200,
        disk_path=".",
        gpus=(),
        apple_silicon=False,
        cpu_features=frozenset({"avx2"}),
        power_plan="performance",
        python_bits=64,
    )
    data.update(kwargs)
    return HardwareProfile(**data)


def test_catalog_ids_unique():
    from localai.catalog import CATALOG

    ids = [model.id for model in CATALOG]
    assert len(ids) == len(set(ids))
    assert all(model.filename.endswith(".gguf") for model in CATALOG)


def test_cpu_16gb_modes_diverge():
    machine = hw()
    recs = recommend_all(machine, gpu_offload=False)
    assert recs["fast"].fits
    assert recs["smart"].fits
    assert recs["fast"].model.params_b <= 1.7
    assert recs["fast"].model.quant.startswith("Q4")
    assert recs["balance"].model.params_b == 4
    assert recs["smart"].model.params_b == 8
    assert recs["smart"].model.params_b > recs["fast"].model.params_b
    assert recs["fast"].placement.device == "cpu"


def test_fast_never_larger_than_smart():
    profiles = [
        hw(ram_total_gb=8, ram_available_gb=4.2, cpu_physical=4),
        hw(ram_total_gb=16, ram_available_gb=11),
        hw(ram_total_gb=32, ram_available_gb=22, cpu_physical=8),
        hw(
            ram_total_gb=32,
            ram_available_gb=20,
            gpus=(GpuInfo("RTX 4070", 12, "nvidia"),),
        ),
        hw(
            ram_total_gb=64,
            ram_available_gb=40,
            gpus=(GpuInfo("RTX 4090", 24, "nvidia"),),
        ),
        hw(
            os_name="Darwin",
            arch="arm64",
            apple_silicon=True,
            ram_total_gb=8,
            ram_available_gb=5,
            gpus=(GpuInfo("Apple M1", 0, "apple"),),
        ),
    ]
    for machine in profiles:
        offload = machine.apple_silicon or machine.vram_gb > 0
        fast = recommend(machine, "fast", gpu_offload=offload)
        smart = recommend(machine, "smart", gpu_offload=offload)
        assert fast.model.params_b <= smart.model.params_b + 0.01


def test_gpu_24_smart_is_32b_fast_is_14b():
    machine = hw(
        ram_total_gb=64,
        ram_available_gb=40,
        gpus=(GpuInfo("RTX 4090", 24, "nvidia"),),
    )
    fast = recommend(machine, "fast", gpu_offload=True)
    smart = recommend(machine, "smart", gpu_offload=True)
    assert fast.model.params_b == 14
    assert fast.placement.device == "gpu"
    assert smart.model.params_b == 32
    assert smart.placement.device == "gpu"


def test_apple_does_not_invent_vram():
    machine = hw(
        os_name="Darwin",
        arch="arm64",
        apple_silicon=True,
        ram_total_gb=8,
        ram_available_gb=5,
        gpus=(GpuInfo("Apple M1", 99, "apple"),),
    )
    assert machine.vram_gb == 0
    rec = recommend(machine, "smart", gpu_offload=True)
    assert rec.placement.device == "metal"
    assert rec.model.params_b <= 4


def test_tiny_pc_does_not_crash():
    machine = hw(ram_total_gb=2, ram_available_gb=0.8, cpu_physical=2, swap_gb=0)
    rec = recommend(machine, "fast", gpu_offload=False)
    assert rec.fits is False
    assert rec.model.params_b == 0.6
    assert rec.notes


def test_12gb_gpu_fast_and_smart_differ():
    machine = hw(
        ram_total_gb=32,
        ram_available_gb=20,
        gpus=(GpuInfo("RTX 4070", 12, "nvidia"),),
    )
    fast = recommend(machine, "fast", gpu_offload=True)
    smart = recommend(machine, "smart", gpu_offload=True)
    assert fast.model.params_b == 4
    assert fast.placement.device == "gpu"
    assert smart.model.params_b == 14
    assert smart.placement.device == "gpu"
    assert smart.placement.n_ctx >= 2048
