from localai.hardware import (
    missing_avx2,
    parse_cpu_flags,
    parse_cpuinfo,
    parse_nvidia_smi,
    parse_powercfg,
    power_is_saving,
    HardwareProfile,
)


def test_nvidia_smi():
    text = "NVIDIA GeForce RTX 3060, 12288\nNVIDIA GeForce RTX 3060, 8192\n"
    gpus = parse_nvidia_smi(text)
    assert len(gpus) == 2
    assert gpus[0].vendor == "nvidia"
    assert abs(gpus[0].vram_gb - 12) < 0.01


def test_cpuinfo_and_flags():
    text = "model name : AMD Ryzen 7 5800X\nflags : fpu avx avx2 avx512f\n"
    assert parse_cpuinfo(text) == "AMD Ryzen 7 5800X"
    flags = parse_cpu_flags(text)
    assert "avx2" in flags


def test_powercfg_and_saving():
    text = "GUID схемы питания: abc  (Экономия энергии)"
    plan = parse_powercfg(text)
    assert plan == "Экономия энергии"
    assert power_is_saving(plan)
    assert power_is_saving("powersave")
    assert not power_is_saving("performance")


def test_missing_avx2_only_when_known():
    slow = HardwareProfile(
        os_name="Linux",
        os_version="6",
        arch="x86_64",
        cpu_name="old",
        cpu_physical=2,
        cpu_logical=2,
        cpu_mhz=None,
        ram_total_gb=8,
        ram_available_gb=4,
        swap_gb=0,
        disk_free_gb=10,
        disk_total_gb=20,
        disk_path=".",
        cpu_features=frozenset({"avx"}),
    )
    assert missing_avx2(slow)
    unknown = HardwareProfile(
        os_name="Windows",
        os_version="11",
        arch="AMD64",
        cpu_name="cpu",
        cpu_physical=4,
        cpu_logical=8,
        cpu_mhz=None,
        ram_total_gb=16,
        ram_available_gb=8,
        swap_gb=0,
        disk_free_gb=10,
        disk_total_gb=20,
        disk_path=".",
    )
    assert not missing_avx2(unknown)
