"""Снимает параметры и настройки компьютера."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_gb: float
    vendor: str


@dataclass(frozen=True)
class HardwareProfile:
    os_name: str
    os_version: str
    arch: str
    cpu_name: str
    cpu_physical: int
    cpu_logical: int
    cpu_mhz: float | None
    ram_total_gb: float
    ram_available_gb: float
    swap_gb: float
    disk_free_gb: float
    disk_total_gb: float
    disk_path: str
    gpus: tuple[GpuInfo, ...] = ()
    apple_silicon: bool = False
    cpu_features: frozenset[str] = field(default_factory=frozenset)
    power_plan: str | None = None
    python_bits: int = 64

    @property
    def vram_gb(self) -> float:
        """Отдельная видеопамять. У Apple Silicon она общая с RAM — сюда не плюсуется."""
        if self.apple_silicon:
            return 0.0
        nvidia = [gpu for gpu in self.gpus if gpu.vendor == "nvidia" and gpu.vram_gb > 0]
        if len(nvidia) >= 2:
            return sum(gpu.vram_gb for gpu in nvidia)
        best = self.best_gpu
        if best is None or best.vendor == "apple":
            return 0.0
        return best.vram_gb

    @property
    def best_gpu(self) -> GpuInfo | None:
        real = [gpu for gpu in self.gpus if gpu.vram_gb > 0 or gpu.vendor == "apple"]
        pool = real or list(self.gpus)
        if not pool:
            return None
        return max(pool, key=lambda gpu: gpu.vram_gb)

    @property
    def os_line(self) -> str:
        return f"{self.os_name} {self.os_version} ({self.arch})".strip()

    @property
    def has_discrete_gpu(self) -> bool:
        return self.vram_gb > 0 and not self.apple_silicon


def detect(disk_path: str | None = None) -> HardwareProfile:
    import psutil

    path = disk_path or os.getcwd()
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        path = os.getcwd()
        usage = shutil.disk_usage(path)
    freq = None
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq is not None:
            freq = float(cpu_freq.max or cpu_freq.current or 0) or None
    except (OSError, RuntimeError, AttributeError):
        freq = None
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    apple = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
    gpus = _detect_gpus(apple)
    return HardwareProfile(
        os_name=platform.system(),
        os_version=platform.release(),
        arch=platform.machine(),
        cpu_name=_cpu_name(),
        cpu_physical=psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        cpu_logical=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        cpu_mhz=freq,
        ram_total_gb=vm.total / (1024**3),
        ram_available_gb=vm.available / (1024**3),
        swap_gb=swap.total / (1024**3),
        disk_free_gb=usage.free / (1024**3),
        disk_total_gb=usage.total / (1024**3),
        disk_path=str(path),
        gpus=tuple(gpus),
        apple_silicon=apple,
        cpu_features=frozenset(_cpu_features()),
        power_plan=_power_plan(),
        python_bits=64 if sys_maxsize_bits() >= 64 else 32,
    )


def sys_maxsize_bits() -> int:
    import sys

    return 128 if sys.maxsize > 2**32 else 32


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("name"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            mib = float(parts[1])
        except ValueError:
            continue
        if not name or name.lower() in {"nvidia-smi", "[not supported]"}:
            continue
        gpus.append(GpuInfo(name=name, vram_gb=mib / 1024, vendor="nvidia"))
    return gpus


def parse_cpuinfo(text: str) -> str | None:
    for line in text.splitlines():
        if "model name" in line.lower() or line.lower().startswith("hardware"):
            _, _, value = line.partition(":")
            name = value.strip()
            if name:
                return name
    return None


def parse_cpu_flags(text: str) -> set[str]:
    features: set[str] = set()
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("flags") or lower.startswith("features"):
            _, _, value = line.partition(":")
            features.update(value.lower().split())
    wanted = {"avx", "avx2", "avx512f", "avx512", "neon", "asimd"}
    return {flag for flag in features if flag in wanted or flag.startswith("avx")}


def parse_governor(text: str) -> str | None:
    value = text.strip().splitlines()[0].strip() if text.strip() else ""
    return value or None


def parse_powercfg(text: str) -> str | None:
    match = re.search(r"\(([^)]+)\)\s*$", text.strip(), re.MULTILINE)
    if match:
        return match.group(1).strip()
    line = text.strip().splitlines()
    return line[-1].strip() if line else None


def power_is_saving(plan: str | None) -> bool:
    if not plan:
        return False
    low = plan.lower()
    hints = ("powersave", "power saver", "экономи", "saver", "low power")
    return any(hint in low for hint in hints)


def missing_avx2(profile: HardwareProfile) -> bool:
    arch = profile.arch.lower()
    if arch not in {"x86_64", "amd64", "x64"}:
        return False
    if not profile.cpu_features:
        return False
    return "avx2" not in profile.cpu_features


def _cpu_name() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            text = open("/proc/cpuinfo", encoding="utf-8", errors="replace").read()
        except OSError:
            text = ""
        name = parse_cpuinfo(text)
        if name:
            return name
    if system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name.strip()
    if system == "Windows":
        name = _run(["wmic", "cpu", "get", "name"])
        if name:
            lines = [line.strip() for line in name.splitlines() if line.strip() and "name" not in line.lower()]
            if lines:
                return lines[0]
    return platform.processor() or "неизвестный CPU"


def _cpu_features() -> set[str]:
    if platform.system() == "Linux":
        try:
            text = open("/proc/cpuinfo", encoding="utf-8", errors="replace").read()
        except OSError:
            return set()
        return parse_cpu_flags(text)
    if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return {"neon"}
    return set()


def _power_plan() -> str | None:
    system = platform.system()
    if system == "Linux":
        path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        try:
            return parse_governor(open(path, encoding="utf-8", errors="replace").read())
        except OSError:
            return None
    if system == "Windows":
        return parse_powercfg(_run(["powercfg", "/getactivescheme"]) or "")
    if system == "Darwin":
        low = _run(["pmset", "-g"])
        if low and "lowpowermode" in low.lower() and re.search(r"lowpowermode\s+1", low.lower()):
            return "low power"
        return None
    return None


def _detect_gpus(apple: bool) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    nvidia = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if nvidia:
        gpus.extend(parse_nvidia_smi(nvidia))
    if gpus:
        return gpus
    if apple:
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
        return [GpuInfo(name=name.strip(), vram_gb=0.0, vendor="apple")]
    if platform.system() == "Windows":
        text = _run(["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"])
        gpus.extend(_parse_wmic_gpu(text or ""))
    elif platform.system() == "Linux":
        text = _run(["lspci"])
        gpus.extend(_parse_lspci(text or ""))
    return gpus


def _parse_wmic_gpu(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or "adapterram" in raw.lower():
            continue
        match = re.search(r"(\d+)\s*$", raw)
        name = raw
        vram = 0.0
        if match:
            name = raw[: match.start()].strip()
            try:
                vram = int(match.group(1)) / (1024**3)
            except ValueError:
                vram = 0.0
        if not name or name.lower() in {"name"}:
            continue
        vendor = "nvidia" if "nvidia" in name.lower() else "amd" if "amd" in name.lower() or "radeon" in name.lower() else "unknown"
        # AdapterRAM на Windows часто врёт и упирается в 4 ГБ. Имя всё равно полезно.
        if vram > 64:
            vram = 0.0
        gpus.append(GpuInfo(name=name, vram_gb=vram, vendor=vendor))
    return gpus


def _parse_lspci(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        lower = line.lower()
        if "vga compatible" not in lower and "3d controller" not in lower and "display controller" not in lower:
            continue
        name = line.split(":", 2)[-1].strip()
        vendor = "unknown"
        if "nvidia" in lower:
            vendor = "nvidia"
        elif "amd" in lower or "radeon" in lower:
            vendor = "amd"
        elif "intel" in lower:
            vendor = "intel"
        gpus.append(GpuInfo(name=name, vram_gb=0.0, vendor=vendor))
    return gpus


def _run(cmd: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 and not completed.stdout.strip():
        return None
    return completed.stdout
