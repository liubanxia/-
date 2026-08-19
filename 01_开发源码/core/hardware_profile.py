from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass


_TRUE = {"1", "true", "yes", "on"}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _total_ram_gb() -> float | None:
    """Read physical RAM without adding a psutil dependency."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        pass

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 ** 3), 2)
    except Exception:
        return None


def _nvidia_smi_name() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                line = line.strip()
                if line:
                    return line
    except Exception:
        pass
    return ""


def _torch_cuda_info() -> tuple[bool, str, tuple[int, int] | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "", None
        name = str(torch.cuda.get_device_name(0) or "")
        capability = tuple(int(x) for x in torch.cuda.get_device_capability(0))
        return True, name, capability
    except Exception:
        return False, "", None


def _is_k420(name: str) -> bool:
    normalized = " ".join((name or "").upper().replace("NVIDIA", "").split())
    return normalized == "QUADRO K420"


@dataclass(frozen=True)
class HardwareProfile:
    mode: str
    ram_gb: float | None
    gpu_name: str
    cuda_available: bool
    cuda_capability: tuple[int, int] | None
    inference_device: str
    heavy_3d_allowed: bool
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware_profile() -> HardwareProfile:
    ram_gb = _total_ram_gb()
    cuda_available, torch_gpu_name, capability = _torch_cuda_info()
    gpu_name = torch_gpu_name or _nvidia_smi_name()

    if _flag("PHOENIX_FORCE_CPU"):
        return HardwareProfile(
            mode="forced_cpu",
            ram_gb=ram_gb,
            gpu_name=gpu_name,
            cuda_available=cuda_available,
            cuda_capability=capability,
            inference_device="cpu",
            heavy_3d_allowed=_flag("PHOENIX_ALLOW_HEAVY_CPU"),
            reason="PHOENIX_FORCE_CPU=1",
        )

    legacy_k420 = _is_k420(gpu_name)
    low_memory = ram_gb is not None and ram_gb <= 10.5

    if legacy_k420 and low_memory:
        return HardwareProfile(
            mode="hospital_light",
            ram_gb=ram_gb,
            gpu_name=gpu_name,
            cuda_available=cuda_available,
            cuda_capability=capability,
            inference_device="cpu",
            heavy_3d_allowed=_flag("PHOENIX_ALLOW_HEAVY_CPU"),
            reason=(
                "检测到 Quadro K420 + 低内存医院工作站；K420仅用于显示，"
                "默认禁止重型3D模型在CPU上无限阻塞"
            ),
        )

    if cuda_available and capability is not None and capability[0] >= 5:
        return HardwareProfile(
            mode="modern_gpu",
            ram_gb=ram_gb,
            gpu_name=gpu_name,
            cuda_available=True,
            cuda_capability=capability,
            inference_device="cuda:0",
            heavy_3d_allowed=True,
            reason="现代CUDA GPU可用",
        )

    if low_memory:
        return HardwareProfile(
            mode="cpu_light",
            ram_gb=ram_gb,
            gpu_name=gpu_name,
            cuda_available=cuda_available,
            cuda_capability=capability,
            inference_device="cpu",
            heavy_3d_allowed=_flag("PHOENIX_ALLOW_HEAVY_CPU"),
            reason="系统内存<=10.5GB，默认采用CPU轻量保护模式",
        )

    return HardwareProfile(
        mode="cpu_standard",
        ram_gb=ram_gb,
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        cuda_capability=capability,
        inference_device="cpu",
        heavy_3d_allowed=True,
        reason="无可用现代CUDA GPU，系统内存允许CPU完整模式",
    )
