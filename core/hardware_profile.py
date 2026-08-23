from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass

TRUE = {"1", "true", "yes", "on"}


def _flag(name): return os.environ.get(name, "").strip().lower() in TRUE


def _total_ram_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 ** 3), 2)
    except Exception:
        pass
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MEMORYSTATUSEX(); status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024 ** 3), 2)
    except Exception:
        pass
    return None


def _torch_cuda_info():
    try:
        import torch
        if not torch.cuda.is_available(): return False, "", None
        return True, str(torch.cuda.get_device_name(0) or ""), tuple(int(x) for x in torch.cuda.get_device_capability(0))
    except Exception:
        return False, "", None


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
    def to_dict(self): return asdict(self)


def detect_hardware_profile():
    ram_gb = _total_ram_gb()
    cuda_available, gpu_name, capability = _torch_cuda_info()
    if _flag("PHOENIX_FORCE_CPU"):
        return HardwareProfile("forced_cpu", ram_gb, gpu_name, cuda_available, capability, "cpu", _flag("PHOENIX_ALLOW_HEAVY_CPU"), "PHOENIX_FORCE_CPU=1")
    if cuda_available and capability and capability[0] >= 5:
        return HardwareProfile("modern_gpu", ram_gb, gpu_name, True, capability, "cuda:0", True, "CUDA GPU available")
    low_memory = ram_gb is not None and ram_gb <= 10.5
    return HardwareProfile("cpu_light" if low_memory else "cpu_standard", ram_gb, gpu_name, cuda_available, capability, "cpu", (not low_memory) or _flag("PHOENIX_ALLOW_HEAVY_CPU"), "Memory-aware CPU mode")
