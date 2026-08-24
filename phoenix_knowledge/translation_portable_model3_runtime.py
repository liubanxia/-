from __future__ import annotations

"""Hardware-adaptive local model3 runtime.

Phoenix must run from the SSD on changing machines.  This layer deliberately
avoids tuning for one GPU model: compatible CUDA is tried when available, older
or unusable CUDA devices fall back to CPU, and CUDA load failures/OOM fall back
to CPU before the caller is allowed to escalate to API.

The translation/quality policy is unchanged.  This module only chooses a safe
execution device for the same local Qwen model3 weights.
"""

import gc
import os

_INSTALLED = False
_MIN_CUDA_MAJOR = 5


def _requested_device() -> str:
    value = os.environ.get("PHOENIX_MODEL3_DEVICE", "auto").strip().lower()
    return value if value in {"auto", "cpu", "cuda"} else "auto"


def choose_device(
    *,
    requested: str = "auto",
    cuda_available: bool,
    capability_major: int | None = None,
    probe_ok: bool = True,
) -> str:
    """Pure device selection contract used by tests and runtime.

    Explicit CPU always wins.  Explicit/automatic CUDA is accepted only when a
    usable CUDA runtime exists and the device is new enough for the current
    PyTorch/transformers path.  Otherwise Phoenix keeps working on CPU.
    """

    mode = str(requested or "auto").strip().lower()
    if mode == "cpu":
        return "cpu"
    if not cuda_available or not probe_ok:
        return "cpu"
    if capability_major is not None and int(capability_major) < _MIN_CUDA_MAJOR:
        return "cpu"
    return "cuda:0"


def _cuda_probe(torch) -> tuple[bool, int | None, str]:
    try:
        if not torch.cuda.is_available():
            return False, None, "CUDA unavailable"
        major, _minor = torch.cuda.get_device_capability(0)
        if int(major) < _MIN_CUDA_MAJOR:
            return False, int(major), f"compute capability {major}.x too old"
        # A tiny real allocation catches drivers that report CUDA but cannot
        # launch kernels with the installed PyTorch build.
        probe = torch.empty((1,), device="cuda:0")
        del probe
        torch.cuda.synchronize(0)
        name = str(torch.cuda.get_device_name(0) or "CUDA GPU")
        return True, int(major), name
    except Exception as exc:
        return False, None, f"CUDA probe failed: {type(exc).__name__}: {exc}"


def _clear_partial(self, torch) -> None:
    self._model = None
    self._processor = getattr(self, "_processor", None)
    self._device = None
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_cpu(self, AutoModelForCausalLM, torch) -> None:
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    # Let transformers honor the checkpoint's native dtype when possible; if
    # the local build does not accept torch_dtype="auto", retry conservatively.
    try:
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype="auto",
            **load_kwargs,
        )
    except (TypeError, ValueError):
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **load_kwargs,
        )
    self._model.to("cpu")
    self._model.eval()
    self._device = "cpu"


def _load_cuda(self, AutoModelForCausalLM, torch, capability_major: int | None) -> None:
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
    }
    if capability_major is not None and int(capability_major) >= 8:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    try:
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **load_kwargs,
        )
    except (TypeError, ValueError):
        load_kwargs.pop("attn_implementation", None)
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **load_kwargs,
        )
    self._model.to("cuda:0")
    self._model.eval()
    self._device = "cuda:0"


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .qwen_local_medical_backend import LocalQwenMedicalBackend

    if bool(getattr(LocalQwenMedicalBackend, "_phoenix_portable_runtime", False)):
        return

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"Local Qwen model missing/incomplete: {self.model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        requested = _requested_device()
        cuda_ok, capability_major, cuda_note = _cuda_probe(torch)
        selected = choose_device(
            requested=requested,
            cuda_available=cuda_ok,
            capability_major=capability_major,
            probe_ok=cuda_ok,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )

        if selected.startswith("cuda"):
            print(
                f"[Phoenix][模型3] 硬件自适应：优先尝试 {cuda_note} / CUDA。",
                flush=True,
            )
            try:
                _load_cuda(self, AutoModelForCausalLM, torch, capability_major)
            except Exception as exc:
                print(
                    "[Phoenix][模型3] 当前CUDA路径不可安全运行："
                    f"{type(exc).__name__}: {exc}；自动回退CPU，不改变翻译链。",
                    flush=True,
                )
                _clear_partial(self, torch)
                _load_cpu(self, AutoModelForCausalLM, torch)
        else:
            reason = "用户指定CPU" if requested == "cpu" else cuda_note
            print(
                f"[Phoenix][模型3] 硬件自适应：使用CPU | {reason}。",
                flush=True,
            )
            _load_cpu(self, AutoModelForCausalLM, torch)

        label = "CPU"
        if str(self._device).startswith("cuda"):
            try:
                label = f"CUDA | {torch.cuda.get_device_name(0)}"
            except Exception:
                label = "CUDA"
        print(
            f"[Phoenix][模型3] 已加载并启用 | {label} | 同一质量规则",
            flush=True,
        )

    LocalQwenMedicalBackend._load = load
    LocalQwenMedicalBackend._phoenix_portable_runtime = True

    print(
        "[Phoenix][硬件兼容] 模型3启用通用运行层：兼容CUDA则用GPU；"
        "旧/不可用/显存失败CUDA自动回CPU；不绑定具体显卡型号。",
        flush=True,
    )
