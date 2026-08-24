from __future__ import annotations

"""Portable runtime for Phoenix local translation stages 1 and 2.

No GPU model is hard-coded.  Compatible CUDA is opportunistic; old/incompatible
CUDA, driver failures and GPU memory failures fall back to CPU while preserving
the same model weights, prompts, quality gates and cascade order.
"""

import gc

from .translation_portable_model3_runtime import _cuda_probe, _requested_device, choose_device

_INSTALLED = False


def _runtime_device(torch) -> tuple[str, str, int | None]:
    requested = _requested_device()
    cuda_ok, capability_major, note = _cuda_probe(torch)
    selected = choose_device(
        requested=requested,
        cuda_available=cuda_ok,
        capability_major=capability_major,
        probe_ok=cuda_ok,
    )
    if requested == "cpu":
        note = "用户指定CPU"
    return selected, note, capability_major


def _clear_backend(backend, torch) -> None:
    backend._model = None
    backend._device = "cpu"
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_seq2seq(self, *, model_kind: str) -> None:
    if self._model is not None:
        return
    if not self.available():
        raise RuntimeError(f"{model_kind}翻译模型未下载: {self.model_path}")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device, note, _capability = _runtime_device(torch)
    tokenizer_kwargs = {"local_files_only": True}
    if model_kind == "NLLB":
        tokenizer_kwargs["src_lang"] = "eng_Latn"
    self._tokenizer = AutoTokenizer.from_pretrained(
        str(self.model_path),
        **tokenizer_kwargs,
    )

    def cpu_load() -> None:
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )
        self._model.to("cpu")
        self._model.eval()
        self._device = "cpu"

    if device.startswith("cuda"):
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._model.to(device)
            self._model.eval()
            self._device = device
            print(
                f"[Phoenix][{model_kind}] 硬件自适应：CUDA可用 | {note}",
                flush=True,
            )
            return
        except Exception as exc:
            print(
                f"[Phoenix][{model_kind}] CUDA运行失败: {type(exc).__name__}: {exc}；"
                "自动回退CPU。",
                flush=True,
            )
            _clear_backend(self, torch)

    cpu_load()
    print(
        f"[Phoenix][{model_kind}] 硬件自适应：CPU | {note}",
        flush=True,
    )


def _load_hymt(self) -> None:
    if self._model is not None and self._tokenizer is not None:
        return
    if not self.available():
        raise RuntimeError(f"HY-MT模型未下载完整: {self.model_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, note, _capability = _runtime_device(torch)
    self._tokenizer = AutoTokenizer.from_pretrained(
        str(self.model_path),
        local_files_only=True,
    )

    def cpu_load() -> None:
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype="auto",
            )
        except (TypeError, ValueError):
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
        self._model.to("cpu")
        self._model.eval()
        self._device = "cpu"

    if device.startswith("cuda"):
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=torch.float16,
            )
            self._model.to(device)
            self._model.eval()
            self._device = device
            print(
                f"[Phoenix][模型2/HY-MT] 硬件自适应：CUDA可用 | {note}",
                flush=True,
            )
            return
        except Exception as exc:
            print(
                "[Phoenix][模型2/HY-MT] CUDA运行失败: "
                f"{type(exc).__name__}: {exc}；自动回退CPU。",
                flush=True,
            )
            _clear_backend(self, torch)

    cpu_load()
    print(
        f"[Phoenix][模型2/HY-MT] 硬件自适应：CPU | {note}",
        flush=True,
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .translation_models import MarianEnZhBackend, NLLBEnZhBackend
    from .hymt_translation_backend import HYMTMedicalTranslationBackend

    if not bool(getattr(MarianEnZhBackend, "_phoenix_portable_runtime", False)):
        def marian_load(self):
            return _load_seq2seq(self, model_kind="Marian")

        MarianEnZhBackend._load = marian_load
        MarianEnZhBackend._phoenix_portable_runtime = True

    if not bool(getattr(NLLBEnZhBackend, "_phoenix_portable_runtime", False)):
        def nllb_load(self):
            return _load_seq2seq(self, model_kind="NLLB")

        NLLBEnZhBackend._load = nllb_load
        NLLBEnZhBackend._phoenix_portable_runtime = True

    if not bool(getattr(HYMTMedicalTranslationBackend, "_phoenix_portable_runtime", False)):
        HYMTMedicalTranslationBackend._load = _load_hymt
        HYMTMedicalTranslationBackend._phoenix_portable_runtime = True

    print(
        "[Phoenix][硬件兼容] 本地翻译模型1/2启用通用运行层："
        "不绑定显卡型号；兼容CUDA优先，异常自动CPU回退。",
        flush=True,
    )
