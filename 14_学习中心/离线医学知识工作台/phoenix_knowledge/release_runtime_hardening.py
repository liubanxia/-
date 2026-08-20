from __future__ import annotations

import importlib

from .config import model_dir_ready

_INSTALLED = False
_LLM_MODULES = (
    "torch",
    "transformers",
    "accelerate",
    "safetensors",
)
_SEQ2SEQ_MODULES = _LLM_MODULES + ("sentencepiece",)


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _modules_ready(names: tuple[str, ...]) -> bool:
    return all(_module_importable(name) for name in names)


def local_generation_runtime_ready() -> bool:
    """Runtime required by local Qwen generation."""

    return _modules_ready(_LLM_MODULES)


def local_seq2seq_runtime_ready() -> bool:
    """Runtime required by local Marian/NLLB fallback translation."""

    return _modules_ready(_SEQ2SEQ_MODULES)


def install() -> None:
    """READY requires real weights plus an importable execution runtime."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .llm import LocalLLM
    from .retrieval import EmbeddingEngine
    from .translation_models import (
        MultiModelTranslationEngine,
        _Seq2SeqBackend,
    )
    from .workbench import MedicalKnowledgeWorkbench

    # A pointer/config/status file by itself is not a usable model.
    EmbeddingEngine.available = lambda self: model_dir_ready(
        self.model_path
    )
    _Seq2SeqBackend.available = lambda self: model_dir_ready(
        self.model_path
    )

    if hasattr(EmbeddingEngine, "readiness"):
        original_embedding_readiness = EmbeddingEngine.readiness

        def embedding_readiness(self):
            payload = dict(original_embedding_readiness(self))
            if not model_dir_ready(self.model_path):
                payload.update(
                    {
                        "state": "model_missing",
                        "label": "语义模型未下载或文件不完整",
                        "ready": False,
                        "model_ready": False,
                        "device": "unavailable",
                    }
                )
            return payload

        EmbeddingEngine.readiness = embedding_readiness

    original_llm_available = LocalLLM.available

    def llm_available(self, profile=None):
        if not original_llm_available(self, profile):
            return False
        backend = self.backend(profile)
        if backend != "transformers_local":
            # Explicitly authorized remote service or loopback local server.
            return True
        return local_generation_runtime_ready()

    LocalLLM.available = llm_available

    original_available_backends = (
        MultiModelTranslationEngine.available_backends
    )
    original_active_backends = MultiModelTranslationEngine.active_backends

    def available_backends(self):
        names = list(original_available_backends(self))
        if local_seq2seq_runtime_ready():
            return names
        local_seq2seq = {self.marian.name, self.nllb.name}
        return [
            name for name in names if name not in local_seq2seq
        ]

    def active_backends(
        self,
        target_language="中文",
        smart_level="smart1",
    ):
        backends = list(
            original_active_backends(
                self,
                target_language,
                smart_level,
            )
        )
        if local_seq2seq_runtime_ready():
            return backends
        local_seq2seq = {self.marian.name, self.nllb.name}
        return [
            backend
            for backend in backends
            if getattr(backend, "name", "") not in local_seq2seq
        ]

    MultiModelTranslationEngine.available_backends = available_backends
    MultiModelTranslationEngine.active_backends = active_backends

    original_status = MedicalKnowledgeWorkbench.status

    def status(self):
        payload = original_status(self)
        payload.update(
            {
                "generator_runtime_available": (
                    local_generation_runtime_ready()
                ),
                "translation_seq2seq_runtime_available": (
                    local_seq2seq_runtime_ready()
                ),
                "generator_fast_ready": self.llm.available("fast"),
                "generator_deep_ready": self.llm.available("deep"),
            }
        )
        return payload

    MedicalKnowledgeWorkbench.status = status
