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
    return _modules_ready(_LLM_MODULES)


def local_seq2seq_runtime_ready() -> bool:
    return _modules_ready(_SEQ2SEQ_MODULES)


def _embedding_count(engine) -> int:
    try:
        with engine.db._lock:
            row = engine.db._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_name=?",
                (engine.model_name,),
            ).fetchone()
        return int(row[0] if row else 0)
    except Exception:
        return 0


def _embedding_readiness(engine) -> dict:
    chunks = int(engine.db.count_chunks())
    vectors = _embedding_count(engine)
    missing = max(0, chunks - vectors)
    model_ready = model_dir_ready(engine.model_path)
    runtime_ready = _module_importable("sentence_transformers")
    ready = bool(
        model_ready
        and runtime_ready
        and missing == 0
    )
    if not model_ready:
        state = "model_missing"
        label = "语义模型未下载或文件不完整"
    elif not runtime_ready:
        state = "runtime_missing"
        label = "语义组件缺失或加载失败"
    elif chunks == 0:
        state = "ready"
        label = "语义检索就绪（资料库为空）"
    elif missing:
        state = "index_incomplete"
        label = f"语义索引 {vectors}/{chunks}"
    else:
        state = "ready"
        label = f"语义索引 {vectors}/{chunks} READY"
    return {
        "state": state,
        "label": label,
        "ready": ready,
        "model_ready": model_ready,
        "runtime_ready": runtime_ready,
        "chunks": chunks,
        "vectors": vectors,
        "missing": missing,
        "device": (
            engine.device
            if model_ready and runtime_ready
            else "unavailable"
        ),
    }


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

    EmbeddingEngine.available = lambda self: model_dir_ready(
        self.model_path
    )
    EmbeddingEngine.readiness = _embedding_readiness
    _Seq2SeqBackend.available = lambda self: model_dir_ready(
        self.model_path
    )

    original_llm_available = LocalLLM.available

    def llm_available(self, profile=None):
        if not original_llm_available(self, profile):
            return False
        backend = self.backend(profile)
        if backend != "transformers_local":
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
