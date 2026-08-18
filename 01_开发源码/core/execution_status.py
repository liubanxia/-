from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


SUCCESS_STATES = {"ok", "success", "completed", "loaded"}
FAILURE_STATES = {"error", "failed", "missing", "not_loaded", "skipped"}


@dataclass
class ModelExecution:
    model_name: str
    selected: bool = True
    loaded: bool = False
    executed: bool = False
    status: str = "not_started"
    error: str = ""
    processed_images: int = 0
    lesion_count: int | None = None
    device: str = ""
    backend: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid_negative(self) -> bool:
        return self.executed and self.status == "success" and self.lesion_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "selected": self.selected,
            "loaded": self.loaded,
            "executed": self.executed,
            "status": self.status,
            "error": self.error,
            "processed_images": self.processed_images,
            "lesion_count": self.lesion_count,
            "device": self.device,
            "backend": self.backend,
            "valid_negative": self.valid_negative,
            "metadata": dict(self.metadata),
        }


def execution_from_raw(
    model_name: str,
    raw: Any,
    load_status: str = "",
    load_error: str = "",
) -> ModelExecution:
    execution = ModelExecution(model_name=model_name)
    execution.loaded = load_status == "loaded"

    if load_status and load_status != "loaded":
        execution.status = load_status
        execution.error = load_error or f"模型未加载: {load_status}"
        return execution

    if not isinstance(raw, dict):
        execution.status = "failed"
        execution.error = f"模型输出类型无效: {type(raw).__name__}"
        return execution

    if raw.get("error"):
        execution.status = "failed"
        execution.error = str(raw.get("error"))
        return execution

    execution.executed = True
    execution.status = "success"
    execution.processed_images = int(raw.get("processed_images", 0) or 0)
    lesions = raw.get("lesions")
    execution.lesion_count = len(lesions) if isinstance(lesions, list) else 0
    execution.device = str(raw.get("device", "") or "")
    execution.backend = str(raw.get("inference_backend", raw.get("backend", "")) or "")
    execution.metadata = {
        k: v
        for k, v in raw.items()
        if k not in {"lesions", "error"}
    }
    return execution


def summarize_executions(executions: Iterable[ModelExecution]) -> Dict[str, Any]:
    items: List[ModelExecution] = list(executions)
    diagnostic_executed = any(x.executed for x in items)
    failures = [x.model_name for x in items if x.status == "failed"]
    return {
        "diagnostic_executed": diagnostic_executed,
        "all_successful": bool(items) and all(x.status == "success" for x in items),
        "failed_models": failures,
        "models": [x.to_dict() for x in items],
    }
