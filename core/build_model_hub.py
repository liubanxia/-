from pathlib import Path

from core.model_hub import ModelHub
from core.model_pool_policy import attach_model_pool_policy
from model_adapters.body_part import BodyPartAdapter
from model_adapters.yolo_lesion import YoloLesionAdapter


def build_model_hub(model_root=None):
    """Build a lightweight local model hub from a caller-supplied model root."""
    root = Path(model_root) if model_root else Path(__file__).resolve().parents[1] / "models"
    hub = ModelHub()
    bpr = root / "BodyPartRegression"
    if bpr.exists(): hub.register(BodyPartAdapter(bpr))
    candidates = {
        "fracture_rescbam": (root / "YOLOv8_ResCBAM.onnx", "detect"),
        "fractureatlas_localization": (root / "fractureatlas_detect.pt", "detect"),
        "fractureatlas_segmentation": (root / "fractureatlas_segment.pt", "segment"),
    }
    for name, (path, task) in candidates.items():
        if path.exists(): hub.register(YoloLesionAdapter(name, path, task=task))
    return attach_model_pool_policy(hub)


def build_full_model_hub(model_root=None): return build_model_hub(model_root)
