from __future__ import annotations

from pathlib import Path

from core.model_hub import ModelHub
from core.model_pool_policy import TEACHER_CORE, attach_model_pool_policy
from model_adapters.body_part import BodyPartAdapter
from model_adapters.yolo_lesion import YoloLesionAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "04_AI模型"


def build_model_hub():
    """Build the hospital frontline pool.

    Only hospital-qualified lightweight models are registered here. Large 3D
    disease models, generic segmenters and VLMs are teacher assets and never
    auto-run on a patient case.
    """
    hub = ModelHub()

    hub.register(
        BodyPartAdapter(
            ROOT
            / "路由模型"
            / "BodyPartRegression"
            / "phoenix_export"
            / "BodyPartRegression_128x128.onnx"
        )
    )

    hub.register(
        YoloLesionAdapter(
            "fracture_rescbam",
            ROOT / "视觉B_骨折防护" / "YOLOv8_ResCBAM.onnx",
            task="detect",
        )
    )

    hub.register(
        YoloLesionAdapter(
            "fractureatlas_localization",
            ROOT
            / "00_批量部署暂存"
            / "原始权重"
            / "yolov8_localization_fractureAtlas.pt",
            task="detect",
        )
    )

    hub.register(
        YoloLesionAdapter(
            "fractureatlas_segmentation",
            ROOT
            / "00_批量部署暂存"
            / "原始权重"
            / "yolov8_segmentation_fractureAtlas.pt",
            task="segment",
        )
    )

    return attach_model_pool_policy(hub)


def add_deployable_ct_diagnostics(hub):
    """Compatibility hook.

    CT students are registered here only after they pass the hospital machine
    gate: executed=True, acceptable RAM and acceptable latency. No large teacher
    is allowed to fill a frontline gap automatically.
    """
    return attach_model_pool_policy(hub)


def build_full_model_hub():
    """Build the complete current hospital-qualified runtime pool."""
    return add_deployable_ct_diagnostics(build_model_hub())


def add_research_teacher_models(hub):
    """Attach teacher metadata without registering teachers for patient runtime."""
    hub.teacher_catalog = dict(TEACHER_CORE)
    return attach_model_pool_policy(hub)


def add_research_segmentation_models(hub):
    """Attach lazy teacher segmentation objects for offline distillation only."""
    from model_adapters.ct_segmentation_runtime import CT_SEGMENTATION_POOL

    hub.teacher_segmentation_pool = dict(CT_SEGMENTATION_POOL)
    return attach_model_pool_policy(hub)
