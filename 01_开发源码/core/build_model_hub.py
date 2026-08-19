from __future__ import annotations

from pathlib import Path

from core.model_hub import ModelHub
from core.model_pool_policy import attach_model_pool_policy
from model_adapters.body_part import BodyPartAdapter
from model_adapters.yolo_lesion import YoloLesionAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "04_AI模型"


def build_model_hub():
    """Build the always-on frontline model pool.

    Only models that materially participate in a hospital diagnostic path are
    registered here. Research encoders, report teachers, generic segmentation
    models and screening-only models are deliberately excluded from the active
    runtime pool.
    """
    hub = ModelHub()

    # CT anatomy router. It is not a disease model, but it is retained because
    # its output directly selects the disease specialist in stage 2.
    hub.register(
        BodyPartAdapter(
            ROOT / "路由模型/BodyPartRegression/phoenix_export/BodyPartRegression_128x128.onnx"
        )
    )

    # Bone DR diagnostic + localization/segmentation helpers.
    hub.register(
        YoloLesionAdapter(
            "fracture_rescbam",
            ROOT / "视觉B_骨折防护/YOLOv8_ResCBAM.onnx",
            task="detect",
        )
    )

    hub.register(
        YoloLesionAdapter(
            "fractureatlas_localization",
            ROOT / "00_批量部署暂存/原始权重/yolov8_localization_fractureAtlas.pt",
            task="detect",
        )
    )

    hub.register(
        YoloLesionAdapter(
            "fractureatlas_segmentation",
            ROOT / "00_批量部署暂存/原始权重/yolov8_segmentation_fractureAtlas.pt",
            task="segment",
        )
    )

    return attach_model_pool_policy(hub)


def add_deployable_ct_diagnostics(hub):
    """Register CT disease models that are deployable on this machine."""
    from model_adapters.blast_ct import BlastCTAdapter

    hub.register(
        BlastCTAdapter(
            ROOT
            / "CT病灶模型"
            / "BLAST_CT_头颅出血"
            / "cache"
        )
    )

    # MONAI lung RetinaNet is a real disease detector, but the hospital
    # i3-12100/8GB/K420 profile cannot run it within the frontline latency and
    # memory budget. Keep it available only on modern-GPU development systems.
    if hub.hardware_profile.heavy_3d_allowed:
        from model_adapters.monai_lung_nodule_ct import MonaiLungNoduleCTAdapter

        hub.register(
            MonaiLungNoduleCTAdapter()
        )

    return attach_model_pool_policy(hub)


def build_full_model_hub():
    """Build the production Phoenix runtime model hub.

    "Full" now means the complete *frontline diagnostic* pool, not every model
    stored on the SSD. Teacher models and experimental components remain on disk
    for distillation/research but are not registered into patient-case runtime.
    """
    hub = build_model_hub()
    hub = add_deployable_ct_diagnostics(hub)
    return attach_model_pool_policy(hub)


# ---------------------------------------------------------------------------
# Research/archive registration is intentionally opt-in.
# These helpers are kept so downloaded assets remain reusable for distillation
# and experiments, but PhoenixRuntime never calls them during clinical cases.
# ---------------------------------------------------------------------------

def add_research_teacher_models(hub):
    from model_adapters.hf_directory import HFDirectoryAdapter
    from model_adapters.medsam2 import MedSAM2Adapter

    configs = [
        ("rad_dino", "教师模型/06_RAD_DINO_ModelScope", "image_encoder"),
        ("medgemma_4b", "教师模型/11_MedGemma_1.5_4B_ModelScope", "report_reasoning"),
        ("medgemma_27b", "教师模型/12_MedGemma_27B_ModelScope", "teacher_reasoning"),
        ("medsiglip_448", "教师模型/13_MedSigLIP_448_ModelScope", "image_encoder"),
        ("maira2", "教师模型/14_MAIRA_2_ModelScope", "report_teacher"),
        ("rad_dino_maira2", "教师模型/15_RAD_DINO_MAIRA2_ModelScope", "report_pipeline"),
    ]

    for name, rel_path, role in configs:
        hub.register(
            HFDirectoryAdapter(
                name,
                ROOT / rel_path,
                role,
            )
        )

    hub.register(
        MedSAM2Adapter(
            ROOT / "待接入模型/MedSAM2"
        )
    )
    return attach_model_pool_policy(hub)


def add_research_segmentation_models(hub):
    from model_adapters.sam_med3d import SAMMed3DAdapter
    from model_adapters.totalsegmentator import TotalSegmentatorAdapter

    hub.register(
        SAMMed3DAdapter(
            ROOT / "待接入模型/SAM-Med3D",
            ROOT / "待接入模型/SAM-Med3D/checkpoint/sam_med3d_turbo.pth",
        )
    )
    hub.register(TotalSegmentatorAdapter())
    return attach_model_pool_policy(hub)
