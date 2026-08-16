from pathlib import Path

from core.model_hub import ModelHub
from model_adapters.body_part import BodyPartAdapter
from model_adapters.yolo_lesion import YoloLesionAdapter
from model_adapters.hf_directory import HFDirectoryAdapter
from model_adapters.medsam2 import MedSAM2Adapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "04_AI模型"


def build_model_hub():
    hub = ModelHub()

    hub.register(
        BodyPartAdapter(
            ROOT / "路由模型/BodyPartRegression/phoenix_export/BodyPartRegression_128x128.onnx"
        )
    )

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

    return hub


def add_teacher_models(hub):

    configs = [
        (
            "rad_dino",
            "教师模型/06_RAD_DINO_ModelScope",
            "image_encoder",
        ),
        (
            "medgemma_4b",
            "教师模型/11_MedGemma_1.5_4B_ModelScope",
            "report_reasoning",
        ),
        (
            "medgemma_27b",
            "教师模型/12_MedGemma_27B_ModelScope",
            "teacher_reasoning",
        ),
        (
            "medsiglip_448",
            "教师模型/13_MedSigLIP_448_ModelScope",
            "image_encoder",
        ),
        (
            "maira2",
            "教师模型/14_MAIRA_2_ModelScope",
            "report_teacher",
        ),
        (
            "rad_dino_maira2",
            "教师模型/15_RAD_DINO_MAIRA2_ModelScope",
            "report_pipeline",
        ),
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

    return hub


def build_full_model_hub():
    hub = build_model_hub()
    return add_teacher_models(hub)


def add_segmentation_models(hub):
    from model_adapters.sam_med3d import SAMMed3DAdapter
    from model_adapters.totalsegmentator import TotalSegmentatorAdapter

    hub.register(
        SAMMed3DAdapter(
            ROOT / "待接入模型/SAM-Med3D",
            ROOT / "待接入模型/SAM-Med3D/checkpoint/sam_med3d_turbo.pth",
        )
    )

    hub.register(
        TotalSegmentatorAdapter()
    )

    return hub


def add_ct_lesion_models(hub):
    from model_adapters.blast_ct import BlastCTAdapter

    hub.register(
        BlastCTAdapter(
            ROOT
            / "CT病灶模型"
            / "BLAST_CT_头颅出血"
            / "cache"
        )
    )

    return hub
