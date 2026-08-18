from model_adapters.monai_lung_nodule_ct import MonaiLungNoduleCTAdapter
from pathlib import Path

from core.model_hub import ModelHub
from model_adapters.body_part import BodyPartAdapter
from model_adapters.yolo_lesion import YoloLesionAdapter
from model_adapters.hf_directory import HFDirectoryAdapter
from model_adapters.medsam2 import MedSAM2Adapter
from model_adapters.torchxrayvision_chest import TorchXRayVisionChestAdapter


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
        TorchXRayVisionChestAdapter(
            ROOT / "教师模型/00_源码_TorchXRayVision",
            ROOT / "胸片模型/TorchXRayVision_weights",
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
    if "monai_lung_nodule_ct" not in hub.models:
        hub.register(
            MonaiLungNoduleCTAdapter()
        )

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


def add_extension_pools(hub):
    """
    Phoenix 扩展专家层。

    这里只挂登记对象，不加载权重、不运行前向。
    因此不会改变既有 CT/DR 主链的运行行为。
    """

    from model_adapters.phoenix_extended_pool import EXTENDED_POOL
    from model_adapters.specialist_pool import SPECIALIST_POOL
    from ai_models.component_registry import PhoenixComponentRegistry

    hub.extended_pool = EXTENDED_POOL
    hub.specialist_pool = SPECIALIST_POOL
    hub.component_registry = PhoenixComponentRegistry()

    return hub


def build_full_model_hub():
    """
    Phoenix 完整模型 Hub。

    已验证主链继续使用 ModelHub.register()。
    新增实验专家通过 extension pools 挂载，
    默认不加载、默认不自动推理。
    """

    hub = build_model_hub()

    hub = add_teacher_models(hub)
    hub = add_segmentation_models(hub)
    hub = add_ct_lesion_models(hub)

    hub = add_extension_pools(hub)

    hub = attach_all_expert_services(hub)

    hub = attach_phoenix_expert_stack(hub)
    hub = attach_clinical_output_pipeline(hub)
    hub = attach_clinical_case_controller(hub)
    hub = attach_expert_scheduler(hub)
    return hub


def attach_all_expert_services(hub):
    """
    Phoenix 全专家统一入口。
    此函数只挂接，不主动加载、不启动推理。
    """
    from core.expert_router import EXPERT_ROUTER
    from ai_models.expert_catalog import EXPERT_CATALOG
    from model_adapters.native_specialist_runtime import NATIVE_SPECIALISTS

    hub.expert_catalog = EXPERT_CATALOG
    hub.expert_router = EXPERT_ROUTER
    hub.native_specialists = NATIVE_SPECIALISTS

    return hub


def attach_phoenix_expert_stack(hub):
    from core.phoenix_expert_stack import (
        PHOENIX_EXPERT_STACK,
    )

    hub.expert_stack = PHOENIX_EXPERT_STACK
    return hub


def attach_clinical_output_pipeline(hub):
    from core.clinical_output_pipeline import (
        CLINICAL_OUTPUT_PIPELINE,
    )
    from core.report_teacher_pool import (
        REPORT_TEACHER_POOL,
    )

    hub.clinical_output_pipeline = CLINICAL_OUTPUT_PIPELINE
    hub.report_teacher_pool = REPORT_TEACHER_POOL
    return hub


def attach_clinical_case_controller(hub):
    from core.clinical_case_controller import (
        CLINICAL_CASE_CONTROLLER,
    )
    from output.clinical_delivery_bridge import (
        CLINICAL_DELIVERY_BRIDGE,
    )

    hub.clinical_case_controller = CLINICAL_CASE_CONTROLLER
    hub.clinical_delivery_bridge = CLINICAL_DELIVERY_BRIDGE

    return hub


def attach_expert_scheduler(hub):
    from core.expert_inference_scheduler import (
        EXPERT_INFERENCE_SCHEDULER,
    )
    from core.expert_feature_memory import (
        EXPERT_FEATURE_MEMORY,
    )

    hub.expert_scheduler = EXPERT_INFERENCE_SCHEDULER
    hub.expert_feature_memory = EXPERT_FEATURE_MEMORY

    return hub
