from __future__ import annotations

from typing import Dict, Iterable


MODEL_ROLES: Dict[str, str] = {
    "body_part_regression": "router",

    # Current hospital-qualified bone DR chain.
    "fracture_rescbam": "diagnostic",
    "fractureatlas_localization": "localization",
    "fractureatlas_segmentation": "segmentation",

    # Planned lightweight students.
    "ich_2p5d_student": "diagnostic",
    "ich_segmentation_student": "segmentation",
    "brain_infarct_2p5d_student": "diagnostic",
    "brain_atrophy_quant_student": "diagnostic",
    "chest_dr_nano_detector": "diagnostic",
    "renal_stone_student": "diagnostic",
    "sbo_2p5d_student": "diagnostic",
    "appendicitis_2p5d_student": "diagnostic",
    "lumbar_mri_student": "diagnostic",

    # Teacher-only components.
    "blast_ct_head": "teacher",
    "monai_lung_nodule_ct": "teacher",
    "torchxrayvision_chest": "screening",
    "medsam2": "teacher_segmentation",
    "vista3d": "teacher_segmentation",
    "segvol": "teacher_segmentation",
    "wholebody_ct_seg": "teacher_segmentation",
    "m3d_clip": "teacher_encoder",
    "rad_dino_maira2": "teacher_encoder",
    "medgemma_1p5_4b": "teacher_reasoning",
}


HELPER_ROLES = {
    "router",
    "segmentation",
    "localization",
    "anatomy_segmentation",
    "lesion_segmentation",
    "encoder",
    "embedding",
}

TEACHER_ROLES = {
    "teacher",
    "teacher_segmentation",
    "teacher_encoder",
    "teacher_reasoning",
}


def model_role(model_name: str) -> str:
    return MODEL_ROLES.get(str(model_name), "unknown")


def is_diagnostic_model(model_name: str) -> bool:
    return model_role(model_name) == "diagnostic"


def is_screening_model(model_name: str) -> bool:
    return model_role(model_name) == "screening"


def is_helper_model(model_name: str) -> bool:
    return model_role(model_name) in HELPER_ROLES


def is_teacher_model(model_name: str) -> bool:
    return model_role(model_name) in TEACHER_ROLES


def roles_for(names: Iterable[str]) -> dict[str, str]:
    return {str(name): model_role(str(name)) for name in names}
