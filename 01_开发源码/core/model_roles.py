from __future__ import annotations

from typing import Dict


MODEL_ROLES: Dict[str, str] = {
    "body_part_regression": "router",
    "blast_ct_head": "diagnostic",
    "monai_lung_nodule_ct": "diagnostic",
    "torchxrayvision_chest": "screening",
    "fracture_rescbam": "diagnostic",
    "fractureatlas_localization": "localization",
    "fractureatlas_segmentation": "segmentation",
}


NON_DIAGNOSTIC_ROLES = {
    "router",
    "segmentation",
    "localization",
    "encoder",
    "embedding",
}


def model_role(model_name: str) -> str:
    return MODEL_ROLES.get(str(model_name), "unknown")


def is_diagnostic_model(model_name: str) -> bool:
    role = model_role(model_name)
    return role not in NON_DIAGNOSTIC_ROLES and role != "unknown"
