MODEL_ROLES = {
    "body_part_regression": "router",
    "fracture_rescbam": "diagnostic",
    "fractureatlas_localization": "localization",
    "fractureatlas_segmentation": "segmentation",
    "ich_2p5d_student": "diagnostic",
    "ich_segmentation_student": "segmentation",
    "brain_infarct_2p5d_student": "diagnostic",
    "brain_atrophy_quant_student": "diagnostic",
    "chest_dr_nano_detector": "diagnostic",
    "renal_stone_student": "diagnostic",
    "sbo_2p5d_student": "diagnostic",
    "appendicitis_2p5d_student": "diagnostic",
    "lumbar_mri_student": "diagnostic",
    "torchxrayvision_chest": "screening",
}
HELPER_ROLES = {"router", "segmentation", "localization", "anatomy_segmentation", "lesion_segmentation", "encoder", "embedding"}
TEACHER_ROLES = {"teacher", "teacher_segmentation", "teacher_encoder", "teacher_reasoning"}


def model_role(name): return MODEL_ROLES.get(str(name), "unknown")
def is_diagnostic_model(name): return model_role(name) == "diagnostic"
def is_screening_model(name): return model_role(name) == "screening"
def is_helper_model(name): return model_role(name) in HELPER_ROLES
def is_teacher_model(name): return model_role(name) in TEACHER_ROLES
