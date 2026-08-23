CAPABILITY_CHAIN = ("anatomy_localization", "anatomy_segmentation", "lesion_localization", "lesion_segmentation", "characterization", "differential")
MODEL_REGION_COVERAGE = {
    "ich_2p5d_student": {"head"}, "brain_infarct_2p5d_student": {"head"}, "brain_atrophy_quant_student": {"head"},
    "renal_stone_student": {"abdomen", "pelvis"}, "sbo_2p5d_student": {"abdomen", "pelvis"}, "appendicitis_2p5d_student": {"abdomen", "pelvis"},
}
TEACHER_CORE = {
    "blast_ct_head": {"capabilities": ["lesion_localization", "lesion_segmentation", "characterization"]},
    "medsam2": {"capabilities": ["lesion_segmentation"]},
    "vista3d": {"capabilities": ["anatomy_segmentation", "lesion_segmentation"]},
    "segvol": {"capabilities": ["anatomy_segmentation", "lesion_segmentation"]},
    "m3d_clip": {"capabilities": ["feature_embedding", "characterization"]},
    "rad_dino_maira2": {"capabilities": ["feature_embedding", "characterization"]},
}


def model_pool_snapshot(hub):
    profile = getattr(hub, "hardware_profile", None)
    return {
        "active_registered_models": list(getattr(hub, "models", {}).keys()),
        "capability_chain": list(CAPABILITY_CHAIN),
        "teacher_core": dict(TEACHER_CORE),
        "hardware_mode": getattr(profile, "mode", "") if profile else "",
        "heavy_3d_allowed": bool(getattr(profile, "heavy_3d_allowed", False)) if profile else False,
    }


def attach_model_pool_policy(hub):
    hub.model_pool_policy = model_pool_snapshot(hub)
    return hub
