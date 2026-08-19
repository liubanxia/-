from __future__ import annotations


CAPABILITY_CHAIN = (
    "anatomy_localization",
    "anatomy_segmentation",
    "lesion_localization",
    "lesion_segmentation",
    "characterization",
    "differential",
)


FRONTLINE_ACTIVE = {
    "body_part_regression": {
        "role": "router",
        "region": "ct",
        "capabilities": ["anatomy_localization"],
        "clinical_value": "CT anatomy routing for downstream disease chains",
    },
    "fracture_rescbam": {
        "role": "diagnostic",
        "region": "bone_dr",
        "capabilities": ["lesion_localization", "characterization", "differential"],
        "clinical_value": "bone DR fracture detection",
    },
    "fractureatlas_localization": {
        "role": "localization",
        "region": "bone_dr",
        "capabilities": ["lesion_localization"],
        "clinical_value": "fracture localization support",
    },
    "fractureatlas_segmentation": {
        "role": "segmentation",
        "region": "bone_dr",
        "capabilities": ["lesion_segmentation"],
        "clinical_value": "fracture segmentation support",
    },
}


FRONTLINE_PLANNED = {
    "ich_2p5d_student": {
        "region": "head_ct",
        "capabilities": ["lesion_localization", "characterization", "differential"],
        "teachers": ["rsna_ich_teacher"],
    },
    "ich_segmentation_student": {
        "region": "head_ct",
        "capabilities": ["lesion_segmentation"],
        "teachers": ["blast_ct_head", "medsam2"],
    },
    "brain_infarct_2p5d_student": {
        "region": "head_ct",
        "capabilities": ["lesion_localization", "lesion_segmentation", "characterization"],
        "teachers": ["aisd_teacher"],
    },
    "brain_atrophy_quant_student": {
        "region": "head_ct",
        "capabilities": ["anatomy_segmentation", "characterization"],
        "teachers": ["ctseg"],
    },
    "chest_dr_nano_detector": {
        "region": "chest_dr",
        "capabilities": ["lesion_localization", "characterization", "differential"],
        "teachers": ["rsna_pneumonia", "vindr_cxr", "rad_dino_maira2"],
    },
    "renal_stone_student": {
        "region": "abdomen_pelvis_ct",
        "capabilities": ["anatomy_localization", "lesion_localization", "lesion_segmentation", "characterization"],
        "teachers": ["wholebody_ct_seg", "renal_cect_seg"],
    },
    "sbo_2p5d_student": {
        "region": "abdomen_pelvis_ct",
        "capabilities": ["anatomy_localization", "lesion_localization", "characterization", "differential"],
        "teachers": ["dba_drp_sbo", "wholebody_ct_seg"],
    },
    "appendicitis_2p5d_student": {
        "region": "abdomen_pelvis_ct",
        "capabilities": ["anatomy_localization", "lesion_localization", "lesion_segmentation", "characterization", "differential"],
        "teachers": ["ai_ppendix", "wholebody_ct_seg", "medsam2"],
    },
    "lumbar_mri_student": {
        "region": "lumbar_mri",
        "capabilities": ["anatomy_localization", "characterization", "differential"],
        "teachers": ["spinenet_v2"],
    },
}


TEACHER_CORE = {
    "rsna_ich_teacher": {
        "capabilities": ["lesion_localization", "characterization", "differential"],
        "storage_policy": "keep_until_student_passes",
    },
    "blast_ct_head": {
        "capabilities": ["lesion_localization", "lesion_segmentation", "characterization"],
        "storage_policy": "teacher_only",
    },
    "ctseg": {
        "capabilities": ["anatomy_segmentation", "quantification"],
        "storage_policy": "keep_teacher",
    },
    "wholebody_ct_seg": {
        "capabilities": ["anatomy_segmentation"],
        "storage_policy": "keep_teacher",
    },
    "pancreas_dints_seg": {
        "capabilities": ["anatomy_segmentation", "lesion_segmentation"],
        "storage_policy": "keep_teacher",
    },
    "renal_cect_seg": {
        "capabilities": ["anatomy_segmentation"],
        "storage_policy": "keep_teacher",
    },
    "medsam2": {
        "capabilities": ["lesion_segmentation"],
        "storage_policy": "keep_teacher",
    },
    "vista3d": {
        "capabilities": ["anatomy_segmentation", "lesion_segmentation"],
        "storage_policy": "keep_single_canonical_copy",
    },
    "segvol": {
        "capabilities": ["anatomy_segmentation", "lesion_segmentation"],
        "storage_policy": "keep_single_canonical_copy",
    },
    "m3d_clip": {
        "capabilities": ["feature_embedding", "characterization"],
        "storage_policy": "keep_teacher",
    },
    "rad_dino_maira2": {
        "capabilities": ["feature_embedding", "characterization"],
        "storage_policy": "keep_teacher",
    },
    "medgemma_1p5_4b": {
        "capabilities": ["characterization", "differential"],
        "storage_policy": "keep_one_reasoning_teacher",
    },
    "merlin_disease_backbone": {
        "capabilities": ["feature_embedding", "characterization"],
        "storage_policy": "keep_single_checkpoint",
    },
    "ai_ppendix": {
        "capabilities": ["anatomy_localization", "lesion_localization", "lesion_segmentation", "characterization", "differential"],
        "storage_policy": "download_teacher_when_needed",
    },
    "spinenet_v2": {
        "capabilities": ["anatomy_localization", "characterization", "differential"],
        "storage_policy": "research_teacher_noncommercial",
    },
}


DELETE_NOW = {
    "torchxrayvision_chest": "screening-only and replaced by bbox-based chest DR plan",
    "sam_med3d": "large generic segmenter overlaps MedSAM2/VISTA3D/SegVol",
    "monai_multi_organ_seg": "large overlap with whole-body 104-structure segmenter",
    "monai_renal_unest": "heavier duplicate renal segmentation teacher",
    "monai_spleen": "single-organ duplicate covered by whole-body segmentation",
    "monai_wholebrain_unest": "MRI-oriented model was incorrectly kept in CT specialist pool",
    "monai_pediatric_abdomen": "not in current adult hospital target set",
    "suprem_extra_backbones": "keep only the smallest selected SegResNet teacher checkpoint",
    "duplicate_vista3d": "keep only CT_分割/VISTA3D-HF canonical copy",
    "duplicate_segvol": "keep only CT_分割/SegVol canonical copy",
}


EXTRACT_OR_DROP_HEAVY_TEACHERS = {
    "Lingshu-32B": "drop full language-only package",
    "MedGemma-27B": "drop full 27B package; retain one smaller reasoning teacher",
    "HealthGPT-Pro-8B": "drop redundant language-only package",
    "Lingshu-7B": "drop redundant language-only package",
    "Fleming-VL-8B": "drop unvalidated redundant VLM package",
    "Lingshu-I-8B": "drop unvalidated redundant VLM package",
    "LLaVA-Med-7B": "drop unvalidated redundant VLM package",
    "Hulu-Med-4B": "drop unvalidated redundant VLM package",
    "HealthGPT-Pro-4B": "drop redundant language-only package",
    "MedGemma-4B-old": "drop duplicate old 4B package",
    "MAIRA-2-full": "keep RAD-DINO/MAIRA vision teacher; drop duplicate full report stack",
}


ARCHIVED_FROM_FRONTLINE = {
    **{name: reason for name, reason in DELETE_NOW.items()},
    "blast_ct_head": "teacher only; too heavy for routine hospital CPU use",
    "monai_lung_nodule_ct": "teacher/development benchmark only until a lightweight student passes hospital execution",
    "medsam2": "teacher segmentation component; not auto-run in patient frontline",
    "vista3d": "teacher segmentation component; not auto-run in patient frontline",
    "segvol": "teacher segmentation component; not auto-run in patient frontline",
    "wholebody_ct_seg": "teacher anatomy segmentation component; not auto-run in patient frontline",
    "m3d_clip": "teacher feature encoder; not auto-run in patient frontline",
    "rad_dino_maira2": "teacher feature encoder; not auto-run in patient frontline",
    "medgemma_1p5_4b": "reasoning teacher; not auto-run in patient frontline",
}


CLINICAL_GAPS = {
    "head_ct": "lightweight ICH/infarct/atrophy students not yet hospital-qualified",
    "chest_ct": "no hospital-qualified lightweight chest CT disease chain yet",
    "abdomen_pelvis_ct": "renal stone/SBO/appendicitis students not yet hospital-qualified",
    "chest_dr": "Nano detector not yet hospital-qualified",
    "lumbar_mri": "lightweight student not yet hospital-qualified",
}


MODEL_REGION_COVERAGE = {
    "ich_2p5d_student": {"head"},
    "brain_infarct_2p5d_student": {"head"},
    "brain_atrophy_quant_student": {"head"},
    "renal_stone_student": {"abdomen", "pelvis"},
    "sbo_2p5d_student": {"abdomen", "pelvis"},
    "appendicitis_2p5d_student": {"abdomen", "pelvis"},
}


def model_pool_snapshot(hub) -> dict:
    registered = list(getattr(hub, "models", {}).keys())
    profile = getattr(hub, "hardware_profile", None)

    return {
        "active_registered_models": registered,
        "frontline_active": list(FRONTLINE_ACTIVE),
        "frontline_planned": dict(FRONTLINE_PLANNED),
        "teacher_core": dict(TEACHER_CORE),
        "delete_now": dict(DELETE_NOW),
        "extract_or_drop_heavy_teachers": dict(EXTRACT_OR_DROP_HEAVY_TEACHERS),
        "archived_from_frontline": dict(ARCHIVED_FROM_FRONTLINE),
        "clinical_gaps": dict(CLINICAL_GAPS),
        "capability_chain": list(CAPABILITY_CHAIN),
        "hardware_mode": getattr(profile, "mode", "") if profile else "",
        "heavy_3d_allowed": bool(
            getattr(profile, "heavy_3d_allowed", False)
        ) if profile else False,
    }


def attach_model_pool_policy(hub):
    hub.model_pool_policy = model_pool_snapshot(hub)
    return hub
