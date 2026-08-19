from __future__ import annotations


ALWAYS_FRONTLINE = {
    "body_part_regression": {
        "role": "router",
        "clinical_value": "CT anatomy routing for second-stage disease models",
    },
    "blast_ct_head": {
        "role": "diagnostic",
        "clinical_value": "head CT hemorrhage diagnosis/localization",
    },
    "fracture_rescbam": {
        "role": "diagnostic",
        "clinical_value": "bone DR fracture detection",
    },
    "fractureatlas_localization": {
        "role": "localization",
        "clinical_value": "fracture localization support",
    },
    "fractureatlas_segmentation": {
        "role": "segmentation",
        "clinical_value": "fracture segmentation support",
    },
}

CONDITIONAL_FRONTLINE = {
    "monai_lung_nodule_ct": {
        "role": "diagnostic",
        "requirement": "modern GPU / heavy_3d_allowed",
        "clinical_value": "chest CT pulmonary nodule detection",
    },
}

ARCHIVED_FROM_FRONTLINE = {
    "torchxrayvision_chest": "screening-only; not accepted as a formal diagnostic model",
    "rad_dino": "image encoder; no direct disease diagnosis",
    "medsiglip_448": "image encoder; no direct disease diagnosis",
    "medgemma_4b": "reasoning/report teacher; not frontline image diagnosis",
    "medgemma_27b": "teacher model; not deployable on hospital workstation",
    "maira2": "report teacher; not frontline disease detector",
    "rad_dino_maira2": "report pipeline/teacher; not frontline disease detector",
    "medsam2": "generic segmentation; not a disease diagnosis model",
    "sam_med3d": "generic segmentation; not a disease diagnosis model",
    "totalsegmentator": "anatomy segmentation; not a disease diagnosis model",
}


CLINICAL_GAPS = {
    "chest_ct_hospital": "no hospital-deployable pulmonary nodule/disease detector yet",
    "abdomen_pelvis_ct": "no frontline abdominal/pelvic disease detector yet",
    "chest_dr": "screening model removed; diagnostic detector/localizer replacement required",
}


def model_pool_snapshot(hub) -> dict:
    registered = list(getattr(hub, "models", {}).keys())
    profile = getattr(hub, "hardware_profile", None)

    return {
        "active_registered_models": registered,
        "always_frontline": list(ALWAYS_FRONTLINE),
        "conditional_frontline": list(CONDITIONAL_FRONTLINE),
        "archived_from_frontline": dict(ARCHIVED_FROM_FRONTLINE),
        "clinical_gaps": dict(CLINICAL_GAPS),
        "hardware_mode": getattr(profile, "mode", "") if profile else "",
        "heavy_3d_allowed": bool(
            getattr(profile, "heavy_3d_allowed", False)
        ) if profile else False,
    }


def attach_model_pool_policy(hub):
    hub.model_pool_policy = model_pool_snapshot(hub)
    return hub
