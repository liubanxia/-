from __future__ import annotations

import pydicom

from .ct_chest_gate import is_chest_ct
from .ct_head_gate import is_head_ct


XRAY_MODALITIES = {"DX", "DR", "CR", "XR"}
MRI_MODALITIES = {"MR", "MRI"}

CHEST_WORDS = {
    "CHEST", "THORAX", "LUNG",
    "胸", "肺",
}

BONE_WORDS = {
    "HAND", "WRIST", "FOREARM", "ELBOW",
    "HUMERUS", "SHOULDER",
    "FOOT", "ANKLE", "TIBIA", "FIBULA",
    "KNEE", "FEMUR", "HIP", "PELVIS",
    "SPINE", "CERVICAL", "THORACIC",
    "LUMBAR", "SACRUM", "SKULL",
    "手", "腕", "前臂", "肘", "肱骨", "肩",
    "足", "踝", "胫骨", "腓骨", "膝", "股骨",
    "髋", "骨盆", "脊柱", "颈椎", "胸椎",
    "腰椎", "骶骨", "颅骨",
}

_HEAD_ROUTER_WORDS = {
    "HEAD", "BRAIN", "CRANIUM", "SKULL",
    "头", "脑", "颅",
}

_CHEST_ROUTER_WORDS = {
    "CHEST", "THORAX", "LUNG",
    "胸", "肺",
}

_ABDOMEN_ROUTER_WORDS = {
    "ABDOMEN", "ABDOMINAL", "LIVER", "HEPATIC",
    "PANCREAS", "KIDNEY", "RENAL", "SPLEEN",
    "BOWEL", "APPENDIX",
    "腹", "肝", "胰", "肾", "脾", "肠", "阑尾",
}

_PELVIS_ROUTER_WORDS = {
    "PELVIS", "PELVIC", "BLADDER", "PROSTATE", "UTERUS",
    "骨盆", "盆腔", "膀胱", "前列腺", "子宫",
}

_LUMBAR_WORDS = {
    "LUMBAR", "L-SPINE", "L SPINE", "LSPINE",
    "腰椎", "腰骶",
}


def get_modalities(case):
    return {
        str(getattr(series, "modality", "")).upper()
        for series in getattr(case, "series", []) or []
    }


def _series_text(series) -> str:
    text = []

    for attr in (
        "description",
        "series_description",
        "study_description",
        "protocol_name",
        "body_part",
    ):
        value = getattr(series, attr, None)
        if value:
            text.append(str(value).upper())

    files = getattr(series, "files", []) or []
    if files:
        try:
            ds = pydicom.dcmread(
                str(files[0]),
                stop_before_pixels=True,
                force=True,
            )
        except Exception:
            ds = None

        if ds is not None:
            for key in (
                "BodyPartExamined",
                "StudyDescription",
                "SeriesDescription",
                "ProtocolName",
            ):
                value = str(getattr(ds, key, "")).upper()
                if value:
                    text.append(value)

    return " ".join(text)


def get_xray_region(case):
    joined = " ".join(
        _series_text(series)
        for series in getattr(case, "series", []) or []
        if str(getattr(series, "modality", "")).upper() in XRAY_MODALITIES
    )

    if any(word in joined for word in CHEST_WORDS):
        return "chest"

    if any(word in joined for word in BONE_WORDS):
        return "bone"

    return "other"


def get_mri_region(case):
    joined = " ".join(
        _series_text(series)
        for series in getattr(case, "series", []) or []
        if str(getattr(series, "modality", "")).upper() in MRI_MODALITIES
    )

    if any(word in joined for word in _LUMBAR_WORDS):
        return "lumbar_spine"

    return "other"


def _normalize_router_regions(router_result) -> set[str]:
    if not isinstance(router_result, dict):
        return set()

    regions: set[str] = set()
    raw_regions = router_result.get("active_body_regions") or []
    if isinstance(raw_regions, str):
        raw_regions = [raw_regions]

    for item in raw_regions:
        value = str(item or "").strip().lower()
        if value:
            regions.add(value)

    text_values = [
        router_result.get("body_part_examined_tag"),
        router_result.get("body_part_display"),
        router_result.get("body_part_examined_tag_raw"),
    ]
    text = " ".join(str(x or "") for x in text_values).upper()

    if any(word.upper() in text for word in _HEAD_ROUTER_WORDS):
        regions.add("head")
    if any(word.upper() in text for word in _CHEST_ROUTER_WORDS):
        regions.add("chest")
    if any(word.upper() in text for word in _ABDOMEN_ROUTER_WORDS):
        regions.add("abdomen")
    if any(word.upper() in text for word in _PELVIS_ROUTER_WORDS):
        regions.add("pelvis")

    return regions


def ct_route_decision(case, router_result=None) -> dict:
    router_regions = _normalize_router_regions(router_result)
    metadata_head = bool(is_head_ct(case))
    metadata_chest = bool(is_chest_ct(case))

    return {
        "metadata_head": metadata_head,
        "metadata_chest": metadata_chest,
        "router_regions": sorted(router_regions),
        "head": metadata_head or "head" in router_regions,
        "chest": metadata_chest or "chest" in router_regions,
        "abdomen": "abdomen" in router_regions,
        "pelvis": "pelvis" in router_regions,
    }


def select_ct_specialists(case, router_result=None):
    """Return the *planned lightweight* second-stage diagnostic chain.

    Teacher models are deliberately not returned here. If a student has not
    been registered yet, Phoenix records it as unavailable instead of silently
    falling back to a large teacher model.
    """
    decision = ct_route_decision(case, router_result)
    models = []

    if decision["head"]:
        models.extend([
            "ich_2p5d_student",
            "ich_segmentation_student",
            "brain_infarct_2p5d_student",
            "brain_atrophy_quant_student",
        ])

    if decision["abdomen"] or decision["pelvis"]:
        models.extend([
            "renal_stone_student",
            "sbo_2p5d_student",
            "appendicitis_2p5d_student",
        ])

    # Chest CT remains a declared gap until a lightweight disease chain passes
    # the hospital executed=True + RAM + latency gate.
    return list(dict.fromkeys(models))


def select_initial_models(case):
    modalities = get_modalities(case)

    if "CT" in modalities:
        return ["body_part_regression"]

    if modalities & XRAY_MODALITIES:
        region = get_xray_region(case)

        if region == "chest":
            return ["chest_dr_nano_detector"]

        if region == "bone":
            return [
                "fracture_rescbam",
                "fractureatlas_localization",
                "fractureatlas_segmentation",
            ]

    if modalities & MRI_MODALITIES:
        if get_mri_region(case) == "lumbar_spine":
            return ["lumbar_mri_student"]

    return []


def select_models(case, router_result=None):
    modalities = get_modalities(case)

    if "CT" in modalities:
        return [
            "body_part_regression",
            *select_ct_specialists(case, router_result),
        ]

    return select_initial_models(case)
