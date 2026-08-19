from __future__ import annotations

import pydicom

from .ct_chest_gate import is_chest_ct
from .ct_head_gate import is_head_ct


XRAY_MODALITIES = {
    "DX", "DR", "CR", "XR",
}

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


def get_modalities(case):
    return {
        str(getattr(series, "modality", "")).upper()
        for series in getattr(case, "series", []) or []
    }


def get_xray_region(case):
    text = []

    for series in getattr(case, "series", []) or []:
        if str(getattr(series, "modality", "")).upper() not in XRAY_MODALITIES:
            continue

        # Prefer already parsed series metadata when the PACS adapter provides
        # it. Fall back to DICOM header reads for old adapters.
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
        if not files:
            continue

        try:
            ds = pydicom.dcmread(
                str(files[0]),
                stop_before_pixels=True,
                force=True,
            )
        except Exception:
            continue

        for key in (
            "BodyPartExamined",
            "StudyDescription",
            "SeriesDescription",
            "ProtocolName",
        ):
            value = str(getattr(ds, key, "")).upper()
            if value:
                text.append(value)

    joined = " ".join(text)

    if any(word in joined for word in CHEST_WORDS):
        return "chest"

    if any(word in joined for word in BONE_WORDS):
        return "bone"

    return "other"


def _normalize_router_regions(router_result) -> set[str]:
    """Extract canonical anatomy from BodyPartRegression output.

    BodyPartRegression returns both ``active_body_regions`` and a normalized
    ``body_part_examined_tag``. The previous Phoenix pipeline ran this router
    but decided specialists before the router result existed. This helper makes
    the model output an actual routing signal while keeping DICOM metadata as a
    fallback.
    """
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
    }


def select_ct_specialists(case, router_result=None):
    decision = ct_route_decision(case, router_result)
    models = []

    # Head and chest specialists are independent. Each adapter performs its own
    # anatomy-aware series binding before inference, so a multi-region Study can
    # safely invoke both.
    if decision["head"]:
        models.append("blast_ct_head")

    if decision["chest"]:
        models.append("monai_lung_nodule_ct")

    return models


def select_initial_models(case):
    modalities = get_modalities(case)

    if "CT" in modalities:
        # True stage 1: run the anatomy router first. Specialist selection is
        # intentionally deferred until its result is available.
        return ["body_part_regression"]

    if modalities & XRAY_MODALITIES:
        region = get_xray_region(case)

        if region == "chest":
            return ["torchxrayvision_chest"]

        if region == "bone":
            return [
                "fracture_rescbam",
                "fractureatlas_localization",
                "fractureatlas_segmentation",
            ]

    return []


def select_models(case, router_result=None):
    """Compatibility selector plus complete CT plan when router output exists."""
    modalities = get_modalities(case)

    if "CT" in modalities:
        return [
            "body_part_regression",
            *select_ct_specialists(case, router_result),
        ]

    return select_initial_models(case)
