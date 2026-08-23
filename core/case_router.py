from __future__ import annotations

import pydicom

from .ct_chest_gate import is_chest_ct
from .ct_head_gate import is_head_ct

XRAY_MODALITIES = {"DX", "DR", "CR", "XR"}
MRI_MODALITIES = {"MR", "MRI"}
CHEST_WORDS = {"CHEST", "THORAX", "LUNG", "胸", "肺"}
BONE_WORDS = {"HAND", "WRIST", "FOREARM", "ELBOW", "HUMERUS", "SHOULDER", "FOOT", "ANKLE", "TIBIA", "FIBULA", "KNEE", "FEMUR", "HIP", "PELVIS", "SPINE", "CERVICAL", "THORACIC", "LUMBAR", "SACRUM", "SKULL", "手", "腕", "前臂", "肘", "肱骨", "肩", "足", "踝", "胫骨", "腓骨", "膝", "股骨", "髋", "骨盆", "脊柱", "颈椎", "胸椎", "腰椎", "骶骨", "颅骨"}
HEAD_WORDS = {"HEAD", "BRAIN", "CRANIUM", "SKULL", "头", "脑", "颅"}
ABDOMEN_WORDS = {"ABDOMEN", "ABDOMINAL", "LIVER", "HEPATIC", "PANCREAS", "KIDNEY", "RENAL", "SPLEEN", "BOWEL", "APPENDIX", "腹", "肝", "胰", "肾", "脾", "肠", "阑尾"}
PELVIS_WORDS = {"PELVIS", "PELVIC", "BLADDER", "PROSTATE", "UTERUS", "骨盆", "盆腔", "膀胱", "前列腺", "子宫"}
LUMBAR_WORDS = {"LUMBAR", "L-SPINE", "L SPINE", "LSPINE", "腰椎", "腰骶"}


def get_modalities(case):
    return {str(getattr(series, "modality", "")).upper() for series in getattr(case, "series", []) or []}


def _series_text(series):
    text = []
    for attr in ("description", "series_description", "study_description", "protocol_name", "body_part"):
        value = getattr(series, attr, None)
        if value:
            text.append(str(value).upper())
    files = getattr(series, "files", []) or []
    if files:
        try:
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
            for key in ("BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName"):
                value = str(getattr(ds, key, "")).upper()
                if value:
                    text.append(value)
        except Exception:
            pass
    return " ".join(text)


def get_xray_region(case):
    joined = " ".join(_series_text(s) for s in getattr(case, "series", []) or [] if str(getattr(s, "modality", "")).upper() in XRAY_MODALITIES)
    if any(word in joined for word in CHEST_WORDS):
        return "chest"
    if any(word in joined for word in BONE_WORDS):
        return "bone"
    return "other"


def get_mri_region(case):
    joined = " ".join(_series_text(s) for s in getattr(case, "series", []) or [] if str(getattr(s, "modality", "")).upper() in MRI_MODALITIES)
    return "lumbar_spine" if any(word in joined for word in LUMBAR_WORDS) else "other"


def _normalize_router_regions(router_result):
    if not isinstance(router_result, dict):
        return set()
    regions = {str(x or "").strip().lower() for x in (router_result.get("active_body_regions") or []) if str(x or "").strip()}
    text = " ".join(str(router_result.get(key) or "") for key in ("body_part_examined_tag", "body_part_display", "body_part_examined_tag_raw")).upper()
    if any(word in text for word in HEAD_WORDS): regions.add("head")
    if any(word in text for word in CHEST_WORDS): regions.add("chest")
    if any(word in text for word in ABDOMEN_WORDS): regions.add("abdomen")
    if any(word in text for word in PELVIS_WORDS): regions.add("pelvis")
    return regions


def ct_route_decision(case, router_result=None):
    regions = _normalize_router_regions(router_result)
    return {
        "metadata_head": bool(is_head_ct(case)), "metadata_chest": bool(is_chest_ct(case)),
        "router_regions": sorted(regions),
        "head": bool(is_head_ct(case)) or "head" in regions,
        "chest": bool(is_chest_ct(case)) or "chest" in regions,
        "abdomen": "abdomen" in regions, "pelvis": "pelvis" in regions,
    }


def select_ct_specialists(case, router_result=None):
    decision = ct_route_decision(case, router_result)
    models = []
    if decision["head"]:
        models += ["ich_2p5d_student", "ich_segmentation_student", "brain_infarct_2p5d_student", "brain_atrophy_quant_student"]
    if decision["abdomen"] or decision["pelvis"]:
        models += ["renal_stone_student", "sbo_2p5d_student", "appendicitis_2p5d_student"]
    return list(dict.fromkeys(models))


def select_initial_models(case):
    modalities = get_modalities(case)
    if "CT" in modalities:
        return ["body_part_regression"]
    if modalities & XRAY_MODALITIES:
        region = get_xray_region(case)
        if region == "chest": return ["chest_dr_nano_detector"]
        if region == "bone": return ["fracture_rescbam", "fractureatlas_localization", "fractureatlas_segmentation"]
    if modalities & MRI_MODALITIES and get_mri_region(case) == "lumbar_spine":
        return ["lumbar_mri_student"]
    return []


def select_models(case, router_result=None):
    return ["body_part_regression", *select_ct_specialists(case, router_result)] if "CT" in get_modalities(case) else select_initial_models(case)
