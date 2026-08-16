import pydicom

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


def get_modalities(case):
    return {
        str(series.modality).upper()
        for series in case.series
    }


def get_xray_region(case):
    text = []

    for series in case.series:
        if str(series.modality).upper() not in XRAY_MODALITIES:
            continue

        if not series.files:
            continue

        try:
            ds = pydicom.dcmread(
                str(series.files[0]),
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


def select_models(case):
    modalities = get_modalities(case)

    if "CT" in modalities:
        models = ["body_part_regression"]

        if is_head_ct(case):
            models.append("blast_ct_head")

        return models

    if modalities & XRAY_MODALITIES:
        region = get_xray_region(case)

        if region == "chest":
            return [
                "torchxrayvision_chest",
            ]

        if region == "bone":
            return [
                "fracture_rescbam",
                "fractureatlas_localization",
                "fractureatlas_segmentation",
            ]

        # 其他DR等待Hulu通用主模型。
        return []

    return []
