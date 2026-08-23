from dataclasses import dataclass

CHEST_WORDS = ("CHEST", "THORAX", "LUNG", "PULMONARY", "胸", "肺")
HEAD_WORDS = ("HEAD", "BRAIN", "CRANIUM", "SKULL", "头", "颅", "脑")
LOCALIZER_WORDS = ("LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW", "定位", "定位像")


@dataclass(frozen=True)
class SeriesScore:
    series: object
    score: float
    text: str
    positive_match: bool


def _series_text(series):
    values = []
    for attr in ("series_description", "protocol_name", "description", "body_part", "study_description"):
        value = getattr(series, attr, None)
        if value: values.append(str(value))
    files = getattr(series, "files", []) or []
    if files:
        try:
            import pydicom
            ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True, specific_tags=["BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName", "ImageType"])
            for attr in ("BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName", "ImageType"):
                value = getattr(ds, attr, None)
                if value: values.append(str(value))
        except Exception: pass
    return " ".join(values).upper()


def select_ct_series(case, anatomy, minimum_images=16):
    anatomy = str(anatomy).lower().strip()
    words = CHEST_WORDS if anatomy == "chest" else HEAD_WORDS if anatomy == "head" else None
    if words is None: raise ValueError(f"unsupported CT anatomy selector: {anatomy}")
    candidates = []
    for series in getattr(case, "series", []) or []:
        if str(getattr(series, "modality", "")).upper() != "CT": continue
        files = getattr(series, "files", []) or []
        if len(files) < minimum_images: continue
        text = _series_text(series)
        positive = any(word.upper() in text for word in words)
        localizer = any(word.upper() in text for word in LOCALIZER_WORDS)
        score = (1000 if positive else 0) - (1000 if localizer else 0) + min(float(len(files)), 800.0)
        candidates.append(SeriesScore(series, score, text, positive))
    valid = [x for x in candidates if x.positive_match and not any(word in x.text for word in LOCALIZER_WORDS)]
    if not valid:
        raise RuntimeError(f"没有找到明确匹配{anatomy}的诊断CT序列；拒绝跨部位自动选择。")
    return max(valid, key=lambda item: item.score).series
