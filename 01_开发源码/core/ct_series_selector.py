from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


CHEST_WORDS = (
    "CHEST", "THORAX", "LUNG", "PULMONARY",
    "胸", "肺",
)

HEAD_WORDS = (
    "HEAD", "BRAIN", "CRANIUM", "SKULL",
    "头", "颅", "脑",
)

LOCALIZER_WORDS = (
    "LOCALIZER", "SCOUT", "TOPOGRAM", "SURVIEW",
    "定位", "定位像",
)


@dataclass(frozen=True)
class SeriesScore:
    series: object
    score: float
    text: str
    positive_match: bool


def _series_text(series) -> str:
    values = []

    for attr in (
        "series_description",
        "protocol_name",
        "description",
        "body_part",
        "study_description",
    ):
        value = getattr(series, attr, None)
        if value:
            values.append(str(value))

    files = getattr(series, "files", []) or []

    if files:
        try:
            import pydicom

            ds = pydicom.dcmread(
                str(files[0]),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "BodyPartExamined",
                    "StudyDescription",
                    "SeriesDescription",
                    "ProtocolName",
                    "ImageType",
                ],
            )

            for attr in (
                "BodyPartExamined",
                "StudyDescription",
                "SeriesDescription",
                "ProtocolName",
                "ImageType",
            ):
                value = getattr(ds, attr, None)
                if value:
                    values.append(str(value))
        except Exception:
            pass

    return " ".join(values).upper()


def _score_series(series, positive_words: Tuple[str, ...]) -> SeriesScore:
    text = _series_text(series)
    files = getattr(series, "files", []) or []
    count = len(files)

    positive = any(word.upper() in text for word in positive_words)
    localizer = any(word.upper() in text for word in LOCALIZER_WORDS)

    score = 0.0

    if positive:
        score += 1000.0

    if localizer:
        score -= 1000.0

    score += min(float(count), 800.0)

    if count < 16:
        score -= 500.0

    return SeriesScore(
        series=series,
        score=score,
        text=text,
        positive_match=positive,
    )


def select_ct_series(case, anatomy: str, minimum_images: int = 16):
    anatomy = str(anatomy).lower().strip()

    if anatomy == "chest":
        positive_words = CHEST_WORDS
    elif anatomy == "head":
        positive_words = HEAD_WORDS
    else:
        raise ValueError(f"unsupported CT anatomy selector: {anatomy}")

    candidates: List[SeriesScore] = []

    for series in getattr(case, "series", []) or []:
        if str(getattr(series, "modality", "")).upper() != "CT":
            continue

        files = getattr(series, "files", []) or []
        if len(files) < minimum_images:
            continue

        candidates.append(
            _score_series(series, positive_words)
        )

    positive_candidates = [
        item
        for item in candidates
        if item.positive_match
        and not any(word in item.text for word in LOCALIZER_WORDS)
    ]

    if not positive_candidates:
        raise RuntimeError(
            f"没有找到明确匹配{anatomy}的诊断CT序列，"
            "为避免跨部位误推理，Phoenix拒绝自动选择最长序列。"
        )

    return max(
        positive_candidates,
        key=lambda item: item.score,
    ).series
