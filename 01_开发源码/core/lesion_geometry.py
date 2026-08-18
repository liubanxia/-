from __future__ import annotations

from typing import Iterable, Optional

from core.dicom_geometry import nearest_slice_for_world_point


def _series_by_uid(case, uid: str):
    if not uid:
        return None
    for series in getattr(case, "series", []) or []:
        if str(getattr(series, "series_uid", "")) == str(uid):
            return series
    return None


def _default_ct_series(case):
    candidates = [
        series
        for series in getattr(case, "series", []) or []
        if str(getattr(series, "modality", "")).upper() == "CT"
        and getattr(series, "files", None)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: len(getattr(s, "files", []) or []))


def resolve_lesion_geometry(case, lesion) -> bool:
    """Resolve world LPS lesion coordinates to a concrete DICOM slice and pixel point."""
    if getattr(lesion, "image_index", None) is not None and getattr(lesion, "point", None) is not None:
        return True

    world_point = getattr(lesion, "world_point_lps", None)
    if not world_point:
        return False

    series = _series_by_uid(case, getattr(lesion, "series_uid", ""))
    if series is None:
        series = _default_ct_series(case)

    if series is None:
        return False

    match = nearest_slice_for_world_point(
        getattr(series, "files", []) or [],
        world_point,
    )
    if match is None:
        return False

    image_index, _geometry, point = match
    lesion.series_uid = str(getattr(series, "series_uid", ""))
    lesion.image_index = int(image_index)
    lesion.point = (float(point[0]), float(point[1]))
    return True


def resolve_lesions_for_case(case, lesions: Iterable) -> int:
    resolved = 0
    for lesion in lesions:
        try:
            if resolve_lesion_geometry(case, lesion):
                resolved += 1
        except Exception:
            continue
    return resolved
