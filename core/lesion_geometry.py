from core.dicom_geometry import nearest_slice_for_world_point


def _series_by_uid(case, uid):
    for series in getattr(case, "series", []) or []:
        if uid and str(getattr(series, "series_uid", "")) == str(uid):
            return series
    return None


def _default_ct_series(case):
    candidates = [s for s in getattr(case, "series", []) or [] if str(getattr(s, "modality", "")).upper() == "CT" and getattr(s, "files", None)]
    return max(candidates, key=lambda s: len(getattr(s, "files", []) or [])) if candidates else None


def resolve_lesion_geometry(case, lesion):
    if getattr(lesion, "image_index", None) is not None and getattr(lesion, "point", None) is not None:
        return True
    world_point = getattr(lesion, "world_point_lps", None)
    if not world_point:
        return False
    series = _series_by_uid(case, getattr(lesion, "series_uid", "")) or _default_ct_series(case)
    if series is None:
        return False
    match = nearest_slice_for_world_point(getattr(series, "files", []) or [], world_point)
    if match is None:
        return False
    image_index, _, point = match
    lesion.series_uid = str(getattr(series, "series_uid", ""))
    lesion.image_index = int(image_index)
    lesion.point = (float(point[0]), float(point[1]))
    return True


def resolve_lesions_for_case(case, lesions):
    resolved = 0
    for lesion in lesions:
        try:
            resolved += int(resolve_lesion_geometry(case, lesion))
        except Exception:
            pass
    return resolved
