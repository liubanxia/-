from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .contracts import AnalysisResult, Lesion


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _tuple_or_none(value: Any) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return tuple(float(x) for x in value)
    except Exception:
        return None
    return None


def _point_or_none(value: Any) -> Optional[Tuple[float, float]]:
    coords = _tuple_or_none(value)
    if coords and len(coords) >= 2:
        return float(coords[0]), float(coords[1])
    return None


def _label(item: Dict[str, Any]) -> str:
    for key in ("finding", "label_name", "name", "type", "label"):
        value = item.get(key)
        if value is None or value == "":
            continue
        if key == "label" and isinstance(value, (int, float)):
            continue
        return str(value)
    return "异常候选灶"


def _normalise_lesion(model_name: str, item: Dict[str, Any]) -> Lesion:
    geometry = item.get("geometry")
    if not isinstance(geometry, dict):
        geometry = {}

    box = _tuple_or_none(item.get("box") if item.get("box") is not None else geometry.get("box"))
    box_3d = _tuple_or_none(item.get("box_3d") if item.get("box_3d") is not None else geometry.get("box_3d"))
    world_point_lps = _tuple_or_none(item.get("world_point_lps"))
    geometry_mode = str(item.get("geometry_mode", "") or "")

    confidence = item.get("confidence")
    if confidence is None:
        confidence = item.get("score")
    if confidence is None:
        confidence = item.get("label_score")

    return Lesion(
        label=_label(item),
        confidence=_float(confidence, 0.0),
        series_uid=str(item.get("series_uid", "") or ""),
        image_index=_int_or_none(item.get("image_index")),
        point=_point_or_none(item.get("point")),
        box=box,
        box_3d=box_3d,
        world_point_lps=(tuple(world_point_lps[:3]) if world_point_lps is not None and len(world_point_lps) >= 3 else None),
        geometry_mode=geometry_mode,
        voxel_count=int(item.get("voxel_count", 0) or 0),
        source_model=model_name,
        finding=str(item.get("finding", "") or ""),
        metadata={key: value for key, value in item.items() if key not in {"label", "label_name", "name", "finding", "type", "confidence", "score", "label_score", "series_uid", "image_index", "point", "box", "box_3d", "world_point_lps", "geometry_mode", "voxel_count", "geometry"}},
    )


def fuse_results(raw_results: Dict[str, Any]):
    result = AnalysisResult()
    result.raw_model_results = raw_results
    for model_name, data in raw_results.items():
        if not isinstance(data, dict):
            continue
        lesions = data.get("lesions", [])
        if not isinstance(lesions, list):
            continue
        for item in lesions:
            if not isinstance(item, dict):
                continue
            try:
                result.lesions.append(_normalise_lesion(model_name, item))
            except Exception as exc:
                result.warnings.append(f"{model_name}病灶结果解析失败: {type(exc).__name__}: {exc}")
    return result
