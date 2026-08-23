from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DicomPlaneGeometry:
    path: Path
    image_position: Tuple[float, float, float]
    row_direction: Tuple[float, float, float]
    column_direction: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    row_spacing: float
    column_spacing: float
    rows: int
    columns: int
    instance_number: int


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))  # type: ignore[return-value]


def _cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: Sequence[float]) -> Tuple[float, float, float]:
    n = _norm(v)
    if n <= 0:
        raise ValueError("zero-length direction vector")
    return tuple(float(x) / n for x in v)  # type: ignore[return-value]


def read_plane_geometry(path: str | Path) -> Optional[DicomPlaneGeometry]:
    try:
        import pydicom

        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "PixelSpacing",
                "Rows",
                "Columns",
                "InstanceNumber",
            ],
        )

        ipp = tuple(float(x) for x in ds.ImagePositionPatient)
        iop = tuple(float(x) for x in ds.ImageOrientationPatient)
        spacing = tuple(float(x) for x in ds.PixelSpacing)

        if len(ipp) != 3 or len(iop) != 6 or len(spacing) != 2:
            return None

        row_direction = _normalize(iop[:3])
        column_direction = _normalize(iop[3:])
        normal = _normalize(_cross(row_direction, column_direction))

        return DicomPlaneGeometry(
            path=Path(path),
            image_position=ipp,  # type: ignore[arg-type]
            row_direction=row_direction,
            column_direction=column_direction,
            normal=normal,
            row_spacing=float(spacing[0]),
            column_spacing=float(spacing[1]),
            rows=int(getattr(ds, "Rows", 0) or 0),
            columns=int(getattr(ds, "Columns", 0) or 0),
            instance_number=int(getattr(ds, "InstanceNumber", 0) or 0),
        )
    except Exception:
        return None


def plane_coordinate(geometry: DicomPlaneGeometry) -> float:
    return _dot(geometry.image_position, geometry.normal)


def world_lps_to_pixel(
    world_point_lps: Sequence[float],
    geometry: DicomPlaneGeometry,
    clamp: bool = True,
) -> Tuple[float, float, float]:
    if len(world_point_lps) < 3:
        raise ValueError("world_point_lps must contain x,y,z")

    delta = _sub(world_point_lps[:3], geometry.image_position)
    x = _dot(delta, geometry.row_direction) / geometry.column_spacing
    y = _dot(delta, geometry.column_direction) / geometry.row_spacing
    plane_offset = _dot(delta, geometry.normal)

    if clamp:
        if geometry.columns > 0:
            x = min(max(x, 0.0), float(geometry.columns - 1))
        if geometry.rows > 0:
            y = min(max(y, 0.0), float(geometry.rows - 1))

    return x, y, plane_offset


def nearest_slice_for_world_point(
    files: Iterable[str | Path],
    world_point_lps: Sequence[float],
) -> Optional[Tuple[int, DicomPlaneGeometry, Tuple[float, float]]]:
    geometries = []
    for original_index, path in enumerate(files):
        geometry = read_plane_geometry(path)
        if geometry is None:
            continue
        try:
            x, y, offset = world_lps_to_pixel(world_point_lps, geometry, clamp=True)
        except Exception:
            continue
        geometries.append((abs(offset), original_index, geometry, (x, y)))

    if not geometries:
        return None

    _, index, geometry, point = min(geometries, key=lambda item: item[0])
    return index, geometry, point
