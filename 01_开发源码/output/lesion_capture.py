from __future__ import annotations

from core.dicom_pixels import read_dicom_image
from core.image_preprocess import to_uint8


def capture_lesions(case, lesions, memory):
    series_map = {
        str(s.series_uid): s
        for s in case.series
    }

    for i, lesion in enumerate(lesions):
        series = series_map.get(
            str(lesion.series_uid)
        )

        if series is None:
            continue

        index = lesion.image_index

        if index is None:
            continue

        try:
            index = int(index)
        except Exception:
            continue

        if index < 0 or index >= len(series.files):
            continue

        try:
            image, _, _ = read_dicom_image(
                series.files[index]
            )
            image = to_uint8(image)
        except Exception:
            continue

        memory.put(
            f"lesion_{i}",
            {
                "image": image,
                "point": lesion.point,
                "label": lesion.label,
                "finding": lesion.finding,
                "confidence": lesion.confidence,
                "voxel_count": lesion.voxel_count,
                "series_uid": lesion.series_uid,
                "image_index": index,
                "source_model": lesion.source_model,
                "world_point_lps": lesion.world_point_lps,
                "box_3d": lesion.box_3d,
                "geometry_mode": lesion.geometry_mode,
            }
        )
