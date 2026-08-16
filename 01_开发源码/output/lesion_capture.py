from core.dicom_pixels import read_dicom_image
from core.image_preprocess import to_uint8


def capture_lesions(case, lesions, memory):
    series_map = {
        s.series_uid: s
        for s in case.series
    }

    for i, lesion in enumerate(lesions):
        series = series_map.get(
            lesion.series_uid
        )

        if series is None:
            continue

        index = lesion.image_index

        if index is None:
            continue

        if index >= len(series.files):
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
                "confidence": lesion.confidence,
                "voxel_count": lesion.voxel_count,
            }
        )
