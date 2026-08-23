def build_overlays(lesions):
    return [{"type": "thin_arrow", "point": lesion.point, "image_index": lesion.image_index, "series_uid": lesion.series_uid, "label": lesion.label, "confidence": lesion.confidence, "source_model": lesion.source_model} for lesion in lesions if lesion.point is not None]
