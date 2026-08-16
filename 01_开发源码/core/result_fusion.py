from .contracts import AnalysisResult, Lesion


def fuse_results(raw_results):
    result = AnalysisResult()
    result.raw_model_results = raw_results

    for model_name, data in raw_results.items():
        if not isinstance(data, dict):
            continue

        for item in data.get("lesions", []):
            result.lesions.append(
                Lesion(
                    label=item.get("label", "异常"),
                    confidence=float(item.get("confidence", 0)),
                    series_uid=item.get("series_uid", ""),
                    image_index=item.get("image_index"),
                    point=item.get("point"),
                    source_model=model_name,
                )
            )

    return result
