from core.expert_result_fusion import ExpertFinding


def _first(d, *keys, default=None):
    if not isinstance(d, dict):
        return default

    for key in keys:
        value = d.get(key)
        if value not in (None, "", [], {}):
            return value

    return default


class LegacyResultBridge:
    """
    把 Phoenix 已有 CT/DR 模型输出转换为统一 ExpertFinding。
    对未知字段采用宽松读取，不伪造诊断。
    """

    def convert_one(self, result):
        if isinstance(result, ExpertFinding):
            return result

        if not isinstance(result, dict):
            if hasattr(result, "model_dump"):
                try:
                    result = result.model_dump()
                except Exception:
                    pass

            if not isinstance(result, dict) and hasattr(result, "to_dict"):
                try:
                    result = result.to_dict()
                except Exception:
                    pass

            if not isinstance(result, dict) and hasattr(result, "__dict__"):
                try:
                    result = {
                        k: v for k, v in vars(result).items()
                        if not k.startswith("_")
                    }
                except Exception:
                    pass

        if not isinstance(result, dict):
            return None

        expert_id = str(
            _first(
                result,
                "expert_id",
                "model_id",
                "model",
                "source",
                default="legacy_model",
            )
        )

        location = str(
            _first(
                result,
                "location",
                "anatomy",
                "body_part",
                "region",
                default="",
            )
        )

        finding = str(
            _first(
                result,
                "finding",
                "description",
                "lesion",
                "text",
                "label",
                default="",
            )
        )

        impression = str(
            _first(
                result,
                "impression",
                "diagnosis",
                "diagnostic_tendency",
                default="",
            )
        )

        score = _first(
            result,
            "score",
            "confidence",
            "probability",
        )

        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None

        geometry = result.get("geometry")

        if geometry is None:
            bbox = _first(
                result,
                "bbox",
                "box",
                "bounding_box",
            )

            point = _first(
                result,
                "point",
                "center",
                "tip",
            )

            slice_index = _first(
                result,
                "slice_index",
                "slice",
                "instance_index",
            )

            if bbox is not None or point is not None:
                geometry = {
                    "bbox": bbox,
                    "point": point,
                    "slice_index": slice_index,
                }

        if not finding and not impression and geometry is None:
            return None

        return ExpertFinding(
            expert_id=expert_id,
            task=str(
                _first(
                    result,
                    "task",
                    "type",
                    default="legacy_inference",
                )
            ),
            location=location,
            finding=finding,
            impression=impression,
            geometry=geometry,
            score=score,
            metadata={
                "legacy": True,
            },
        )

    def convert(self, results):
        if results is None:
            return []

        if isinstance(results, dict):
            for key in ("results", "findings", "lesions", "items"):
                nested = results.get(key)

                if isinstance(nested, list):
                    results = nested
                    break
            else:
                results = [results]

        if not isinstance(results, (list, tuple)):
            results = [results]

        output = []

        for result in results:
            item = self.convert_one(result)

            if item is not None:
                output.append(item)

        return output


LEGACY_RESULT_BRIDGE = LegacyResultBridge()
