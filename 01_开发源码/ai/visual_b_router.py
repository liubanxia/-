class VisualBRouteDecision:
    """
    视觉B路由决策结果。

    selected_model_id:
        实际选择的模型ID；
        None表示当前没有安全可用的视觉B子模型。

    reason:
        可解释的路由原因。
    """

    def __init__(
        self,
        selected_model_id,
        reason,
    ):
        self.selected_model_id = selected_model_id
        self.reason = str(reason)

    @property
    def has_model(self):
        return self.selected_model_id is not None

    def __repr__(self):
        return (
            "VisualBRouteDecision("
            f"selected_model_id={self.selected_model_id!r}, "
            f"reason={self.reason!r}"
            ")"
        )


class VisualBRouter:
    """
    Project Phoenix 视觉B模型路由器。

    当前原则：
    1. 只根据明确的DICOM上下文路由；
    2. 不看图猜部位；
    3. 不自动调用任何模型；
    4. 没有适用模型时返回“无模型”，不越域推理；
    5. 后续可继续增加成人腕部、其他四肢DR、CT骨折模型。
    """

    PEDIATRIC_WRIST_MODEL_ID = (
        "yolov8_rescbam_wrist_dx_v1"
    )

    WRIST_KEYWORDS = (
        "WRIST",
        "CARPAL",
        "腕",
        "腕关节",
    )

    @staticmethod
    def _patient_age_to_years(
        patient_age,
    ):
        value = str(
            patient_age
        ).strip().upper()

        if len(value) != 4:
            return None

        number_text = value[:3]
        unit = value[3]

        if not number_text.isdigit():
            return None

        number = int(number_text)

        if unit == "Y":
            return float(number)

        if unit == "M":
            return number / 12.0

        if unit == "W":
            return number / 52.0

        if unit == "D":
            return number / 365.25

        return None

    @classmethod
    def route(
        cls,
        series_context,
    ):
        if not isinstance(
            series_context,
            dict,
        ):
            raise TypeError(
                "视觉B series_context必须是dict"
            )

        modality = str(
            series_context.get(
                "modality",
                "",
            )
        ).strip().upper()

        body_part = str(
            series_context.get(
                "body_part_examined",
                "",
            )
        ).strip()

        study_description = str(
            series_context.get(
                "study_description",
                "",
            )
        ).strip()

        series_description = str(
            series_context.get(
                "series_description",
                "",
            )
        ).strip()

        searchable_text = " ".join(
            (
                body_part,
                study_description,
                series_description,
            )
        ).upper()

        # ------------------------------------------
        # CT
        # ------------------------------------------
        if modality == "CT":
            return VisualBRouteDecision(
                selected_model_id=None,
                reason=(
                    "当前尚未接入CT骨折视觉B子模型"
                ),
            )

        # ------------------------------------------
        # 非DX
        # ------------------------------------------
        if modality != "DX":
            return VisualBRouteDecision(
                selected_model_id=None,
                reason=(
                    f"当前视觉B尚不支持模态：{modality or 'UNKNOWN'}"
                ),
            )

        is_wrist = any(
            keyword.upper()
            in searchable_text
            for keyword
            in cls.WRIST_KEYWORDS
        )

        # ------------------------------------------
        # DX，但不是明确腕部
        # ------------------------------------------
        if not is_wrist:
            return VisualBRouteDecision(
                selected_model_id=None,
                reason=(
                    "当前DX未匹配已接入的视觉B专科模型"
                ),
            )

        patient_age = (
            series_context.get(
                "patient_age",
                "",
            )
        )

        age_years = (
            cls._patient_age_to_years(
                patient_age
            )
        )

        # ------------------------------------------
        # 腕部，但年龄未知
        # ------------------------------------------
        if age_years is None:
            return VisualBRouteDecision(
                selected_model_id=None,
                reason=(
                    "腕部DX年龄信息不足，"
                    "不允许自动选择儿童模型"
                ),
            )

        # ------------------------------------------
        # 儿童腕部DX
        # ------------------------------------------
        if age_years < 18.0:
            return VisualBRouteDecision(
                selected_model_id=(
                    cls.PEDIATRIC_WRIST_MODEL_ID
                ),
                reason=(
                    "明确DX + 腕部 + 儿童，"
                    "选择YOLOv8_ResCBAM"
                ),
            )

        # ------------------------------------------
        # 成人腕部DX
        # ------------------------------------------
        return VisualBRouteDecision(
            selected_model_id=None,
            reason=(
                "当前为成人腕部DX，"
                "尚未接入成人腕部骨折视觉B子模型"
            ),
        )
