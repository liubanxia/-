from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FractureCandidate:
    """
    Project Phoenix 视觉B统一骨折候选结果。

    该结构只表示“候选”，不表示最终诊断。
    """

    slice_index: int
    sop_instance_uid: str
    confidence: float
    region_type: str
    region: object

    source: str = "视觉B"
    task_type: str = "骨折候选"

    def __post_init__(self):
        # --------------------------------------------------
        # 来源与任务类型
        # --------------------------------------------------
        if self.source != "视觉B":
            raise ValueError(
                "骨折候选来源必须是视觉B"
            )

        if self.task_type != "骨折候选":
            raise ValueError(
                "视觉B任务类型必须是骨折候选"
            )

        # --------------------------------------------------
        # CT切片位置
        # --------------------------------------------------
        if (
            not isinstance(self.slice_index, int)
            or isinstance(self.slice_index, bool)
        ):
            raise TypeError(
                "骨折候选slice_index必须是整数"
            )

        if self.slice_index < 0:
            raise ValueError(
                "骨折候选slice_index不能小于0"
            )

        # --------------------------------------------------
        # DICOM切片身份
        # --------------------------------------------------
        if self.sop_instance_uid is None:
            raise ValueError(
                "骨折候选必须绑定SOPInstanceUID"
            )

        sop_uid = str(
            self.sop_instance_uid
        ).strip()

        if not sop_uid:
            raise ValueError(
                "骨折候选必须绑定SOPInstanceUID"
            )

        object.__setattr__(
            self,
            "sop_instance_uid",
            sop_uid,
        )

        # --------------------------------------------------
        # 模型原始置信度
        # --------------------------------------------------
        if isinstance(self.confidence, bool):
            raise ValueError(
                "骨折候选confidence必须是有效数值"
            )

        try:
            confidence = float(
                self.confidence
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "骨折候选confidence必须是有效数值"
            ) from exc

        if not math.isfinite(confidence):
            raise ValueError(
                "骨折候选confidence必须是有限数值"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "骨折候选confidence必须位于0~1"
            )

        object.__setattr__(
            self,
            "confidence",
            confidence,
        )

        # --------------------------------------------------
        # 空间区域
        # --------------------------------------------------
        region_type = str(
            self.region_type
        ).strip().lower()

        if region_type not in (
            "bbox",
            "mask",
        ):
            raise ValueError(
                "骨折候选region_type仅允许bbox或mask"
            )

        if self.region is None:
            raise ValueError(
                "骨折候选必须包含候选空间区域"
            )

        if region_type == "bbox":
            if isinstance(
                self.region,
                (str, bytes, dict),
            ):
                raise TypeError(
                    "骨折候选bbox必须是4个数值坐标"
                )

            try:
                coordinates = tuple(
                    self.region
                )
            except TypeError as exc:
                raise TypeError(
                    "骨折候选bbox必须是4个数值坐标"
                ) from exc

            if len(coordinates) != 4:
                raise ValueError(
                    "骨折候选bbox必须包含4个坐标"
                )

            normalized_coordinates = []

            for value in coordinates:
                if isinstance(value, bool):
                    raise ValueError(
                        "骨折候选bbox坐标必须是有限数值"
                    )

                try:
                    value = float(
                        value
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "骨折候选bbox坐标必须是有限数值"
                    ) from exc

                if not math.isfinite(value):
                    raise ValueError(
                        "骨折候选bbox坐标必须是有限数值"
                    )

                normalized_coordinates.append(
                    value
                )

            x1, y1, x2, y2 = normalized_coordinates

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    "骨折候选bbox必须具有正面积"
                )

            object.__setattr__(
                self,
                "region",
                tuple(normalized_coordinates),
            )

        object.__setattr__(
            self,
            "region_type",
            region_type,
        )

    def to_dict(self):
        """
        转换为Phoenix上层UI/记录流程可使用的字典。
        """

        return {
            "source": self.source,
            "task_type": self.task_type,
            "slice_index": self.slice_index,
            "sop_instance_uid": self.sop_instance_uid,
            "confidence": self.confidence,
            "region_type": self.region_type,
            "region": self.region,
        }
