from collections.abc import Iterable
import math

from ai.fracture_candidate import FractureCandidate
from ai.visual_b_model_contract import VisualBModelContract
from dicom.reader import read_dicom


class VisualBOutputParser:
    """
    Project Phoenix 视觉B统一输出解析器。

    职责：
    1. 接收ONNX原始输出；
    2. 调用外部明确提供的模型专用decoder；
    3. 将decoder输出绑定到当前安全CT Series；
    4. 读取真实DICOM SOPInstanceUID；
    5. 生成统一FractureCandidate。

    本类不猜测任何具体ONNX模型输出格式。
    """

    REQUIRED_FIELDS = (
        "slice_index",
        "confidence",
        "region_type",
        "region",
    )

    def __init__(
        self,
        decoder,
        model_contract,
    ):
        if not callable(decoder):
            raise TypeError(
                "视觉B decoder必须是可调用函数"
            )

        if not isinstance(
            model_contract,
            VisualBModelContract,
        ):
            raise TypeError(
                "视觉B parser的model_contract必须是VisualBModelContract"
            )

        self.decoder = decoder
        self.model_contract = model_contract

    def __call__(
        self,
        raw_outputs,
        series_context,
    ):
        if not isinstance(series_context, dict):
            raise TypeError(
                "视觉B series_context必须是字典"
            )

        series_files = series_context.get(
            "series_files"
        )

        if not isinstance(
            series_files,
            (tuple, list),
        ):
            raise TypeError(
                "视觉B series_files必须是tuple或list"
            )

        if not series_files:
            raise ValueError(
                "视觉B series_files不能为空"
            )

        decoded_candidates = self.decoder(
            raw_outputs
        )

        if isinstance(
            decoded_candidates,
            (str, bytes, dict),
        ):
            raise TypeError(
                "视觉B decoder必须返回候选序列"
            )

        if not isinstance(
            decoded_candidates,
            Iterable,
        ):
            raise TypeError(
                "视觉B decoder必须返回可迭代候选序列"
            )

        candidates = []

        for draft in decoded_candidates:
            if not self._passes_model_contract(
                draft
            ):
                continue

            candidate = self._build_candidate(
                draft=draft,
                series_files=series_files,
            )

            candidates.append(
                candidate
            )

        return tuple(candidates)

    def _passes_model_contract(
        self,
        draft,
    ):
        """
        在读取目标DICOM前执行模型级候选约束。

        返回：
        - True：候选达到模型阈值，可继续绑定DICOM；
        - False：候选低于模型阈值，安全过滤。

        非法候选结构不得被静默过滤，必须显式报错。
        """

        if not isinstance(draft, dict):
            raise TypeError(
                "视觉B标准候选草稿必须是字典"
            )

        missing_fields = [
            field
            for field in self.REQUIRED_FIELDS
            if field not in draft
        ]

        if missing_fields:
            raise ValueError(
                "视觉B候选草稿缺少字段："
                + ", ".join(missing_fields)
            )

        confidence = draft[
            "confidence"
        ]

        if isinstance(confidence, bool):
            raise ValueError(
                "视觉B候选confidence必须是有效数值"
            )

        try:
            confidence = float(
                confidence
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "视觉B候选confidence必须是有效数值"
            ) from exc

        if not math.isfinite(confidence):
            raise ValueError(
                "视觉B候选confidence必须是有限数值"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "视觉B候选confidence必须位于0~1"
            )

        region_type = str(
            draft["region_type"]
        ).strip().lower()

        if (
            region_type
            not in self.model_contract.allowed_region_types
        ):
            raise ValueError(
                "视觉B候选region_type不符合当前模型契约："
                f"{region_type}"
            )

        return (
            confidence
            >= self.model_contract.confidence_threshold
        )

    def _build_candidate(
        self,
        draft,
        series_files,
    ):
        if not isinstance(draft, dict):
            raise TypeError(
                "视觉B标准候选草稿必须是字典"
            )

        missing_fields = [
            field
            for field in self.REQUIRED_FIELDS
            if field not in draft
        ]

        if missing_fields:
            raise ValueError(
                "视觉B候选草稿缺少字段："
                + ", ".join(missing_fields)
            )

        slice_index = draft[
            "slice_index"
        ]

        if (
            not isinstance(slice_index, int)
            or isinstance(slice_index, bool)
        ):
            raise TypeError(
                "视觉B候选slice_index必须是整数"
            )

        if slice_index < 0:
            raise ValueError(
                "视觉B候选slice_index不能小于0"
            )

        if slice_index >= len(series_files):
            raise IndexError(
                "视觉B候选slice_index超出当前CT Series范围"
            )

        file_path = series_files[
            slice_index
        ]

        dataset = read_dicom(
            file_path
        )

        sop_instance_uid = str(
            getattr(
                dataset,
                "SOPInstanceUID",
                "",
            )
        ).strip()

        if not sop_instance_uid:
            raise RuntimeError(
                "视觉B候选目标切片缺失SOPInstanceUID"
            )

        return FractureCandidate(
            slice_index=slice_index,
            sop_instance_uid=sop_instance_uid,
            confidence=draft["confidence"],
            region_type=draft["region_type"],
            region=draft["region"],
        )
