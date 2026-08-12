from collections.abc import Iterable

from ai.fracture_candidate import FractureCandidate
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

    def __init__(self, decoder):
        if not callable(decoder):
            raise TypeError(
                "视觉B decoder必须是可调用函数"
            )

        self.decoder = decoder

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
            candidate = self._build_candidate(
                draft=draft,
                series_files=series_files,
            )

            candidates.append(
                candidate
            )

        return tuple(candidates)

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
