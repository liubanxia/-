from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VisualBModelContract:
    """
    Project Phoenix 视觉B真实模型配置契约。

    本结构只描述模型接入所必须显式声明的信息。

    安全原则：
    1. 不自动猜测模型输入名；
    2. 不自动猜测模型输出名；
    3. 不自动猜测input_builder；
    4. 不自动猜测decoder；
    5. 不自动猜测置信度阈值；
    6. 不负责加载ONNX模型；
    7. 不负责执行医学推理。
    """

    model_id: str
    input_builder_id: str
    decoder_id: str

    expected_input_names: tuple
    expected_output_names: tuple

    confidence_threshold: float

    allowed_region_types: tuple = (
        "bbox",
        "mask",
    )

    def __post_init__(self):
        # --------------------------------------------------
        # 模型身份
        # --------------------------------------------------
        model_id = str(
            self.model_id
        ).strip()

        if not model_id:
            raise ValueError(
                "视觉B model_id不能为空"
            )

        object.__setattr__(
            self,
            "model_id",
            model_id,
        )

        # --------------------------------------------------
        # input_builder / decoder 身份
        # --------------------------------------------------
        input_builder_id = str(
            self.input_builder_id
        ).strip()

        if not input_builder_id:
            raise ValueError(
                "视觉B input_builder_id不能为空"
            )

        object.__setattr__(
            self,
            "input_builder_id",
            input_builder_id,
        )

        decoder_id = str(
            self.decoder_id
        ).strip()

        if not decoder_id:
            raise ValueError(
                "视觉B decoder_id不能为空"
            )

        object.__setattr__(
            self,
            "decoder_id",
            decoder_id,
        )

        # --------------------------------------------------
        # ONNX输入 / 输出名称
        # --------------------------------------------------
        input_names = self._normalize_names(
            self.expected_input_names,
            "expected_input_names",
        )

        output_names = self._normalize_names(
            self.expected_output_names,
            "expected_output_names",
        )

        object.__setattr__(
            self,
            "expected_input_names",
            input_names,
        )

        object.__setattr__(
            self,
            "expected_output_names",
            output_names,
        )

        # --------------------------------------------------
        # 候选置信度阈值
        # --------------------------------------------------
        if isinstance(
            self.confidence_threshold,
            bool,
        ):
            raise ValueError(
                "视觉B confidence_threshold必须是有效数值"
            )

        try:
            threshold = float(
                self.confidence_threshold
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "视觉B confidence_threshold必须是有效数值"
            ) from exc

        if not math.isfinite(threshold):
            raise ValueError(
                "视觉B confidence_threshold必须是有限数值"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "视觉B confidence_threshold必须位于0~1"
            )

        object.__setattr__(
            self,
            "confidence_threshold",
            threshold,
        )

        # --------------------------------------------------
        # 空间结果类型
        # --------------------------------------------------
        if not isinstance(
            self.allowed_region_types,
            (tuple, list),
        ):
            raise TypeError(
                "视觉B allowed_region_types必须是tuple或list"
            )

        region_types = tuple(
            str(item).strip().lower()
            for item in self.allowed_region_types
        )

        if not region_types:
            raise ValueError(
                "视觉B allowed_region_types不能为空"
            )

        if len(set(region_types)) != len(
            region_types
        ):
            raise ValueError(
                "视觉B allowed_region_types不得重复"
            )

        invalid_region_types = [
            item
            for item in region_types
            if item not in (
                "bbox",
                "mask",
            )
        ]

        if invalid_region_types:
            raise ValueError(
                "视觉B allowed_region_types仅允许bbox或mask"
            )

        object.__setattr__(
            self,
            "allowed_region_types",
            region_types,
        )

    @staticmethod
    def _normalize_names(
        values,
        field_name,
    ):
        if not isinstance(
            values,
            (tuple, list),
        ):
            raise TypeError(
                f"视觉B {field_name}必须是tuple或list"
            )

        normalized = tuple(
            str(item).strip()
            for item in values
        )

        if not normalized:
            raise ValueError(
                f"视觉B {field_name}不能为空"
            )

        if any(
            not item
            for item in normalized
        ):
            raise ValueError(
                f"视觉B {field_name}不得包含空名称"
            )

        if len(set(normalized)) != len(
            normalized
        ):
            raise ValueError(
                f"视觉B {field_name}不得包含重复名称"
            )

        return normalized

    def validate_onnx_metadata(
        self,
        input_metadata,
        output_metadata,
    ):
        """
        校验真实ONNX模型的输入/输出名称是否与契约完全一致。

        注意：
        - 本方法不加载ONNX模型；
        - metadata必须由OnnxCpuModelAdapter显式提供；
        - 不自动容忍缺失、额外或顺序不同的名称；
        - 当前阶段不猜测shape和数据类型。
        """

        actual_input_names = self._extract_metadata_names(
            input_metadata,
            "input_metadata",
        )

        actual_output_names = self._extract_metadata_names(
            output_metadata,
            "output_metadata",
        )

        if actual_input_names != self.expected_input_names:
            raise RuntimeError(
                "视觉B ONNX输入名称与模型契约不一致："
                f"expected={self.expected_input_names}, "
                f"actual={actual_input_names}"
            )

        if actual_output_names != self.expected_output_names:
            raise RuntimeError(
                "视觉B ONNX输出名称与模型契约不一致："
                f"expected={self.expected_output_names}, "
                f"actual={actual_output_names}"
            )

        return True

    @staticmethod
    def _extract_metadata_names(
        metadata,
        field_name,
    ):
        if not isinstance(
            metadata,
            (tuple, list),
        ):
            raise TypeError(
                f"视觉B {field_name}必须是tuple或list"
            )

        if not metadata:
            raise ValueError(
                f"视觉B {field_name}不能为空"
            )

        names = []

        for item in metadata:
            if not isinstance(item, dict):
                raise TypeError(
                    f"视觉B {field_name}中的每一项必须是字典"
                )

            if "name" not in item:
                raise ValueError(
                    f"视觉B {field_name}缺少name字段"
                )

            name = str(
                item["name"]
            ).strip()

            if not name:
                raise ValueError(
                    f"视觉B {field_name}不得包含空name"
                )

            names.append(name)

        names = tuple(names)

        if len(set(names)) != len(names):
            raise ValueError(
                f"视觉B {field_name}不得包含重复name"
            )

        return names

    def validate_input_feed(
        self,
        input_feed,
    ):
        """
        校验input_builder生成的ONNX输入字典。

        安全原则：
        - 输入必须是非空字典；
        - 输入名称必须与模型契约完全一致；
        - 不允许缺少输入；
        - 不允许出现额外输入；
        - 当前阶段不猜测数组shape或dtype。
        """

        if not isinstance(
            input_feed,
            dict,
        ):
            raise TypeError(
                "视觉B input_feed必须是字典"
            )

        if not input_feed:
            raise ValueError(
                "视觉B input_feed不能为空"
            )

        actual_input_names = tuple(
            input_feed.keys()
        )

        if set(actual_input_names) != set(self.expected_input_names):
            raise RuntimeError(
                "视觉B input_builder输出名称与模型契约不一致："
                f"expected={self.expected_input_names}, "
                f"actual={actual_input_names}"
            )

        return True
