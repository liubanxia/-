from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VisualBModelContract:
    """Explicit contract for a visual-B ONNX model. No metadata is guessed."""

    model_id: str
    input_builder_id: str
    decoder_id: str
    expected_input_names: tuple
    expected_output_names: tuple
    confidence_threshold: float
    allowed_region_types: tuple = ("bbox", "mask")

    def __post_init__(self):
        for field_name in ("model_id", "input_builder_id", "decoder_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"视觉B {field_name}不能为空")
            object.__setattr__(self, field_name, value)

        object.__setattr__(self, "expected_input_names", self._normalize_names(self.expected_input_names, "expected_input_names"))
        object.__setattr__(self, "expected_output_names", self._normalize_names(self.expected_output_names, "expected_output_names"))

        if isinstance(self.confidence_threshold, bool):
            raise ValueError("视觉B confidence_threshold必须是有效数值")
        threshold = float(self.confidence_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("视觉B confidence_threshold必须位于0~1且为有限数值")
        object.__setattr__(self, "confidence_threshold", threshold)

        if not isinstance(self.allowed_region_types, (tuple, list)):
            raise TypeError("视觉B allowed_region_types必须是tuple或list")
        region_types = tuple(str(item).strip().lower() for item in self.allowed_region_types)
        if not region_types or len(set(region_types)) != len(region_types):
            raise ValueError("视觉B allowed_region_types不能为空或重复")
        if any(item not in ("bbox", "mask") for item in region_types):
            raise ValueError("视觉B allowed_region_types仅允许bbox或mask")
        object.__setattr__(self, "allowed_region_types", region_types)

    @staticmethod
    def _normalize_names(values, field_name):
        if not isinstance(values, (tuple, list)):
            raise TypeError(f"视觉B {field_name}必须是tuple或list")
        normalized = tuple(str(item).strip() for item in values)
        if not normalized or any(not item for item in normalized):
            raise ValueError(f"视觉B {field_name}不能为空或包含空名称")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"视觉B {field_name}不得包含重复名称")
        return normalized

    @staticmethod
    def _extract_metadata_names(metadata, field_name):
        if not isinstance(metadata, (tuple, list)) or not metadata:
            raise TypeError(f"视觉B {field_name}必须是非空tuple或list")
        names = []
        for item in metadata:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError(f"视觉B {field_name}中的每项必须包含name")
            name = str(item["name"]).strip()
            if not name:
                raise ValueError(f"视觉B {field_name}不得包含空name")
            names.append(name)
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError(f"视觉B {field_name}不得包含重复name")
        return names

    def validate_onnx_metadata(self, input_metadata, output_metadata):
        actual_inputs = self._extract_metadata_names(input_metadata, "input_metadata")
        actual_outputs = self._extract_metadata_names(output_metadata, "output_metadata")
        if actual_inputs != self.expected_input_names:
            raise RuntimeError(f"视觉B ONNX输入名称与模型契约不一致：expected={self.expected_input_names}, actual={actual_inputs}")
        if actual_outputs != self.expected_output_names:
            raise RuntimeError(f"视觉B ONNX输出名称与模型契约不一致：expected={self.expected_output_names}, actual={actual_outputs}")
        return True

    def validate_input_feed(self, input_feed):
        if not isinstance(input_feed, dict):
            raise TypeError("视觉B input_feed必须是字典")
        if not input_feed:
            raise ValueError("视觉B input_feed不能为空")
        actual = tuple(input_feed.keys())
        if set(actual) != set(self.expected_input_names):
            raise RuntimeError(f"视觉B input_builder输出名称与模型契约不一致：expected={self.expected_input_names}, actual={actual}")
        return True
