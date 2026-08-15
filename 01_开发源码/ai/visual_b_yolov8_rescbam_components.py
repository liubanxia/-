from ai.visual_b_component_registry import (
    VisualBComponentRegistry,
)
from ai.visual_b_model_contract import (
    VisualBModelContract,
)
from ai.visual_b_output_parser import (
    VisualBOutputParser,
)
from ai.visual_b_yolov8_rescbam import (
    build_yolov8_rescbam_input,
    decode_yolov8_rescbam_fractures,
)


MODEL_ID = "yolov8_rescbam_wrist_dx_v1"

INPUT_BUILDER_ID = (
    "yolov8_rescbam_letterbox_1024_v1"
)

DECODER_ID = (
    "yolov8_rescbam_fracture_decoder_v1"
)


YOLOV8_RESCBAM_MODEL_CONTRACT = (
    VisualBModelContract(
        model_id=MODEL_ID,
        input_builder_id=INPUT_BUILDER_ID,
        decoder_id=DECODER_ID,

        # 已从真实ONNX metadata确认。
        expected_input_names=(
            "images",
        ),
        expected_output_names=(
            "output0",
        ),

        # 作者原始NMS默认阈值。
        # Phoenix后续可通过模型配置版本化调整。
        confidence_threshold=0.25,

        # 当前模型接入仅向Phoenix输出bbox。
        allowed_region_types=(
            "bbox",
        ),
    )
)


def create_yolov8_rescbam_registry():
    """
    创建YOLOv8_ResCBAM专用组件注册表。

    所有组件均显式注册；
    不进行自动搜索、自动猜测或静默替换。
    """

    registry = VisualBComponentRegistry()

    registry.register_input_builder(
        INPUT_BUILDER_ID,
        build_yolov8_rescbam_input,
    )

    registry.register_decoder(
        DECODER_ID,
        decode_yolov8_rescbam_fractures,
    )

    return registry


def create_yolov8_rescbam_parser():
    """
    根据模型契约显式解析组件，
    并创建Phoenix统一VisualBOutputParser。
    """

    registry = (
        create_yolov8_rescbam_registry()
    )

    components = (
        registry.resolve_contract_components(
            YOLOV8_RESCBAM_MODEL_CONTRACT
        )
    )

    return VisualBOutputParser(
        decoder=components["decoder"],
        model_contract=(
            YOLOV8_RESCBAM_MODEL_CONTRACT
        ),
    )


def resolve_yolov8_rescbam_components():
    """
    返回真实模型装配所需组件。

    供后续OnnxVisualB实例化使用。
    """

    registry = (
        create_yolov8_rescbam_registry()
    )

    components = (
        registry.resolve_contract_components(
            YOLOV8_RESCBAM_MODEL_CONTRACT
        )
    )

    parser = VisualBOutputParser(
        decoder=components["decoder"],
        model_contract=(
            YOLOV8_RESCBAM_MODEL_CONTRACT
        ),
    )

    return {
        "model_contract": (
            YOLOV8_RESCBAM_MODEL_CONTRACT
        ),
        "input_builder": (
            components["input_builder"]
        ),
        "decoder": components["decoder"],
        "output_parser": parser,
    }
