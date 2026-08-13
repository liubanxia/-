from ai.model_path_resolver import resolve_visual_b_model_path
from ai.onnx_visuals import OnnxVisualB
from ai.visual_b_component_registry import VisualBComponentRegistry
from ai.visual_b_model_contract import VisualBModelContract
from ai.visual_b_output_parser import VisualBOutputParser


class VisualBModelAssembler:
    """
    Project Phoenix 视觉B真实模型装配器。

    职责：
    - 接收明确的VisualBModelContract；
    - 从组件注册表解析input_builder和decoder；
    - 调用显式模型路径解析器；
    - 创建VisualBOutputParser；
    - 创建OnnxVisualB。

    安全原则：
    - 不自动搜索模型；
    - 不猜测input_builder；
    - 不猜测decoder；
    - 不执行医学推理；
    - 不主动加载ONNX Session；
    - 不绕过医生主动启动门控。
    """

    def __init__(
        self,
        component_registry,
        model_path_resolver=resolve_visual_b_model_path,
    ):
        if not isinstance(
            component_registry,
            VisualBComponentRegistry,
        ):
            raise TypeError(
                "视觉B component_registry必须是VisualBComponentRegistry"
            )

        if not callable(model_path_resolver):
            raise TypeError(
                "视觉B model_path_resolver必须是可调用对象"
            )

        self.component_registry = component_registry
        self.model_path_resolver = model_path_resolver

    def assemble(
        self,
        model_contract,
    ):
        """
        根据显式模型契约装配一个OnnxVisualB实例。

        本方法只创建对象，不执行infer。
        """

        if not isinstance(
            model_contract,
            VisualBModelContract,
        ):
            raise TypeError(
                "视觉B model_contract必须是VisualBModelContract"
            )

        components = (
            self.component_registry.resolve_contract_components(
                model_contract
            )
        )

        input_builder = components[
            "input_builder"
        ]

        decoder = components[
            "decoder"
        ]

        model_path = self.model_path_resolver()

        output_parser = VisualBOutputParser(
            decoder=decoder,
            model_contract=model_contract,
        )

        return OnnxVisualB(
            model_path=model_path,
            input_builder=input_builder,
            model_contract=model_contract,
            output_parser=output_parser,
        )
