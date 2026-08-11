from ai.onnx_cpu_adapter import OnnxCpuModelAdapter
from ai.visual_interfaces import VisualAInterface, VisualBInterface


class OnnxVisualA(VisualAInterface):
    """
    视觉A：ONNX综合阅片模型包装层。

    当前只负责：
    - 连接Phoenix统一视觉接口；
    - 连接ONNX CPU适配器；
    - 接收外部明确提供的输入预处理函数；
    - 不自行猜测医学模型输入格式。
    """

    def __init__(
        self,
        model_path,
        input_builder,
        output_parser=None,
    ):
        if not callable(input_builder):
            raise TypeError(
                "视觉A input_builder必须是可调用函数"
            )

        if (
            output_parser is not None
            and not callable(output_parser)
        ):
            raise TypeError(
                "视觉A output_parser必须是可调用函数或None"
            )

        self.input_builder = input_builder
        self.output_parser = output_parser

        self.adapter = OnnxCpuModelAdapter(
            model_path=model_path,
            model_name=self.name,
        )

    @property
    def is_loaded(self):
        return self.adapter.is_loaded

    def unload(self):
        """
        显式释放当前视觉通路的ONNX模型Session。
        """

        return self.adapter.unload()

    def infer(self, series_context):
        input_feed = self.input_builder(
            series_context
        )

        if not isinstance(input_feed, dict):
            raise TypeError(
                "视觉A预处理结果必须是ONNX输入字典"
            )

        raw_outputs = self.adapter.infer(
            input_feed
        )

        if self.output_parser is not None:
            return self.output_parser(
                raw_outputs
            )

        return {
            "source": self.name,
            "status": "onnx_success",
            "raw_outputs": raw_outputs,
        }


class OnnxVisualB(VisualBInterface):
    """
    视觉B：ONNX骨折漏诊防护模型包装层。

    与视觉A保持独立模型、独立预处理及独立输出解析。
    """

    def __init__(
        self,
        model_path,
        input_builder,
        output_parser=None,
    ):
        if not callable(input_builder):
            raise TypeError(
                "视觉B input_builder必须是可调用函数"
            )

        if (
            output_parser is not None
            and not callable(output_parser)
        ):
            raise TypeError(
                "视觉B output_parser必须是可调用函数或None"
            )

        self.input_builder = input_builder
        self.output_parser = output_parser

        self.adapter = OnnxCpuModelAdapter(
            model_path=model_path,
            model_name=self.name,
        )

    @property
    def is_loaded(self):
        return self.adapter.is_loaded

    def unload(self):
        """
        显式释放当前视觉通路的ONNX模型Session。
        """

        return self.adapter.unload()

    def infer(self, series_context):
        input_feed = self.input_builder(
            series_context
        )

        if not isinstance(input_feed, dict):
            raise TypeError(
                "视觉B预处理结果必须是ONNX输入字典"
            )

        raw_outputs = self.adapter.infer(
            input_feed
        )

        if self.output_parser is not None:
            return self.output_parser(
                raw_outputs
            )

        return {
            "source": self.name,
            "status": "onnx_success",
            "raw_outputs": raw_outputs,
        }
