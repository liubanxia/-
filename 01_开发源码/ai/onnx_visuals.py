from ai.onnx_cpu_adapter import OnnxCpuModelAdapter
from ai.visual_b_model_contract import VisualBModelContract
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
        model_contract,
        output_parser=None,
    ):
        if not callable(input_builder):
            raise TypeError(
                "视觉B input_builder必须是可调用函数"
            )

        if not isinstance(
            model_contract,
            VisualBModelContract,
        ):
            raise TypeError(
                "视觉B model_contract必须是VisualBModelContract"
            )

        if (
            output_parser is not None
            and not callable(output_parser)
        ):
            raise TypeError(
                "视觉B output_parser必须是可调用函数或None"
            )

        if output_parser is not None:
            parser_contract = getattr(
                output_parser,
                "model_contract",
                None,
            )

            if not isinstance(
                parser_contract,
                VisualBModelContract,
            ):
                raise TypeError(
                    "视觉B output_parser必须显式绑定VisualBModelContract"
                )

            if parser_contract != model_contract:
                raise ValueError(
                    "视觉B output_parser模型契约与OnnxVisualB不一致"
                )

        self.input_builder = input_builder
        self.model_contract = model_contract
        self.output_parser = output_parser

        self.adapter = OnnxCpuModelAdapter(
            model_path=model_path,
            model_name=self.name,
        )

        # 契约验证延迟到第一次真实infer。
        # 创建主窗口或视觉对象时不得自动加载ONNX模型。
        self._contract_validated = False

    @property
    def is_loaded(self):
        return self.adapter.is_loaded

    def unload(self):
        """
        显式释放当前视觉通路的ONNX模型Session。

        Session释放后必须重新进行模型契约验证。
        """

        was_loaded = self.adapter.unload()
        self._contract_validated = False

        return was_loaded

    def infer(self, series_context):
        input_feed = self.input_builder(
            series_context
        )

        if not isinstance(input_feed, dict):
            raise TypeError(
                "视觉B预处理结果必须是ONNX输入字典"
            )

        self.model_contract.validate_input_feed(
            input_feed
        )

        if not self._contract_validated:
            self.model_contract.validate_onnx_metadata(
                input_metadata=self.adapter.get_input_metadata(),
                output_metadata=self.adapter.get_output_metadata(),
            )
            self._contract_validated = True

        raw_outputs = self.adapter.infer(
            input_feed
        )

        if self.output_parser is not None:
            return self.output_parser(
                raw_outputs,
                series_context,
            )

        return {
            "source": self.name,
            "status": "onnx_success",
            "raw_outputs": raw_outputs,
        }
