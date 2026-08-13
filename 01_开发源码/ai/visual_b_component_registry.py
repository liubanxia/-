from ai.visual_b_model_contract import VisualBModelContract


class VisualBComponentRegistry:
    """
    Project Phoenix 视觉B组件注册表。

    当前职责：
    - 显式注册input_builder；
    - 显式注册decoder；
    - 根据VisualBModelContract中的ID解析真实可调用组件；
    - 禁止自动猜测、自动搜索或静默覆盖。

    当前不负责：
    - 模型文件路径解析；
    - ONNX Session加载；
    - 医学推理；
    - 专科路由；
    - UI控制。
    """

    def __init__(self):
        self._input_builders = {}
        self._decoders = {}

    @staticmethod
    def _normalize_component_id(
        component_id,
        field_name,
    ):
        if not isinstance(component_id, str):
            raise TypeError(
                f"{field_name}必须是字符串"
            )

        component_id = component_id.strip()

        if not component_id:
            raise ValueError(
                f"{field_name}不能为空"
            )

        return component_id

    @staticmethod
    def _validate_callable(
        component,
        field_name,
    ):
        if not callable(component):
            raise TypeError(
                f"{field_name}必须是可调用对象"
            )

        return component

    def register_input_builder(
        self,
        input_builder_id,
        input_builder,
    ):
        input_builder_id = self._normalize_component_id(
            input_builder_id,
            "视觉B input_builder_id",
        )

        input_builder = self._validate_callable(
            input_builder,
            "视觉B input_builder",
        )

        if input_builder_id in self._input_builders:
            raise ValueError(
                "视觉B input_builder_id已注册："
                f"{input_builder_id}"
            )

        self._input_builders[
            input_builder_id
        ] = input_builder

        return input_builder_id

    def register_decoder(
        self,
        decoder_id,
        decoder,
    ):
        decoder_id = self._normalize_component_id(
            decoder_id,
            "视觉B decoder_id",
        )

        decoder = self._validate_callable(
            decoder,
            "视觉B decoder",
        )

        if decoder_id in self._decoders:
            raise ValueError(
                "视觉B decoder_id已注册："
                f"{decoder_id}"
            )

        self._decoders[
            decoder_id
        ] = decoder

        return decoder_id

    def get_input_builder(
        self,
        input_builder_id,
    ):
        input_builder_id = self._normalize_component_id(
            input_builder_id,
            "视觉B input_builder_id",
        )

        try:
            return self._input_builders[
                input_builder_id
            ]
        except KeyError as exc:
            raise KeyError(
                "视觉B input_builder_id未注册："
                f"{input_builder_id}"
            ) from exc

    def get_decoder(
        self,
        decoder_id,
    ):
        decoder_id = self._normalize_component_id(
            decoder_id,
            "视觉B decoder_id",
        )

        try:
            return self._decoders[
                decoder_id
            ]
        except KeyError as exc:
            raise KeyError(
                "视觉B decoder_id未注册："
                f"{decoder_id}"
            ) from exc

    def resolve_contract_components(
        self,
        model_contract,
    ):
        """
        根据真实模型契约解析对应的input_builder和decoder。

        只允许显式ID映射，不进行任何自动猜测。
        """

        if not isinstance(
            model_contract,
            VisualBModelContract,
        ):
            raise TypeError(
                "视觉B model_contract必须是VisualBModelContract"
            )

        return {
            "input_builder": self.get_input_builder(
                model_contract.input_builder_id
            ),
            "decoder": self.get_decoder(
                model_contract.decoder_id
            ),
        }
