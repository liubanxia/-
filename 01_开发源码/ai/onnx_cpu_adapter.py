from pathlib import Path

import onnxruntime as ort


class OnnxCpuModelAdapter:
    """
    Project Phoenix ONNX CPU 模型适配器。

    安全原则：
    1. 只允许 CPUExecutionProvider；
    2. 初始化适配器时不加载模型；
    3. 第一次实际 infer() 时才延迟建立 ONNX Session；
    4. 不允许自动切换 Azure、GPU 或其他推理 Provider。
    """

    REQUIRED_PROVIDER = "CPUExecutionProvider"

    def __init__(self, model_path, model_name):
        self.model_path = Path(model_path)
        self.model_name = str(model_name).strip()

        if not self.model_name:
            raise ValueError("ONNX模型名称不能为空")

        # 延迟加载：
        # 创建 Phoenix 主窗口时不会自动加载AI模型。
        self._session = None

    @property
    def is_loaded(self):
        return self._session is not None

    def _load_session(self):
        """
        第一次真实推理时才加载模型。
        """

        if self._session is not None:
            return self._session

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX模型文件不存在：{self.model_path}"
            )

        if not self.model_path.is_file():
            raise ValueError(
                f"ONNX模型路径不是文件：{self.model_path}"
            )

        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(
                f"模型文件不是.onnx格式：{self.model_path}"
            )

        available_providers = ort.get_available_providers()

        if self.REQUIRED_PROVIDER not in available_providers:
            raise RuntimeError(
                "CPUExecutionProvider不可用，禁止启动ONNX推理"
            )

        session = ort.InferenceSession(
            str(self.model_path),
            providers=[self.REQUIRED_PROVIDER],
        )

        active_providers = session.get_providers()

        if active_providers != [self.REQUIRED_PROVIDER]:
            raise RuntimeError(
                "ONNX实际推理Provider不符合Phoenix CPU-only要求："
                f"{active_providers}"
            )

        self._session = session
        return self._session

    def unload(self):
        """
        显式释放ONNX Session。

        注意：
        医生点击“停止双视觉AI”时不自动调用本方法。
        本方法主要用于模型切换、程序退出或主动释放内存。
        """

        was_loaded = self._session is not None
        self._session = None

        return was_loaded

    def get_input_metadata(self):
        """
        返回模型输入信息。

        调用本方法会加载模型。
        """

        session = self._load_session()

        return [
            {
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            }
            for item in session.get_inputs()
        ]

    def get_output_metadata(self):
        """
        返回模型输出信息。

        调用本方法会加载模型。
        """

        session = self._load_session()

        return [
            {
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            }
            for item in session.get_outputs()
        ]

    def infer(self, input_feed, output_names=None):
        """
        执行一次ONNX CPU推理。

        input_feed:
            ONNX Runtime要求的 {输入名: numpy数组} 字典。

        output_names:
            None表示返回模型全部输出。
        """

        if not isinstance(input_feed, dict):
            raise TypeError(
                "ONNX推理输入必须是字典"
            )

        if not input_feed:
            raise ValueError(
                "ONNX推理输入不能为空"
            )

        session = self._load_session()

        return session.run(
            output_names,
            input_feed,
        )
