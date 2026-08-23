from pathlib import Path

import onnxruntime as ort


class OnnxCpuModelAdapter:
    """Lazy ONNX Runtime adapter restricted to CPUExecutionProvider."""

    REQUIRED_PROVIDER = "CPUExecutionProvider"

    def __init__(self, model_path, model_name):
        self.model_path = Path(model_path)
        self.model_name = str(model_name).strip()
        if not self.model_name:
            raise ValueError("ONNX模型名称不能为空")
        self._session = None

    @property
    def is_loaded(self):
        return self._session is not None

    def _load_session(self):
        if self._session is not None:
            return self._session
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX模型文件不存在：{self.model_path}")
        if not self.model_path.is_file():
            raise ValueError(f"ONNX模型路径不是文件：{self.model_path}")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError(f"模型文件不是.onnx格式：{self.model_path}")
        available = ort.get_available_providers()
        if self.REQUIRED_PROVIDER not in available:
            raise RuntimeError("CPUExecutionProvider不可用，禁止启动ONNX推理")
        session = ort.InferenceSession(str(self.model_path), providers=[self.REQUIRED_PROVIDER])
        if session.get_providers() != [self.REQUIRED_PROVIDER]:
            raise RuntimeError(f"ONNX实际推理Provider不符合CPU-only要求：{session.get_providers()}")
        self._session = session
        return session

    def unload(self):
        was_loaded = self._session is not None
        self._session = None
        return was_loaded

    def get_input_metadata(self):
        return [{"name": item.name, "shape": item.shape, "type": item.type} for item in self._load_session().get_inputs()]

    def get_output_metadata(self):
        return [{"name": item.name, "shape": item.shape, "type": item.type} for item in self._load_session().get_outputs()]

    def infer(self, input_feed, output_names=None):
        if not isinstance(input_feed, dict):
            raise TypeError("ONNX推理输入必须是字典")
        if not input_feed:
            raise ValueError("ONNX推理输入不能为空")
        return self._load_session().run(output_names, input_feed)
