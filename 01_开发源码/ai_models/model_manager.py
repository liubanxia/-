
from pathlib import Path
import json


class PhoenixModelManager:
    """
    Phoenix统一模型管理器。

    硬规则：
    - 医生未启动病例AI，不得加载模型。
    - 所有模型延迟加载。
    - 病例结束统一释放。
    """

    def __init__(self, registry_path):
        self.registry_path = Path(registry_path)

        with open(
            self.registry_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.registry = json.load(f)

        self.case_active = False
        self.loaded_models = {}

    def activate_for_case(self):
        self.case_active = True
        print("[Phoenix] AI病例会话已激活")

    def deactivate_case(self):
        self.loaded_models.clear()
        self.case_active = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print("[Phoenix] AI病例会话已关闭")

    def get_model_info(self, model_name):
        models = self.registry.get("模型", {})

        if model_name not in models:
            raise KeyError(
                f"模型未注册：{model_name}"
            )

        return models[model_name]

    def get_model_path(self, model_name):
        info = self.get_model_info(model_name)

        model_path = info.get("主模型")

        if not model_path:
            raise FileNotFoundError(
                f"{model_name}没有主模型"
            )

        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    def load_model(self, model_name):

        if not self.case_active:
            raise RuntimeError(
                "医生尚未点击启动AI，禁止自动加载模型。"
            )

        if model_name in self.loaded_models:
            return self.loaded_models[model_name]

        info = self.get_model_info(model_name)
        path = self.get_model_path(model_name)

        layer = str(info.get("层级", ""))
        suffix = path.suffix.lower()

        # ----------------------------------------------------
        # YOLO视觉模型
        #
        # 对视觉B使用Ultralytics统一前/后处理。
        # .pt和Ultralytics导出的.onnx均可由YOLO接口调用。
        # ----------------------------------------------------
        if "视觉B" in layer:
            from ultralytics import YOLO

            model = YOLO(str(path))

        # ----------------------------------------------------
        # 非YOLO ONNX
        # ----------------------------------------------------
        elif suffix == ".onnx":
            import onnxruntime as ort

            available = ort.get_available_providers()

            providers = (
                [
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )

            model = ort.InferenceSession(
                str(path),
                providers=providers
            )

        # ----------------------------------------------------
        elif suffix == ".pt":
            from ultralytics import YOLO
            model = YOLO(str(path))

        else:
            raise RuntimeError(
                f"尚未接入模型格式：{suffix}"
            )

        self.loaded_models[model_name] = model

        print(
            f"[Phoenix] 已延迟加载：{model_name}"
        )

        return model

    def list_ready_models(self):
        result = []

        for name, info in self.registry.get(
            "模型", {}
        ).items():

            path = info.get("主模型")

            if (
                info.get("MVP启用") is True
                and path
                and Path(path).exists()
            ):
                result.append(name)

        return result
