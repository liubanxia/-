
from pathlib import Path
import json

try:
    from .model_manager import PhoenixModelManager
except ImportError:
    from model_manager import PhoenixModelManager


class PhoenixAIOrchestrator:

    XRAY_MODALITIES = {"DX", "DR", "CR", "XR"}

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent

        base_dir = Path(base_dir)

        self.registry_path = base_dir / "模型注册表.json"
        self.config_path = base_dir / "MVP调用链.json"

        self.manager = PhoenixModelManager(
            self.registry_path
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.config = json.load(f)

        self.active = False
        self.modality = None

    def doctor_start(self, modality):
        """
        只能由医生点击AI按钮后调用。
        """
        modality = str(modality).upper().strip()

        if modality == "CT":
            key = "CT"
        elif modality in self.XRAY_MODALITIES:
            key = "DR"
        else:
            raise ValueError(
                f"当前MVP暂不支持模态：{modality}"
            )

        self.manager.activate_for_case()

        self.active = True
        self.modality = key

        print(
            f"[Phoenix] 医生已启动AI：{key}"
        )

        return self.get_plan()

    def get_plan(self):
        if not self.active:
            raise RuntimeError(
                "医生尚未点击启动AI。"
            )

        return self.config[self.modality]

    def load_model(self, model_name):
        if not self.active:
            raise RuntimeError(
                "医生尚未点击启动AI，禁止加载模型。"
            )

        plan = self.get_plan()

        allowed = (
            plan.get("路由模型", [])
            + plan.get("视觉A", [])
            + plan.get("视觉B", [])
        )

        if model_name not in allowed:
            raise RuntimeError(
                f"当前病例禁止调用模型：{model_name}"
            )

        return self.manager.load_model(model_name)

    def doctor_finish(self):
        self.manager.deactivate_case()
        self.active = False
        self.modality = None

        print(
            "[Phoenix] 病例AI流程结束，模型已释放"
        )
