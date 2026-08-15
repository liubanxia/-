from pathlib import Path

from ai.onnx_visuals import OnnxVisualB
from ai.visual_b_router import VisualBRouter
from ai.visual_b_yolov8_rescbam_components import (
    resolve_yolov8_rescbam_components,
)


class VisualBModelManager:
    """
    Project Phoenix 视觉B模型管理器。

    职责：
    1. 根据VisualBRouter选择专科子模型；
    2. 只有真正需要推理时才创建对应OnnxVisualB；
    3. 不在初始化阶段自动加载ONNX Session；
    4. 缺少适用模型时明确返回“未运行”；
    5. 支持后续增加更多视觉B子模型；
    6. 支持显式卸载当前模型。

    注意：
    - 路由器本身不运行模型；
    - 创建OnnxVisualB也不会加载ONNX Session；
    - ONNX Session只在infer()首次调用时加载。
    """

    PEDIATRIC_WRIST_MODEL_ID = (
        "yolov8_rescbam_wrist_dx_v1"
    )

    def __init__(
        self,
        model_root,
    ):
        self.model_root = Path(
            model_root
        )

        self._active_model_id = None
        self._active_visual = None

    @property
    def active_model_id(self):
        return self._active_model_id

    @property
    def active_visual(self):
        return self._active_visual

    @property
    def is_loaded(self):
        if self._active_visual is None:
            return False

        return bool(
            self._active_visual.is_loaded
        )

    def _create_pediatric_wrist_model(
        self,
    ):
        model_path = (
            self.model_root
            / "YOLOv8_ResCBAM.onnx"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                "视觉B儿童腕部模型不存在："
                f"{model_path}"
            )

        components = (
            resolve_yolov8_rescbam_components()
        )

        return OnnxVisualB(
            model_path=str(model_path),
            input_builder=(
                components["input_builder"]
            ),
            model_contract=(
                components["model_contract"]
            ),
            output_parser=(
                components["output_parser"]
            ),
        )

    def _create_model(
        self,
        model_id,
    ):
        if (
            model_id
            == self.PEDIATRIC_WRIST_MODEL_ID
        ):
            return (
                self._create_pediatric_wrist_model()
            )

        raise KeyError(
            "视觉B模型管理器未注册模型："
            f"{model_id}"
        )

    def _activate_model(
        self,
        model_id,
    ):
        # 已经是当前模型，则直接复用对象。
        if (
            self._active_visual is not None
            and self._active_model_id == model_id
        ):
            return self._active_visual

        # 切换专科模型前，先释放旧Session。
        self.unload()

        visual = self._create_model(
            model_id
        )

        self._active_model_id = model_id
        self._active_visual = visual

        return visual

    def infer(
        self,
        series_context,
    ):
        """
        根据当前影像上下文路由并执行视觉B。

        返回统一字典：
        {
            "status": ...,
            "route": ...,
            "model_id": ...,
            "candidates": ...
        }
        """

        decision = VisualBRouter.route(
            series_context
        )
        if not decision.has_model:
            # 当前没有安全适用模型时，
            # 不允许沿用上一病例模型。
            self.unload()

            return {
                "status": "no_applicable_model",
                "route": decision.reason,
                "model_id": None,
                "candidates": (),
            }

        visual = self._activate_model(
            decision.selected_model_id
        )

        # 只有到这里才真正触发ONNX推理。
        candidates = visual.infer(
            series_context
        )

        return {
            "status": "success",
            "route": decision.reason,
            "model_id": (
                decision.selected_model_id
            ),
            "candidates": candidates,
        }

    def unload(self):
        """
        显式释放当前视觉B模型Session，
        并清除当前活动模型身份。
        """

        was_loaded = False

        if self._active_visual is not None:
            was_loaded = bool(
                self._active_visual.unload()
            )

        self._active_visual = None
        self._active_model_id = None

        return was_loaded
