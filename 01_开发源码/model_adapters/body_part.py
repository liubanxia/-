from pathlib import Path

from core.model_adapter import ModelAdapter
from ai_models.ct_bodypart_adapter import BodyPartRegressionCTAdapter


class BodyPartAdapter(ModelAdapter):

    name = "body_part_regression"

    def __init__(self, model_path=None):
        self.project_root = (
            Path(__file__).resolve().parents[2]
        )

        self.repo_path = (
            self.project_root
            / "04_AI模型"
            / "路由模型"
            / "BodyPartRegression"
        )

        self.model_dir = (
            self.repo_path
            / "weights"
            / "public_bpr_model"
            / "public_bpr_model"
        )

        self.backend = None

    def load(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                str(self.repo_path)
            )

        if not self.model_dir.exists():
            raise FileNotFoundError(
                str(self.model_dir)
            )

        self.backend = BodyPartRegressionCTAdapter(
            project_root=self.project_root,
            repo_path=self.repo_path,
            model_dir=self.model_dir,
        )

    def predict(self, case):
        if not case.series:
            return {
                "model": self.name,
                "error": "病例没有CT序列",
            }

        # 当前MVP先处理切片最多的CT series
        series = max(
            case.series,
            key=lambda x: len(x.files),
        )

        if not series.files:
            return {
                "model": self.name,
                "error": "CT序列没有影像",
            }

        # 旧适配器支持直接传入一个DICOM，
        # 并自动寻找其所属完整Series。
        input_path = series.files[0]

        result = self.backend.run(
            str(input_path)
        )

        if not isinstance(result, dict):
            result = {
                "result": result
            }

        result["model"] = self.name
        result["processed_images"] = len(
            series.files
        )

        return result

    def unload(self):
        if self.backend is not None:
            try:
                self.backend.release()
            except Exception:
                pass

        self.backend = None
