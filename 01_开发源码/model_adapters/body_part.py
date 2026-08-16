from pathlib import Path
import numpy as np

from core.model_adapter import ModelAdapter


class BodyPartAdapter(ModelAdapter):

    name = "body_part_regression"

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.session = None

    def load(self):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, case):
        return {
            "model": self.name,
            "status": "ready",
            "case_id": case.case_id,
        }

    def unload(self):
        self.session = None
