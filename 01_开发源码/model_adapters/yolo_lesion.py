from pathlib import Path

from core.model_adapter import ModelAdapter


class YoloLesionAdapter(ModelAdapter):

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))

    def predict(self, case):
        return {
            "model": self.name,
            "status": "ready",
            "case_id": case.case_id,
        }

    def unload(self):
        self.model = None
