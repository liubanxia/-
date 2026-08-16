from pathlib import Path
from core.model_adapter import ModelAdapter


class ReportModelAdapter(ModelAdapter):

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = Path(model_path)
        self.model = None
        self.processor = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                self.model_path
            )

    def predict(self, case):
        return {
            "model": self.name,
            "status": "report_backend_available",
            "case_id": case.case_id,
        }

    def unload(self):
        self.model = None
        self.processor = None
