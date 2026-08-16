from pathlib import Path

from core.model_adapter import ModelAdapter


class HFDirectoryAdapter(ModelAdapter):

    def __init__(self, name, model_path, role):
        self.name = name
        self.model_path = Path(model_path)
        self.role = role
        self.model = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                str(self.model_path)
            )

    def predict(self, case):
        return {
            "model": self.name,
            "role": self.role,
            "status": "available",
            "case_id": case.case_id,
        }

    def unload(self):
        self.model = None
