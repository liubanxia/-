from pathlib import Path

from core.model_adapter import ModelAdapter


class HFDirectoryAdapter(ModelAdapter):

    def __init__(self, name, model_path, role):
        self.name = name
        self.model_path = Path(model_path)
        self.role = role

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                str(self.model_path)
            )

    def predict(self, case):
        raise RuntimeError(
            f"{self.name} 真实前向推理尚未接入"
        )

    def unload(self):
        pass
