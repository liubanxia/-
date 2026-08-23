from pathlib import Path
from core.model_adapter import ModelAdapter


class MedSAM2Adapter(ModelAdapter):
    name = "medsam2"

    def __init__(self, repo_path, checkpoint=None):
        self.repo_path = Path(repo_path)
        self.checkpoint = Path(checkpoint) if checkpoint else None

    def load(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(str(self.repo_path))

    def predict(self, case):
        raise RuntimeError("MedSAM2真实自动病灶推理尚未接入")

    def unload(self):
        pass
