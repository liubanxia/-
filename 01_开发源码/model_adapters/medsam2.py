from pathlib import Path

from core.model_adapter import ModelAdapter


class MedSAM2Adapter(ModelAdapter):

    name = "medsam2"

    def __init__(self, repo_path, checkpoint=None):
        self.repo_path = Path(repo_path)
        self.checkpoint = (
            Path(checkpoint)
            if checkpoint else None
        )

    def load(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(
                str(self.repo_path)
            )

    def predict(self, case):
        return {
            "model": self.name,
            "status": "adapter_ready",
            "checkpoint": (
                str(self.checkpoint)
                if self.checkpoint else None
            ),
        }
