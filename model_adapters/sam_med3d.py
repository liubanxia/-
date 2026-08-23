from pathlib import Path
from core.model_adapter import ModelAdapter


class SAMMed3DAdapter(ModelAdapter):
    name = "sam_med3d"

    def __init__(self, repo_path, checkpoint):
        self.repo_path = Path(repo_path)
        self.checkpoint = Path(checkpoint)

    def load(self):
        if not self.repo_path.exists():
            raise FileNotFoundError(self.repo_path)
        if not self.checkpoint.exists():
            raise FileNotFoundError(self.checkpoint)

    def predict(self, case):
        return {"model": self.name, "status": "ready_for_prompt_segmentation", "case_id": case.case_id}
