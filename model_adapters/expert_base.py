from pathlib import Path


class PhoenixExpertAdapter:

    model_id = "unknown"
    task = "unknown"

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.loaded = False

    def validate_assets(self):
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        return True

    def load(self):
        raise NotImplementedError

    def run(self, case):
        raise NotImplementedError

    def unload(self):
        self.model = None
        self.loaded = False

    def describe(self):
        return {
            "model_id": self.model_id,
            "task": self.task,
            "path": str(self.model_path),
            "loaded": self.loaded,
        }
