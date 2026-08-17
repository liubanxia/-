from pathlib import Path


class HFLocalEncoderAdapter:

    def __init__(self, model_id, path, task):
        self.model_id = model_id
        self.path = Path(path)
        self.task = task

        self.model = None
        self.processor = None

        self.status = "ADAPTER_READY_UNTESTED"

    def validate_assets(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return True

    def load(self, device="cpu"):
        self.validate_assets()

        import torch
        from transformers import (
            AutoModel,
            AutoProcessor,
        )

        self.model = AutoModel.from_pretrained(
            str(self.path),
            local_files_only=True,
            trust_remote_code=True,
        )

        try:
            self.processor = AutoProcessor.from_pretrained(
                str(self.path),
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception:
            self.processor = None

        self.model.eval()
        self.model.to(device)

        self.status = "LOADED_UNTESTED"

        return self

    def unload(self):
        self.model = None
        self.processor = None
        self.status = "ADAPTER_READY_UNTESTED"

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
