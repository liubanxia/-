from pathlib import Path


class LazyEncoderAdapter:
    def __init__(self, model_id, path, task):
        self.model_id = model_id
        self.path = Path(path)
        self.task = task
        self.model = None
        self.status = "REGISTERED_UNTESTED"

    def validate_assets(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return True

    def unload(self):
        self.model = None


def build_encoder_pool(model_root):
    """Build the local encoder pool without hard-coded workstation paths."""
    root = Path(model_root)
    return {
        "m3d_clip": LazyEncoderAdapter("m3d_clip", root / "M3D-CLIP", "ct_3d_encoder"),
        "medsiglip": LazyEncoderAdapter("medsiglip", root / "MedSigLIP", "medical_vision_encoder"),
        "rad_dino": LazyEncoderAdapter("rad_dino", root / "RAD-DINO", "radiology_vision_encoder"),
        "biovil_text": LazyEncoderAdapter("biovil_text", root / "BioViL-T", "cxr_text_encoder"),
        "merlin_ct": LazyEncoderAdapter("merlin_ct", root / "Merlin", "ct_3d_encoder"),
    }


ENCODER_POOL = {}
