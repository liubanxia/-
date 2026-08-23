from pathlib import Path


class LazyNativeExpert:
    def __init__(self, model_id, task, factory=None, asset_path=None):
        self.model_id = model_id; self.task = task; self.factory = factory
        self.asset_path = Path(asset_path) if asset_path else None
        self.model = None; self.status = "INTEGRATED_UNTESTED"

    def load(self):
        if self.model is not None: return self
        if self.factory is None:
            self.status = "ASSET_INTEGRATED_UNTESTED"; return self
        self.model = self.factory(); self.status = "LOADED_UNTESTED"; return self

    def unload(self): self.model = None; self.status = "INTEGRATED_UNTESTED"


def build_native_specialists(model_root):
    root = Path(model_root)
    def sam_med3d():
        from model_adapters.sam_med3d import SAMMed3DAdapter
        return SAMMed3DAdapter(root / "SAM-Med3D", root / "SAM-Med3D" / "checkpoint" / "sam_med3d_turbo.pth")
    def totalsegmentator():
        from model_adapters.totalsegmentator import TotalSegmentatorAdapter
        return TotalSegmentatorAdapter()
    return {
        "sam_med3d": LazyNativeExpert("sam_med3d", "ct_prompt_segmentation", factory=sam_med3d),
        "totalsegmentator": LazyNativeExpert("totalsegmentator", "ct_anatomy_segmentation", factory=totalsegmentator),
        "merlin": LazyNativeExpert("merlin", "ct_3d_encoder", asset_path=root / "Merlin"),
        "medsam2": LazyNativeExpert("medsam2", "medical_segmentation", asset_path=root / "MedSAM2"),
    }


NATIVE_SPECIALISTS = {}
