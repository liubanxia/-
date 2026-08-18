from pathlib import Path

from core.environment_paths import resolve_project_root


ROOT = resolve_project_root() / "04_AI模型"


class LazyNativeExpert:
    def __init__(
        self,
        model_id,
        task,
        factory=None,
        asset_path=None,
    ):
        self.model_id = model_id
        self.task = task
        self.factory = factory
        self.asset_path = (
            Path(asset_path)
            if asset_path
            else None
        )

        self.model = None
        self.status = "INTEGRATED_UNTESTED"

    def load(self):
        if self.model is not None:
            return self

        if self.factory is None:
            self.status = "ASSET_INTEGRATED_UNTESTED"
            return self

        self.model = self.factory()
        self.status = "LOADED_UNTESTED"
        return self

    def unload(self):
        self.model = None
        self.status = "INTEGRATED_UNTESTED"


def _sam_med3d():
    from model_adapters.sam_med3d import SAMMed3DAdapter

    return SAMMed3DAdapter(
        ROOT / "待接入模型/SAM-Med3D",
        ROOT / "待接入模型/SAM-Med3D/checkpoint/sam_med3d_turbo.pth",
    )


def _totalsegmentator():
    from model_adapters.totalsegmentator import TotalSegmentatorAdapter

    return TotalSegmentatorAdapter()


NATIVE_SPECIALISTS = {
    "sam_med3d": LazyNativeExpert(
        "sam_med3d",
        "ct_prompt_segmentation",
        factory=_sam_med3d,
    ),
    "totalsegmentator": LazyNativeExpert(
        "totalsegmentator",
        "ct_anatomy_segmentation",
        factory=_totalsegmentator,
    ),
    "merlin": LazyNativeExpert(
        "merlin",
        "ct_3d_encoder",
        asset_path=ROOT / "批量专家池/CT_通用/Merlin",
    ),
    "medsam2": LazyNativeExpert(
        "medsam2",
        "medical_segmentation",
        asset_path=ROOT / "待接入模型/MedSAM2",
    ),
}
