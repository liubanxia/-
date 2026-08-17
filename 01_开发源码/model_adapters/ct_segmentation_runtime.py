from pathlib import Path
import gc


class VISTA3DAdapter:
    def __init__(self):
        self.model_id = "vista3d"
        self.root = Path(
            r"D:\project_phoenix\04_AI模型\批量专家池\CT_分割\VISTA3D-HF"
        )
        self.pretrained = self.root / "vista3d_pretrained_model"
        self.weight = self.pretrained / "model.safetensors"

        self.task = "ct_3d_segmentation"
        self.model = None
        self.status = "ADAPTER_READY_UNTESTED"

    def validate_assets(self):
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        if not self.weight.exists():
            raise FileNotFoundError(self.weight)
        return True

    def load(self, device="cpu"):
        self.validate_assets()

        import sys
        from safetensors.torch import load_file

        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        from vista3d_config import VISTA3DConfig
        from vista3d_model import VISTA3DModel

        config = VISTA3DConfig.from_pretrained(
            str(self.pretrained),
            local_files_only=True,
        )

        model = VISTA3DModel(config)

        state = load_file(
            str(self.weight),
            device="cpu",
        )

        result = model.load_state_dict(
            state,
            strict=False,
        )

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "VISTA3D checkpoint mismatch: "
                f"missing={len(result.missing_keys)}, "
                f"unexpected={len(result.unexpected_keys)}"
            )

        model.eval()
        model.to(device)

        self.model = model
        self.status = "LOADED_UNTESTED"
        return self

    def run(
        self,
        volume,
        class_vector=None,
        point_coords=None,
        point_labels=None,
    ):
        if self.model is None:
            raise RuntimeError("VISTA3D is not loaded")

        if class_vector is None and point_coords is None:
            raise ValueError(
                "VISTA3D requires class_vector or point prompt"
            )

        import torch

        with torch.inference_mode():
            return self.model.network(
                volume,
                class_vector=class_vector,
                point_coords=point_coords,
                point_labels=point_labels,
            )

    def unload(self):
        self.model = None
        self.status = "ADAPTER_READY_UNTESTED"
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class SegVolAdapter:
    def __init__(self):
        self.model_id = "segvol"
        self.root = Path(
            r"D:\project_phoenix\04_AI模型\批量专家池\CT_分割\SegVol"
        )

        self.task = "prompted_ct_3d_segmentation"
        self.model = None
        self.processor = None
        self.status = "ADAPTER_READY_UNTESTED"

    def validate_assets(self):
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        return True

    def load(self, device="cpu"):
        self.validate_assets()

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(self.root),
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )

        model.eval()
        model.to(device)

        self.model = model
        self.processor = getattr(model, "processor", None)

        self.status = "LOAD_COMPAT_PASS_FORWARD_PENDING"
        return self

    def run(
        self,
        image,
        zoomed_image=None,
        text_prompt=None,
        bbox_prompt_group=None,
        point_prompt_group=None,
        use_zoom=False,
    ):
        if self.model is None:
            raise RuntimeError("SegVol is not loaded")

        if (
            text_prompt is None
            and bbox_prompt_group is None
            and point_prompt_group is None
        ):
            raise ValueError("SegVol requires at least one prompt")

        if zoomed_image is None:
            zoomed_image = image

        import torch

        with torch.inference_mode():
            return self.model(
                image=image,
                zoomed_image=zoomed_image,
                text_prompt=text_prompt,
                bbox_prompt_group=bbox_prompt_group,
                point_prompt_group=point_prompt_group,
                use_zoom=use_zoom,
            )

    def unload(self):
        self.model = None
        self.processor = None
        self.status = "ADAPTER_READY_UNTESTED"
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


VISTA3D = VISTA3DAdapter()
SEGVOL = SegVolAdapter()

CT_SEGMENTATION_POOL = {
    "vista3d": VISTA3D,
    "segvol": SEGVOL,
}
