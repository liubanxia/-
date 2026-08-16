import sys
from pathlib import Path

import pydicom
import torch

from core.model_adapter import ModelAdapter


XRAY_MODALITIES = {"DX", "DR", "CR", "XR"}


class TorchXRayVisionChestAdapter(ModelAdapter):

    name = "torchxrayvision_chest"

    def __init__(self, source_dir, cache_dir):
        self.source_dir = Path(source_dir)
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.xrv = None

    def load(self):
        source = str(self.source_dir)

        if source not in sys.path:
            sys.path.insert(0, source)

        import torchxrayvision as xrv

        self.xrv = xrv

        self.model = xrv.models.DenseNet(
            weights="densenet121-res224-all",
            cache_dir=str(self.cache_dir),
        )

        self.model.cpu()
        self.model.eval()

    def _pick_image(self, case):
        fallback = None

        for series in case.series:
            if str(series.modality).upper() not in XRAY_MODALITIES:
                continue

            for path in series.files:
                if fallback is None:
                    fallback = path

                try:
                    ds = pydicom.dcmread(
                        str(path),
                        stop_before_pixels=True,
                        force=True,
                    )
                except Exception:
                    continue

                view = str(
                    getattr(ds, "ViewPosition", "")
                ).upper()

                if view in {"PA", "AP"}:
                    return path

        return fallback

    def predict(self, case):
        if self.model is None:
            raise RuntimeError(
                "TorchXRayVision尚未加载"
            )

        path = self._pick_image(case)

        if path is None:
            return {
                "model": self.name,
                "processed_images": 0,
                "scores": {},
                "ranked_candidates": [],
                "lesions": [],
            }

        img = self.xrv.utils.load_image(
            str(path)
        )

        img = self.xrv.datasets.XRayCenterCrop()(
            img
        )

        img = self.xrv.datasets.XRayResizer(
            224
        )(img)

        tensor = torch.from_numpy(
            img
        ).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(
                tensor
            )[0].cpu().numpy()

        scores = {
            name: float(score)
            for name, score in zip(
                self.model.pathologies,
                output,
            )
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            "model": self.name,
            "processed_images": 1,
            "image_path": str(path),
            "scores": scores,
            "ranked_candidates": [
                {
                    "label": name,
                    "score": score,
                }
                for name, score in ranked
            ],
            "lesions": [],
        }

    def unload(self):
        self.model = None
        self.xrv = None
