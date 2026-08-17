import gc
import inspect
import torch

from core.segmentation_prompt_memory import SEGMENTATION_PROMPT_MEMORY
from core.expert_runtime_bridge import EXPERT_RUNTIME_BRIDGE


class PromptSegmentationRunner:

    def _bbox3d(self, item, shape):
        bbox = item.get("bbox")

        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None

        _, _, d, h, w = shape
        x1, y1, x2, y2 = map(float, bbox[:4])

        z = item.get("slice_index")
        z = d / 2 if z is None else float(z)

        z1 = max(0.0, z - 2)
        z2 = min(float(d - 1), z + 2)

        return torch.tensor(
            [[[z1, y1, x1], [z2, y2, x2]]],
            dtype=torch.float32,
        )

    def run_segvol(self, volume):
        prompts = SEGMENTATION_PROMPT_MEMORY.all()

        if not prompts:
            return []

        adapter = EXPERT_RUNTIME_BRIDGE.resolve(
            "SegVol::segmentation_model"
        )

        if adapter is None:
            return []

        outputs = []

        try:
            adapter.load("cpu")

            for item in prompts:
                box = self._bbox3d(item, volume.shape)

                if box is None:
                    continue

                with torch.inference_mode():
                    y = adapter.run(
                        image=volume,
                        zoomed_image=volume,
                        bbox_prompt_group=(box, None),
                        use_zoom=False,
                    )

                outputs.append({
                    "expert_id": "SegVol",
                    "task": "prompt_segmentation",
                    "tensor": y.detach().cpu(),
                    "source_prompt": item,
                })

        finally:
            adapter.unload()
            gc.collect()

        return outputs

    def run_sam_med3d(self, volume):
        native = EXPERT_RUNTIME_BRIDGE.resolve("sam_med3d")

        if native is None:
            return []

        prompts = SEGMENTATION_PROMPT_MEMORY.all()

        if not prompts:
            return []

        outputs = []

        try:
            native.load()
            target = getattr(native, "model", None)

            if target is None:
                return []

            if hasattr(target, "load"):
                try:
                    target.load()
                except Exception:
                    pass

            method = None

            for name in ("run", "predict", "segment"):
                fn = getattr(target, name, None)
                if callable(fn):
                    method = fn
                    break

            if method is None:
                return []

            params = inspect.signature(method).parameters

            for item in prompts:
                kwargs = {}

                if "image" in params:
                    kwargs["image"] = volume
                elif "volume" in params:
                    kwargs["volume"] = volume

                if "bbox" in params:
                    kwargs["bbox"] = item.get("bbox")
                elif "boxes" in params:
                    kwargs["boxes"] = item.get("bbox")

                if "point" in params:
                    kwargs["point"] = item.get("point")
                elif "points" in params:
                    kwargs["points"] = item.get("point")

                try:
                    y = method(**kwargs)

                    outputs.append({
                        "expert_id": "SAM-Med3D",
                        "task": "prompt_segmentation",
                        "tensor": y,
                        "source_prompt": item,
                    })
                except Exception:
                    continue

        finally:
            try:
                native.unload()
            except Exception:
                pass
            gc.collect()

        return outputs


PROMPT_SEGMENTATION_RUNNER = PromptSegmentationRunner()
