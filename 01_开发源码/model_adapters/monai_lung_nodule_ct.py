from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

from core.ct_nifti import series_to_nifti
from core.ct_series_selector import select_ct_series
from core.model_adapter import ModelAdapter


class MonaiLungNoduleCTAdapter(ModelAdapter):
    """Direct inference adapter for MONAI lung_nodule_ct_detection."""

    name = "monai_lung_nodule_ct"

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.bundle_root = (
            self.project_root
            / "04_AI模型"
            / "批量专家池"
            / "MONAI_CT专科"
            / "lung_nodule_ct_detection"
        )
        self.config = self.bundle_root / "configs" / "inference.json"
        self.weights = self.bundle_root / "models" / "model.pt"

        self.loaded = False
        self.device = None
        self.network = None
        self.detector = None
        self.preprocessing = None
        self.postprocessing = None

        self.roi_size = (512, 512, 192)
        self.score_threshold = 0.02
        self.nms_threshold = 0.22

    def _select_device(self):
        import torch

        if not torch.cuda.is_available():
            return torch.device("cpu")

        try:
            major, _minor = torch.cuda.get_device_capability(0)
            if major < 5:
                return torch.device("cpu")
        except Exception:
            return torch.device("cpu")

        return torch.device("cuda:0")

    def _load_checkpoint(self):
        import torch

        try:
            checkpoint = torch.load(
                str(self.weights),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                str(self.weights),
                map_location="cpu",
            )

        if not isinstance(checkpoint, dict):
            raise RuntimeError("MONAI肺结节checkpoint格式无法识别")

        for key in ("model", "state_dict", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state_dict = value
                break
        else:
            state_dict = checkpoint

        try:
            self.network.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError:
            pass

        cleaned = {}
        for key, value in state_dict.items():
            new_key = str(key)
            for prefix in ("module.", "model.", "network."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value

        self.network.load_state_dict(cleaned, strict=True)

    def load(self):
        if self.loaded:
            return

        if not self.config.exists():
            raise RuntimeError(f"MONAI inference.json不存在: {self.config}")
        if not self.weights.exists():
            raise RuntimeError(f"MONAI肺结节权重不存在: {self.weights}")

        from monai.bundle import ConfigParser

        self.device = self._select_device()

        parser = ConfigParser()
        parser.read_config(str(self.config))

        transforms = parser["preprocessing"]["transforms"]
        transforms[0]["_disabled_"] = False
        transforms[1]["_disabled_"] = True
        transforms[4]["_disabled_"] = False

        post_transforms = parser["postprocessing"]["transforms"]
        post_transforms[1]["affine_lps_to_ras"] = False

        parser["bundle_root"] = str(self.bundle_root)
        parser["device"] = str(self.device)
        parser["amp"] = False
        parser["load_pretrain"] = False
        parser.parse(reset=True)

        self.network = parser.get_parsed_content("network")
        self.detector = parser.get_parsed_content("detector")
        self.preprocessing = parser.get_parsed_content("preprocessing")
        self.postprocessing = parser.get_parsed_content("postprocessing")

        self.detector.set_target_keys(
            box_key="box",
            label_key="label",
        )
        self.detector.set_box_selector_parameters(
            score_thresh=self.score_threshold,
            topk_candidates_per_level=1000,
            nms_thresh=self.nms_threshold,
            detections_per_img=300,
        )
        self.detector.set_sliding_window_inferer(
            roi_size=self.roi_size,
            overlap=0.25,
            sw_batch_size=1,
            mode="constant",
            device="cpu",
        )

        self._load_checkpoint()

        self.network.to(self.device)
        self.network.eval()
        self.detector.to(self.device)
        self.detector.eval()

        self.loaded = True
        print(f"MONAI_LUNG_DIRECT_READY device={self.device}")

    def unload(self):
        self.network = None
        self.detector = None
        self.preprocessing = None
        self.postprocessing = None
        self.loaded = False

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _select_ct_series(self, case):
        return select_ct_series(
            case,
            anatomy="chest",
            minimum_images=16,
        )

    @staticmethod
    def _cpu_value(value):
        try:
            import torch

            if torch.is_tensor(value):
                return value.detach().cpu()
        except Exception:
            pass
        return value

    @staticmethod
    def _python_value(value):
        try:
            import torch

            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
        except Exception:
            pass

        try:
            if hasattr(value, "tolist"):
                return value.tolist()
        except Exception:
            pass

        return value

    def _run_direct(self, series, work_dir):
        import torch

        input_nii = work_dir / "ct.nii.gz"
        series_to_nifti(series, input_nii)

        sample = self.preprocessing({"image": str(input_nii)})
        image = sample.get("image")

        if image is None:
            raise RuntimeError("MONAI预处理没有生成image")

        image_for_inference = image.to(self.device)
        sliding_window_size = math.prod(self.roi_size)
        image_size = image_for_inference[0, ...].numel()
        use_inferer = image_size >= sliding_window_size

        self.network.eval()
        self.detector.eval()
        self.detector.network = self.network
        self.detector.training = False

        with torch.inference_mode():
            prediction = self.detector(
                [image_for_inference],
                use_inferer=use_inferer,
            )

        if not prediction:
            return {
                "box": [],
                "label": [],
                "label_scores": [],
            }

        pred = prediction[0]
        if not isinstance(pred, dict):
            raise RuntimeError(
                "MONAI检测器输出格式异常: "
                f"{type(pred).__name__}"
            )

        post_input = {
            key: self._cpu_value(value)
            for key, value in pred.items()
        }
        post_input["image"] = image.cpu()

        processed = self.postprocessing(post_input)

        return {
            "box": self._python_value(processed.get("box", [])),
            "label": self._python_value(processed.get("label", [])),
            "label_scores": self._python_value(
                processed.get("label_scores", [])
            ),
        }

    def _to_lesions(self, data, series):
        boxes = data.get("box", []) if isinstance(data, dict) else []
        scores = data.get("label_scores", []) if isinstance(data, dict) else []
        labels = data.get("label", []) if isinstance(data, dict) else []

        lesions = []

        for index, box in enumerate(boxes or []):
            try:
                box = [float(value) for value in box]
            except Exception:
                continue

            if len(box) < 6:
                continue

            try:
                score = float(scores[index])
            except Exception:
                score = 0.0

            try:
                class_id = int(labels[index])
            except Exception:
                class_id = 0

            lesions.append(
                {
                    "type": "lung_nodule",
                    "finding": "肺结节候选灶",
                    "location": "肺",
                    "score": score,
                    "label": class_id,
                    "series_uid": str(getattr(series, "series_uid", "")),
                    "world_point_lps": box[:3],
                    "geometry_mode": "cccwhd_lps",
                    "geometry": {
                        "box_3d": box,
                    },
                    "source": self.name,
                }
            )

        return lesions

    def predict(self, case):
        if not self.loaded:
            raise RuntimeError("monai_lung_nodule_ct 尚未load")

        series = self._select_ct_series(case)

        temp_root = self.project_root / "08_temp_cache"
        temp_root.mkdir(parents=True, exist_ok=True)

        work_dir = Path(
            tempfile.mkdtemp(
                prefix="monai_lung_",
                dir=str(temp_root),
            )
        )

        try:
            raw = self._run_direct(series, work_dir)
            lesions = self._to_lesions(raw, series)

            return {
                "processed_images": len(getattr(series, "files", []) or []),
                "series_uid": str(getattr(series, "series_uid", "")),
                "lesions": lesions,
                "raw_prediction_count": len(lesions),
                "inference_backend": "direct_monai_retinanet",
                "device": str(self.device),
                "score_threshold": self.score_threshold,
                "nms_threshold": self.nms_threshold,
            }
        finally:
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )
