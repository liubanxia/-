from pathlib import Path
import numpy as np

from core.model_adapter import ModelAdapter
from core.dicom_pixels import read_dicom_image
from core.image_preprocess import to_rgb


class YoloLesionAdapter(ModelAdapter):
    def __init__(self, name, model_path, task="detect"):
        self.name = name
        self.model_path = Path(model_path)
        self.task = task
        self.model = None

    def load(self):
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path), task=self.task)

    def predict(self, case):
        lesions, errors = [], []
        processed = 0
        for series in case.series:
            for index, path in enumerate(series.files):
                try:
                    image, _, ds = read_dicom_image(path)
                    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
                        image = float(np.max(image)) + float(np.min(image)) - image
                    outputs = self.model.predict(to_rgb(image), device="cpu", verbose=False)
                    processed += 1
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
                    continue
                for output in outputs:
                    boxes = getattr(output, "boxes", None)
                    if boxes is None:
                        continue
                    masks = getattr(output, "masks", None)
                    try:
                        polygons = masks.xy if masks is not None else []
                    except Exception:
                        polygons = []
                    names = output.names
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
                        conf = float(box.conf[0].detach().cpu())
                        cls_id = int(box.cls[0].detach().cpu())
                        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                        point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                        if i < len(polygons):
                            poly = np.asarray(polygons[i])
                            if poly.ndim == 2 and len(poly) >= 3:
                                point = (int(poly[:, 0].mean()), int(poly[:, 1].mean()))
                        lesions.append({
                            "label": label, "confidence": conf, "series_uid": series.series_uid,
                            "image_index": index, "point": point,
                            "box": [int(x1), int(y1), int(x2), int(y2)],
                        })
        if processed == 0:
            return {"model": self.name, "error": errors[0] if errors else "没有成功处理影像"}
        return {"model": self.name, "task": self.task, "processed_images": processed, "lesions": lesions, "warnings": errors}

    def unload(self):
        self.model = None
