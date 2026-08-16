from pathlib import Path

from core.model_adapter import ModelAdapter
from core.dicom_pixels import read_dicom_image
from core.image_preprocess import to_rgb


class YoloLesionAdapter(ModelAdapter):

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = Path(model_path)
        self.model = None

    def load(self):
        from ultralytics import YOLO

        self.model = YOLO(
            str(self.model_path)
        )

    def predict(self, case):
        lesions = []

        for series in case.series:
            for index, path in enumerate(series.files):
                try:
                    image, _, _ = read_dicom_image(
                        path
                    )

                    rgb = to_rgb(image)

                    outputs = self.model.predict(
                        rgb,
                        verbose=False,
                    )

                except Exception:
                    continue

                for output in outputs:
                    boxes = getattr(
                        output,
                        "boxes",
                        None,
                    )

                    if boxes is None:
                        continue

                    for box in boxes:
                        xyxy = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .tolist()
                        )

                        conf = float(
                            box.conf[0]
                            .detach()
                            .cpu()
                        )

                        cls_id = int(
                            box.cls[0]
                            .detach()
                            .cpu()
                        )

                        names = output.names
                        label = names.get(
                            cls_id,
                            str(cls_id),
                        )

                        x1, y1, x2, y2 = xyxy

                        lesions.append({
                            "label": label,
                            "confidence": conf,
                            "series_uid": series.series_uid,
                            "image_index": index,
                            "point": (
                                int((x1 + x2) / 2),
                                int((y1 + y2) / 2),
                            ),
                            "box": [
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2),
                            ],
                        })

        return {
            "model": self.name,
            "lesions": lesions,
        }

    def unload(self):
        self.model = None
