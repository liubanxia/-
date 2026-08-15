from pathlib import Path
import numpy as np

class PacsFractureAI:
    def __init__(self, root):
        root = Path(root)
        self.loc_path = root / "yolov8_localization_fractureAtlas.pt"
        self.seg_path = root / "yolov8_segmentation_fractureAtlas.pt"
        self.loc = None
        self.seg = None

    def _load(self):
        from ultralytics import YOLO
        if self.loc is None:
            self.loc = YOLO(str(self.loc_path))
        if self.seg is None:
            self.seg = YOLO(str(self.seg_path))

    def infer(self, image):
        self._load()
        arr = np.asarray(image)

        r1 = self.loc.predict(arr, verbose=False)[0]
        r2 = self.seg.predict(arr, verbose=False)[0]

        boxes = []
        if r1.boxes is not None:
            for xyxy, conf in zip(r1.boxes.xyxy, r1.boxes.conf):
                boxes.append({
                    "bbox_xyxy": [float(x) for x in xyxy.tolist()],
                    "confidence": float(conf),
                })

        masks = []
        if r2.masks is not None:
            for poly in r2.masks.xy:
                masks.append({
                    "polygon_xy": np.asarray(poly).tolist()
                })

        n = len(boxes)

        report = (
            "【AI影像学所见】\n"
            + (
                f"检测到骨折候选 {n} 处，请核对标记区域。"
                if n else
                "当前未检出超过阈值的骨折候选。"
            )
            + f"\n分割区域：{len(masks)}处。"
            + "\n\n【诊断意见】\n"
            "AI辅助结果，最终诊断由医生结合PACS原始影像审核。"
        )

        return {
            "boxes": boxes,
            "masks": masks,
            "report": report,
            "ram_only": True,
        }
