#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class ScanRegion:
    name: str
    x: float
    y: float
    w: float
    h: float
    imgsz: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline whole-frame vs source-pixel tiled visible-human benchmark")
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--confidence", type=float, default=0.12)
    return p.parse_args()


def plan_regions(width: int, height: int) -> list[ScanRegion]:
    # Mirror the iOS PixelScalePlanner intent, but run the most demanding small-target plan for
    # this offline gate so we can directly measure whether preserving source pixels adds signal.
    regions = [ScanRegion("full", 0.0, 0.0, 1.0, 1.0, 640)]
    for yi, y in enumerate((0.0, 0.46)):
        for xi, x in enumerate((0.0, 0.46)):
            regions.append(ScanRegion(f"tile_{yi}_{xi}", x, y, 0.54, 0.54, 640))
    regions.append(ScanRegion("center_band", 0.12, 0.20, 0.76, 0.62, 640))
    regions.append(ScanRegion("center_crop", 0.24, 0.18, 0.52, 0.64, 768))
    return regions


def run_yolo(model: YOLO, image: np.ndarray, imgsz: int, conf: float) -> list[tuple[float, float, float, float, float]]:
    results = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=0.45,
        classes=[0],  # COCO generic person only. No team/enemy identity.
        device="cpu",
        verbose=False,
        max_det=20,
    )
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
        for b, s in zip(xyxy, scores)
    ]


def detect_whole(model: YOLO, frame: np.ndarray, conf: float) -> list[Detection]:
    return [
        Detection(x1, y1, x2, y2, score, "whole_640")
        for x1, y1, x2, y2, score in run_yolo(model, frame, 640, conf)
    ]


def detect_tiled(model: YOLO, frame: np.ndarray, conf: float) -> list[Detection]:
    h, w = frame.shape[:2]
    candidates: list[Detection] = []
    for region in plan_regions(w, h):
        x1 = max(0, min(w - 1, int(round(region.x * w))))
        y1 = max(0, min(h - 1, int(round(region.y * h))))
        x2 = max(x1 + 1, min(w, int(round((region.x + region.w) * w))))
        y2 = max(y1 + 1, min(h, int(round((region.y + region.h) * h))))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        for bx1, by1, bx2, by2, score in run_yolo(model, crop, region.imgsz, conf):
            candidates.append(
                Detection(
                    bx1 + x1,
                    by1 + y1,
                    bx2 + x1,
                    by2 + y1,
                    score,
                    region.name,
                )
            )
    return nms(candidates, iou_threshold=0.45)


def iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def nms(items: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(items, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [d for d in ordered if iou(best, d) < iou_threshold]
    return kept


def nearest_match(det: Detection, others: list[Detection]) -> bool:
    return any(iou(det, other) >= 0.30 for other in others)


def draw(frame: np.ndarray, whole: list[Detection], tiled: list[Detection], name: str) -> np.ndarray:
    canvas = frame.copy()
    # Whole-frame detections: blue. Pixel-preserving fusion detections: red.
    for d in whole:
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 90, 30), 2)
        cv2.putText(canvas, f"W {d.confidence:.2f}", (int(d.x1), max(16, int(d.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 90, 30), 1, cv2.LINE_AA)
    for d in tiled:
        thickness = 2 if nearest_match(d, whole) else 3
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (30, 45, 255), thickness)
        label = f"P {d.confidence:.2f} {d.source}"
        cv2.putText(canvas, label, (int(d.x1), min(canvas.shape[0] - 5, int(d.y2) + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 45, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"{name}  whole={len(whole)}  pixel_fusion={len(tiled)}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    if not images:
        return
    thumb_w = 480
    thumbs: list[np.ndarray] = []
    for image in images[:16]:
        scale = thumb_w / image.shape[1]
        thumb = cv2.resize(image, (thumb_w, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
    row_h = max(i.shape[0] for i in thumbs)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((row_h * rows, thumb_w * cols, 3), dtype=np.uint8)
    for idx, image in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * row_h:r * row_h + image.shape[0], c * thumb_w:(c + 1) * thumb_w] = image
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> int:
    args = parse_args()
    frames_dir = Path(args.frames)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise SystemExit("no benchmark frames")

    model = YOLO(args.model)
    rows: list[dict[str, object]] = []
    visual_frames: list[np.ndarray] = []
    whole_confs: list[float] = []
    tiled_confs: list[float] = []
    whole_total = 0
    tiled_total = 0
    whole_positive_frames = 0
    tiled_positive_frames = 0
    tile_gain_frames = 0
    tile_only_detections = 0
    tile_only_small_detections = 0

    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        whole = detect_whole(model, frame, args.confidence)
        tiled = detect_tiled(model, frame, args.confidence)
        whole_total += len(whole)
        tiled_total += len(tiled)
        whole_positive_frames += int(bool(whole))
        tiled_positive_frames += int(bool(tiled))
        tile_gain_frames += int(len(tiled) > len(whole))
        whole_confs.extend(d.confidence for d in whole)
        tiled_confs.extend(d.confidence for d in tiled)

        tile_only = [d for d in tiled if not nearest_match(d, whole)]
        tile_only_detections += len(tile_only)
        small = [d for d in tile_only if (d.height / max(h, 1)) < 0.10]
        tile_only_small_detections += len(small)

        rows.append({
            "frame": path.name,
            "width": w,
            "height": h,
            "whole_count": len(whole),
            "pixel_fusion_count": len(tiled),
            "tile_only_count": len(tile_only),
            "tile_only_small_count": len(small),
            "whole_max_confidence": max((d.confidence for d in whole), default=0.0),
            "pixel_fusion_max_confidence": max((d.confidence for d in tiled), default=0.0),
        })
        visual_frames.append(draw(frame, whole, tiled, path.name))

    processed = len(rows)
    if processed == 0:
        raise SystemExit("frames could not be decoded")

    with (out / "detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_contact_sheet(visual_frames, out / "contact_sheet.jpg")

    report = {
        "benchmark": "LiteView public gameplay generic-visible-human detector signal",
        "source_url": args.source_url,
        "model": args.model,
        "confidence_threshold": args.confidence,
        "processed_frames": processed,
        "whole_frame_640": {
            "frames_with_person_signal": whole_positive_frames,
            "positive_frame_rate": whole_positive_frames / processed,
            "total_detections": whole_total,
            "median_confidence": median(whole_confs),
        },
        "pixel_preserving_fusion": {
            "regions_per_frame": 7,
            "frames_with_person_signal": tiled_positive_frames,
            "positive_frame_rate": tiled_positive_frames / processed,
            "total_detections_after_nms": tiled_total,
            "median_confidence": median(tiled_confs),
            "frames_with_more_detections_than_whole": tile_gain_frames,
            "tile_only_detections": tile_only_detections,
            "tile_only_small_detections_under_10pct_frame_height": tile_only_small_detections,
        },
        "signal_delta": {
            "positive_frames": tiled_positive_frames - whole_positive_frames,
            "detections": tiled_total - whole_total,
        },
        "interpretation_limit": "This is detector-signal validation on public gameplay, not manual ground-truth recall. Boxes must be visually inspected before calling a detection correct. It does not infer team/enemy identity.",
    }
    (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = f"""## LiteView public gameplay pixel benchmark\n\n- Source: `{args.source_url}`\n- Model: `{args.model}` (generic COCO person)\n- Frames processed: **{processed}**\n- Whole-frame 640: person signal on **{whole_positive_frames}/{processed}** frames; total boxes **{whole_total}**\n- Pixel-preserving fusion: person signal on **{tiled_positive_frames}/{processed}** frames; total boxes after NMS **{tiled_total}**\n- Frames where tiled fusion produced more boxes: **{tile_gain_frames}/{processed}**\n- Tile-only boxes: **{tile_only_detections}**\n- Tile-only small boxes (<10% frame height): **{tile_only_small_detections}**\n\n**Important:** this is a real model-execution signal check, not ground-truth accuracy. The contact sheet must be inspected before treating any box as a correct visible-human detection. No team/enemy identity is inferred.\n"""
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
