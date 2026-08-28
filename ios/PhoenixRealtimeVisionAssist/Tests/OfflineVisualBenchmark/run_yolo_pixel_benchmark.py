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
    seg_overlap: float = 0.0
    track_hits: int = 1
    static_score: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) * 0.5

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) * 0.5


@dataclass(frozen=True)
class ScanRegion:
    name: str
    x: float
    y: float
    w: float
    h: float
    imgsz: int


@dataclass
class TrackState:
    track_id: int
    box: Detection
    hits: int
    misses: int
    age: int
    centers: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline pixel-preserving visible-human benchmark with segmentation and temporal confirmation")
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--seg-model", default="yolo11n-seg.pt")
    p.add_argument("--confidence", type=float, default=0.12)
    p.add_argument("--seg-confidence", type=float, default=0.10)
    p.add_argument("--min-track-hits", type=int, default=2)
    return p.parse_args()


def plan_regions(width: int, height: int) -> list[ScanRegion]:
    regions = [ScanRegion("full", 0.0, 0.0, 1.0, 1.0, 640)]
    for yi, y in enumerate((0.0, 0.46)):
        for xi, x in enumerate((0.0, 0.46)):
            regions.append(ScanRegion(f"tile_{yi}_{xi}", x, y, 0.54, 0.54, 640))
    regions.append(ScanRegion("center_band", 0.12, 0.20, 0.76, 0.62, 640))
    regions.append(ScanRegion("center_crop", 0.24, 0.18, 0.52, 0.64, 768))
    return regions


def run_yolo(model: YOLO, image: np.ndarray, imgsz: int, conf: float) -> list[tuple[float, float, float, float, float]]:
    results = model.predict(source=image, imgsz=imgsz, conf=conf, iou=0.45, classes=[0], device="cpu", verbose=False, max_det=30)
    if not results:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)) for b, s in zip(xyxy, scores)]


def detect_whole(model: YOLO, frame: np.ndarray, conf: float) -> list[Detection]:
    return [Detection(x1, y1, x2, y2, score, "whole_640") for x1, y1, x2, y2, score in run_yolo(model, frame, 640, conf)]


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
            candidates.append(Detection(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, score, region.name))
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


def expand_box(d: Detection, w: int, h: int, margin: float = 0.20) -> tuple[int, int, int, int]:
    dx = d.width * margin
    dy = d.height * margin
    x1 = max(0, int(math.floor(d.x1 - dx)))
    y1 = max(0, int(math.floor(d.y1 - dy)))
    x2 = min(w, int(math.ceil(d.x2 + dx)))
    y2 = min(h, int(math.ceil(d.y2 + dy)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def segmentation_overlap(seg_model: YOLO, frame: np.ndarray, det: Detection, conf: float) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(det, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    results = seg_model.predict(source=crop, imgsz=512, conf=conf, iou=0.45, classes=[0], device="cpu", verbose=False, max_det=8, retina_masks=True)
    if not results:
        return 0.0
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0 or r.masks is None or r.masks.data is None:
        return 0.0
    masks = r.masks.data.cpu().numpy()
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    target_area = max(1.0, det.width * det.height)
    best = 0.0
    for box, score, mask in zip(boxes, scores, masks):
        if float(score) < conf:
            continue
        bx1, by1, bx2, by2 = [float(v) for v in box]
        gx1, gy1, gx2, gy2 = bx1 + x1, by1 + y1, bx2 + x1, by2 + y1
        candidate = Detection(gx1, gy1, gx2, gy2, float(score), "seg")
        spatial = iou(det, candidate)
        mask_ratio = float((mask > 0.5).mean())
        area_ratio = min(candidate.width * candidate.height, target_area) / max(candidate.width * candidate.height, target_area)
        best = max(best, 0.60 * spatial + 0.25 * min(1.0, mask_ratio * 4.0) + 0.15 * area_ratio)
    return float(best)


def update_tracks(tracks: list[TrackState], detections: list[Detection], frame_w: int, frame_h: int, next_track_id: int) -> tuple[list[TrackState], list[Detection], int]:
    unmatched = set(range(len(detections)))
    confirmed: list[Detection] = []
    for t in tracks:
        best_idx = None
        best_score = -1.0
        for idx in list(unmatched):
            d = detections[idx]
            overlap = iou(t.box, d)
            dx = (t.box.cx - d.cx) / max(frame_w, 1)
            dy = (t.box.cy - d.cy) / max(frame_h, 1)
            center_dist = math.hypot(dx, dy)
            score = overlap * 0.75 + max(0.0, 1.0 - center_dist / 0.12) * 0.25
            if score > best_score and (overlap >= 0.12 or center_dist <= 0.06):
                best_idx = idx
                best_score = score
        t.age += 1
        if best_idx is None:
            t.misses += 1
            continue
        d = detections[best_idx]
        unmatched.remove(best_idx)
        t.box = d
        t.hits += 1
        t.misses = 0
        t.centers.append((d.cx / max(frame_w, 1), d.cy / max(frame_h, 1)))
        if len(t.centers) > 12:
            t.centers.pop(0)
        static_score = 0.0
        if len(t.centers) >= 4:
            xs = [p[0] for p in t.centers]
            ys = [p[1] for p in t.centers]
            static_score = max(0.0, 1.0 - (statistics.pstdev(xs) + statistics.pstdev(ys)) / 0.012)
        confirmed.append(Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.source, d.seg_overlap, t.hits, static_score))

    tracks = [t for t in tracks if t.misses <= 2]
    for idx in unmatched:
        d = detections[idx]
        tracks.append(TrackState(next_track_id, d, 1, 0, 1, [(d.cx / max(frame_w, 1), d.cy / max(frame_h, 1))]))
        confirmed.append(Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.source, d.seg_overlap, 1, 0.0))
        next_track_id += 1
    return tracks, confirmed, next_track_id


def is_hud_like(d: Detection, frame_w: int, frame_h: int) -> bool:
    cx = d.cx / max(frame_w, 1)
    cy = d.cy / max(frame_h, 1)
    h_ratio = d.height / max(frame_h, 1)
    w_ratio = d.width / max(frame_w, 1)
    bottom_weapon_zone = cy > 0.72 and 0.22 < cx < 0.78 and (w_ratio > 0.10 or h_ratio > 0.18)
    static_overlay = d.track_hits >= 4 and d.static_score >= 0.92
    extreme_edge = (cx < 0.04 or cx > 0.96) and h_ratio < 0.18
    return bottom_weapon_zone or static_overlay or extreme_edge


def final_accept(d: Detection, frame_w: int, frame_h: int, min_track_hits: int) -> bool:
    h_ratio = d.height / max(frame_h, 1)
    strong_seg = d.seg_overlap >= 0.36
    strong_det = d.confidence >= 0.24
    temporal = d.track_hits >= min_track_hits
    tiny_target = h_ratio < 0.10
    evidence_ok = strong_seg and temporal if tiny_target else strong_seg and (temporal or strong_det)
    return evidence_ok and not is_hud_like(d, frame_w, frame_h)


def draw(frame: np.ndarray, whole: list[Detection], tiled: list[Detection], final: list[Detection], name: str) -> np.ndarray:
    canvas = frame.copy()
    for d in whole:
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (255, 90, 30), 1)
    for d in tiled:
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (30, 45, 255), 1)
    for d in final:
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (70, 220, 70), 2)
        cv2.putText(canvas, f"V {d.confidence:.2f} S{d.seg_overlap:.2f} T{d.track_hits}", (int(d.x1), max(16, int(d.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70, 220, 70), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"{name} whole={len(whole)} tiled={len(tiled)} verified={len(final)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    if not images:
        return
    thumb_w = 480
    thumbs: list[np.ndarray] = []
    for image in images[:20]:
        scale = thumb_w / image.shape[1]
        thumbs.append(cv2.resize(image, (thumb_w, max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    row_h = max(i.shape[0] for i in thumbs)
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.zeros((row_h * rows, thumb_w * cols, 3), dtype=np.uint8)
    for idx, image in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * row_h:r * row_h + image.shape[0], c * thumb_w:(c + 1) * thumb_w] = image
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


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

    detector = YOLO(args.model)
    segmenter = YOLO(args.seg_model)
    tracks: list[TrackState] = []
    next_track_id = 1
    rows: list[dict[str, object]] = []
    visual_frames: list[np.ndarray] = []
    whole_total = tiled_total = seg_confirmed_total = verified_total = 0
    whole_positive = tiled_positive = verified_positive = 0
    rejected_hud = rejected_seg = rejected_temporal = 0
    seg_scores: list[float] = []

    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        whole = detect_whole(detector, frame, args.confidence)
        tiled = detect_tiled(detector, frame, args.confidence)
        with_seg: list[Detection] = []
        for d in tiled:
            seg = segmentation_overlap(segmenter, frame, d, args.seg_confidence)
            seg_scores.append(seg)
            if seg >= 0.18:
                with_seg.append(Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.source, seg, 1, 0.0))
            else:
                rejected_seg += 1
        tracks, tracked, next_track_id = update_tracks(tracks, with_seg, w, h, next_track_id)
        final: list[Detection] = []
        for d in tracked:
            if is_hud_like(d, w, h):
                rejected_hud += 1
                continue
            if d.track_hits < args.min_track_hits and d.confidence < 0.24:
                rejected_temporal += 1
            if final_accept(d, w, h, args.min_track_hits):
                final.append(d)

        whole_total += len(whole)
        tiled_total += len(tiled)
        seg_confirmed_total += len(with_seg)
        verified_total += len(final)
        whole_positive += int(bool(whole))
        tiled_positive += int(bool(tiled))
        verified_positive += int(bool(final))
        rows.append({
            "frame": path.name,
            "width": w,
            "height": h,
            "whole_count": len(whole),
            "tiled_count": len(tiled),
            "seg_candidate_count": len(with_seg),
            "verified_count": len(final),
            "max_verified_conf": max((d.confidence for d in final), default=0.0),
            "max_verified_seg": max((d.seg_overlap for d in final), default=0.0),
            "max_verified_track_hits": max((d.track_hits for d in final), default=0),
        })
        visual_frames.append(draw(frame, whole, tiled, final, path.name))

    processed = len(rows)
    if processed == 0:
        raise SystemExit("frames could not be decoded")

    with (out / "detections.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(visual_frames, out / "contact_sheet_verified.jpg")

    report = {
        "benchmark": "LiteView offline visible-human evidence chain",
        "source_url": args.source_url,
        "detector_model": args.model,
        "segmenter_model": args.seg_model,
        "processed_frames": processed,
        "whole_frame_640": {"positive_frames": whole_positive, "total_boxes": whole_total},
        "pixel_tiled_candidates": {"positive_frames": tiled_positive, "total_boxes": tiled_total},
        "segmentation_review": {"total_candidates_passing_loose_seg_gate": seg_confirmed_total, "median_seg_overlap_score": median(seg_scores), "rejected_by_segmentation": rejected_seg},
        "temporal_and_hud_filter": {"verified_positive_frames": verified_positive, "verified_total_boxes": verified_total, "rejected_hud_or_static_overlay": rejected_hud, "rejected_low_temporal_evidence": rejected_temporal},
        "interpretation_limit": "Offline public-gameplay visible-human evidence check only. No team/enemy identity, aiming point, hidden target, live game coordinates, or anti-cheat behavior is produced.",
    }
    (out / "benchmark_verified.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = f"""## LiteView offline visible-human evidence chain\n\n- Frames: **{processed}**\n- Whole-frame person signal: **{whole_positive}/{processed}**, boxes **{whole_total}**\n- Pixel tiled candidates: **{tiled_positive}/{processed}**, boxes **{tiled_total}**\n- Segmentation-reviewed candidates: **{seg_confirmed_total}**\n- Final temporally verified visible-human signal: **{verified_positive}/{processed}**, boxes **{verified_total}**\n- Rejected by segmentation: **{rejected_seg}**\n- Rejected as HUD/static overlay: **{rejected_hud}**\n- Rejected for weak temporal evidence: **{rejected_temporal}**\n\nThis is an offline visible-human replay-analysis gate only. It does not infer team/enemy identity or produce live-game assistance.\n"""
    (out / "summary_verified.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
