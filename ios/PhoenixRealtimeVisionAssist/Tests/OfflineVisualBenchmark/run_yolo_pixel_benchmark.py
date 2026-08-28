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
    pose_score: float = 0.0
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

    @property
    def aspect_hw(self) -> float:
        return self.height / max(self.width, 1e-6)


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
    p = argparse.ArgumentParser(description="Offline source-pixel visible-human evidence benchmark")
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--seg-model", default="yolo11n-seg.pt")
    p.add_argument("--pose-model", default="yolo11n-pose.pt")
    p.add_argument("--confidence", type=float, default=0.12)
    p.add_argument("--seg-confidence", type=float, default=0.10)
    p.add_argument("--pose-confidence", type=float, default=0.10)
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
    return nms(candidates, 0.45)


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
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


def expand_box(d: Detection, w: int, h: int, margin: float = 0.22) -> tuple[int, int, int, int]:
    dx, dy = d.width * margin, d.height * margin
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
        candidate = Detection(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, float(score), "seg")
        spatial = iou(det, candidate)
        mask_ratio = float((mask > 0.5).mean())
        area_ratio = min(candidate.width * candidate.height, target_area) / max(candidate.width * candidate.height, target_area)
        best = max(best, 0.60 * spatial + 0.25 * min(1.0, mask_ratio * 4.0) + 0.15 * area_ratio)
    return float(best)


def pose_evidence(pose_model: YOLO, frame: np.ndarray, det: Detection, conf: float) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(det, w, h, margin=0.35)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    results = pose_model.predict(source=crop, imgsz=512, conf=conf, iou=0.45, device="cpu", verbose=False, max_det=6)
    if not results:
        return 0.0
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
        return 0.0
    boxes = r.boxes.xyxy.cpu().numpy()
    box_scores = r.boxes.conf.cpu().numpy()
    kp_conf = r.keypoints.conf
    kp_conf_np = kp_conf.cpu().numpy() if kp_conf is not None else None
    best = 0.0
    for idx, (box, box_score) in enumerate(zip(boxes, box_scores)):
        bx1, by1, bx2, by2 = [float(v) for v in box]
        candidate = Detection(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, float(box_score), "pose")
        spatial = iou(det, candidate)
        if spatial < 0.10:
            continue
        if kp_conf_np is None or idx >= len(kp_conf_np):
            valid = 0
            group_count = 0
            kp_mean = 0.0
        else:
            arr = kp_conf_np[idx]
            valid_mask = arr >= 0.18
            valid = int(valid_mask.sum())
            groups = ((0, 5), (5, 11), (11, 17))
            group_count = sum(int(valid_mask[a:b].sum()) > 0 for a, b in groups)
            kp_mean = float(arr[valid_mask].mean()) if valid else 0.0
        keypoint_score = min(1.0, valid / 7.0) * 0.55 + min(1.0, group_count / 2.0) * 0.25 + min(1.0, kp_mean / 0.45) * 0.20
        best = max(best, 0.50 * spatial + 0.50 * keypoint_score)
    return float(best)


def detect_scope_circle(frame: np.ndarray) -> tuple[float, float, float] | None:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.8)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.25, minDist=h * 0.35, param1=90, param2=34, minRadius=int(h * 0.22), maxRadius=int(h * 0.52))
    if circles is None:
        return None
    cx0, cy0 = w * 0.5, h * 0.52
    choices = []
    for cx, cy, r in circles[0]:
        center_distance = math.hypot((cx - cx0) / max(w, 1), (cy - cy0) / max(h, 1))
        if center_distance <= 0.24:
            choices.append((center_distance, float(cx), float(cy), float(r)))
    if not choices:
        return None
    _, cx, cy, r = min(choices, key=lambda x: x[0])
    return cx, cy, r


def scope_ring_reject(det: Detection, circle: tuple[float, float, float] | None) -> bool:
    if circle is None:
        return False
    cx, cy, r = circle
    d = math.hypot(det.cx - cx, det.cy - cy)
    # Real targets seen through the optic should lie inside the clear aperture. Boxes centered
    # on the metal/black ring are a dominant hard negative in the public replay sample.
    return d > r * 0.76 and d < r * 1.30


def weapon_zone_risk(det: Detection, frame_w: int, frame_h: int) -> bool:
    cx = det.cx / max(frame_w, 1)
    cy = det.cy / max(frame_h, 1)
    h_ratio = det.height / max(frame_h, 1)
    w_ratio = det.width / max(frame_w, 1)
    return cy > 0.61 and 0.28 < cx < 0.84 and (h_ratio > 0.14 or w_ratio > 0.09)


def update_tracks(tracks: list[TrackState], detections: list[Detection], frame_w: int, frame_h: int, next_track_id: int) -> tuple[list[TrackState], list[Detection], int]:
    unmatched = set(range(len(detections)))
    tracked_out: list[Detection] = []
    for t in tracks:
        best_idx, best_score = None, -1.0
        for idx in list(unmatched):
            d = detections[idx]
            overlap = iou(t.box, d)
            dx = (t.box.cx - d.cx) / max(frame_w, 1)
            dy = (t.box.cy - d.cy) / max(frame_h, 1)
            center_dist = math.hypot(dx, dy)
            score = overlap * 0.72 + max(0.0, 1.0 - center_dist / 0.11) * 0.28
            if score > best_score and (overlap >= 0.12 or center_dist <= 0.055):
                best_idx, best_score = idx, score
        t.age += 1
        if best_idx is None:
            t.misses += 1
            continue
        d = detections[best_idx]
        unmatched.remove(best_idx)
        t.box, t.hits, t.misses = d, t.hits + 1, 0
        t.centers.append((d.cx / max(frame_w, 1), d.cy / max(frame_h, 1)))
        if len(t.centers) > 14:
            t.centers.pop(0)
        static_score = 0.0
        if len(t.centers) >= 4:
            xs, ys = [p[0] for p in t.centers], [p[1] for p in t.centers]
            static_score = max(0.0, 1.0 - (statistics.pstdev(xs) + statistics.pstdev(ys)) / 0.011)
        tracked_out.append(Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.source, d.seg_overlap, d.pose_score, t.hits, static_score))
    tracks = [t for t in tracks if t.misses <= 2]
    for idx in unmatched:
        d = detections[idx]
        tracks.append(TrackState(next_track_id, d, 1, 0, 1, [(d.cx / max(frame_w, 1), d.cy / max(frame_h, 1))]))
        tracked_out.append(d)
        next_track_id += 1
    return tracks, tracked_out, next_track_id


def static_overlay_reject(d: Detection) -> bool:
    return d.track_hits >= 4 and d.static_score >= 0.92


def final_accept(d: Detection, frame_w: int, frame_h: int, min_track_hits: int, circle: tuple[float, float, float] | None) -> bool:
    if scope_ring_reject(d, circle) or static_overlay_reject(d):
        return False
    h_ratio = d.height / max(frame_h, 1)
    tiny = h_ratio < 0.10
    temporal = d.track_hits >= min_track_hits
    seg_ok = d.seg_overlap >= (0.38 if tiny else 0.34)
    pose_ok = d.pose_score >= (0.34 if tiny else 0.30)
    det_strong = d.confidence >= 0.28
    if weapon_zone_risk(d, frame_w, frame_h):
        return seg_ok and pose_ok and temporal
    if d.aspect_hw < 0.82 and not pose_ok:
        return False
    if tiny:
        return seg_ok and pose_ok and temporal
    return seg_ok and (pose_ok or (temporal and det_strong))


def draw(frame: np.ndarray, tiled: list[Detection], final: list[Detection], circle: tuple[float, float, float] | None, name: str) -> np.ndarray:
    canvas = frame.copy()
    if circle is not None:
        cx, cy, r = circle
        cv2.circle(canvas, (int(cx), int(cy)), int(r * 0.76), (220, 220, 0), 1)
    for d in tiled:
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (30, 45, 255), 1)
    for d in final:
        cv2.rectangle(canvas, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (70, 220, 70), 2)
        cv2.putText(canvas, f"V D{d.confidence:.2f} S{d.seg_overlap:.2f} P{d.pose_score:.2f} T{d.track_hits}", (int(d.x1), max(16, int(d.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 220, 70), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, f"{name} tiled={len(tiled)} verified={len(final)} scope={int(circle is not None)}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_contact_sheet(images: list[np.ndarray], output: Path) -> None:
    if not images:
        return
    thumb_w = 480
    thumbs = []
    for image in images[:24]:
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


def main() -> int:
    args = parse_args()
    frames_dir, out = Path(args.frames), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise SystemExit("no benchmark frames")
    detector, segmenter, poser = YOLO(args.model), YOLO(args.seg_model), YOLO(args.pose_model)
    tracks: list[TrackState] = []
    next_track_id = 1
    rows, visuals = [], []
    stats = {"whole_boxes": 0, "tiled_boxes": 0, "seg_pass": 0, "pose_reviewed": 0, "verified": 0, "verified_frames": 0, "seg_reject": 0, "scope_ring_reject": 0, "weapon_risk_reject": 0, "static_reject": 0, "shape_reject": 0}

    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        whole = detect_whole(detector, frame, args.confidence)
        tiled = detect_tiled(detector, frame, args.confidence)
        circle = detect_scope_circle(frame)
        evidence: list[Detection] = []
        for d in tiled:
            seg = segmentation_overlap(segmenter, frame, d, args.seg_confidence)
            if seg < 0.18:
                stats["seg_reject"] += 1
                continue
            pose = pose_evidence(poser, frame, d, args.pose_confidence)
            stats["pose_reviewed"] += 1
            evidence.append(Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.source, seg, pose, 1, 0.0))
        tracks, tracked, next_track_id = update_tracks(tracks, evidence, w, h, next_track_id)
        final = []
        for d in tracked:
            if scope_ring_reject(d, circle):
                stats["scope_ring_reject"] += 1
                continue
            if static_overlay_reject(d):
                stats["static_reject"] += 1
                continue
            accepted = final_accept(d, w, h, args.min_track_hits, circle)
            if not accepted:
                if weapon_zone_risk(d, w, h):
                    stats["weapon_risk_reject"] += 1
                elif d.aspect_hw < 0.82 and d.pose_score < 0.30:
                    stats["shape_reject"] += 1
                continue
            final.append(d)
        stats["whole_boxes"] += len(whole)
        stats["tiled_boxes"] += len(tiled)
        stats["seg_pass"] += len(evidence)
        stats["verified"] += len(final)
        stats["verified_frames"] += int(bool(final))
        rows.append({"frame": path.name, "width": w, "height": h, "scope_mode": int(circle is not None), "whole_count": len(whole), "tiled_count": len(tiled), "seg_pass_count": len(evidence), "verified_count": len(final), "max_pose_score": max((d.pose_score for d in final), default=0.0)})
        visuals.append(draw(frame, tiled, final, circle, path.name))

    processed = len(rows)
    with (out / "detections_v2.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    make_contact_sheet(visuals, out / "contact_sheet_verified_v2.jpg")
    report = {"benchmark": "LiteView offline visible-human evidence chain v2", "source_url": args.source_url, "processed_frames": processed, "models": {"detector": args.model, "segmenter": args.seg_model, "pose": args.pose_model}, "stats": stats, "interpretation_limit": "Offline replay analysis only; no team/enemy identity, aiming, hidden-position inference, live coordinates, or anti-cheat behavior."}
    (out / "benchmark_verified_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = f"""## LiteView offline visible-human evidence chain v2\n\n- Frames: **{processed}**\n- Pixel tiled candidate boxes: **{stats['tiled_boxes']}**\n- Passed loose segmentation gate: **{stats['seg_pass']}**\n- Pose-reviewed candidates: **{stats['pose_reviewed']}**\n- Final verified boxes: **{stats['verified']}** on **{stats['verified_frames']}/{processed}** frames\n- Segmentation rejects: **{stats['seg_reject']}**\n- Scope-ring rejects: **{stats['scope_ring_reject']}**\n- Weapon-risk rejects: **{stats['weapon_risk_reject']}**\n- Static-overlay rejects: **{stats['static_reject']}**\n- Shape rejects: **{stats['shape_reject']}**\n\nOffline replay-analysis validation only. The contact sheet must still be visually audited before claiming accuracy.\n"""
    (out / "summary_verified_v2.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
