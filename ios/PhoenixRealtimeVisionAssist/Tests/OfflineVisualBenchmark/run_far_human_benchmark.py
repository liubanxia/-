#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_yolo_pixel_benchmark as base
import run_domain_calibrated_benchmark as v4


@dataclass(frozen=True)
class Region:
    name: str
    x: float
    y: float
    w: float
    h: float
    imgsz: int
    conf_delta: float = 0.0


@dataclass(frozen=True)
class FinalSample:
    crop: np.ndarray
    label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline far-visible-human multiscale benchmark")
    for name in (
        "train-frames",
        "validation-a-frames",
        "validation-b-frames",
        "output",
        "train-source-url",
        "validation-a-source-url",
        "validation-b-source-url",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--memory-model", default="yolo11n.pt")
    p.add_argument("--memory-seg-model", default="yolo11n-seg.pt")
    p.add_argument("--memory-pose-model", default="yolo11n-pose.pt")
    p.add_argument("--model", default="yolo11s.pt")
    p.add_argument("--seg-model", default="yolo11s-seg.pt")
    p.add_argument("--pose-model", default="yolo11s-pose.pt")
    p.add_argument("--confidence", type=float, default=0.08)
    p.add_argument("--seg-confidence", type=float, default=0.08)
    p.add_argument("--pose-confidence", type=float, default=0.08)
    p.add_argument("--min-track-hits", type=int, default=3)
    return p.parse_args()


def regions() -> list[Region]:
    out = [Region("full", 0.0, 0.0, 1.0, 1.0, 768, 0.02)]
    for yi, y in enumerate((0.0, 0.45)):
        for xi, x in enumerate((0.0, 0.45)):
            out.append(Region(f"p2_{yi}_{xi}", x, y, 0.55, 0.55, 896, 0.0))
    for yi, y in enumerate((0.0, 0.27, 0.54)):
        for xi, x in enumerate((0.0, 0.30, 0.60)):
            out.append(Region(f"p3_{yi}_{xi}", x, y, 0.40, 0.46, 960, -0.02))
    out.append(Region("center_band", 0.06, 0.12, 0.88, 0.72, 960, -0.01))
    return out


def predict_people(model: YOLO, image: np.ndarray, imgsz: int, conf: float) -> list[base.Detection]:
    r = model.predict(
        source=image,
        imgsz=imgsz,
        conf=max(0.04, conf),
        iou=0.45,
        classes=[0],
        device="cpu",
        verbose=False,
        max_det=45,
    )
    if not r or r[0].boxes is None or len(r[0].boxes) == 0:
        return []
    boxes = r[0].boxes.xyxy.cpu().numpy()
    scores = r[0].boxes.conf.cpu().numpy()
    return [base.Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), "local") for b, s in zip(boxes, scores)]


def cluster_candidates(items: list[base.Detection]) -> list[base.Detection]:
    remaining = sorted(items, key=lambda d: d.confidence, reverse=True)
    groups: list[list[base.Detection]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        keep = []
        for d in remaining:
            center_dist = math.hypot(seed.cx - d.cx, seed.cy - d.cy)
            size_gate = max(8.0, 0.42 * min(max(seed.width, 1.0), max(seed.height, 1.0)))
            if base.iou(seed, d) >= 0.24 or center_dist <= size_gate:
                group.append(d)
            else:
                keep.append(d)
        groups.append(group)
        remaining = keep

    fused: list[base.Detection] = []
    for group in groups:
        region_names = {d.source for d in group}
        votes = len(region_names)
        weights = np.array([max(0.05, d.confidence) for d in group], dtype=np.float32)
        weights /= weights.sum()
        x1 = float(sum(w * d.x1 for w, d in zip(weights, group)))
        y1 = float(sum(w * d.y1 for w, d in zip(weights, group)))
        x2 = float(sum(w * d.x2 for w, d in zip(weights, group)))
        y2 = float(sum(w * d.y2 for w, d in zip(weights, group)))
        conf = min(0.99, max(d.confidence for d in group) + 0.035 * max(0, votes - 1))
        fused.append(base.Detection(x1, y1, x2, y2, conf, f"far_votes={votes}"))
    return base.nms(fused, 0.48)


def detect_far(model: YOLO, frame: np.ndarray, conf: float) -> list[base.Detection]:
    h, w = frame.shape[:2]
    raw: list[base.Detection] = []
    for region in regions():
        x1 = max(0, min(w - 1, int(round(region.x * w))))
        y1 = max(0, min(h - 1, int(round(region.y * h))))
        x2 = max(x1 + 1, min(w, int(round((region.x + region.w) * w))))
        y2 = max(y1 + 1, min(h, int(round((region.y + region.h) * h))))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        local = predict_people(model, crop, region.imgsz, conf + region.conf_delta)
        for d in local:
            raw.append(base.Detection(d.x1 + x1, d.y1 + y1, d.x2 + x1, d.y2 + y1, d.confidence, region.name))
    return cluster_candidates(raw)


def votes(d: base.Detection) -> int:
    if d.source.startswith("far_votes="):
        try:
            return max(1, int(d.source.split("=", 1)[1]))
        except ValueError:
            pass
    return 1


def segmentation_overlap_far(seg_model: YOLO, frame: np.ndarray, det: base.Detection, conf: float) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = base.expand_box(det, w, h, margin=0.34)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    imgsz = 768 if det.height <= 72 else 576
    r = seg_model.predict(source=crop, imgsz=imgsz, conf=conf, iou=0.45, classes=[0], device="cpu", verbose=False, max_det=8, retina_masks=True)
    if not r or r[0].boxes is None or len(r[0].boxes) == 0 or r[0].masks is None:
        return 0.0
    boxes = r[0].boxes.xyxy.cpu().numpy()
    scores = r[0].boxes.conf.cpu().numpy()
    masks = r[0].masks.data.cpu().numpy()
    best = 0.0
    for box, score, mask in zip(boxes, scores, masks):
        if float(score) < conf:
            continue
        bx1, by1, bx2, by2 = [float(v) for v in box]
        candidate = base.Detection(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, float(score), "seg")
        spatial = base.iou(det, candidate)
        mask_ratio = float((mask > 0.5).mean())
        area_ratio = min(candidate.width * candidate.height, det.width * det.height) / max(candidate.width * candidate.height, det.width * det.height, 1.0)
        best = max(best, 0.62 * spatial + 0.23 * min(1.0, mask_ratio * 4.0) + 0.15 * area_ratio)
    return float(best)


def evidence(frame: np.ndarray, candidates: list[base.Detection], seg_model: YOLO, pose_model: YOLO, a: argparse.Namespace) -> list[base.Detection]:
    out = []
    for d in candidates:
        seg = segmentation_overlap_far(seg_model, frame, d, a.seg_confidence)
        if seg < 0.12:
            continue
        pose = base.pose_evidence(pose_model, frame, d, a.pose_confidence) if d.height >= 22 else 0.0
        out.append(base.Detection(d.x1, d.y1, d.x2, d.y2, d.confidence, d.source, seg, pose, 1, 0.0))
    return out


def far_accept(d: base.Detection, w: int, h: int, min_hits: int, circle) -> bool:
    if base.scope_ring_reject(d, circle) or base.static_overlay_reject(d):
        return False
    if d.aspect_hw < 0.70:
        return False
    v = votes(d)
    temporal = d.track_hits >= min_hits
    if not temporal:
        return False
    if base.weapon_zone_risk(d, w, h):
        return d.seg_overlap >= 0.36 and d.pose_score >= 0.28 and v >= 2

    px = d.height
    if px < 24:
        return v >= 3 and d.seg_overlap >= 0.28 and (d.pose_score >= 0.10 or d.confidence >= 0.24)
    if px < 40:
        return v >= 2 and d.seg_overlap >= 0.25 and (d.pose_score >= 0.12 or d.confidence >= 0.20)
    if px < 64:
        return v >= 2 and d.seg_overlap >= 0.23 and (d.pose_score >= 0.14 or d.confidence >= 0.18)
    return base.final_accept(d, w, h, min_hits, circle)


def bucket(px: float) -> str:
    if px < 24:
        return "lt24"
    if px < 40:
        return "24_39"
    if px < 64:
        return "40_63"
    if px < 96:
        return "64_95"
    return "ge96"


def crop_with_margin(frame: np.ndarray, d: base.Detection) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = base.expand_box(d, w, h, margin=0.55)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def draw(frame: np.ndarray, candidates: list[base.Detection], final: list[tuple[base.Detection, float]], circle, name: str) -> np.ndarray:
    c = frame.copy()
    if circle is not None:
        cx, cy, r = circle
        cv2.circle(c, (int(cx), int(cy)), int(r * 0.76), (220, 220, 0), 1)
    for d in candidates:
        cv2.rectangle(c, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (30, 45, 245), 1)
    for d, sim in final:
        cv2.rectangle(c, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (70, 220, 70), 2)
        text = f"FAR {int(d.height)}px V{votes(d)} S{d.seg_overlap:.2f} P{d.pose_score:.2f} N{sim:.2f}"
        cv2.putText(c, text, (int(d.x1), max(14, int(d.y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 220, 70), 1, cv2.LINE_AA)
    cv2.rectangle(c, (0, 0), (c.shape[1], 29), (0, 0, 0), -1)
    cv2.putText(c, f"{name} pyramid={len(candidates)} accepted={len(final)}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return c


def contact_sheet(images: list[np.ndarray], out: Path) -> None:
    if not images:
        return
    ims = []
    for im in images[:36]:
        scale = 520 / im.shape[1]
        ims.append(cv2.resize(im, (520, max(1, int(im.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    rh = max(i.shape[0] for i in ims)
    rows = math.ceil(len(ims) / 2)
    canvas = np.zeros((rows * rh, 1040, 3), np.uint8)
    for idx, im in enumerate(ims):
        r, c = divmod(idx, 2)
        canvas[r * rh:r * rh + im.shape[0], c * 520:(c + 1) * 520] = im
    cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def gallery(samples: list[FinalSample], out: Path) -> None:
    if not samples:
        return
    samples = samples[:48]
    tw, th, cols = 190, 170, 6
    rows = math.ceil(len(samples) / cols)
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for idx, sample in enumerate(samples):
        im = sample.crop
        scale = min((tw - 10) / im.shape[1], (th - 34) / im.shape[0])
        rs = cv2.resize(im, (max(1, int(im.shape[1] * scale)), max(1, int(im.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
        rr, cc = divmod(idx, cols)
        y0, x0 = rr * th, cc * tw
        ix = x0 + (tw - rs.shape[1]) // 2
        iy = y0 + 25 + max(0, (th - 30 - rs.shape[0]) // 2)
        canvas[iy:iy + rs.shape[0], ix:ix + rs.shape[1]] = rs
        cv2.putText(canvas, sample.label[:27], (x0 + 4, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def validate(name: str, folder: Path, detector: YOLO, segmenter: YOLO, poser: YOLO, mem, a: argparse.Namespace):
    tracks = []
    tid = 1
    rows = []
    visuals = []
    samples: list[FinalSample] = []
    stats = {
        "frames": 0,
        "pyramid_candidates": 0,
        "seg_evidence": 0,
        "accepted": 0,
        "accepted_frames": 0,
        "negative_memory_reject": 0,
        "accepted_by_height": {"lt24": 0, "24_39": 0, "40_63": 0, "64_95": 0, "ge96": 0},
    }
    for p in sorted(folder.glob("frame_*.jpg")):
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        stats["frames"] += 1
        h, w = frame.shape[:2]
        circle = base.detect_scope_circle(frame)
        candidates = detect_far(detector, frame, a.confidence)
        ev = evidence(frame, candidates, segmenter, poser, a)
        tracks, tracked, tid = base.update_tracks(tracks, ev, w, h, tid)
        final: list[tuple[base.Detection, float]] = []
        rejected = 0
        for d in tracked:
            if not far_accept(d, w, h, a.min_track_hits, circle):
                continue
            fv = v4.feat(frame, d, circle)
            if fv is None:
                continue
            sim = v4.negative_similarity(mem, fv[0])
            if sim >= mem["threshold"] and d.pose_score < 0.34:
                rejected += 1
                continue
            final.append((d, sim))
            stats["accepted_by_height"][bucket(d.height)] += 1
            crop = crop_with_margin(frame, d)
            if crop is not None:
                samples.append(FinalSample(crop, f"{name} {int(d.height)}px V{votes(d)} S{d.seg_overlap:.2f} P{d.pose_score:.2f}"))
        stats["pyramid_candidates"] += len(candidates)
        stats["seg_evidence"] += len(ev)
        stats["negative_memory_reject"] += rejected
        stats["accepted"] += len(final)
        stats["accepted_frames"] += int(bool(final))
        rows.append({
            "source": name,
            "frame": p.name,
            "width": w,
            "height": h,
            "pyramid_candidates": len(candidates),
            "seg_evidence": len(ev),
            "memory_reject": rejected,
            "accepted": len(final),
            "lt24": sum(1 for d, _ in final if bucket(d.height) == "lt24"),
            "24_39": sum(1 for d, _ in final if bucket(d.height) == "24_39"),
            "40_63": sum(1 for d, _ in final if bucket(d.height) == "40_63"),
        })
        if final or candidates:
            visuals.append(draw(frame, candidates, final, circle, f"{name}/{p.name}"))
    return stats, rows, visuals, samples


def main() -> int:
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    memory_detector = YOLO(a.memory_model)
    memory_segmenter = YOLO(a.memory_seg_model)
    memory_poser = YOLO(a.memory_pose_model)
    negatives, mining = v4.mine_negatives(Path(a.train_frames), memory_detector, memory_segmenter, memory_poser, a)
    v4.bank_sheet(negatives, out / "hard_negative_memory_bank_v5.jpg")
    mem, memory_stats = v4.build_memory(negatives)
    v4.save_memory(mem, out / "fps_hard_negative_memory_v5.npz")

    detector = YOLO(a.model)
    segmenter = YOLO(a.seg_model)
    poser = YOLO(a.pose_model)

    a_stats, a_rows, a_visuals, a_samples = validate("A", Path(a.validation_a_frames), detector, segmenter, poser, mem, a)
    b_stats, b_rows, b_visuals, b_samples = validate("B", Path(a.validation_b_frames), detector, segmenter, poser, mem, a)

    rows = a_rows + b_rows
    if not rows:
        raise SystemExit("no validation rows")
    with (out / "far_validation_v5.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    contact_sheet(a_visuals, out / "validation_A_contact_sheet_v5.jpg")
    contact_sheet(b_visuals, out / "validation_B_contact_sheet_v5.jpg")
    gallery(a_samples + b_samples, out / "far_candidate_gallery_v5.jpg")

    combined = {k: 0 for k in ("frames", "pyramid_candidates", "seg_evidence", "accepted", "accepted_frames", "negative_memory_reject")}
    combined["accepted_by_height"] = {"lt24": 0, "24_39": 0, "40_63": 0, "64_95": 0, "ge96": 0}
    for st in (a_stats, b_stats):
        for k in ("frames", "pyramid_candidates", "seg_evidence", "accepted", "accepted_frames", "negative_memory_reject"):
            combined[k] += st[k]
        for k, v in st["accepted_by_height"].items():
            combined["accepted_by_height"][k] += v

    report = {
        "benchmark": "LiteView offline far-visible-human pyramid validation v5",
        "training_source_url": a.train_source_url,
        "validation_sources": {
            "A": a.validation_a_source_url,
            "B": a.validation_b_source_url,
        },
        "models": {
            "memory_detector": a.memory_model,
            "far_detector": a.model,
            "far_segmenter": a.seg_model,
            "far_pose": a.pose_model,
        },
        "negative_memory": {"mining": mining, "memory": memory_stats},
        "validation_A": a_stats,
        "validation_B": b_stats,
        "combined": combined,
        "important_limit": "Accepted-box counts are not ground-truth recall or precision. Manually inspect both contact sheets and the far-candidate gallery before changing the mobile runtime.",
        "scope_limit": "Offline replay analysis only; no team/enemy identity, aiming, hidden-position inference, live coordinates, or anti-cheat behavior.",
    }
    (out / "far_benchmark_v5.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    b = combined["accepted_by_height"]
    summary = f"""## LiteView offline far-visible-human pyramid v5

- Negative-memory samples: **{memory_stats['samples']}**
- Validation frames: **{combined['frames']}** across two independent public clips
- Pyramid candidates: **{combined['pyramid_candidates']}**
- Segmentation-supported candidates: **{combined['seg_evidence']}**
- Accepted visible-human candidates: **{combined['accepted']}** on **{combined['accepted_frames']}/{combined['frames']}** frames
- Negative-memory rejects: **{combined['negative_memory_reject']}**
- Accepted height <24 px: **{b['lt24']}**
- Accepted height 24-39 px: **{b['24_39']}**
- Accepted height 40-63 px: **{b['40_63']}**
- Accepted height 64-95 px: **{b['64_95']}**
- Accepted height >=96 px: **{b['ge96']}**

This is a candidate-generation benchmark, not an accuracy claim. The two contact sheets and far-candidate gallery are the acceptance gate.
"""
    (out / "summary_far_v5.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
