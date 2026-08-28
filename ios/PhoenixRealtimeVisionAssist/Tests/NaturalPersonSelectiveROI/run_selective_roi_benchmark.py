#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PersonGT:
    box: tuple[float, float, float, float]
    reference_height_px: float
    bucket: str


@dataclass(frozen=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Natural-scene selective ROI second-look benchmark")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--proposal-model", default="yolo11n.pt")
    p.add_argument("--second-model", default="yolo11s.pt")
    p.add_argument("--proposal-conf", type=float, default=0.01)
    p.add_argument("--final-conf", type=float, default=0.05)
    p.add_argument("--second-conf", type=float, default=0.08)
    p.add_argument("--seed-conf-max", type=float, default=0.14)
    p.add_argument("--seed-height-max", type=float, default=64.0)
    p.add_argument("--reference-imgsz", type=int, default=640)
    p.add_argument("--max-rois", type=int, default=2)
    p.add_argument("--roi-fraction", type=float, default=0.34)
    p.add_argument("--roi-expand", type=float, default=5.0)
    p.add_argument("--iou-threshold", type=float, default=0.30)
    return p.parse_args()


def find_root(root: Path, kind: str) -> Path:
    for p in (root / kind / "train2017", root / "coco128" / kind / "train2017", root / kind):
        if p.exists():
            return p
    raise FileNotFoundError(f"cannot locate {kind} under {root}")


def projected_height(shape, box, imgsz: int) -> float:
    h, w = shape[:2]
    scale = min(imgsz / max(w, 1), imgsz / max(h, 1))
    return max(0.0, box[3] - box[1]) * scale


def bucket(px: float) -> str:
    if px < 16:
        return "lt16"
    if px < 24:
        return "16_23"
    if px < 32:
        return "24_31"
    if px < 48:
        return "32_47"
    if px < 64:
        return "48_63"
    if px < 96:
        return "64_95"
    return "ge96"


def load_dataset(root: Path, reference_imgsz: int):
    image_root = find_root(root, "images")
    label_root = find_root(root, "labels")
    by_image: dict[Path, list[PersonGT]] = {}

    for label_path in sorted(label_root.rglob("*.txt")):
        rel = label_path.relative_to(label_root)
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = image_root / rel.with_suffix(ext)
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        people: list[PersonGT] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5 or int(float(parts[0])) != 0:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            box = (
                (cx - bw / 2.0) * w,
                (cy - bh / 2.0) * h,
                (cx + bw / 2.0) * w,
                (cy + bh / 2.0) * h,
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            px = projected_height(image.shape, box, reference_imgsz)
            people.append(PersonGT(box, px, bucket(px)))
        if people:
            by_image[image_path] = people
    return by_image


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + ba - inter, 1e-9)


def predict(model: YOLO, image: np.ndarray, imgsz: int, conf: float, max_det: int = 120):
    t0 = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=0.50,
        classes=[0],
        device="cpu",
        verbose=False,
        max_det=max_det,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    preds = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        preds = [(tuple(float(v) for v in box), float(score)) for box, score in zip(boxes, scores)]
    return preds, elapsed_ms


def nms(preds, threshold: float = 0.50):
    ordered = sorted(preds, key=lambda p: p[1], reverse=True)
    keep = []
    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        ordered = [p for p in ordered if iou(best[0], p[0]) < threshold]
    return keep


def roi_iou(a: ROI, b: ROI) -> float:
    return iou((a.x1, a.y1, a.x2, a.y2), (b.x1, b.y1, b.x2, b.y2))


def make_roi(image_shape, box, score: float, min_fraction: float, expand: float) -> ROI:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    rw = max(w * min_fraction, bw * expand)
    rh = max(h * min_fraction, bh * expand)
    rw = min(rw, w * 0.62)
    rh = min(rh, h * 0.62)

    rx1 = int(round(cx - rw * 0.5))
    ry1 = int(round(cy - rh * 0.5))
    rx2 = int(round(cx + rw * 0.5))
    ry2 = int(round(cy + rh * 0.5))

    if rx1 < 0:
        rx2 -= rx1
        rx1 = 0
    if ry1 < 0:
        ry2 -= ry1
        ry1 = 0
    if rx2 > w:
        rx1 -= rx2 - w
        rx2 = w
    if ry2 > h:
        ry1 -= ry2 - h
        ry2 = h

    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(w, max(rx1 + 1, rx2))
    ry2 = min(h, max(ry1 + 1, ry2))
    return ROI(rx1, ry1, rx2, ry2, score)


def merge_rois(rois: list[ROI], max_rois: int) -> list[ROI]:
    ordered = sorted(rois, key=lambda r: r.score, reverse=True)
    merged: list[ROI] = []
    for roi in ordered:
        hit = None
        for idx, current in enumerate(merged):
            cx1 = (current.x1 + current.x2) * 0.5
            cy1 = (current.y1 + current.y2) * 0.5
            cx2 = (roi.x1 + roi.x2) * 0.5
            cy2 = (roi.y1 + roi.y2) * 0.5
            norm = max(1.0, min(current.width, current.height, roi.width, roi.height))
            center_near = math.hypot(cx1 - cx2, cy1 - cy2) <= 0.32 * norm
            if roi_iou(current, roi) >= 0.18 or center_near:
                hit = idx
                break
        if hit is None:
            merged.append(roi)
        else:
            current = merged[hit]
            merged[hit] = ROI(
                min(current.x1, roi.x1),
                min(current.y1, roi.y1),
                max(current.x2, roi.x2),
                max(current.y2, roi.y2),
                max(current.score, roi.score),
            )
    return sorted(merged, key=lambda r: r.score, reverse=True)[:max_rois]


def propose_rois(
    image: np.ndarray,
    raw_preds,
    final_preds,
    reference_imgsz: int,
    seed_conf_max: float,
    seed_height_max: float,
    min_fraction: float,
    expand: float,
    max_rois: int,
):
    seeds = []
    for box, score in raw_preds:
        if score >= seed_conf_max:
            continue
        ref_h = projected_height(image.shape, box, reference_imgsz)
        if ref_h > seed_height_max:
            continue
        if any(iou(box, accepted[0]) >= 0.18 for accepted in final_preds):
            continue
        x1, y1, x2, y2 = box
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        if bh / bw < 0.65:
            continue
        priority = score + 0.035 * max(0.0, 1.0 - ref_h / max(seed_height_max, 1.0))
        seeds.append((box, priority, score, ref_h))

    seeds.sort(key=lambda x: x[1], reverse=True)
    candidates = [make_roi(image.shape, box, priority, min_fraction, expand) for box, priority, _, _ in seeds[:8]]
    return merge_rois(candidates, max_rois), seeds


def selective_predict(proposal_model: YOLO, second_model: YOLO, image: np.ndarray, a):
    raw, base_ms = predict(proposal_model, image, 640, a.proposal_conf, max_det=160)
    base_final = [(box, score) for box, score in raw if score >= a.final_conf]
    rois, seeds = propose_rois(
        image=image,
        raw_preds=raw,
        final_preds=base_final,
        reference_imgsz=a.reference_imgsz,
        seed_conf_max=a.seed_conf_max,
        seed_height_max=a.seed_height_max,
        min_fraction=a.roi_fraction,
        expand=a.roi_expand,
        max_rois=a.max_rois,
    )

    merged = list(base_final)
    total_ms = base_ms
    for roi in rois:
        crop = image[roi.y1:roi.y2, roi.x1:roi.x2]
        local, elapsed = predict(second_model, crop, 640, a.second_conf, max_det=40)
        total_ms += elapsed
        for box, score in local:
            x1, y1, x2, y2 = box
            mapped = (x1 + roi.x1, y1 + roi.y1, x2 + roi.x1, y2 + roi.y1)
            merged.append((mapped, score))
    return nms(merged, 0.50), total_ms, rois, len(seeds), base_final, raw


def greedy_match(preds, gts: list[PersonGT], hit_iou: float):
    pred_order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    unmatched_gt = set(range(len(gts)))
    matched_pred = set()
    matched_gt = set()
    for pi in pred_order:
        best_gi = None
        best_iou = 0.0
        for gi in unmatched_gt:
            ov = iou(preds[pi][0], gts[gi].box)
            if ov > best_iou:
                best_iou = ov
                best_gi = gi
        if best_gi is not None and best_iou >= hit_iou:
            matched_pred.add(pi)
            matched_gt.add(best_gi)
            unmatched_gt.remove(best_gi)
    return matched_pred, matched_gt


def record(mode: str, image_name: str, preds, elapsed: float, gts: list[PersonGT], hit_iou: float, roi_count=0, seed_count=0):
    matched_pred, matched_gt = greedy_match(preds, gts, hit_iou)
    row = {
        "mode": mode,
        "image": image_name,
        "cpu_ms": round(elapsed, 3),
        "roi_count": roi_count,
        "seed_count": seed_count,
        "predictions": len(preds),
        "matched_predictions": len(matched_pred),
        "false_positive_predictions": len(preds) - len(matched_pred),
        "gt_count": len(gts),
        "matched_gt_count": len(matched_gt),
    }
    for b in ("lt16", "16_23", "24_31", "32_47", "48_63", "64_95", "ge96"):
        indices = [idx for idx, gt in enumerate(gts) if gt.bucket == b]
        row[f"gt_{b}"] = len(indices)
        row[f"hit_{b}"] = sum(1 for idx in indices if idx in matched_gt)
    return row


def summarize(rows, mode: str):
    subset = [r for r in rows if r["mode"] == mode]
    pred_total = sum(r["predictions"] for r in subset)
    tp_total = sum(r["matched_predictions"] for r in subset)
    gt_total = sum(r["gt_count"] for r in subset)
    matched_total = sum(r["matched_gt_count"] for r in subset)
    buckets = {}
    for b in ("lt16", "16_23", "24_31", "32_47", "48_63", "64_95", "ge96"):
        gt_n = sum(r[f"gt_{b}"] for r in subset)
        hit_n = sum(r[f"hit_{b}"] for r in subset)
        buckets[b] = {"gt": gt_n, "hits": hit_n, "recall": hit_n / gt_n if gt_n else None}
    return {
        "images": len(subset),
        "predictions": pred_total,
        "true_positive_predictions": tp_total,
        "false_positive_predictions": pred_total - tp_total,
        "precision_iou30": tp_total / pred_total if pred_total else None,
        "person_gt": gt_total,
        "matched_person_gt": matched_total,
        "overall_recall_iou30": matched_total / gt_total if gt_total else None,
        "median_cpu_ms_per_image": float(np.median([r["cpu_ms"] for r in subset])) if subset else None,
        "average_rois_per_image": float(np.mean([r["roi_count"] for r in subset])) if subset else 0.0,
        "p95_rois_per_image": float(np.percentile([r["roi_count"] for r in subset], 95)) if subset else 0.0,
        "images_with_second_pass": sum(1 for r in subset if r["roi_count"] > 0),
        "average_seed_count": float(np.mean([r["seed_count"] for r in subset])) if subset else 0.0,
        "by_reference_height": buckets,
    }


def make_gallery(items, out: Path):
    cards = []
    for image, gts, base_final, selective, rois in items[:12]:
        canvas = image.copy()
        for gt in gts:
            if gt.reference_height_px < 32:
                x1, y1, x2, y2 = [int(round(v)) for v in gt.box]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
        for box, _ in base_final:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 60, 235), 1)
        for box, _ in selective:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (235, 180, 40), 1)
        for roi in rois:
            cv2.rectangle(canvas, (roi.x1, roi.y1), (roi.x2, roi.y2), (220, 90, 220), 1)
        cv2.putText(
            canvas,
            "green=GT<32 red=base blue=selective magenta=ROI",
            (7, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        scale = 520 / max(canvas.shape[1], 1)
        cards.append(cv2.resize(canvas, (520, max(1, int(canvas.shape[0] * scale))), interpolation=cv2.INTER_AREA))
    if not cards:
        return
    rh = max(card.shape[0] for card in cards)
    cols = 2
    rows = math.ceil(len(cards) / cols)
    sheet = np.zeros((rows * rh, cols * 520, 3), np.uint8)
    for idx, card in enumerate(cards):
        rr, cc = divmod(idx, cols)
        sheet[rr * rh:rr * rh + card.shape[0], cc * 520:(cc + 1) * 520] = card
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main() -> int:
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(Path(a.dataset), a.reference_imgsz)
    if not dataset:
        raise SystemExit("no person annotations found")

    proposal_model = YOLO(a.proposal_model)
    second_model = YOLO(a.second_model)
    rows = []
    gallery = []

    for image_path, gts in dataset.items():
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        raw640, ms640 = predict(proposal_model, image, 640, a.proposal_conf, max_det=160)
        full640 = [(box, score) for box, score in raw640 if score >= a.final_conf]
        rows.append(record("n_full640", image_path.name, full640, ms640, gts, a.iou_threshold))

        full960, ms960 = predict(proposal_model, image, 960, a.final_conf, max_det=120)
        rows.append(record("n_full960", image_path.name, full960, ms960, gts, a.iou_threshold))

        selective, ms_sel, rois, seed_count, base_final, _ = selective_predict(proposal_model, second_model, image, a)
        rows.append(record("n640_selective_s640", image_path.name, selective, ms_sel, gts, a.iou_threshold, len(rois), seed_count))

        if any(gt.reference_height_px < 24 for gt in gts) and len(gallery) < 12:
            gallery.append((image.copy(), gts, base_final, selective, rois))

    with (out / "selective_roi_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    modes = ("n_full640", "n_full960", "n640_selective_s640")
    summaries = {mode: summarize(rows, mode) for mode in modes}
    report = {
        "benchmark": "Natural-scene selective ROI second-look v1",
        "dataset": "COCO128 original full scenes and person annotations",
        "proposal_model": a.proposal_model,
        "second_model": a.second_model,
        "proposal_conf": a.proposal_conf,
        "final_conf": a.final_conf,
        "second_conf": a.second_conf,
        "seed_conf_max": a.seed_conf_max,
        "seed_height_max": a.seed_height_max,
        "reference_imgsz": a.reference_imgsz,
        "max_rois": a.max_rois,
        "roi_fraction": a.roi_fraction,
        "roi_expand": a.roi_expand,
        "iou_hit_threshold": a.iou_threshold,
        "modes": summaries,
        "method_guard": "Second-pass ROIs are selected only from low-confidence detector proposals and geometry; ground truth is used only for evaluation.",
        "scope_limit": "Generic offline visible-person benchmark only; no gameplay team/enemy identity, hidden-position inference, live coordinates, or anti-cheat behavior.",
    }
    (out / "selective_roi_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_gallery(gallery, out / "selective_roi_gallery.jpg")

    def pct(v):
        return "n/a" if v is None else f"{100.0 * v:.1f}%"

    lines = [
        "## Natural-scene selective ROI second-look v1",
        "",
        f"- Images with person annotations: **{len(dataset)}**",
        f"- Proposal: **{Path(a.proposal_model).stem}@640, conf {a.proposal_conf:.2f}**",
        f"- Final baseline threshold: **{a.final_conf:.2f}**",
        f"- Second look: **{Path(a.second_model).stem}@640, conf {a.second_conf:.2f}**",
        f"- Maximum second-pass ROIs: **{a.max_rois}**",
        "",
        "| mode | <16 | 16-23 | 24-31 | overall recall | precision | median CPU ms | avg ROI/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        s = summaries[mode]
        b = s["by_reference_height"]
        lines.append(
            f"| {mode} | {pct(b['lt16']['recall'])} | {pct(b['16_23']['recall'])} | {pct(b['24_31']['recall'])} | "
            f"{pct(s['overall_recall_iou30'])} | {pct(s['precision_iou30'])} | {s['median_cpu_ms_per_image']:.1f} | {s['average_rois_per_image']:.2f} |"
        )
    lines += [
        "",
        "Selective ROI is worth keeping only if it recovers a meaningful share of tiny-person recall without the ~5x compute and precision collapse of unconditional four-tile scanning.",
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary_selective_roi_benchmark.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
