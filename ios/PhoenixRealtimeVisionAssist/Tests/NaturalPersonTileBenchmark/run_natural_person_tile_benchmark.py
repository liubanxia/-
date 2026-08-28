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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Natural-scene visible-person overlap-tile benchmark")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    p.add_argument("--confidence", type=float, default=0.05)
    p.add_argument("--iou-threshold", type=float, default=0.30)
    p.add_argument("--reference-imgsz", type=int, default=640)
    p.add_argument("--tile-fraction", type=float, default=0.58)
    return p.parse_args()


def find_root(root: Path, kind: str) -> Path:
    for p in (root / kind / "train2017", root / "coco128" / kind / "train2017", root / kind):
        if p.exists():
            return p
    raise FileNotFoundError(f"cannot locate {kind} under {root}")


def projected_height(shape, box, imgsz: int) -> float:
    h, w = shape[:2]
    scale = min(imgsz / max(w, 1), imgsz / max(h, 1))
    return (box[3] - box[1]) * scale


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
        people = []
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


def predict(model: YOLO, image: np.ndarray, imgsz: int, conf: float):
    t0 = time.perf_counter()
    result = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        iou=0.50,
        classes=[0],
        device="cpu",
        verbose=False,
        max_det=100,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    preds = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        preds = [(tuple(float(v) for v in box), float(score)) for box, score in zip(boxes, scores)]
    return preds, elapsed_ms


def tile_rects(w: int, h: int, fraction: float):
    tw = max(w // 2, min(w, int(round(w * fraction))))
    th = max(h // 2, min(h, int(round(h * fraction))))
    x1 = w - tw
    y1 = h - th
    return [(0, 0, tw, th), (x1, 0, w, th), (0, y1, tw, h), (x1, y1, w, h)]


def nms(preds, threshold: float = 0.50):
    ordered = sorted(preds, key=lambda p: p[1], reverse=True)
    keep = []
    while ordered:
        best = ordered.pop(0)
        keep.append(best)
        ordered = [p for p in ordered if iou(best[0], p[0]) < threshold]
    return keep


def tiled_predictions(model: YOLO, image: np.ndarray, conf: float, tile_fraction: float):
    full, full_ms = predict(model, image, 640, conf)
    merged = list(full)
    total_ms = full_ms
    h, w = image.shape[:2]
    for x1, y1, x2, y2 in tile_rects(w, h, tile_fraction):
        crop = image[y1:y2, x1:x2]
        local, elapsed = predict(model, crop, 640, conf)
        total_ms += elapsed
        for box, score in local:
            bx1, by1, bx2, by2 = box
            merged.append(((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), score))
    return nms(merged), total_ms


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


def summarize_mode(records, mode: str):
    subset = [r for r in records if r["mode"] == mode]
    pred_total = sum(r["predictions"] for r in subset)
    tp_total = sum(r["matched_predictions"] for r in subset)
    fp_total = pred_total - tp_total
    gt_total = sum(r["gt_count"] for r in subset)
    matched_gt_total = sum(r["matched_gt_count"] for r in subset)
    order = ["lt16", "16_23", "24_31", "32_47", "48_63", "64_95", "ge96"]
    by_bucket = {}
    for b in order:
        gt_n = sum(r[f"gt_{b}"] for r in subset)
        hit_n = sum(r[f"hit_{b}"] for r in subset)
        by_bucket[b] = {
            "gt": gt_n,
            "hits": hit_n,
            "recall": (hit_n / gt_n) if gt_n else None,
        }
    return {
        "images": len(subset),
        "predictions": pred_total,
        "true_positive_predictions": tp_total,
        "false_positive_predictions": fp_total,
        "precision_iou30": (tp_total / pred_total) if pred_total else None,
        "person_gt": gt_total,
        "matched_person_gt": matched_gt_total,
        "overall_recall_iou30": (matched_gt_total / gt_total) if gt_total else None,
        "median_cpu_ms_per_image": float(np.median([r["cpu_ms"] for r in subset])) if subset else None,
        "by_reference_height": by_bucket,
    }


def evaluate_model(model_name: str, dataset, a):
    model = YOLO(model_name)
    records = []
    gallery = []
    order = ["lt16", "16_23", "24_31", "32_47", "48_63", "64_95", "ge96"]

    for image_path, gts in dataset.items():
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        mode_results = {}
        full640, ms640 = predict(model, image, 640, a.confidence)
        full960, ms960 = predict(model, image, 960, a.confidence)
        tiled, mstile = tiled_predictions(model, image, a.confidence, a.tile_fraction)
        mode_results["full640"] = (full640, ms640)
        mode_results["full960"] = (full960, ms960)
        mode_results["full640_plus_4tiles"] = (tiled, mstile)

        for mode, (preds, elapsed) in mode_results.items():
            matched_pred, matched_gt = greedy_match(preds, gts, a.iou_threshold)
            row = {
                "model": Path(model_name).stem,
                "mode": mode,
                "image": image_path.name,
                "cpu_ms": round(elapsed, 3),
                "predictions": len(preds),
                "matched_predictions": len(matched_pred),
                "false_positive_predictions": len(preds) - len(matched_pred),
                "gt_count": len(gts),
                "matched_gt_count": len(matched_gt),
            }
            for b in order:
                indices = [i for i, gt in enumerate(gts) if gt.bucket == b]
                row[f"gt_{b}"] = len(indices)
                row[f"hit_{b}"] = sum(1 for i in indices if i in matched_gt)
            records.append(row)

        if Path(model_name).stem == "yolo11s" and any(gt.reference_height_px < 24 for gt in gts) and len(gallery) < 12:
            gallery.append((image_path, image.copy(), gts, full640, tiled))

    modes = ["full640", "full960", "full640_plus_4tiles"]
    return records, {mode: summarize_mode(records, mode) for mode in modes}, gallery


def draw_gallery(items, out: Path):
    cards = []
    for image_path, image, gts, full, tiled in items:
        canvas = image.copy()
        for gt in gts:
            if gt.reference_height_px < 32:
                x1, y1, x2, y2 = [int(round(v)) for v in gt.box]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)
        for box, _ in full:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 60, 235), 1)
        for box, _ in tiled:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (235, 180, 40), 1)
        cv2.putText(canvas, "green=GT<32px red=full640 blue=full+tiles", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
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

    all_records = []
    model_summaries = {}
    gallery_items = []
    for model_name in a.models:
        records, summary, gallery = evaluate_model(model_name, dataset, a)
        all_records.extend(records)
        model_summaries[Path(model_name).stem] = summary
        gallery_items.extend(gallery)

    with (out / "natural_person_tile_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)

    report = {
        "benchmark": "Natural-scene visible-person overlap-tile benchmark v1",
        "dataset": "COCO128 original full scenes and person annotations",
        "reference_imgsz": a.reference_imgsz,
        "tile_fraction": a.tile_fraction,
        "tile_count": 4,
        "confidence": a.confidence,
        "iou_hit_threshold": a.iou_threshold,
        "models": model_summaries,
        "interpretation_limit": "Generic offline visible-person benchmark only. It measures full-scene/tiled person detection and does not evaluate gameplay, team identity, hidden-position inference, or live assistance.",
    }
    (out / "natural_person_tile_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_gallery(gallery_items[:12], out / "natural_person_tile_gallery.jpg")

    lines = [
        "## Natural-scene visible-person overlap-tile benchmark v1",
        "",
        f"- Images with person annotations: **{len(dataset)}**",
        f"- Fixed reference buckets: **projected height at {a.reference_imgsz}px**",
        f"- Tile layout: **4 overlapping crops, fraction={a.tile_fraction:.2f}**",
        f"- Hit/match rule: **IoU >= {a.iou_threshold:.2f}**",
        "",
        "| model | mode | <16 recall | 16-23 recall | 24-31 recall | overall recall | precision | median CPU ms/image |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, modes in model_summaries.items():
        for mode in ("full640", "full960", "full640_plus_4tiles"):
            s = modes[mode]
            def pct(v):
                return "n/a" if v is None else f"{100*v:.1f}%"
            lines.append(
                f"| {model} | {mode} | {pct(s['by_reference_height']['lt16']['recall'])} | "
                f"{pct(s['by_reference_height']['16_23']['recall'])} | {pct(s['by_reference_height']['24_31']['recall'])} | "
                f"{pct(s['overall_recall_iou30'])} | {pct(s['precision_iou30'])} | {s['median_cpu_ms_per_image']:.1f} |"
            )
    lines += ["", "The tiled mode is useful only if tiny-person recall improves enough to justify its extra compute and false positives."]
    text = "\n".join(lines) + "\n"
    (out / "summary_natural_person_tile_benchmark.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
