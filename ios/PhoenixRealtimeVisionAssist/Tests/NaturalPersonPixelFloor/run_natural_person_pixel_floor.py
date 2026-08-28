#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PersonGT:
    image_path: Path
    box: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Natural-scene visible-person pixel-floor benchmark")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 960])
    p.add_argument("--reference-imgsz", type=int, default=640)
    p.add_argument("--confidence", type=float, default=0.05)
    p.add_argument("--iou-threshold", type=float, default=0.30)
    return p.parse_args()


def find_root(root: Path, kind: str) -> Path:
    for p in (root / kind / "train2017", root / "coco128" / kind / "train2017", root / kind):
        if p.exists():
            return p
    raise FileNotFoundError(f"cannot locate {kind} under {root}")


def load_people(root: Path) -> list[PersonGT]:
    image_root = find_root(root, "images")
    label_root = find_root(root, "labels")
    people: list[PersonGT] = []
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
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5 or int(float(parts[0])) != 0:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            if x2 <= x1 or y2 <= y1:
                continue
            people.append(PersonGT(image_path, (x1, y1, x2, y2)))
    return people


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


def projected_height(image_shape, box, imgsz: int) -> float:
    h, w = image_shape[:2]
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


def run_config(
    model_name: str,
    imgsz: int,
    reference_imgsz: int,
    people: list[PersonGT],
    confidence: float,
    hit_iou: float,
):
    model = YOLO(model_name)
    by_image: dict[Path, list[PersonGT]] = {}
    for gt in people:
        by_image.setdefault(gt.image_path, []).append(gt)

    rows = []
    for image_path, gts in by_image.items():
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=confidence,
            iou=0.50,
            classes=[0],
            device="cpu",
            verbose=False,
            max_det=100,
        )[0]
        preds = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            preds = [(tuple(float(v) for v in b), float(s)) for b, s in zip(boxes, scores)]

        for gt in gts:
            actual_px = projected_height(image.shape, gt.box, imgsz)
            reference_px = projected_height(image.shape, gt.box, reference_imgsz)
            best_iou = 0.0
            best_conf = 0.0
            for pred, score in preds:
                ov = iou(gt.box, pred)
                if ov > best_iou:
                    best_iou = ov
                    best_conf = score
            rows.append({
                "model": model_name,
                "imgsz": imgsz,
                "image": image_path.name,
                "reference_imgsz": reference_imgsz,
                "reference_person_height_px": round(reference_px, 3),
                "actual_person_height_px": round(actual_px, 3),
                "reference_bucket": bucket(reference_px),
                "best_iou": round(best_iou, 5),
                "best_confidence": round(best_conf, 5),
                "hit_iou30": int(best_iou >= hit_iou),
            })

    order = ["lt16", "16_23", "24_31", "32_47", "48_63", "64_95", "ge96"]
    grouped = {}
    for name in order:
        subset = [r for r in rows if r["reference_bucket"] == name]
        n = len(subset)
        grouped[name] = {
            "samples": n,
            "recall_iou30": (sum(r["hit_iou30"] for r in subset) / n) if n else None,
            "median_best_iou": float(np.median([r["best_iou"] for r in subset])) if subset else None,
            "median_best_confidence": float(np.median([r["best_confidence"] for r in subset])) if subset else None,
        }
    return rows, {
        "model": model_name,
        "imgsz": imgsz,
        "reference_imgsz": reference_imgsz,
        "by_reference_height": grouped,
    }


def main() -> int:
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    people = load_people(Path(a.dataset))
    if not people:
        raise SystemExit("no person annotations found")

    all_rows = []
    configs = []
    for model_name in a.models:
        for imgsz in a.imgsz:
            rows, summary = run_config(
                model_name,
                imgsz,
                a.reference_imgsz,
                people,
                a.confidence,
                a.iou_threshold,
            )
            all_rows.extend(rows)
            configs.append(summary)

    with (out / "natural_person_pixel_floor.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "benchmark": "Natural-scene visible-person pixel-floor v2 fixed-reference buckets",
        "dataset": "COCO128 original full scenes and person annotations",
        "person_annotations": len(people),
        "reference_imgsz": a.reference_imgsz,
        "confidence": a.confidence,
        "iou_hit_threshold": a.iou_threshold,
        "configs": configs,
        "interpretation_limit": "Original full-scene generic visible-person benchmark. Every model/input configuration is grouped by the same reference 640px projected-person buckets so comparisons use the same people. It does not evaluate gameplay, team identity, hidden-person inference, or live assistance.",
    }
    (out / "natural_person_pixel_floor.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    order = ["lt16", "16_23", "24_31", "32_47", "48_63", "64_95", "ge96"]
    lines = [
        "## Natural-scene visible-person pixel-floor v2",
        "",
        f"- Person annotations: **{len(people)}**",
        "- Source: **COCO128 original full scenes**",
        f"- Fixed reference buckets: **projected height at {a.reference_imgsz}px input**",
        f"- Hit rule: **IoU >= {a.iou_threshold:.2f}**",
        "",
        "| model | input | <16 | 16-23 | 24-31 | 32-47 | 48-63 | 64-95 | >=96 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in configs:
        cells = []
        for b in order:
            d = cfg["by_reference_height"][b]
            if not d["samples"]:
                cells.append("n/a")
            else:
                cells.append(f"{100*d['recall_iou30']:.1f}% (n={d['samples']})")
        lines.append(f"| {Path(cfg['model']).stem} | {cfg['imgsz']} | " + " | ".join(cells) + " |")
    lines += ["", "All rows use identical 640-reference person buckets, so 640 vs 960 is a same-person comparison."]
    text = "\n".join(lines) + "\n"
    (out / "summary_natural_person_pixel_floor.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
