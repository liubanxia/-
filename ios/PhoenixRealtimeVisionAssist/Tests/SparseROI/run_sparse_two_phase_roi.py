#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class PersonGT:
    image_path: Path
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    box: tuple[float, float, float, float]
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic two-phase sparse ROI benchmark")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--reference-imgsz", type=int, default=640)
    p.add_argument("--confidence", type=float, default=0.05)
    p.add_argument("--hit-iou", type=float, default=0.30)
    p.add_argument("--merge-iou", type=float, default=0.50)
    p.add_argument("--coverages", nargs="+", type=float, default=[0.60, 0.68, 0.75])
    return p.parse_args()


def find_root(root: Path, kind: str) -> Path:
    candidates = (
        root / kind / "train2017",
        root / "coco128" / kind / "train2017",
        root / kind,
    )
    for p in candidates:
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
            if x2 > x1 and y2 > y1:
                people.append(PersonGT(image_path, (x1, y1, x2, y2)))
    return people


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def projected_height(image_shape, box, imgsz: int) -> float:
    h, w = image_shape[:2]
    scale = min(imgsz / max(w, 1), imgsz / max(h, 1))
    return (box[3] - box[1]) * scale


def make_two_phase_rois(width: int, height: int, coverage: float) -> list[tuple[int, int, int, int]]:
    coverage = min(max(coverage, 0.51), 0.95)
    if width >= height:
        roi_w = min(width, max(2, int(round(width * coverage))))
        return [(0, 0, roi_w, height), (width - roi_w, 0, width, height)]
    roi_h = min(height, max(2, int(round(height * coverage))))
    return [(0, 0, width, roi_h), (0, height - roi_h, width, height)]


def run_person_model(model: YOLO, image: np.ndarray, imgsz: int, confidence: float) -> tuple[list[Prediction], float]:
    started = time.perf_counter()
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
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    predictions: list[Prediction] = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, score in zip(boxes, scores):
            predictions.append(Prediction(tuple(float(v) for v in box), float(score)))
    return predictions, elapsed_ms


def map_crop_predictions(
    predictions: list[Prediction], roi: tuple[int, int, int, int]
) -> list[Prediction]:
    x1, y1, _, _ = roi
    result: list[Prediction] = []
    for pred in predictions:
        bx1, by1, bx2, by2 = pred.box
        result.append(Prediction((bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), pred.score))
    return result


def nms(predictions: list[Prediction], threshold: float) -> list[Prediction]:
    remaining = sorted(predictions, key=lambda p: p.score, reverse=True)
    kept: list[Prediction] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [p for p in remaining if iou(best.box, p.box) < threshold]
    return kept


def greedy_stats(
    gts: list[PersonGT], predictions: list[Prediction], hit_iou: float
) -> tuple[int, int]:
    unmatched = set(range(len(gts)))
    tp = 0
    fp = 0
    for pred in sorted(predictions, key=lambda p: p.score, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx in unmatched:
            ov = iou(gts[idx].box, pred.box)
            if ov > best_iou:
                best_iou = ov
                best_idx = idx
        if best_idx is not None and best_iou >= hit_iou:
            unmatched.remove(best_idx)
            tp += 1
        else:
            fp += 1
    return tp, fp


def recall_for_gts(gts: list[PersonGT], predictions: list[Prediction], hit_iou: float) -> tuple[int, int]:
    hits = 0
    for gt in gts:
        if any(iou(gt.box, pred.box) >= hit_iou for pred in predictions):
            hits += 1
    return hits, len(gts)


def summarize_counts(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    people = load_people(Path(args.dataset))
    if not people:
        raise SystemExit("no person annotations found")
    by_image: dict[Path, list[PersonGT]] = {}
    for gt in people:
        by_image.setdefault(gt.image_path, []).append(gt)

    model = YOLO(args.model)
    first_image = cv2.imread(str(next(iter(by_image))))
    if first_image is not None:
        run_person_model(model, first_image, args.imgsz, args.confidence)

    baseline = {
        "hits": 0,
        "total": 0,
        "tiny_hits": 0,
        "tiny_total": 0,
        "tp": 0,
        "fp": 0,
        "times": [],
    }
    configs = {
        coverage: {
            "cycle_hits": 0,
            "cycle_total": 0,
            "cycle_tiny_hits": 0,
            "cycle_tiny_total": 0,
            "cycle_tp": 0,
            "cycle_fp": 0,
            "phase_hits": [0, 0],
            "phase_total": [0, 0],
            "phase_tiny_hits": [0, 0],
            "phase_tiny_total": [0, 0],
            "phase_times": [[], []],
            "cycle_times": [],
        }
        for coverage in args.coverages
    }

    for image_path, gts in by_image.items():
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        tiny_gts = [
            gt for gt in gts
            if projected_height(image.shape, gt.box, args.reference_imgsz) < 32.0
        ]

        global_preds, global_ms = run_person_model(model, image, args.imgsz, args.confidence)
        hits, total = recall_for_gts(gts, global_preds, args.hit_iou)
        tiny_hits, tiny_total = recall_for_gts(tiny_gts, global_preds, args.hit_iou)
        tp, fp = greedy_stats(gts, global_preds, args.hit_iou)
        baseline["hits"] += hits
        baseline["total"] += total
        baseline["tiny_hits"] += tiny_hits
        baseline["tiny_total"] += tiny_total
        baseline["tp"] += tp
        baseline["fp"] += fp
        baseline["times"].append(global_ms)

        for coverage in args.coverages:
            state = configs[coverage]
            phase_predictions: list[list[Prediction]] = []
            cycle_ms = 0.0
            for phase_index, roi in enumerate(make_two_phase_rois(width, height, coverage)):
                x1, y1, x2, y2 = roi
                crop = image[y1:y2, x1:x2]
                crop_preds, elapsed = run_person_model(model, crop, args.imgsz, args.confidence)
                mapped = map_crop_predictions(crop_preds, roi)
                phase_predictions.append(mapped)
                cycle_ms += elapsed
                state["phase_times"][phase_index].append(elapsed)

                phase_hits, phase_total = recall_for_gts(gts, mapped, args.hit_iou)
                phase_tiny_hits, phase_tiny_total = recall_for_gts(tiny_gts, mapped, args.hit_iou)
                state["phase_hits"][phase_index] += phase_hits
                state["phase_total"][phase_index] += phase_total
                state["phase_tiny_hits"][phase_index] += phase_tiny_hits
                state["phase_tiny_total"][phase_index] += phase_tiny_total

            merged = nms(phase_predictions[0] + phase_predictions[1], args.merge_iou)
            cycle_hits, cycle_total = recall_for_gts(gts, merged, args.hit_iou)
            cycle_tiny_hits, cycle_tiny_total = recall_for_gts(tiny_gts, merged, args.hit_iou)
            cycle_tp, cycle_fp = greedy_stats(gts, merged, args.hit_iou)
            state["cycle_hits"] += cycle_hits
            state["cycle_total"] += cycle_total
            state["cycle_tiny_hits"] += cycle_tiny_hits
            state["cycle_tiny_total"] += cycle_tiny_total
            state["cycle_tp"] += cycle_tp
            state["cycle_fp"] += cycle_fp
            state["cycle_times"].append(cycle_ms)

    baseline_precision = baseline["tp"] / max(baseline["tp"] + baseline["fp"], 1)
    baseline_result = {
        "overall_recall": summarize_counts(baseline["hits"], baseline["total"]),
        "under32_recall": summarize_counts(baseline["tiny_hits"], baseline["tiny_total"]),
        "precision": baseline_precision,
        "median_ms_per_scan": float(np.median(baseline["times"])),
        "p90_ms_per_scan": float(np.percentile(baseline["times"], 90)),
        "tp": baseline["tp"],
        "fp": baseline["fp"],
    }

    results = []
    for coverage in args.coverages:
        state = configs[coverage]
        precision = state["cycle_tp"] / max(state["cycle_tp"] + state["cycle_fp"], 1)
        phase_overall = [
            summarize_counts(state["phase_hits"][i], state["phase_total"][i]) for i in range(2)
        ]
        phase_tiny = [
            summarize_counts(state["phase_tiny_hits"][i], state["phase_tiny_total"][i]) for i in range(2)
        ]
        all_phase_times = state["phase_times"][0] + state["phase_times"][1]
        results.append({
            "coverage": coverage,
            "cycle_overall_recall": summarize_counts(state["cycle_hits"], state["cycle_total"]),
            "cycle_under32_recall": summarize_counts(state["cycle_tiny_hits"], state["cycle_tiny_total"]),
            "cycle_precision": precision,
            "average_single_phase_overall_recall": float(np.mean(phase_overall)),
            "average_single_phase_under32_recall": float(np.mean(phase_tiny)),
            "median_ms_per_phase_scan": float(np.median(all_phase_times)),
            "p90_ms_per_phase_scan": float(np.percentile(all_phase_times, 90)),
            "median_ms_per_two_phase_cycle": float(np.median(state["cycle_times"])),
            "tp": state["cycle_tp"],
            "fp": state["cycle_fp"],
        })

    report = {
        "benchmark": "Deterministic two-phase sparse long-axis ROI scan",
        "model": args.model,
        "imgsz": args.imgsz,
        "reference_imgsz": args.reference_imgsz,
        "confidence": args.confidence,
        "person_annotations": len(people),
        "baseline": baseline_result,
        "two_phase": results,
        "method": (
            "Each source image is split along its longer axis into two deterministic overlapping ROIs. "
            "Each ROI is independently resized to the same 384 model input; detections are remapped and merged. "
            "No annotation is used to choose the ROIs. One phase represents one detector invocation; a complete "
            "two-phase cycle represents two successive sparse scans on a static scene."
        ),
        "limitations": (
            "Static generic visible-person benchmark only. Union-of-two-phases is an upper bound for a temporal "
            "scan cycle because real objects and cameras can move between phases. It does not evaluate gameplay, "
            "identity, hidden-object inference, or anti-cheat behavior."
        ),
    }
    (output / "sparse_two_phase_roi.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "## Sparse two-phase 384 ROI benchmark",
        "",
        f"- Model: **{Path(args.model).stem}**",
        f"- Person annotations: **{len(people)}**",
        f"- Confidence: **{args.confidence:.2f}**",
        "- ROI selection: **deterministic long-axis split; no ground-truth-guided crop selection**",
        "",
        "| mode | coverage | <32 recall | overall recall | precision | avg one-phase <32 | avg one-phase overall | median ms / phase | p90 ms / phase | median ms / full cycle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| global baseline | 100% | {baseline_result['under32_recall']*100:.1f}% | {baseline_result['overall_recall']*100:.1f}% | {baseline_result['precision']*100:.1f}% | — | — | {baseline_result['median_ms_per_scan']:.1f} | {baseline_result['p90_ms_per_scan']:.1f} | — |",
    ]
    for r in results:
        lines.append(
            f"| two-phase ROI | {r['coverage']*100:.0f}% | "
            f"{r['cycle_under32_recall']*100:.1f}% | {r['cycle_overall_recall']*100:.1f}% | "
            f"{r['cycle_precision']*100:.1f}% | {r['average_single_phase_under32_recall']*100:.1f}% | "
            f"{r['average_single_phase_overall_recall']*100:.1f}% | {r['median_ms_per_phase_scan']:.1f} | "
            f"{r['p90_ms_per_phase_scan']:.1f} | {r['median_ms_per_two_phase_cycle']:.1f} |"
        )

    viable = [
        r for r in results
        if r["cycle_precision"] >= baseline_result["precision"] - 0.03
        and r["cycle_under32_recall"] > baseline_result["under32_recall"]
    ]
    lines += ["", "### Decision"]
    if viable:
        best = max(
            viable,
            key=lambda r: (
                r["cycle_under32_recall"] - baseline_result["under32_recall"],
                -r["median_ms_per_phase_scan"],
            ),
        )
        lines.append(
            f"- Best candidate: **{best['coverage']*100:.0f}% two-phase ROI**. "
            f"Cycle <32 recall gain: **{(best['cycle_under32_recall']-baseline_result['under32_recall'])*100:+.1f} pp**; "
            f"precision delta: **{(best['cycle_precision']-baseline_result['precision'])*100:+.1f} pp**."
        )
        lines.append(
            "- Keep as a validation candidate only. Next step must measure sequential Core ML peak memory and temporal reacquisition cost before any runtime integration."
        )
    else:
        lines.append(
            "- No deterministic two-phase ROI setting improved <32 recall while staying within 3 precision points of the baseline. Do not integrate this strategy."
        )
    lines += [
        "",
        "Two-phase cycle recall is measured on static images and is an upper bound for a temporal scan cycle; moving video can perform worse.",
    ]
    text = "\n".join(lines) + "\n"
    (output / "summary_sparse_two_phase_roi.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
