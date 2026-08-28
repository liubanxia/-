#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def load_sparse_module(path: Path):
    spec = importlib.util.spec_from_file_location("sparse_roi_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep slice-only confidence for two-phase ROI inference")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base-script", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--reference-imgsz", type=int, default=640)
    p.add_argument("--baseline-confidence", type=float, default=0.05)
    p.add_argument("--slice-floor-confidence", type=float, default=0.05)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.07, 0.09, 0.12, 0.15])
    p.add_argument("--coverages", nargs="+", type=float, default=[0.68, 0.75])
    p.add_argument("--hit-iou", type=float, default=0.30)
    p.add_argument("--merge-iou", type=float, default=0.50)
    return p.parse_args()


def filtered(predictions, threshold: float):
    return [p for p in predictions if p.score >= threshold]


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base = load_sparse_module(Path(args.base_script))

    people = base.load_people(Path(args.dataset))
    if not people:
        raise SystemExit("no person annotations found")
    by_image = {}
    for gt in people:
        by_image.setdefault(gt.image_path, []).append(gt)

    model = YOLO(args.model)
    first = cv2.imread(str(next(iter(by_image))))
    if first is not None:
        base.run_person_model(model, first, args.imgsz, args.slice_floor_confidence)

    baseline = {"hits": 0, "total": 0, "tiny_hits": 0, "tiny_total": 0, "tp": 0, "fp": 0, "times": []}
    states = {
        (coverage, threshold): {
            "hits": 0, "total": 0, "tiny_hits": 0, "tiny_total": 0,
            "tp": 0, "fp": 0, "phase_times": [], "cycle_times": []
        }
        for coverage in args.coverages for threshold in args.thresholds
    }

    for image_path, gts in by_image.items():
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        tiny_gts = [
            gt for gt in gts
            if base.projected_height(image.shape, gt.box, args.reference_imgsz) < 32.0
        ]

        global_preds, global_ms = base.run_person_model(
            model, image, args.imgsz, args.baseline_confidence
        )
        hits, total = base.recall_for_gts(gts, global_preds, args.hit_iou)
        tiny_hits, tiny_total = base.recall_for_gts(tiny_gts, global_preds, args.hit_iou)
        tp, fp = base.greedy_stats(gts, global_preds, args.hit_iou)
        baseline["hits"] += hits
        baseline["total"] += total
        baseline["tiny_hits"] += tiny_hits
        baseline["tiny_total"] += tiny_total
        baseline["tp"] += tp
        baseline["fp"] += fp
        baseline["times"].append(global_ms)

        for coverage in args.coverages:
            mapped_phases = []
            phase_times = []
            for roi in base.make_two_phase_rois(w, h, coverage):
                x1, y1, x2, y2 = roi
                crop = image[y1:y2, x1:x2]
                preds, elapsed = base.run_person_model(
                    model, crop, args.imgsz, args.slice_floor_confidence
                )
                mapped_phases.append(base.map_crop_predictions(preds, roi))
                phase_times.append(elapsed)

            for threshold in args.thresholds:
                state = states[(coverage, threshold)]
                merged = base.nms(
                    filtered(mapped_phases[0], threshold) + filtered(mapped_phases[1], threshold),
                    args.merge_iou,
                )
                hits, total = base.recall_for_gts(gts, merged, args.hit_iou)
                tiny_hits, tiny_total = base.recall_for_gts(tiny_gts, merged, args.hit_iou)
                tp, fp = base.greedy_stats(gts, merged, args.hit_iou)
                state["hits"] += hits
                state["total"] += total
                state["tiny_hits"] += tiny_hits
                state["tiny_total"] += tiny_total
                state["tp"] += tp
                state["fp"] += fp
                state["phase_times"].extend(phase_times)
                state["cycle_times"].append(sum(phase_times))

    baseline_precision = baseline["tp"] / max(baseline["tp"] + baseline["fp"], 1)
    baseline_under32 = baseline["tiny_hits"] / max(baseline["tiny_total"], 1)
    baseline_overall = baseline["hits"] / max(baseline["total"], 1)
    baseline_result = {
        "confidence": args.baseline_confidence,
        "under32_recall": baseline_under32,
        "overall_recall": baseline_overall,
        "precision": baseline_precision,
        "median_ms": float(np.median(baseline["times"])),
        "p90_ms": float(np.percentile(baseline["times"], 90)),
        "tp": baseline["tp"],
        "fp": baseline["fp"],
    }

    rows = []
    for coverage in args.coverages:
        for threshold in args.thresholds:
            s = states[(coverage, threshold)]
            precision = s["tp"] / max(s["tp"] + s["fp"], 1)
            rows.append({
                "coverage": coverage,
                "slice_confidence": threshold,
                "under32_recall": s["tiny_hits"] / max(s["tiny_total"], 1),
                "overall_recall": s["hits"] / max(s["total"], 1),
                "precision": precision,
                "median_ms_per_phase": float(np.median(s["phase_times"])),
                "p90_ms_per_phase": float(np.percentile(s["phase_times"], 90)),
                "median_ms_per_cycle": float(np.median(s["cycle_times"])),
                "tp": s["tp"],
                "fp": s["fp"],
            })

    viable = [
        r for r in rows
        if r["precision"] >= baseline_precision - 0.03
        and r["under32_recall"] > baseline_under32
    ]
    best = max(
        viable,
        key=lambda r: (r["under32_recall"], r["overall_recall"], r["precision"], -r["median_ms_per_phase"]),
    ) if viable else None

    report = {
        "benchmark": "two-phase ROI slice-confidence sweep",
        "model": args.model,
        "imgsz": args.imgsz,
        "person_annotations": len(people),
        "baseline": baseline_result,
        "rows": rows,
        "best_viable": best,
        "notes": (
            "Each ROI inference is executed once at the lowest slice confidence; higher thresholds are applied "
            "to the same detections offline so runtime timing remains comparable. ROI placement is deterministic "
            "and never ground-truth guided."
        ),
    }
    (out / "sparse_roi_confidence_sweep.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "## Sparse ROI slice-confidence sweep",
        "",
        f"Baseline YOLO11n 384 @ {args.baseline_confidence:.2f}: **<32 {baseline_under32*100:.1f}% / overall {baseline_overall*100:.1f}% / precision {baseline_precision*100:.1f}%**",
        "",
        "| coverage | slice conf | <32 recall | overall recall | precision | median ms / phase | median ms / cycle |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['coverage']*100:.0f}% | {r['slice_confidence']:.2f} | "
            f"{r['under32_recall']*100:.1f}% | {r['overall_recall']*100:.1f}% | "
            f"{r['precision']*100:.1f}% | {r['median_ms_per_phase']:.1f} | {r['median_ms_per_cycle']:.1f} |"
        )
    lines += ["", "### Decision"]
    if best is None:
        lines.append("- No ROI confidence setting beats baseline tiny-person recall while staying within 3 precision points. Stop two-phase sliced inference for this runtime.")
    else:
        lines.append(
            f"- Best candidate: **{best['coverage']*100:.0f}% coverage / {best['slice_confidence']:.2f} slice confidence**."
        )
        lines.append(
            f"- Versus baseline: <32 recall **{(best['under32_recall']-baseline_under32)*100:+.1f} pp**, "
            f"overall recall **{(best['overall_recall']-baseline_overall)*100:+.1f} pp**, "
            f"precision **{(best['precision']-baseline_precision)*100:+.1f} pp**."
        )
        lines.append("- This remains a static validation candidate only; do not integrate until sequential Core ML memory and temporal behavior are measured.")
    lines.append("")
    lines.append("This benchmark covers generic visible-person detection only; it does not evaluate hidden-object inference, identity, gameplay advantage, or anti-cheat behavior.")
    text = "\n".join(lines) + "\n"
    (out / "summary_sparse_roi_confidence_sweep.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
