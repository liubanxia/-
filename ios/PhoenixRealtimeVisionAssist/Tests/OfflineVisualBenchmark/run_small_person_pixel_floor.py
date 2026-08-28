#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class SourcePerson:
    image_path: Path
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Scene:
    image_path: Path
    target_height: int
    gt: tuple[float, float, float, float]
    source_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic small-person pixel-floor benchmark")
    p.add_argument("--dataset", required=True, help="COCO128 root")
    p.add_argument("--output", required=True)
    p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    p.add_argument("--heights", nargs="+", type=int, default=[16, 24, 32, 48, 64, 96])
    p.add_argument("--imgsz", nargs="+", type=int, default=[640, 960])
    p.add_argument("--samples-per-height", type=int, default=24)
    p.add_argument("--confidence", type=float, default=0.05)
    p.add_argument("--iou-threshold", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--canvas", type=int, default=640)
    return p.parse_args()


def find_split_root(root: Path, kind: str) -> Path:
    candidates = [
        root / kind / "train2017",
        root / "coco128" / kind / "train2017",
        root / kind,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"cannot find {kind} under {root}")


def read_sources(dataset_root: Path) -> list[SourcePerson]:
    image_root = find_split_root(dataset_root, "images")
    label_root = find_split_root(dataset_root, "labels")
    sources: list[SourcePerson] = []

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
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls != 0:
                continue
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            ph = y2 - y1
            pw = x2 - x1
            if ph < 80 or pw < 18:
                continue
            if ph > h * 0.92 or pw > w * 0.82:
                continue
            aspect_hw = ph / max(pw, 1.0)
            if not (1.1 <= aspect_hw <= 5.0):
                continue
            if x1 < 1 or y1 < 1 or x2 > w - 1 or y2 > h - 1:
                continue
            sources.append(SourcePerson(image_path, x1, y1, x2, y2))
    return sources


def paste_scaled_scene(
    source: SourcePerson,
    target_height: int,
    canvas_edge: int,
    output_path: Path,
) -> Scene | None:
    image = cv2.imread(str(source.image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = target_height / max(source.height, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(image, (new_w, new_h), interpolation=interp)

    sx1, sy1 = source.x1 * scale, source.y1 * scale
    sx2, sy2 = source.x2 * scale, source.y2 * scale
    pcx = (sx1 + sx2) * 0.5
    pcy = (sy1 + sy2) * 0.5

    if new_w <= canvas_edge and new_h <= canvas_edge:
        canvas = np.full((canvas_edge, canvas_edge, 3), 114, np.uint8)
        off_x = (canvas_edge - new_w) // 2
        off_y = (canvas_edge - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = scaled
        gt = (sx1 + off_x, sy1 + off_y, sx2 + off_x, sy2 + off_y)
    else:
        left = int(round(pcx - canvas_edge * 0.5))
        top = int(round(pcy - canvas_edge * 0.5))
        left = max(0, min(max(0, new_w - canvas_edge), left))
        top = max(0, min(max(0, new_h - canvas_edge), top))
        right = min(new_w, left + canvas_edge)
        bottom = min(new_h, top + canvas_edge)

        crop = scaled[top:bottom, left:right]
        canvas = np.full((canvas_edge, canvas_edge, 3), 114, np.uint8)
        off_x = (canvas_edge - crop.shape[1]) // 2
        off_y = (canvas_edge - crop.shape[0]) // 2
        canvas[off_y:off_y + crop.shape[0], off_x:off_x + crop.shape[1]] = crop
        gt = (
            sx1 - left + off_x,
            sy1 - top + off_y,
            sx2 - left + off_x,
            sy2 - top + off_y,
        )

    gx1, gy1, gx2, gy2 = gt
    if gx1 < 0 or gy1 < 0 or gx2 > canvas_edge or gy2 > canvas_edge:
        return None
    if gy2 - gy1 < max(4, target_height * 0.75):
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return Scene(output_path, target_height, gt, source.image_path.name)


def build_scenes(
    sources: list[SourcePerson],
    heights: list[int],
    samples_per_height: int,
    canvas_edge: int,
    out: Path,
    seed: int,
) -> list[Scene]:
    rng = random.Random(seed)
    pool = list(sources)
    rng.shuffle(pool)
    if len(pool) < samples_per_height:
        raise RuntimeError(f"only {len(pool)} valid source persons; need {samples_per_height}")

    chosen = pool[:samples_per_height]
    scenes: list[Scene] = []
    for target_height in heights:
        for idx, source in enumerate(chosen):
            scene = paste_scaled_scene(
                source,
                target_height,
                canvas_edge,
                out / "generated" / f"h{target_height:03d}" / f"scene_{idx:03d}.jpg",
            )
            if scene is not None:
                scenes.append(scene)
    return scenes


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def expanded_contains_center(gt: tuple[float, float, float, float], pred: tuple[float, float, float, float]) -> bool:
    gx1, gy1, gx2, gy2 = gt
    px1, py1, px2, py2 = pred
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    gw, gh = gx2 - gx1, gy2 - gy1
    return (
        gx1 - 0.25 * gw <= pcx <= gx2 + 0.25 * gw
        and gy1 - 0.18 * gh <= pcy <= gy2 + 0.18 * gh
    )


def predictions_for(model: YOLO, paths: list[str], imgsz: int, conf: float):
    return model.predict(
        source=paths,
        imgsz=imgsz,
        conf=conf,
        iou=0.50,
        classes=[0],
        device="cpu",
        verbose=False,
        max_det=50,
        stream=True,
    )


def evaluate_config(
    model_name: str,
    imgsz: int,
    scenes: list[Scene],
    confidence: float,
    iou_threshold: float,
) -> tuple[list[dict], dict]:
    model = YOLO(model_name)
    scene_by_path = {str(s.image_path): s for s in scenes}
    rows: list[dict] = []

    results = predictions_for(model, [str(s.image_path) for s in scenes], imgsz, confidence)
    for result in results:
        scene = scene_by_path[str(Path(result.path))]
        preds: list[tuple[tuple[float, float, float, float], float]] = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for b, score in zip(boxes, scores):
                preds.append(((float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(score)))

        best_iou = 0.0
        best_conf = 0.0
        center_hit = False
        best_box = None
        for box, score in preds:
            ov = iou(scene.gt, box)
            center = expanded_contains_center(scene.gt, box)
            if ov > best_iou or (math.isclose(ov, best_iou) and score > best_conf):
                best_iou = ov
                best_conf = score
                best_box = box
            center_hit = center_hit or center

        hit_iou = best_iou >= iou_threshold
        rows.append({
            "model": model_name,
            "imgsz": imgsz,
            "target_height": scene.target_height,
            "source_name": scene.source_name,
            "scene": str(scene.image_path),
            "predictions": len(preds),
            "best_iou": round(best_iou, 5),
            "best_confidence": round(best_conf, 5),
            "hit_iou30": int(hit_iou),
            "center_hit": int(center_hit),
            "best_box": list(best_box) if best_box is not None else None,
            "gt": list(scene.gt),
        })

    grouped: dict[str, dict] = {}
    for height in sorted({s.target_height for s in scenes}):
        subset = [r for r in rows if r["target_height"] == height]
        n = len(subset)
        grouped[str(height)] = {
            "samples": n,
            "recall_iou30": sum(r["hit_iou30"] for r in subset) / max(n, 1),
            "recall_center": sum(r["center_hit"] for r in subset) / max(n, 1),
            "median_best_iou": float(np.median([r["best_iou"] for r in subset])) if subset else 0.0,
            "median_best_confidence": float(np.median([r["best_confidence"] for r in subset])) if subset else 0.0,
        }

    summary = {
        "model": model_name,
        "imgsz": imgsz,
        "by_target_height": grouped,
    }
    return rows, summary


def annotate(scene: Scene, rows_by_config: list[dict], canvas_edge: int) -> np.ndarray:
    im = cv2.imread(str(scene.image_path))
    if im is None:
        return np.zeros((canvas_edge, canvas_edge, 3), np.uint8)

    gx1, gy1, gx2, gy2 = [int(round(v)) for v in scene.gt]
    cv2.rectangle(im, (gx1, gy1), (gx2, gy2), (70, 220, 70), 2)
    y = 20
    for r in rows_by_config:
        label = f"{Path(r['model']).stem}@{r['imgsz']} iou={r['best_iou']:.2f} hit={r['hit_iou30']}"
        cv2.putText(im, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 17
    cv2.putText(
        im,
        f"GT person height={scene.target_height}px",
        (8, canvas_edge - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return im


def write_gallery(scenes: list[Scene], rows: list[dict], out: Path, seed: int) -> None:
    rng = random.Random(seed)
    selected: list[Scene] = []
    for height in sorted({s.target_height for s in scenes}):
        subset = [s for s in scenes if s.target_height == height]
        rng.shuffle(subset)
        selected.extend(subset[:4])

    cards = []
    for scene in selected:
        matching = [r for r in rows if r["scene"] == str(scene.image_path)]
        cards.append(annotate(scene, matching, 640))

    if not cards:
        return
    card_w = 400
    resized = [cv2.resize(c, (card_w, card_w), interpolation=cv2.INTER_AREA) for c in cards]
    cols = 4
    rows_n = math.ceil(len(resized) / cols)
    canvas = np.zeros((rows_n * card_w, cols * card_w, 3), np.uint8)
    for idx, card in enumerate(resized):
        rr, cc = divmod(idx, cols)
        canvas[rr * card_w:(rr + 1) * card_w, cc * card_w:(cc + 1) * card_w] = card
    cv2.imwrite(str(out), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    a = parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)

    dataset = Path(a.dataset)
    sources = read_sources(dataset)
    scenes = build_scenes(
        sources=sources,
        heights=a.heights,
        samples_per_height=a.samples_per_height,
        canvas_edge=a.canvas,
        out=out,
        seed=a.seed,
    )

    all_rows: list[dict] = []
    configs = []
    for model_name in a.models:
        for imgsz in a.imgsz:
            rows, summary = evaluate_config(
                model_name=model_name,
                imgsz=imgsz,
                scenes=scenes,
                confidence=a.confidence,
                iou_threshold=a.iou_threshold,
            )
            all_rows.extend(rows)
            configs.append(summary)

    csv_path = out / "small_person_pixel_floor.csv"
    if all_rows:
        fieldnames = [
            "model", "imgsz", "target_height", "source_name", "scene",
            "predictions", "best_iou", "best_confidence", "hit_iou30",
            "center_hit", "best_box", "gt",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                row = dict(row)
                row["best_box"] = json.dumps(row["best_box"])
                row["gt"] = json.dumps(row["gt"])
                writer.writerow(row)

    report = {
        "benchmark": "LiteView generic small-person pixel-floor v1",
        "dataset": "COCO128 person annotations",
        "source_persons_available": len(sources),
        "source_persons_sampled_per_height": a.samples_per_height,
        "target_heights_px": a.heights,
        "canvas_edge_px": a.canvas,
        "confidence": a.confidence,
        "iou_hit_threshold": a.iou_threshold,
        "configs": configs,
        "interpretation_limit": (
            "Synthetic scale-normalized benchmark using real COCO person scenes. "
            "It measures detector pixel sensitivity, not gameplay accuracy or hidden-person inference."
        ),
    }
    (out / "small_person_pixel_floor.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "## LiteView generic small-person pixel-floor v1",
        "",
        f"- Valid COCO source persons: **{len(sources)}**",
        f"- Samples per target height: **{a.samples_per_height}**",
        f"- Heights tested: **{', '.join(str(x) for x in a.heights)} px**",
        f"- Hit rule: IoU >= **{a.iou_threshold:.2f}** against ground-truth person box",
        "",
        "| model | input | 16px | 24px | 32px | 48px | 64px | 96px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg in configs:
        vals = []
        for h in (16, 24, 32, 48, 64, 96):
            item = cfg["by_target_height"].get(str(h), {})
            vals.append(f"{100.0 * item.get('recall_iou30', 0.0):.1f}%")
        lines.append(
            f"| {Path(cfg['model']).stem} | {cfg['imgsz']} | "
            + " | ".join(vals)
            + " |"
        )
    lines.extend([
        "",
        "This is a pixel-sensitivity benchmark, not an accuracy claim for any live application.",
        "The gallery is the visual acceptance check.",
        "",
    ])
    summary = "\n".join(lines)
    (out / "summary_small_person_pixel_floor.md").write_text(summary, encoding="utf-8")
    write_gallery(scenes, all_rows, out / "small_person_pixel_floor_gallery.jpg", a.seed)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
