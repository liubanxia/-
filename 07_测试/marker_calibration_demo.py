from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "01_开发源码"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from output.marker_style import draw_precision_arrow, scaled_point


def _first_ct_dicom(root: Path) -> Path:
    candidates = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )
        except Exception:
            continue

        if str(getattr(ds, "Modality", "")).upper() != "CT":
            continue

        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue

        candidates.append(path)

    if not candidates:
        raise RuntimeError(f"未找到CT DICOM: {root}")

    candidates.sort()
    return candidates[len(candidates) // 2]


def _first_number(value, default):
    try:
        if isinstance(value, (list, tuple)):
            return float(value[0])
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return float(list(value)[0])
        return float(value)
    except Exception:
        return float(default)


def _to_uint8(ds) -> np.ndarray:
    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        center = _first_number(wc, 40.0)
        width = max(_first_number(ww, 400.0), 1.0)
        low = center - width / 2.0
        high = center + width / 2.0
    else:
        low = float(np.percentile(pixels, 1.0))
        high = float(np.percentile(pixels, 99.0))
        if high <= low:
            high = low + 1.0

    pixels = np.clip(pixels, low, high)
    pixels = (pixels - low) / (high - low)
    return (pixels * 255.0).astype(np.uint8)


def build_demo(dicom_root: Path) -> Path:
    dicom_path = _first_ct_dicom(dicom_root)

    ds = pydicom.dcmread(
        str(dicom_path),
        force=True,
    )

    pixels = _to_uint8(ds)
    image = Image.fromarray(pixels).convert("RGB")

    original_shape = pixels.shape
    image.thumbnail((1024, 1024))

    h, w = original_shape[:2]

    targets = [
        (int(w * 0.08), int(h * 0.08), "EDGE-1"),
        (int(w * 0.25), int(h * 0.25), "A"),
        (int(w * 0.50), int(h * 0.50), "B"),
        (int(w * 0.75), int(h * 0.75), "C"),
        (int(w * 0.92), int(h * 0.92), "EDGE-2"),
    ]

    measured = []

    for x, y, label in targets:
        tip = draw_precision_arrow(
            image=image,
            point=(x, y),
            original_shape=original_shape,
            fill="red",
        )
        expected = scaled_point(
            (x, y),
            original_shape,
            (image.width, image.height),
        )
        measured.append(
            (
                label,
                math.hypot(
                    tip[0] - expected[0],
                    tip[1] - expected[1],
                ),
            )
        )

    draw = ImageDraw.Draw(image)

    for x, y, label in targets:
        px, py = scaled_point(
            (x, y),
            original_shape,
            (image.width, image.height),
        )

        draw.ellipse(
            (px - 2, py - 2, px + 2, py + 2),
            fill="lime",
        )
        draw.text(
            (px + 8, py + 8),
            label,
            fill="lime",
        )

    max_error = max(
        (error for _label, error in measured),
        default=0.0,
    )

    output_dir = PROJECT_ROOT / "08_temp_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "marker_calibration.png"
    image.save(output_path)

    print(f"CALIBRATION_DICOM={dicom_path}")
    print(f"CALIBRATION_OUTPUT={output_path}")
    print(f"MAX_TIP_ERROR_PX={max_error:.6f}")
    print("CHECK=每个红色箭头尖端应精确落在对应绿色参考点上")

    if max_error > 0.01:
        raise RuntimeError(
            f"箭头尖端坐标校准失败: max_error={max_error:.6f}px"
        )

    return output_path


def main():
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "07_测试" / "DICOM"
    )

    build_demo(root)


if __name__ == "__main__":
    main()
