from pathlib import Path
from statistics import median

import pydicom


def inspect_ct_series(root):
    root = Path(root)

    series = {}

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

        uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "UNKNOWN",
            )
        )

        z = None

        ipp = getattr(
            ds,
            "ImagePositionPatient",
            None,
        )

        if ipp is not None and len(ipp) >= 3:
            try:
                z = float(ipp[2])
            except Exception:
                pass

        if z is None:
            try:
                z = float(ds.SliceLocation)
            except Exception:
                pass

        series.setdefault(uid, []).append(
            {
                "path": str(path),
                "z": z,
            }
        )

    output = []

    for uid, items in series.items():
        positions = sorted(
            x["z"]
            for x in items
            if x["z"] is not None
        )

        info = {
            "series_uid": uid,
            "images": len(items),
            "positioned_images": len(positions),
            "duplicate_positions": 0,
            "spacing_min": None,
            "spacing_median": None,
            "spacing_max": None,
            "max_gap_ratio": None,
        }

        if len(positions) >= 2:
            diffs = [
                abs(b - a)
                for a, b in zip(
                    positions[:-1],
                    positions[1:],
                )
            ]

            info["duplicate_positions"] = sum(
                1 for d in diffs if d < 1e-5
            )

            usable = [
                d for d in diffs
                if d >= 1e-5
            ]

            if usable:
                med = median(usable)

                info["spacing_min"] = min(usable)
                info["spacing_median"] = med
                info["spacing_max"] = max(usable)

                if med > 0:
                    info["max_gap_ratio"] = (
                        max(usable) / med
                    )

        output.append(info)

    return output
