from statistics import median

import pydicom


def _inspect_files(files, series_uid=""):
    positions = []

    for path in files:
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

        z = None

        ipp = getattr(ds, "ImagePositionPatient", None)
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

        if z is not None:
            positions.append(z)

    positions.sort()

    info = {
        "series_uid": series_uid,
        "images": len(files),
        "positioned_images": len(positions),
        "duplicate_positions": 0,
        "spacing_min": None,
        "spacing_median": None,
        "spacing_max": None,
        "max_gap_ratio": None,
    }

    if len(positions) < 2:
        return info

    diffs = [
        abs(b - a)
        for a, b in zip(positions[:-1], positions[1:])
    ]

    info["duplicate_positions"] = sum(
        d < 1e-5 for d in diffs
    )

    usable = [d for d in diffs if d >= 1e-5]

    if usable:
        med = median(usable)

        info["spacing_min"] = min(usable)
        info["spacing_median"] = med
        info["spacing_max"] = max(usable)

        if med > 0:
            info["max_gap_ratio"] = max(usable) / med

    return info


def inspect_case_ct(case):
    results = []

    for series in case.series:
        if str(series.modality).upper() != "CT":
            continue

        results.append(
            _inspect_files(
                series.files,
                series.series_uid,
            )
        )

    return results


def build_ct_quality_warnings(case):
    warnings = []

    for info in inspect_case_ct(case):
        uid = info["series_uid"]

        if info["duplicate_positions"] > 0:
            warnings.append(
                f"CT序列 {uid} 检测到"
                f"{info['duplicate_positions']}处重复层位置。"
            )

        med = info["spacing_median"]
        minimum = info["spacing_min"]
        maximum = info["spacing_max"]

        if med and minimum and maximum:
            low_ratio = minimum / med
            high_ratio = maximum / med

            if low_ratio < 0.75 or high_ratio > 1.25:
                warnings.append(
                    "CT序列层间距不均匀："
                    f"最小{minimum:.2f} mm，"
                    f"中位{med:.2f} mm，"
                    f"最大{maximum:.2f} mm。"
                    "此提示不自动判定缺层，请结合原始序列确认。"
                )

    return warnings
