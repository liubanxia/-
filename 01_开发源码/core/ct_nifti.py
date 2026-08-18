from __future__ import annotations

from pathlib import Path

import pydicom
import SimpleITK as sitk

from core.dicom_geometry import plane_coordinate, read_plane_geometry


def _fallback_sort(path):
    try:
        ds = pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "SliceLocation",
                "InstanceNumber",
            ],
        )
    except Exception:
        return 3, 0.0, str(path)

    try:
        return 1, float(ds.SliceLocation), str(path)
    except Exception:
        pass

    try:
        return 2, float(ds.InstanceNumber), str(path)
    except Exception:
        return 3, 0.0, str(path)


def _sort_key(path):
    geometry = read_plane_geometry(path)

    if geometry is not None:
        try:
            return 0, float(plane_coordinate(geometry)), str(path)
        except Exception:
            pass

    return _fallback_sort(path)


def sort_dicom_series_files(files):
    return sorted(
        [Path(path) for path in files],
        key=_sort_key,
    )


def series_to_nifti(series, output_path):
    files = sort_dicom_series_files(
        getattr(series, "files", []) or []
    )

    if not files:
        raise RuntimeError("CT序列没有可转换的DICOM文件")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(
        [str(path) for path in files]
    )

    image = reader.Execute()
    sitk.WriteImage(
        image,
        str(output_path),
        True,
    )

    return files
