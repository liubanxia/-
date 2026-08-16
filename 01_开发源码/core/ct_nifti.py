from pathlib import Path
import pydicom
import SimpleITK as sitk


def _sort_key(path):
    ds = pydicom.dcmread(
        str(path),
        stop_before_pixels=True,
        force=True,
    )

    try:
        return float(ds.ImagePositionPatient[2])
    except Exception:
        pass

    try:
        return float(ds.SliceLocation)
    except Exception:
        pass

    return float(
        getattr(ds, "InstanceNumber", 0)
    )


def series_to_nifti(series, output_path):
    files = sorted(
        series.files,
        key=_sort_key,
    )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(
        [str(x) for x in files]
    )

    image = reader.Execute()

    sitk.WriteImage(
        image,
        str(output_path),
        True,
    )

    return files
