import numpy as np


def read_dicom_image(path):
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))

    arr = arr * slope + intercept

    modality = str(
        getattr(ds, "Modality", "")
    ).upper()

    return arr, modality, ds
