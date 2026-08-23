import numpy as np


def read_dicom_image(path):
    import pydicom
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    arr = arr * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    modality = str(getattr(ds, "Modality", "")).upper()
    return arr, modality, ds
