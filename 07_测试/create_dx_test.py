from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


project_root = Path(__file__).resolve().parents[1]
output_file = project_root / "07_测试" / "synthetic_DX.dcm"

file_meta = FileMetaDataset()
file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = generate_uid()

ds = FileDataset(
    str(output_file),
    {},
    file_meta=file_meta,
    preamble=b"\0" * 128,
)

ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()

ds.Modality = "DX"

ds.PatientName = "SYNTHETIC^TEST"
ds.PatientID = "TEST0001"

ds.StudyDate = "20260807"
ds.SeriesDescription = "Synthetic DX Test"

ds.Rows = 64
ds.Columns = 64
ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"

ds.BitsAllocated = 16
ds.BitsStored = 12
ds.HighBit = 11
ds.PixelRepresentation = 0

pixel_array = np.arange(
    ds.Rows * ds.Columns,
    dtype=np.uint16,
).reshape(ds.Rows, ds.Columns)

ds.PixelData = pixel_array.tobytes()

ds.save_as(output_file, enforce_file_format=True)

print("DX测试文件创建成功：")
print(output_file)