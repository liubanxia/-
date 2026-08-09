from pathlib import Path
from datetime import datetime

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    CTImageStorage,
    generate_uid,
)


OUTPUT_DIR = Path(__file__).parent / "ct_series_test"
SLICE_COUNT = 40
ROWS = 512
COLS = 512


def create_slice(index, study_uid, series_uid, frame_uid):
    image = np.full((ROWS, COLS), -1000, dtype=np.int16)

    yy, xx = np.ogrid[:ROWS, :COLS]
    cx = COLS // 2
    cy = ROWS // 2

    body = ((xx - cx) ** 2 / 210**2 + (yy - cy) ** 2 / 220**2) <= 1
    image[body] = 40

    lung_left = (
        (xx - (cx - 90)) ** 2 / 70**2
        + (yy - cy) ** 2 / 135**2
    ) <= 1

    lung_right = (
        (xx - (cx + 90)) ** 2 / 70**2
        + (yy - cy) ** 2 / 135**2
    ) <= 1

    image[lung_left] = -800
    image[lung_right] = -800

    heart_size = 45 + int(index * 0.8)

    heart = (
        (xx - cx) ** 2 / heart_size**2
        + (yy - (cy + 20)) ** 2 / 85**2
    ) <= 1

    image[heart] = 60

    nodule_x = cx - 90 + index * 3
    nodule_y = cy - 20

    nodule = (
        (xx - nodule_x) ** 2
        + (yy - nodule_y) ** 2
    ) <= 12**2

    image[nodule] = 120

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    file_path = OUTPUT_DIR / f"CT_{index + 1:03d}.dcm"

    ds = FileDataset(
        str(file_path),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )

    now = datetime.now()

    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    ds.PatientName = "Phoenix^CTSeriesTest"
    ds.PatientID = "PHX_CT_SERIES_001"

    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = frame_uid

    ds.StudyDescription = "Synthetic CT Series Test"
    ds.SeriesDescription = "Synthetic Chest CT Series"
    ds.Modality = "CT"

    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")

    ds.Rows = ROWS
    ds.Columns = COLS

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15

    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0

    ds.WindowCenter = 40
    ds.WindowWidth = 400

    ds.InstanceNumber = index + 1

    z_position = float(index * 5.0)

    ds.ImagePositionPatient = [0.0, 0.0, z_position]
    ds.SliceLocation = z_position
    ds.SliceThickness = 5.0

    ds.PixelSpacing = [0.8, 0.8]

    ds.PixelData = image.tobytes()

    ds.save_as(file_path, enforce_file_format=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    for index in range(SLICE_COUNT):
        create_slice(
            index,
            study_uid,
            series_uid,
            frame_uid,
        )

    print("=" * 60)
    print("Project Phoenix 多层 CT 测试序列创建成功")
    print(f"目录: {OUTPUT_DIR}")
    print(f"切片数量: {SLICE_COUNT}")
    print("排序字段:")
    print("  ImagePositionPatient")
    print("  SliceLocation")
    print("  InstanceNumber")
    print("=" * 60)


if __name__ == "__main__":
    main()