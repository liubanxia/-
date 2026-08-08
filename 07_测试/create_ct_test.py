from pathlib import Path
from datetime import datetime

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    generate_uid,
)


# ---------------------------------------------------------
# Project Phoenix
# M7.7 CT 测试 DICOM 生成器
# ---------------------------------------------------------

output_path = Path(__file__).parent / "synthetic_CT.dcm"

# 创建 512 × 512 的模拟 CT 图像
height = 512
width = 512

# 默认空气：-1000 HU
hu_image = np.full((height, width), -1000, dtype=np.int16)

yy, xx = np.ogrid[:height, :width]

# 模拟人体软组织区域：约 +40 HU
body_mask = ((xx - 256) ** 2 / 190**2 +
             (yy - 256) ** 2 / 220**2) <= 1

hu_image[body_mask] = 40

# 模拟肺组织区域：约 -750 HU
left_lung = ((xx - 190) ** 2 / 75**2 +
             (yy - 245) ** 2 / 135**2) <= 1

right_lung = ((xx - 322) ** 2 / 75**2 +
              (yy - 245) ** 2 / 135**2) <= 1

hu_image[left_lung] = -750
hu_image[right_lung] = -750

# 模拟纵隔区域：约 +60 HU
mediastinum = ((xx - 256) ** 2 / 45**2 +
               (yy - 265) ** 2 / 100**2) <= 1

hu_image[mediastinum] = 60

# 模拟骨结构：约 +1000 HU
bone = ((xx - 256) ** 2 +
        (yy - 390) ** 2) <= 35**2

hu_image[bone] = 1000


# ---------------------------------------------------------
# DICOM File Meta
# ---------------------------------------------------------

file_meta = FileMetaDataset()

file_meta.MediaStorageSOPClassUID = CTImageStorage
file_meta.MediaStorageSOPInstanceUID = generate_uid()
file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
file_meta.ImplementationClassUID = generate_uid()


# ---------------------------------------------------------
# 创建 DICOM Dataset
# ---------------------------------------------------------

ds = FileDataset(
    str(output_path),
    {},
    file_meta=file_meta,
    preamble=b"\0" * 128,
)

now = datetime.now()

ds.SOPClassUID = CTImageStorage
ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

ds.PatientName = "Phoenix^CT_Test"
ds.PatientID = "PHOENIX_CT_001"

ds.StudyInstanceUID = generate_uid()
ds.SeriesInstanceUID = generate_uid()

ds.StudyDate = now.strftime("%Y%m%d")
ds.StudyTime = now.strftime("%H%M%S")

ds.Modality = "CT"

ds.StudyDescription = "Synthetic CT Test"
ds.SeriesDescription = "Synthetic Chest CT"

ds.Rows = height
ds.Columns = width

ds.SamplesPerPixel = 1
ds.PhotometricInterpretation = "MONOCHROME2"

ds.BitsAllocated = 16
ds.BitsStored = 16
ds.HighBit = 15
ds.PixelRepresentation = 1

# CT HU 转换参数
ds.RescaleIntercept = 0
ds.RescaleSlope = 1
ds.RescaleType = "HU"

# 默认纵隔窗
ds.WindowCenter = 40
ds.WindowWidth = 400

ds.PixelSpacing = [0.7, 0.7]
ds.SliceThickness = 1.0

ds.PixelData = hu_image.tobytes()

ds.save_as(str(output_path), enforce_file_format=True)

print("=" * 60)
print("Project Phoenix CT 测试 DICOM 创建成功")
print(f"文件: {output_path.name}")
print(f"尺寸: {width} x {height}")
print(f"最小 HU: {hu_image.min()}")
print(f"最大 HU: {hu_image.max()}")
print(f"Window Center: {ds.WindowCenter}")
print(f"Window Width: {ds.WindowWidth}")
print("=" * 60)