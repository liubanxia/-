from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
DX_IMAGE_STORAGE_FOR_PRESENTATION = "1.2.840.10008.5.1.4.1.1.1.1"


def _new_dataset(
    path: Path,
    *,
    modality: str,
    sop_class_uid: str,
    study_uid: str,
    series_uid: str,
    body_part: str,
    series_description: str,
    instance_number: int,
) -> FileDataset:
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.PatientName = "SYNTHETIC^PHOENIX"
    ds.PatientID = "SYNTHETIC-NO-PHI"
    ds.PatientBirthDate = "19000101"
    ds.PatientSex = "O"
    ds.StudyDate = "20000101"
    ds.StudyTime = "000000"
    ds.AccessionNumber = "SYNTHETIC"
    ds.StudyDescription = f"SYNTHETIC {body_part} STUDY"
    ds.SeriesDescription = series_description
    ds.ProtocolName = series_description
    ds.BodyPartExamined = body_part
    ds.InstanceNumber = int(instance_number)
    ds.Rows = 32
    ds.Columns = 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [1.0, 1.0]
    return ds


def create_synthetic_ct_series(
    output_dir: str | Path,
    *,
    body_part: str = "HEAD",
    series_description: str | None = None,
    count: int = 4,
) -> List[Path]:
    """Create a tiny deterministic CT DICOM series containing no real patient data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, int(count))
    description = series_description or f"SYNTHETIC {body_part} CT"
    study_uid = generate_uid()
    series_uid = generate_uid()
    paths: List[Path] = []

    for index in range(count):
        path = output_dir / f"CT_{index + 1:03d}.dcm"
        ds = _new_dataset(
            path,
            modality="CT",
            sop_class_uid=CT_IMAGE_STORAGE,
            study_uid=study_uid,
            series_uid=series_uid,
            body_part=body_part,
            series_description=description,
            instance_number=index + 1,
        )
        z = float(index) * 2.5
        ds.ImagePositionPatient = [0.0, 0.0, z]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.SliceLocation = z
        ds.SliceThickness = 2.5
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        pixels = np.full((32, 32), 1000 + index, dtype=np.int16)
        ds.PixelData = pixels.tobytes()
        ds.save_as(str(path), write_like_original=False)
        paths.append(path)

    return paths


def create_synthetic_dr_image(
    output_path: str | Path,
    *,
    body_part: str = "CHEST",
    series_description: str | None = None,
) -> Path:
    """Create one tiny synthetic DR DICOM image containing no real patient data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    description = series_description or f"SYNTHETIC {body_part} DR"
    ds = _new_dataset(
        output_path,
        modality="DR",
        sop_class_uid=DX_IMAGE_STORAGE_FOR_PRESENTATION,
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        body_part=body_part,
        series_description=description,
        instance_number=1,
    )
    pixels = np.arange(32 * 32, dtype=np.int16).reshape(32, 32)
    ds.PixelData = pixels.tobytes()
    ds.save_as(str(output_path), write_like_original=False)
    return output_path
