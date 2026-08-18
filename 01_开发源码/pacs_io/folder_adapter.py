from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from .base import PacsAdapter
from .contracts import CaseInput, DicomSeries


_DICOM_EXTENSIONS = {".dcm", ".dicom", ".ima"}


def _has_dicom_preamble(path: Path) -> bool:
    try:
        if path.stat().st_size < 132:
            return False
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def _candidate_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < 132:
        return False
    if _has_dicom_preamble(path):
        return True
    return path.suffix.lower() in _DICOM_EXTENSIONS


def _projection_sort_value(ds) -> Tuple[int, float]:
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        ipp = [float(x) for x in ds.ImagePositionPatient]
        if len(iop) == 6 and len(ipp) == 3:
            row = iop[:3]
            col = iop[3:]
            normal = (
                row[1] * col[2] - row[2] * col[1],
                row[2] * col[0] - row[0] * col[2],
                row[0] * col[1] - row[1] * col[0],
            )
            projection = sum(float(a) * float(b) for a, b in zip(ipp, normal))
            return 0, float(projection)
    except Exception:
        pass

    try:
        return 1, float(ds.SliceLocation)
    except Exception:
        pass

    try:
        return 2, float(ds.InstanceNumber)
    except Exception:
        return 3, 0.0


def _read_header(path: Path):
    import pydicom

    try:
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=False,
            specific_tags=[
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "Modality",
                "SeriesDescription",
                "ProtocolName",
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "SliceLocation",
                "InstanceNumber",
            ],
        )
    except Exception:
        if path.suffix.lower() not in _DICOM_EXTENSIONS:
            raise
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=[
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "Modality",
                "SeriesDescription",
                "ProtocolName",
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "SliceLocation",
                "InstanceNumber",
            ],
        )


class FolderPacsAdapter(PacsAdapter):

    def load_case(self, case_ref: str) -> CaseInput:
        try:
            import pydicom  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("缺少 pydicom") from exc

        root = Path(case_ref)
        if not root.exists():
            raise RuntimeError(f"病例目录不存在: {root}")
        if not root.is_dir():
            raise RuntimeError(f"病例路径不是目录: {root}")

        grouped = defaultdict(list)
        study_counts = Counter()

        for path in root.rglob("*"):
            if not _candidate_file(path):
                continue

            try:
                ds = _read_header(path)
            except Exception:
                continue

            study_uid = str(getattr(ds, "StudyInstanceUID", "") or "").strip()
            series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()

            if not study_uid or not series_uid:
                continue

            modality = str(getattr(ds, "Modality", "") or "").upper().strip()
            description = str(getattr(ds, "SeriesDescription", "") or "").strip()
            protocol = str(getattr(ds, "ProtocolName", "") or "").strip()
            sort_key = _projection_sort_value(ds)

            grouped[(study_uid, series_uid)].append(
                (sort_key, path, modality, description, protocol)
            )
            study_counts[study_uid] += 1

        if not grouped:
            raise RuntimeError(f"未在病例目录中找到有效DICOM: {root}")

        warnings = []
        selected_study_uid = study_counts.most_common(1)[0][0]

        if len(study_counts) > 1:
            warnings.append(
                "病例目录包含多个StudyInstanceUID；Phoenix为避免跨病例混合，"
                f"本次仅选择影像数量最多的Study: {selected_study_uid}。"
            )

        series = []

        for (study_uid, series_uid), entries in grouped.items():
            if study_uid != selected_study_uid:
                continue

            entries.sort(
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    str(item[1]),
                )
            )

            first = entries[0]
            files = [entry[1] for entry in entries]

            series.append(
                DicomSeries(
                    study_uid=study_uid,
                    series_uid=series_uid,
                    modality=first[2],
                    files=files,
                    series_description=first[3],
                    protocol_name=first[4],
                )
            )

        series.sort(
            key=lambda item: (
                str(item.modality),
                str(item.series_description),
                str(item.series_uid),
            )
        )

        return CaseInput(
            case_id=f"{root.name}:{selected_study_uid}",
            series=series,
            source="folder",
            study_uid=selected_study_uid,
            source_path=root,
            warnings=warnings,
        )

    def close_case(self):
        pass
