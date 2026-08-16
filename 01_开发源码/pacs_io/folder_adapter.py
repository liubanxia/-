from pathlib import Path
from collections import defaultdict

from .base import PacsAdapter
from .contracts import CaseInput, DicomSeries


class FolderPacsAdapter(PacsAdapter):

    def load_case(self, case_ref: str) -> CaseInput:
        try:
            import pydicom
        except ImportError:
            raise RuntimeError("缺少 pydicom")

        root = Path(case_ref)
        groups = defaultdict(list)
        meta = {}

        for f in root.rglob("*"):
            if not f.is_file():
                continue

            try:
                ds = pydicom.dcmread(
                    str(f),
                    stop_before_pixels=True,
                    force=True,
                )
            except Exception:
                continue

            series_uid = str(
                getattr(ds, "SeriesInstanceUID", "unknown")
            )

            groups[series_uid].append(f)

            meta[series_uid] = (
                str(getattr(ds, "StudyInstanceUID", "")),
                str(getattr(ds, "Modality", "")),
            )

        series = []

        for uid, files in groups.items():
            study_uid, modality = meta[uid]

            series.append(
                DicomSeries(
                    study_uid=study_uid,
                    series_uid=uid,
                    modality=modality,
                    files=sorted(files),
                )
            )

        return CaseInput(
            case_id=root.name,
            series=series,
            source="folder",
        )

    def close_case(self):
        pass
