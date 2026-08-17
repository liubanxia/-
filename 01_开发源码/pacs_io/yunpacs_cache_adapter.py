from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


DEFAULT_YUNPACS_CACHE = Path(
    "D:/YUNPACS/放射诊断/ImageDir_r"
)


@dataclass
class YUNSeries:
    uid: str
    modality: str
    description: str
    files: List[Path]


@dataclass
class YUNCase:
    directory: Path
    study_uid: str
    modality: str
    series: List[YUNSeries]
    file_count: int
    newest_mtime: float

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.study_uid}|"
            f"{self.file_count}|"
            f"{int(self.newest_mtime)}"
        )


class YUNPACSLocalCacheAdapter:

    def __init__(
        self,
        root=DEFAULT_YUNPACS_CACHE,
    ):
        self.root = Path(root)

    def exists(self) -> bool:
        return self.root.exists()

    @staticmethod
    def _dicom_files(folder: Path) -> List[Path]:
        try:
            return sorted(
                p
                for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".dcm"
            )
        except OSError:
            return []

    def case_directories(self) -> List[Path]:
        if not self.root.exists():
            return []

        result = set()

        # YUNPACS:
        # ImageDir_r/YYYY-MM-DD/CASE_ID/*.dcm
        for p in self.root.glob("*/*/*.dcm"):
            if p.is_file():
                result.add(p.parent)

        return list(result)

    def latest_directory(self) -> Optional[Path]:
        candidates = []

        for folder in self.case_directories():
            files = self._dicom_files(folder)

            if not files:
                continue

            try:
                newest = max(
                    p.stat().st_mtime
                    for p in files
                )
            except OSError:
                continue

            candidates.append(
                (newest, folder)
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return candidates[0][1]


    def wait_until_stable(
        self,
        folder: Path,
        stable_seconds: float = 3.0,
        timeout: float = 120.0,
    ) -> List[Path]:

        import time

        deadline = time.time() + timeout
        last_signature = None
        stable_since = None

        while time.time() < deadline:

            files = self._dicom_files(folder)

            signature = []

            for p in files:
                try:
                    s = p.stat()
                    signature.append(
                        (
                            p.name,
                            s.st_size,
                            s.st_mtime_ns,
                        )
                    )
                except OSError:
                    pass

            signature = tuple(signature)

            if signature and signature == last_signature:

                if stable_since is None:
                    stable_since = time.time()

                if time.time() - stable_since >= stable_seconds:
                    return files

            else:
                stable_since = None

            last_signature = signature

            time.sleep(0.5)

        raise TimeoutError(
            f"YUNPACS case download timeout: {folder}"
        )


    @staticmethod
    def _read_header(path: Path):
        import pydicom

        try:
            return pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
            )
        except Exception:
            return None


    def build_case(
        self,
        folder: Path,
        wait: bool = True,
    ) -> YUNCase:

        files = (
            self.wait_until_stable(folder)
            if wait
            else self._dicom_files(folder)
        )

        groups = {}
        study_uid = ""
        modalities = set()
        valid_files = []

        for path in files:

            ds = self._read_header(path)

            if ds is None:
                continue

            suid = str(
                getattr(ds, "StudyInstanceUID", "")
                or ""
            )

            series_uid = str(
                getattr(ds, "SeriesInstanceUID", "")
                or ""
            )

            modality = str(
                getattr(ds, "Modality", "")
                or ""
            ).upper()

            description = str(
                getattr(ds, "SeriesDescription", "")
                or ""
            )

            if suid and not study_uid:
                study_uid = suid

            if modality:
                modalities.add(modality)

            if not series_uid:
                series_uid = "UNKNOWN_SERIES"

            valid_files.append(path)

            item = groups.setdefault(
                series_uid,
                {
                    "modality": modality,
                    "description": description,
                    "files": [],
                },
            )

            item["files"].append(path)

        if not valid_files:
            raise RuntimeError(
                "No readable DICOM files found."
            )

        series = []

        for uid, item in groups.items():

            series.append(
                YUNSeries(
                    uid=uid,
                    modality=item["modality"],
                    description=item["description"],
                    files=sorted(item["files"]),
                )
            )

        if "CT" in modalities:
            case_modality = "CT"
        elif "DX" in modalities:
            case_modality = "DX"
        elif "CR" in modalities:
            case_modality = "CR"
        elif modalities:
            case_modality = sorted(modalities)[0]
        else:
            case_modality = "UNKNOWN"

        newest = max(
            p.stat().st_mtime
            for p in valid_files
        )

        return YUNCase(
            directory=folder,
            study_uid=study_uid or folder.name,
            modality=case_modality,
            series=series,
            file_count=len(valid_files),
            newest_mtime=newest,
        )


    def latest_case(
        self,
        wait: bool = True,
    ) -> Optional[YUNCase]:

        folder = self.latest_directory()

        if folder is None:
            return None

        return self.build_case(
            folder,
            wait=wait,
        )
