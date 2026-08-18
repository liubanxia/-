from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
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
    warnings: List[str] = field(default_factory=list)

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

    def case_directories(
        self,
        max_date_directories: int = 14,
    ) -> List[Path]:
        if not self.root.exists():
            return []

        try:
            date_dirs = [
                p
                for p in self.root.iterdir()
                if p.is_dir()
            ]
        except OSError:
            return []

        def _date_sort_key(path: Path):
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        date_dirs.sort(
            key=_date_sort_key,
            reverse=True,
        )

        result = []

        for date_dir in date_dirs[:max_date_directories]:
            try:
                children = [
                    p
                    for p in date_dir.iterdir()
                    if p.is_dir()
                ]
            except OSError:
                continue

            for case_dir in children:
                try:
                    if any(case_dir.glob("*.dcm")):
                        result.append(case_dir)
                except OSError:
                    continue

        return result

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

            candidates.append((newest, folder))

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
                        (p.name, s.st_size, s.st_mtime_ns)
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

        tags = [
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Modality",
            "SeriesDescription",
        ]

        try:
            return pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=False,
                specific_tags=tags,
            )
        except Exception:
            try:
                return pydicom.dcmread(
                    str(path),
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=tags,
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

        records = []
        study_counts = Counter()

        for path in files:
            ds = self._read_header(path)
            if ds is None:
                continue

            study_uid = str(
                getattr(ds, "StudyInstanceUID", "") or ""
            ).strip()
            series_uid = str(
                getattr(ds, "SeriesInstanceUID", "") or ""
            ).strip()
            modality = str(
                getattr(ds, "Modality", "") or ""
            ).upper()
            description = str(
                getattr(ds, "SeriesDescription", "") or ""
            )

            if not study_uid:
                continue

            records.append(
                (
                    path,
                    study_uid,
                    series_uid or "UNKNOWN_SERIES",
                    modality,
                    description,
                )
            )
            study_counts[study_uid] += 1

        if not records:
            raise RuntimeError("No readable DICOM files found.")

        selected_study_uid = study_counts.most_common(1)[0][0]
        warnings = []

        if len(study_counts) > 1:
            warnings.append(
                "YUNPACS缓存目录内出现多个StudyInstanceUID；"
                "已仅绑定影像数量最多的Study，禁止跨Study混合。"
            )

        groups = {}
        modalities = set()
        selected_files = []

        for path, study_uid, series_uid, modality, description in records:
            if study_uid != selected_study_uid:
                continue

            selected_files.append(path)
            if modality:
                modalities.add(modality)

            item = groups.setdefault(
                series_uid,
                {
                    "modality": modality,
                    "description": description,
                    "files": [],
                },
            )
            item["files"].append(path)

        series = [
            YUNSeries(
                uid=uid,
                modality=item["modality"],
                description=item["description"],
                files=sorted(item["files"]),
            )
            for uid, item in groups.items()
        ]

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
            for p in selected_files
        )

        return YUNCase(
            directory=folder,
            study_uid=selected_study_uid,
            modality=case_modality,
            series=series,
            file_count=len(selected_files),
            newest_mtime=newest,
            warnings=warnings,
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
