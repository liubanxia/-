from __future__ import annotations

from pathlib import Path

from .base import PacsAdapter
from .contracts import CaseInput
from .folder_adapter import FolderPacsAdapter
from .yunpacs_cache_adapter import (
    DEFAULT_YUNPACS_CACHE,
    YUNPACSLocalCacheAdapter,
)


class YUNPACSPacsAdapter(PacsAdapter):

    def __init__(
        self,
        root=DEFAULT_YUNPACS_CACHE,
    ):
        self.cache = YUNPACSLocalCacheAdapter(
            root=root
        )
        self.folder = FolderPacsAdapter()
        self.bound_study_uid = ""
        self.bound_directory = None

    def load_case(self, case_ref: str) -> CaseInput:
        expected_study_uid = ""

        if str(case_ref).lower() in {
            "latest",
            "current",
        }:
            case = self.cache.latest_case(
                wait=True
            )

            if case is None:
                raise RuntimeError(
                    "YUNPACS没有发现可用病例"
                )

            directory = case.directory
            expected_study_uid = str(
                case.study_uid or ""
            ).strip()

        else:
            directory = Path(case_ref)

            if not directory.exists():
                raise RuntimeError(
                    f"YUNPACS病例目录不存在: {directory}"
                )

        loaded = self.folder.load_case(
            str(directory)
        )

        actual_study_uid = str(
            loaded.study_uid or ""
        ).strip()

        if (
            expected_study_uid
            and actual_study_uid
            and expected_study_uid != actual_study_uid
        ):
            raise RuntimeError(
                "YUNPACS病例绑定失败: "
                f"cache StudyUID={expected_study_uid}, "
                f"loaded StudyUID={actual_study_uid}"
            )

        self.bound_study_uid = actual_study_uid
        self.bound_directory = Path(directory)

        warnings = list(loaded.warnings)

        if str(case_ref).lower() in {"latest", "current"}:
            warnings.append(
                "当前病例由YUNPACS缓存最近稳定写入目录绑定；"
                "Phoenix同时以StudyInstanceUID校验实际进入AI的DICOM。"
            )

        return CaseInput(
            case_id=loaded.case_id,
            series=loaded.series,
            source="yunpacs",
            study_uid=actual_study_uid,
            source_path=Path(directory),
            warnings=warnings,
        )

    def close_case(self):
        self.bound_study_uid = ""
        self.bound_directory = None
