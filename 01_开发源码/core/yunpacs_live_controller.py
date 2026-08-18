from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pacs_io.yunpacs_cache_adapter import (
    YUNPACSLocalCacheAdapter,
)


class YUNPACSLiveController:

    def __init__(
        self,
        root="D:/YUNPACS/放射诊断/ImageDir_r",
        runtime=None,
    ):
        self.root = str(root)
        self.runtime = runtime

        self.cache = YUNPACSLocalCacheAdapter(
            root=self.root
        )

        self.current_fingerprint = None
        self.current_case = None
        self.current_case_ref = None

    def _ensure_runtime(self):
        if self.runtime is None:
            from core.runtime import PhoenixRuntime
            self.runtime = PhoenixRuntime()

        return self.runtime

    def _today_case_dirs(self):
        root = Path(self.root)

        today = datetime.now().strftime(
            "%Y-%m-%d"
        )

        day_dir = root / today

        if not day_dir.exists():
            return []

        dirs = [
            p for p in day_dir.iterdir()
            if p.is_dir()
        ]

        dirs.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return dirs

    def _select_current_candidate(self):
        candidates = self._today_case_dirs()

        if not candidates:
            raise RuntimeError(
                "今天YUNPACS缓存没有发现病例。"
                "为防止分析错误患者，Phoenix拒绝使用历史病例。"
            )

        return candidates[0]

    def poll_once(self):
        directory = self._select_current_candidate()

        runtime = self._ensure_runtime()

        loaded = runtime.open_case(
            "yunpacs",
            str(directory),
            root=self.root,
        )

        self.current_case = loaded
        self.current_case_ref = str(directory)

        self.current_fingerprint = (
            f"{directory}:"
            f"{directory.stat().st_mtime_ns}"
        )

        return loaded

    def case_identity(self):
        if self.current_case is None:
            return None

        study_uids = []
        modalities = []
        total_files = 0

        for series in getattr(
            self.current_case,
            "series",
            [],
        ):
            uid = getattr(
                series,
                "study_uid",
                None,
            )

            if uid and uid not in study_uids:
                study_uids.append(uid)

            modality = getattr(
                series,
                "modality",
                None,
            )

            if (
                modality
                and modality not in modalities
            ):
                modalities.append(modality)

            total_files += len(
                getattr(series, "files", []) or []
            )

        return {
            "case_id": getattr(
                self.current_case,
                "case_id",
                None,
            ),
            "path": self.current_case_ref,
            "study_uid": (
                study_uids[0]
                if study_uids
                else None
            ),
            "modalities": modalities,
            "series_count": len(
                getattr(
                    self.current_case,
                    "series",
                    [],
                )
            ),
            "file_count": total_files,
        }

    def analyze_current(self):
        if self.current_case is None:
            raise RuntimeError(
                "当前没有已确认的YUNPACS病例"
            )

        return self._ensure_runtime().analyze()

    def close(self):
        if self.runtime is not None:
            self.runtime.close_case()

        self.current_case = None
        self.current_case_ref = None
        self.current_fingerprint = None

    def shutdown(self):
        if self.runtime is not None:
            self.runtime.shutdown()

        self.current_case = None
        self.current_case_ref = None
        self.current_fingerprint = None
