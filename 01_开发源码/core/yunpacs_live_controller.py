from __future__ import annotations

from core.environment_paths import resolve_image_root


class YUNPACSLiveController:

    def __init__(
        self,
        root=None,
        runtime=None,
    ):
        resolved = resolve_image_root(root)
        self.root = str(
            resolved
            if resolved is not None
            else (root or "D:/YUNPACS/放射诊断/ImageDir_r")
        )
        self.runtime = runtime

        self.current_fingerprint = None
        self.current_case = None
        self.current_case_ref = None

    def _ensure_runtime(self):
        if self.runtime is None:
            from core.runtime import PhoenixRuntime
            self.runtime = PhoenixRuntime()

        return self.runtime

    def poll_once(self):
        runtime = self._ensure_runtime()

        # Important: do not pick a directory by folder mtime here. The
        # YUNPACSPacsAdapter owns stable-cache selection and then verifies that
        # the StudyInstanceUID actually loaded into Phoenix matches the cache
        # candidate before inference is allowed.
        loaded = runtime.open_case(
            "yunpacs",
            "current",
            root=self.root,
        )

        self.current_case = loaded
        self.current_case_ref = str(
            getattr(loaded, "source_path", "") or ""
        )

        self.current_fingerprint = (
            f"{getattr(loaded, 'study_uid', '')}|"
            f"{self.current_case_ref}"
        )

        return loaded

    def case_identity(self):
        if self.current_case is None:
            return None

        modalities = []
        total_files = 0

        for series in getattr(
            self.current_case,
            "series",
            [],
        ):
            modality = str(
                getattr(series, "modality", "") or ""
            ).upper()

            if modality and modality not in modalities:
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
            "study_uid": getattr(
                self.current_case,
                "study_uid",
                None,
            ),
            "modalities": modalities,
            "series_count": len(
                getattr(self.current_case, "series", []) or []
            ),
            "file_count": total_files,
            "warnings": list(
                getattr(self.current_case, "warnings", []) or []
            ),
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
