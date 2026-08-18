from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from core.environment_paths import resolve_project_root


class CaseSession:

    def __init__(self):
        self.case_id = None
        self.temp_dir = None
        self.temp_root = resolve_project_root() / "08_temp_cache"

    def open(self, case_id: str) -> Path:
        self.close()

        self.case_id = str(case_id)
        self.temp_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.temp_dir = Path(
            tempfile.mkdtemp(
                prefix="phoenix_case_",
                dir=str(self.temp_root),
            )
        )
        return self.temp_dir

    def close(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(
                self.temp_dir,
                ignore_errors=True,
            )

        self.case_id = None
        self.temp_dir = None

    def purge_stale(self):
        if not self.temp_root.exists():
            return

        for path in self.temp_root.glob("phoenix_case_*"):
            if path == self.temp_dir:
                continue
            if path.is_dir():
                shutil.rmtree(
                    path,
                    ignore_errors=True,
                )
