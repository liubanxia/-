import shutil
import tempfile
from pathlib import Path


class CaseSession:

    def __init__(self):
        self.case_id = None
        self.temp_dir = None

    def open(self, case_id: str) -> Path:
        self.close()

        self.case_id = case_id
        self.temp_dir = Path(
            tempfile.mkdtemp(prefix="phoenix_case_")
        )
        return self.temp_dir

    def close(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        self.case_id = None
        self.temp_dir = None
