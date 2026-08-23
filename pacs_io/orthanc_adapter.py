import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from .base import PacsAdapter
from .folder_adapter import FolderPacsAdapter


class OrthancAdapter(PacsAdapter):

    def __init__(self, url="http://127.0.0.1:8042"):
        self.url = url.rstrip("/")
        self.temp_dir = None

    def _json(self, path):
        with urlopen(self.url + path) as response:
            return json.loads(response.read())

    def load_case(self, case_ref):
        self.close_case()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="phoenix_pacs_"))
        instances = self._json(f"/studies/{case_ref}/instances")
        for i, item in enumerate(instances):
            instance_id = item["ID"] if isinstance(item, dict) else item
            with urlopen(self.url + f"/instances/{instance_id}/file") as response:
                data = response.read()
            (self.temp_dir / f"{i:06d}.dcm").write_bytes(data)

        case = FolderPacsAdapter().load_case(str(self.temp_dir))
        case.case_id = case_ref
        case.source = "orthanc"
        case.temp_dir = self.temp_dir
        return case

    def close_case(self):
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
