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

    def load_case(self, case_ref: str) -> CaseInput:

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

        else:
            directory = Path(case_ref)

            if not directory.exists():
                raise RuntimeError(
                    f"YUNPACS病例目录不存在: {directory}"
                )

        loaded = self.folder.load_case(
            str(directory)
        )

        return CaseInput(
            case_id=loaded.case_id,
            series=loaded.series,
            source="yunpacs",
        )

    def close_case(self):
        pass
