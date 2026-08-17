from __future__ import annotations

from typing import Callable, Optional

from pacs_io.yunpacs_cache_adapter import (
    YUNCase,
    YUNPACSLocalCacheAdapter,
)
from pacs_io.yunpacs_watcher import YUNPACSWatcher


class YUNPACSPhoenixBridge:

    def __init__(
        self,
        on_case: Callable[[YUNCase], None],
        on_case_close: Optional[Callable[[YUNCase], None]] = None,
    ):
        self.adapter = YUNPACSLocalCacheAdapter()
        self.watcher = YUNPACSWatcher(self.adapter)

        self.on_case = on_case
        self.on_case_close = on_case_close

        self.current_case = None

    def handle_case(self, case: YUNCase):

        if self.current_case is not None:
            if case.study_uid != self.current_case.study_uid:

                if self.on_case_close is not None:
                    self.on_case_close(
                        self.current_case
                    )

        self.current_case = case

        self.on_case(case)

    def run_forever(self):

        for case in self.watcher.watch():
            self.handle_case(case)
