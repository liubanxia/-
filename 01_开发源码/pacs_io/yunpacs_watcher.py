from __future__ import annotations

import time

from pacs_io.yunpacs_cache_adapter import (
    YUNPACSLocalCacheAdapter,
)


class YUNPACSWatcher:

    def __init__(
        self,
        adapter=None,
        poll_seconds=1.0,
    ):
        self.adapter = (
            adapter
            or YUNPACSLocalCacheAdapter()
        )

        self.poll_seconds = float(poll_seconds)
        self.last_fingerprint = None

    def wait_next_case(self):

        while True:

            try:
                case = self.adapter.latest_case(
                    wait=True
                )
            except Exception:
                time.sleep(self.poll_seconds)
                continue

            if case is None:
                time.sleep(self.poll_seconds)
                continue

            if (
                case.fingerprint
                != self.last_fingerprint
            ):
                self.last_fingerprint = (
                    case.fingerprint
                )
                return case

            time.sleep(self.poll_seconds)

    def watch(self):

        while True:
            yield self.wait_next_case()
