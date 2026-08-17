from __future__ import annotations

import threading
import time

from core.yunpacs_live_controller import (
    YUNPACSLiveController,
)


class YUNPACSBackgroundWatcher:

    def __init__(
        self,
        root="D:/YUNPACS/放射诊断/ImageDir_r",
        poll_seconds=1.0,
        on_case_ready=None,
        on_error=None,
    ):
        self.controller = YUNPACSLiveController(
            root=root
        )

        self.poll_seconds = float(poll_seconds)
        self.on_case_ready = on_case_ready
        self.on_error = on_error

        self._stop = threading.Event()
        self._thread = None

    def _loop(self):

        while not self._stop.is_set():

            try:
                case = self.controller.poll_once()

                if case is not None:
                    if self.on_case_ready:
                        self.on_case_ready(case)

            except Exception as exc:
                if self.on_error:
                    self.on_error(exc)

            self._stop.wait(
                self.poll_seconds
            )

    def start(self):

        if (
            self._thread
            and self._thread.is_alive()
        ):
            return

        self._stop.clear()

        self._thread = threading.Thread(
            target=self._loop,
            name="YUNPACSWatcher",
            daemon=True,
        )

        self._thread.start()

    def stop(self):

        self._stop.set()

        if self._thread:
            self._thread.join(
                timeout=5
            )

        self.controller.shutdown()

    def analyze_current(self):
        # ONLY called by explicit doctor action.
        return self.controller.analyze_current()
