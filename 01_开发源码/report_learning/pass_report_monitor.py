from dataclasses import dataclass
from datetime import datetime
import hashlib
import threading

from .report_diff_engine import ReportDiffEngine


def _now_iso():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


@dataclass
class ReportSnapshot:
    stage: str
    text: str
    text_hash: str
    captured_at: str


class MemoryReportReader:
    """
    开发测试用。
    后续真实PASS由UI Automation读取器替换。
    """

    def __init__(self, text=""):
        self.text = text

    def read_text(self):
        return str(self.text)


class PassReportMonitor:

    STATE_IDLE = "IDLE"
    STATE_ARMED = "ARMED"
    STATE_WATCHING = "WATCHING"
    STATE_FINALIZED = "FINALIZED"
    STATE_ERROR = "ERROR"

    def __init__(
        self,
        reader=None,
        diff_engine=None,
    ):
        self.reader = reader

        self.diff_engine = (
            diff_engine
            or ReportDiffEngine()
        )

        self.state = self.STATE_IDLE

        self.case_token = None
        self.ai_draft = ""

        self.baseline_snapshot = None
        self.latest_snapshot = None
        self.final_snapshot = None

        self.snapshots = []

        self._last_hash = None
        self._lock = threading.RLock()

        self.last_error = None

    @staticmethod
    def _hash(text):
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    def set_reader(self, reader):
        with self._lock:
            self.reader = reader

    def arm_case(
        self,
        case_token,
        ai_draft,
    ):
        if self.state == self.STATE_WATCHING:
            raise RuntimeError(
                "当前病例仍在监控"
            )

        self.case_token = str(
            case_token
        )

        self.ai_draft = (
            self.diff_engine.normalize_text(
                ai_draft
            )
        )

        self.baseline_snapshot = None
        self.latest_snapshot = None
        self.final_snapshot = None

        self.snapshots = []
        self._last_hash = None
        self.last_error = None

        self.state = self.STATE_ARMED

    def _read_text(self):
        if self.reader is None:
            raise RuntimeError(
                "PASS文本读取器尚未配置"
            )

        text = self.reader.read_text()

        return self.diff_engine.normalize_text(
            text
        )

    def capture_snapshot(
        self,
        stage,
    ):
        text = self._read_text()

        digest = self._hash(
            text
        )

        snapshot = ReportSnapshot(
            stage=str(stage),
            text=text,
            text_hash=digest,
            captured_at=_now_iso(),
        )

        with self._lock:
            self.latest_snapshot = snapshot

            if digest != self._last_hash:
                self.snapshots.append(
                    snapshot
                )
                self._last_hash = digest

        return snapshot

    def capture_baseline(self):
        if self.state not in {
            self.STATE_ARMED,
            self.STATE_WATCHING,
        }:
            raise RuntimeError(
                "当前病例未进入PASS学习流程"
            )

        snapshot = self.capture_snapshot(
            "pass_baseline"
        )

        self.baseline_snapshot = (
            snapshot
        )

        return snapshot

    def poll_once(self):
        if self.state not in {
            self.STATE_ARMED,
            self.STATE_WATCHING,
        }:
            raise RuntimeError(
                "PASS监控尚未启动"
            )

        return self.capture_snapshot(
            "editing"
        )

    def finalize(self):
        """
        病例完成/切换前调用。

        读取PASS最终报告，
        与AI原始草稿进行差异比较。
        """

        self.stop_monitoring()

        if self.case_token is None:
            raise RuntimeError(
                "当前没有已绑定病例"
            )

        final_snapshot = (
            self.capture_snapshot(
                "final_report"
            )
        )

        self.final_snapshot = (
            final_snapshot
        )

        diff = self.diff_engine.compare(
            self.ai_draft,
            final_snapshot.text,
        )

        self.state = (
            self.STATE_FINALIZED
        )

        return {
            "schema":
                "phoenix.pass_learning.v1",

            "case_token":
                self.case_token,

            "ai_draft":
                self.ai_draft,

            "pass_baseline":
                (
                    self.baseline_snapshot.text
                    if self.baseline_snapshot
                    else None
                ),

            "final_report":
                final_snapshot.text,

            "snapshot_count":
                len(self.snapshots),

            "diff":
                diff,

            "global_keylogging":
                False,

            "pass_database_access":
                False,
        }

    def reset(self):
        self.state = self.STATE_IDLE

        self.case_token = None
        self.ai_draft = ""

        self.baseline_snapshot = None
        self.latest_snapshot = None
        self.final_snapshot = None

        self.snapshots = []
        self._last_hash = None

        self.last_error = None

    def start_monitoring(self, interval_seconds=2.0):
        if self.state != self.STATE_ARMED:
            raise RuntimeError(
                "请先arm_case()"
            )

        self._stop_event = threading.Event()
        self._interval_seconds = max(
            1.0,
            float(interval_seconds)
        )

        self.state = self.STATE_WATCHING

        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )

        self._thread.start()

    def _monitor_loop(self):
        while not self._stop_event.wait(
            self._interval_seconds
        ):
            try:
                self.poll_once()

            except Exception as exc:
                self.last_error = str(exc)
                self.state = self.STATE_ERROR
                return

    def stop_monitoring(self):
        if not hasattr(
            self,
            "_stop_event"
        ):
            return

        self._stop_event.set()

        thread = getattr(
            self,
            "_thread",
            None
        )

        if (
            thread is not None
            and thread.is_alive()
        ):
            thread.join(timeout=3)

        if self.state == self.STATE_WATCHING:
            self.state = self.STATE_ARMED

    def clear_sensitive_memory(self):
        if (
            self.reader is not None
            and hasattr(self.reader, "clear_binding")
        ):
            try:
                self.reader.clear_binding()
            except Exception:
                pass

        self.case_token = None
        self.ai_draft = ""

        self.baseline_snapshot = None
        self.latest_snapshot = None
        self.final_snapshot = None

        self.snapshots.clear()
        self._last_hash = None
        self.last_error = None

        self.state = self.STATE_IDLE

    def finalize_ephemeral(self, consumer=None):
        """
        RAM-only病例结束流程。

        完整报告和Diff仅在consumer执行期间存在。
        无论成功或异常，最后都清空病例级内存。
        """
        result = None

        try:
            result = self.finalize()

            if consumer is not None:
                consumer(result)

            summary = result["diff"]["summary"]

            return {
                "change_count":
                    summary["change_count"],
                "add_count":
                    summary["add_count"],
                "delete_count":
                    summary["delete_count"],
                "replace_count":
                    summary["replace_count"],
                "exact_match":
                    result["diff"]["exact_match"],
            }

        finally:
            result = None
            self.clear_sensitive_memory()
