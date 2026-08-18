"""
Phoenix legacy report-learning compatibility layer.

旧的 PASS/UIA 医生操作监控已经停用。
Phoenix 当前不监控医生如何修改报告，也不依赖 pywinauto。

保留兼容类只是为了让旧 main_window 导入能够继续工作。
"""


def _noop(*args, **kwargs):
    return None


class _DisabledLegacyComponent:
    enabled = False
    running = False

    def __init__(self, *args, **kwargs):
        pass

    def start(self, *args, **kwargs):
        return False

    def stop(self, *args, **kwargs):
        return None

    def close(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        return None

    def read(self, *args, **kwargs):
        return ""

    def read_text(self, *args, **kwargs):
        return ""

    def get_text(self, *args, **kwargs):
        return ""

    def __getattr__(self, name):
        return _noop


class PassReportMonitor(_DisabledLegacyComponent):
    pass


class FocusedUIATextReader(_DisabledLegacyComponent):
    pass


__all__ = [
    "PassReportMonitor",
    "FocusedUIATextReader",
]
