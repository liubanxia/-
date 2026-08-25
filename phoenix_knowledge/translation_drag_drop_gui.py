from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QTabWidget

_INSTALLED = False
_TRANSLATION_SUFFIXES = {".pdf", ".pptx", ".docx"}
_LIBRARY_SUFFIXES = {".pdf"}


def _local_paths(event) -> list[Path]:
    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return []
    paths: list[Path] = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.is_file():
            paths.append(path)
    return paths


def _matching(paths: list[Path], suffixes: set[str]) -> list[Path]:
    return [path for path in paths if path.suffix.lower() in suffixes]


class _DropFilter(QObject):
    def __init__(self, window, gui_module, mode: str):
        super().__init__(window)
        self.window = window
        self.gui_module = gui_module
        self.mode = mode

    def eventFilter(self, watched, event):
        event_type = event.type()
        suffixes = _LIBRARY_SUFFIXES if self.mode == "library" else _TRANSLATION_SUFFIXES

        if event_type == QEvent.Type.DragEnter:
            if _matching(_local_paths(event), suffixes):
                event.acceptProposedAction()
                return True
            return False

        if event_type != QEvent.Type.Drop:
            return False

        paths = _matching(_local_paths(event), suffixes)
        if not paths:
            return False

        if self.mode == "library":
            self._drop_library(paths)
        else:
            self._drop_translation(paths)
        event.acceptProposedAction()
        return True

    def _drop_translation(self, paths: list[Path]) -> None:
        path = paths[0]
        self.window.translation_path.setText(str(path))
        self.window.translation_start_page.setValue(1)
        self.window.translation_status.setText(
            f"已接收拖入文档：{path.name}；点击“开始/继续同格式翻译”。"
        )

        tabs = self.window.findChild(QTabWidget)
        if tabs is not None:
            for index in range(tabs.count()):
                if "翻译" in tabs.tabText(index):
                    tabs.setCurrentIndex(index)
                    break

        self.window.statusBar().showMessage(
            f"翻译文档已载入：{path.name}",
            8000,
        )

    def _drop_library(self, paths: list[Path]) -> None:
        if self.window._busy():
            self.window.ingest_label.setText("当前有任务运行，稍后再拖入PDF。")
            return

        files = [str(path) for path in paths]
        self.window.ingest_progress.setValue(0)
        self.window.ingest_label.setText(
            f"已接收 {len(files)} 个PDF，正在导入资料库……"
        )
        self.window.worker = self.gui_module.IngestWorker(
            self.window.workbench,
            files,
        )
        self.window.worker.progress.connect(self.window._ingest_progress)
        self.window.worker.completed.connect(self.window._ingest_done)
        self.window.worker.failed.connect(self.window._failed)
        self.window.worker.start()


def install(gui_module) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    cls = gui_module.WorkbenchWindow
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        self._phoenix_translation_drop_filter = _DropFilter(
            self,
            gui_module,
            "translation",
        )
        self.translation_path.setAcceptDrops(True)
        self.translation_path.installEventFilter(
            self._phoenix_translation_drop_filter
        )
        self.translation_path.setToolTip(
            "可把 PDF / PPTX / DOCX 直接拖到这里进入同格式翻译。"
        )

        self._phoenix_library_drop_filter = _DropFilter(
            self,
            gui_module,
            "library",
        )
        self.library_list.setAcceptDrops(True)
        self.library_list.installEventFilter(self._phoenix_library_drop_filter)
        self.library_list.setToolTip(
            "可把一个或多个 PDF 直接拖到这里导入知识资料库。"
        )

    cls.__init__ = __init__
    cls.__phoenix_translation_drag_drop__ = True
    _INSTALLED = True
