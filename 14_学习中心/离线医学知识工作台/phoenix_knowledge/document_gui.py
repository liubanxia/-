from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox, QPushButton, QTabWidget


_INSTALLED = False


def install(gui_module) -> None:
    """Upgrade the existing PDF-centric GUI to the unified document workbench."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    cls = gui_module.WorkbenchWindow
    ingest_worker_cls = gui_module.IngestWorker

    original_init = cls.__init__
    original_library_tab = cls._library_tab
    original_qa_tab = cls._qa_tab
    original_organize_tab = cls._organize_tab
    original_organize_done = cls._organize_done
    original_use_for_translation = cls.use_selected_book_for_translation
    original_failed = cls._failed

    def _ingest_worker_run(self):
        total_files = max(1, len(self.files))
        messages: list[str] = []
        failures: list[str] = []
        successes = 0

        for file_index, filename in enumerate(self.files, start=1):
            def callback(done, total, message):
                base = int(((file_index - 1) / total_files) * 100)
                span = 100 / total_files
                pct = base + int((done / max(total, 1)) * span)
                self.progress.emit(pct, 100, f"[{file_index}/{total_files}] {message}")

            try:
                result = self.workbench.ingest(Path(filename), progress=callback)
                successes += 1
                suffix = f" | {result.warning}" if result.warning else ""
                messages.append(
                    f"✓ {Path(filename).name}: {result.pages_indexed}/{result.pages_total} 单元{suffix}"
                )
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                failures.append(f"✗ {Path(filename).name}: {detail}")
                self.progress.emit(
                    int(file_index / total_files * 100),
                    100,
                    f"[{file_index}/{total_files}] 当前文件失败，继续下一份资料",
                )

        summary = [f"批量导入完成：成功 {successes}，失败 {len(failures)}。", *messages, *failures]
        if successes:
            self.completed.emit("\n".join(summary))
        else:
            self.failed.emit("\n".join(summary))

    ingest_worker_cls.run = _ingest_worker_run

    def _library_tab(self):
        widget = original_library_tab(self)
        labels = widget.findChildren(QLabel)
        for label in labels:
            text = label.text()
            if "PDF内容只在本机SSD解析和索引" in text:
                label.setText(
                    "PDF / PPT / PPTX / DOCX / TXT / Markdown 均只在本机SSD解析和索引；"
                    "扫描PDF会自动尝试本地OCR，失败会明确标记OCR_REQUIRED；"
                    "PPT/PPTX保留幻灯片编号、表格文字、备注和关联图片。"
                )
        for button in widget.findChildren(QPushButton):
            if button.text() == "导入PDF":
                button.setText("导入资料")
            elif button.text() == "选中书→整本翻译":
                button.setText("选中PDF→整本翻译")
        return widget

    def _qa_tab(self):
        widget = original_qa_tab(self)
        for label in widget.findChildren(QLabel):
            if "只根据已导入PDF回答" in label.text():
                label.setText(
                    "只根据已导入医学资料回答；PDF保留页码，PPT/PPTX保留幻灯片编号，"
                    "所有结论继续保留来源编号。"
                )
        for button in widget.findChildren(QPushButton):
            if button.text() == "根据PDF回答":
                button.setText("根据资料回答")
        return widget

    def _organize_tab(self):
        widget = original_organize_tab(self)
        if hasattr(self, "multi_book_info"):
            self.multi_book_info.setText(
                "默认从全部 PDF / PPT / PPTX / DOCX / TXT / Markdown 中跨资料检索、去重和合并；"
                "整理完成自动输出带图 PDF + DOCX + Markdown + TXT。"
            )
        for button in widget.findChildren(QPushButton):
            if button.text() == "整理全部书籍":
                button.setText("整理全部资料")
            elif button.text() == "另存TXT":
                try:
                    button.clicked.disconnect()
                except Exception:
                    pass
                button.setText("导出 PDF/DOCX/MD/TXT")
                button.clicked.connect(self.export_organize_bundle)
        return widget

    def _init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tabs = self.centralWidget()
        if isinstance(tabs, QTabWidget):
            names = {0: "医学资料库", 1: "资料问答", 2: "多资料知识整理", 4: "笔记整理"}
            for index, name in names.items():
                if index < tabs.count():
                    tabs.setTabText(index, name)

    def add_documents(self):
        if self._busy():
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择医学资料",
            str(self.workbench.paths.source_root),
            (
                "医学资料 (*.pdf *.ppt *.pptx *.docx *.txt *.md);;"
                "PDF (*.pdf);;PowerPoint (*.ppt *.pptx);;Word (*.docx);;"
                "文本 (*.txt *.md)"
            ),
        )
        if not files:
            return
        self.worker = ingest_worker_cls(self.workbench, files)
        self.worker.progress.connect(self._ingest_progress)
        self.worker.completed.connect(self._ingest_done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def export_organize_bundle(self):
        source = getattr(self, "last_organize_path", None)
        title = (self.topic_title.text().strip() if hasattr(self, "topic_title") else "") or "多资料知识整理"
        try:
            if source is not None and Path(source).is_file():
                bundle = self.workbench.exporter.export_path(Path(source), title=title)
            else:
                bundle = self.workbench.exporter.export_text(self.organize_result.toPlainText(), title=title)
            self.workbench.last_export_bundle = bundle
            self.organize_status.setText("多格式输出完成：带图PDF / DOCX / Markdown / TXT")
            QMessageBox.information(
                self,
                "Phoenix 多格式输出",
                "已生成：\n" + "\n".join(str(path) for path in bundle.output_paths),
            )
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", f"{type(exc).__name__}: {exc}")

    def _organize_done(self, output: str):
        result = original_organize_done(self, output)
        bundle = getattr(self.workbench, "last_export_bundle", None)
        if bundle is not None:
            self.organize_status.setText(
                f"整理完成；已同时生成带图 PDF / DOCX / Markdown / TXT：{bundle.output_dir}"
            )
        return result

    def _failed(self, error: str):
        cleaned = str(error)
        for prefix in ("LegacyPPTConversionError: ", "RuntimeError: "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        return original_failed(self, cleaned)

    def use_selected_book_for_translation(self):
        item = self.library_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(str(path)).suffix.lower() != ".pdf":
            QMessageBox.information(
                self,
                "整本翻译目前仅支持PDF",
                "PPT / PPTX / DOCX 已可进入知识库和多资料整理；整本双语版式翻译仍保留为 PDF 专用功能。",
            )
            return
        return original_use_for_translation(self)

    cls.__init__ = _init
    cls._library_tab = _library_tab
    cls._qa_tab = _qa_tab
    cls._organize_tab = _organize_tab
    cls.add_pdfs = add_documents
    cls.add_documents = add_documents
    cls.export_organize_bundle = export_organize_bundle
    cls._organize_done = _organize_done
    cls._failed = _failed
    cls.use_selected_book_for_translation = use_selected_book_for_translation
