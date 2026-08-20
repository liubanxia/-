from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunker import chunk_text
from .config import WorkbenchPaths
from .db import KnowledgeDB
from .pdf_assets import PDFAssetStore
from .pdf_parser import export_docling_structure, iter_pdf_pages_with_ocr, pdf_page_count


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class IngestResult:
    document_id: int
    pages_total: int
    pages_indexed: int
    chunks_total: int
    empty_pages: int
    copied_to_library: Path
    warning: str = ""
    image_count: int = 0


class LibraryIngestor:
    def __init__(self, db: KnowledgeDB, paths: WorkbenchPaths):
        self.db = db
        self.paths = paths
        self.assets = PDFAssetStore(paths.runtime_root)

    def _library_copy(self, source: Path) -> Path:
        source = Path(source).resolve()
        target = self.paths.source_root / source.name
        if target.resolve() == source:
            return source
        if target.exists():
            try:
                if sha256_file(target) == sha256_file(source):
                    return target
            except OSError:
                pass
            stem, suffix = source.stem, source.suffix
            counter = 2
            while target.exists():
                target = self.paths.source_root / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(source, target)
        return target

    def _existing_document(self, pdf_path: Path):
        target = str(Path(pdf_path).resolve())
        for row in self.db.list_documents():
            if str(row["path"]) == target:
                return row
        return None

    def ingest_pdf(
        self,
        source: Path,
        *,
        copy_into_library: bool = True,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
        extract_images: bool = True,
        refresh_images: bool = False,
    ) -> IngestResult:
        source = Path(source).resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".pdf":
            raise ValueError(f"不是PDF文件: {source}")

        pdf_path = self._library_copy(source) if copy_into_library else source
        digest = sha256_file(pdf_path)
        previous = self._existing_document(pdf_path)
        retry_ocr = bool(
            previous is not None
            and str(previous["status"]) == "ocr_required"
            and str(previous["sha256"]) == digest
        )

        total_pages = pdf_page_count(pdf_path)
        document_id = self.db.upsert_document(pdf_path, digest, pdf_path.stem, total_pages)

        empty_pages = 0
        processed_pages = 0
        ocr_recovered_pages = 0
        ocr_unresolved_pages = 0
        ocr_errors: list[str] = []

        for extracted in iter_pdf_pages_with_ocr(pdf_path):
            page_number = int(extracted.page)
            if stop_event is not None and stop_event.is_set():
                self.db.mark_document(document_id, "paused", "任务被暂停，可继续导入")
                break
            if self.db.page_is_indexed(document_id, page_number) and not retry_ocr:
                processed_pages += 1
                if progress:
                    progress(processed_pages, total_pages, f"跳过已完成第 {page_number} 页")
                continue

            text = (extracted.text or "").strip()
            if extracted.ocr_used:
                ocr_recovered_pages += 1
            if extracted.ocr_attempted and not extracted.ocr_used and len(text) < 24:
                ocr_unresolved_pages += 1
                if extracted.ocr_error and len(ocr_errors) < 3:
                    ocr_errors.append(f"第{page_number}页 {extracted.ocr_error}")

            chunks = chunk_text(text)
            if not chunks:
                empty_pages += 1
                if extracted.ocr_attempted and not extracted.ocr_used:
                    chunks = ["[OCR_REQUIRED：本页疑似扫描页；本机OCR未能提取可检索文字]"]
                else:
                    chunks = ["[本页未提取到可检索文本；可能为空白页或图片页]"]

            self.db.replace_page_chunks(document_id, page_number, chunks)
            processed_pages += 1
            if progress:
                if extracted.ocr_used:
                    message = f"OCR恢复并索引第 {page_number}/{total_pages} 页"
                elif extracted.ocr_attempted and not extracted.ocr_used:
                    message = f"第 {page_number}/{total_pages} 页需要OCR资源，已记录状态"
                else:
                    message = f"已索引第 {page_number}/{total_pages} 页"
                progress(processed_pages, total_pages, message)

        row = self.db.get_document(document_id)
        indexed_pages = int(row["indexed_pages"]) if row else processed_pages

        warning_parts: list[str] = []
        status = "indexed"
        if ocr_recovered_pages:
            warning_parts.append(f"本地OCR已恢复 {ocr_recovered_pages} 页扫描内容")
        if ocr_unresolved_pages:
            status = "ocr_required"
            warning_parts.append(
                f"OCR_REQUIRED: {ocr_unresolved_pages}/{total_pages} 页疑似扫描页仍未获得可靠文字；"
                "请在本机准备Tesseract语言数据（默认 eng+chi_sim）后重新导入。"
            )
            if ocr_errors:
                warning_parts.append("；".join(ocr_errors))
        elif total_pages and empty_pages / total_pages >= 0.2:
            status = "indexed_with_warning"
            warning_parts.append(f"{empty_pages}/{total_pages} 页没有可检索文字；可能为空白页或纯图片页。")

        warning = "；".join(part for part in warning_parts if part)
        if stop_event is None or not stop_event.is_set():
            self.db.mark_document(document_id, status, warning)

        image_count = 0
        if extract_images and (stop_event is None or not stop_event.is_set()):
            try:
                if progress:
                    progress(total_pages, total_pages, "正在提取PDF内图片并建立页码关联……")
                manifest = self.assets.extract(pdf_path, force=refresh_images)
                image_count = int(manifest.get("image_count", 0) or 0)
                if progress:
                    progress(total_pages, total_pages, f"图片资料已保存：{image_count} 张，可用于资料整理和翻译富媒体输出")
            except Exception as exc:
                warning = (warning + "；" if warning else "") + f"图片提取失败: {exc}"
                self.db.mark_document(document_id, status, warning)

        try:
            export_docling_structure(pdf_path, self.paths.structure_root / f"D{document_id}_{pdf_path.stem}.md")
        except Exception as exc:
            warning = (warning + "；" if warning else "") + f"Docling结构导出失败: {exc}"
            self.db.mark_document(document_id, status, warning)

        return IngestResult(
            document_id=document_id,
            pages_total=total_pages,
            pages_indexed=indexed_pages,
            chunks_total=self.db.count_chunks(),
            empty_pages=empty_pages,
            copied_to_library=pdf_path,
            warning=warning,
            image_count=image_count,
        )
