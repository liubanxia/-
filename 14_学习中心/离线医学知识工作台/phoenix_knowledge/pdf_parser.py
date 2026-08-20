from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PDFPageExtraction:
    page: int
    text: str
    ocr_attempted: bool = False
    ocr_used: bool = False
    ocr_error: str = ""


def _fitz_reader(path: Path):
    import fitz

    document = fitz.open(str(path))
    try:
        for index in range(document.page_count):
            page = document.load_page(index)
            yield index + 1, page.get_text("text") or ""
    finally:
        document.close()


def _pypdf_reader(path: Path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for index, page in enumerate(reader.pages):
        yield index + 1, page.extract_text() or ""


def iter_pdf_pages(path: Path) -> Iterator[tuple[int, str]]:
    """Compatibility text iterator used by translation and older callers."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持PDF: {path}")
    try:
        yield from _fitz_reader(path)
        return
    except ImportError:
        pass
    try:
        yield from _pypdf_reader(path)
        return
    except ImportError as exc:
        raise RuntimeError("缺少PDF解析依赖。安装 PyMuPDF（推荐）或 pypdf。") from exc


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _ocr_page_text(page) -> str:
    """Run PyMuPDF/Tesseract OCR using only local resources."""
    language = os.environ.get("PHOENIX_OCR_LANGUAGES", "eng+chi_sim").strip() or "eng"
    try:
        dpi = int(os.environ.get("PHOENIX_OCR_DPI", "200") or 200)
    except ValueError:
        dpi = 200
    dpi = max(120, min(dpi, 300))
    textpage = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
    return (page.get_text("text", textpage=textpage) or "").strip()


def iter_pdf_pages_with_ocr(path: Path, *, min_native_chars: int = 24) -> Iterator[PDFPageExtraction]:
    """Extract native text and OCR likely scanned pages when local OCR exists.

    Missing OCR resources are returned as metadata so ingest can mark the
    document OCR_REQUIRED instead of reporting a false successful import.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持PDF: {path}")
    try:
        import fitz
    except ImportError:
        for page_number, text in _pypdf_reader(path):
            yield PDFPageExtraction(page=page_number, text=text)
        return

    disable_ocr = _flag("PHOENIX_OCR_DISABLE", default=False)
    min_native_chars = max(0, int(min_native_chars))
    document = fitz.open(str(path))
    try:
        for index in range(document.page_count):
            page = document.load_page(index)
            native = (page.get_text("text") or "").strip()
            if disable_ocr or len(native) >= min_native_chars:
                yield PDFPageExtraction(page=index + 1, text=native)
                continue
            try:
                has_images = bool(page.get_images(full=True))
            except Exception:
                has_images = False
            if not has_images:
                yield PDFPageExtraction(page=index + 1, text=native)
                continue
            try:
                ocr_text = _ocr_page_text(page)
                chosen = ocr_text if len(ocr_text) > len(native) else native
                yield PDFPageExtraction(
                    page=index + 1,
                    text=chosen,
                    ocr_attempted=True,
                    ocr_used=bool(ocr_text and len(ocr_text) > len(native)),
                )
            except Exception as exc:
                yield PDFPageExtraction(
                    page=index + 1,
                    text=native,
                    ocr_attempted=True,
                    ocr_used=False,
                    ocr_error=f"{type(exc).__name__}: {exc}",
                )
    finally:
        document.close()


def pdf_page_count(path: Path) -> int:
    try:
        import fitz
        document = fitz.open(str(path))
        try:
            return int(document.page_count)
        finally:
            document.close()
    except ImportError:
        from pypdf import PdfReader
        return len(PdfReader(str(path)).pages)


def export_docling_structure(path: Path, output_path: Path) -> bool:
    """Optional structural export; never enables network on its own."""
    if os.environ.get("PHOENIX_ENABLE_DOCLING", "0") != "1":
        return False
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = DocumentConverter().convert(str(path))
    output_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    return True
