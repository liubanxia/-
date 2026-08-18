from __future__ import annotations

import hashlib
import os
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
        raise RuntimeError(
            "缺少PDF解析依赖。安装 PyMuPDF（推荐）或 pypdf。"
        ) from exc


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
