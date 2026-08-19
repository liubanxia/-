from __future__ import annotations

import html
import math
from pathlib import Path


LAYOUT_ORIGINAL_BILINGUAL = "original_bilingual"
LAYOUT_TEXT_BILINGUAL = "text_bilingual"
LAYOUT_TRANSLATED_ONLY = "translated_only"


class TranslationPDFBuilder:
    """Create readable translation PDFs while preserving the source page.

    The default layout keeps the exact original PDF page at the top and places
    the Chinese translation below it. The source page is embedded with
    ``show_pdf_page`` instead of being rasterized, so figures, vector graphics
    and original page layout remain visible for medical reference.
    """

    def __init__(self, source_pdf: Path, pages_root: Path, output_root: Path):
        self.source_pdf = Path(source_pdf)
        self.pages_root = Path(pages_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _translation_html(text: str, *, title: str = "中文译文") -> str:
        escaped = html.escape((text or "").strip()).replace("\n", "<br>")
        return (
            f"<h3>{html.escape(title)}</h3>"
            f"<div class='translation'>{escaped}</div>"
        )

    @staticmethod
    def _source_html(text: str) -> str:
        escaped = html.escape((text or "").strip()).replace("\n", "<br>")
        return f"<h3>English / 原文</h3><div class='source'>{escaped}</div>"

    @staticmethod
    def _css(font_size: float = 10.5) -> str:
        return (
            "body{font-family:sans-serif;line-height:1.45;}"
            f".translation{{font-size:{font_size}pt;}}"
            f".source{{font-size:{max(8.5, font_size - 1)}pt;}}"
            "h3{font-size:11pt;margin:0 0 8pt 0;}"
        )

    @staticmethod
    def _estimated_text_height(text: str, width: float, font_size: float = 10.5) -> float:
        text = text or ""
        chars_per_line = max(18, int(width / max(font_size * 0.58, 1)))
        logical_lines = 0
        for paragraph in text.splitlines() or [text]:
            logical_lines += max(1, math.ceil(max(len(paragraph), 1) / chars_per_line))
        return max(220.0, logical_lines * font_size * 1.72 + 90.0)

    @staticmethod
    def _insert_html(page, rect, markup: str, *, font_size: float = 10.5) -> None:
        try:
            page.insert_htmlbox(
                rect,
                markup,
                css=TranslationPDFBuilder._css(font_size),
                scale_low=0.45,
            )
        except TypeError:
            page.insert_htmlbox(
                rect,
                markup,
                css=TranslationPDFBuilder._css(font_size),
            )

    def _translated_text(self, page_number: int) -> str:
        page_file = self.pages_root / f"{int(page_number):06d}.txt"
        if not page_file.is_file():
            raise RuntimeError(f"无法生成PDF：第 {page_number} 页尚未完成翻译")
        return page_file.read_text(encoding="utf-8", errors="replace").strip()

    def _append_original_bilingual_page(self, out_doc, source_doc, page_index: int, page_number: int) -> None:
        source_page = source_doc[page_index]
        source_rect = source_page.rect
        width = float(source_rect.width)
        source_height = float(source_rect.height)
        translated = self._translated_text(page_number)
        translation_height = self._estimated_text_height(translated, width - 52)
        gap = 16.0
        output_height = source_height + gap + translation_height

        page = out_doc.new_page(width=width, height=output_height)
        page.show_pdf_page(
            page.rect.__class__(0, 0, width, source_height),
            source_doc,
            page_index,
        )
        translation_rect = page.rect.__class__(
            26,
            source_height + gap,
            width - 26,
            output_height - 22,
        )
        self._insert_html(
            page,
            translation_rect,
            self._translation_html(translated),
        )

    def _append_text_bilingual_page(self, out_doc, source_doc, page_index: int, page_number: int) -> None:
        source_page = source_doc[page_index]
        width = max(595.0, float(source_page.rect.width))
        source_text = source_page.get_text("text").strip()
        translated = self._translated_text(page_number)
        source_height = self._estimated_text_height(source_text, width - 52, font_size=9.5)
        translation_height = self._estimated_text_height(translated, width - 52)
        page_height = source_height + translation_height + 80
        page = out_doc.new_page(width=width, height=page_height)
        source_rect = page.rect.__class__(26, 24, width - 26, 24 + source_height)
        translation_rect = page.rect.__class__(
            26,
            42 + source_height,
            width - 26,
            page_height - 22,
        )
        self._insert_html(page, source_rect, self._source_html(source_text), font_size=9.5)
        self._insert_html(page, translation_rect, self._translation_html(translated))

    def _append_translated_only_page(self, out_doc, source_doc, page_index: int, page_number: int) -> None:
        source_page = source_doc[page_index]
        width = max(595.0, float(source_page.rect.width))
        translated = self._translated_text(page_number)
        translation_height = self._estimated_text_height(translated, width - 52)
        page_height = translation_height + 56
        page = out_doc.new_page(width=width, height=page_height)
        rect = page.rect.__class__(26, 24, width - 26, page_height - 20)
        self._insert_html(
            page,
            rect,
            self._translation_html(translated, title=f"第 {page_number} 页 · 中文译文"),
        )

    def build(
        self,
        *,
        start_page: int,
        total_pages: int,
        layout: str = LAYOUT_ORIGINAL_BILINGUAL,
        part_pages: int = 50,
        progress=None,
    ) -> tuple[Path, tuple[Path, ...]]:
        import fitz

        layout = layout if layout in {
            LAYOUT_ORIGINAL_BILINGUAL,
            LAYOUT_TEXT_BILINGUAL,
            LAYOUT_TRANSLATED_ONLY,
        } else LAYOUT_ORIGINAL_BILINGUAL
        part_pages = max(1, int(part_pages))
        start_page = max(1, int(start_page))
        total_pages = max(start_page, int(total_pages))

        if layout == LAYOUT_ORIGINAL_BILINGUAL:
            complete_name = "完整双语译本_原页在上中文在下.pdf"
        elif layout == LAYOUT_TEXT_BILINGUAL:
            complete_name = "完整双语译本_英文在上中文在下.pdf"
        else:
            complete_name = "完整中文译本.pdf"

        complete_path = self.output_root / complete_name
        parts_root = self.output_root / "PDF分册"
        parts_root.mkdir(parents=True, exist_ok=True)
        for old in parts_root.glob("第*.pdf"):
            old.unlink(missing_ok=True)

        source_doc = fitz.open(self.source_pdf)
        out_doc = fitz.open()
        try:
            selected_total = total_pages - start_page + 1
            for offset, page_number in enumerate(range(start_page, total_pages + 1), start=1):
                source_index = page_number - 1
                if source_index < 0 or source_index >= source_doc.page_count:
                    raise RuntimeError(f"源PDF不存在第 {page_number} 页")
                if layout == LAYOUT_ORIGINAL_BILINGUAL:
                    self._append_original_bilingual_page(out_doc, source_doc, source_index, page_number)
                elif layout == LAYOUT_TEXT_BILINGUAL:
                    self._append_text_bilingual_page(out_doc, source_doc, source_index, page_number)
                else:
                    self._append_translated_only_page(out_doc, source_doc, source_index, page_number)
                if progress:
                    progress(
                        offset,
                        selected_total,
                        f"正在生成PDF：第 {page_number}/{total_pages} 页",
                    )

            if complete_path.exists():
                complete_path.unlink()
            out_doc.save(complete_path, garbage=3, deflate=True)
        finally:
            out_doc.close()
            source_doc.close()

        complete = fitz.open(complete_path)
        part_paths: list[Path] = []
        try:
            total_output_pages = complete.page_count
            for part_index, start_index in enumerate(range(0, total_output_pages, part_pages), start=1):
                end_index = min(total_output_pages - 1, start_index + part_pages - 1)
                first_source_page = start_page + start_index
                last_source_page = start_page + end_index
                part_path = parts_root / (
                    f"第{part_index:03d}册_{first_source_page:04d}-{last_source_page:04d}.pdf"
                )
                part_doc = fitz.open()
                try:
                    part_doc.insert_pdf(complete, from_page=start_index, to_page=end_index)
                    part_doc.save(part_path, garbage=3, deflate=True)
                finally:
                    part_doc.close()
                part_paths.append(part_path)
        finally:
            complete.close()

        return complete_path, tuple(part_paths)
