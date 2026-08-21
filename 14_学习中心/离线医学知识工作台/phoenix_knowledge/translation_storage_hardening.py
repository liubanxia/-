from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

_INSTALLED = False


def _human_size(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}GB"


def _atomic_pdf_save(doc, path: Path, *, compact: bool = True) -> None:
    """Write one PDF atomically without the old expensive garbage=3 pass."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    temp.unlink(missing_ok=True)
    kwargs = {
        "garbage": 0,
        "deflate": True,
    }
    if compact:
        kwargs.update(
            {
                "deflate_images": True,
                "deflate_fonts": True,
                "use_objstms": 1,
            }
        )
    try:
        try:
            doc.save(str(temp), **kwargs)
        except TypeError:
            doc.save(str(temp), garbage=0, deflate=True)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _remove_old_parts(parts_root: Path) -> None:
    if not parts_root.exists():
        return
    for old in parts_root.glob("第*.pdf"):
        old.unlink(missing_ok=True)
    try:
        if not any(parts_root.iterdir()):
            parts_root.rmdir()
    except OSError:
        pass


def _insert_compact_translation_text(builder, page, rect, translated: str) -> None:
    """Add Chinese text with the lightest available PDF text path.

    ``china-s`` is PyMuPDF's built-in Simplified Chinese font alias. It avoids
    repeatedly carrying a large external CJK font through hundreds of textbook
    pages. Builds that cannot use the alias transparently fall back to the
    existing HTML text path. A genuine overflow is still treated as a hard
    output-integrity error.
    """

    import fitz

    payload = "中文译文\n" + (translated or "").strip()
    try:
        spare = page.insert_textbox(
            rect,
            payload,
            fontsize=10.5,
            fontname="china-s",
            align=fitz.TEXT_ALIGN_LEFT,
        )
    except Exception:
        builder._insert_html(
            page,
            rect,
            builder._translation_html(translated),
        )
        return

    if float(spare) < -0.01:
        raise RuntimeError(
            "译文文本框空间不足，Phoenix已阻止输出被截断的成品。"
        )


def _extend_copied_page_with_translation(builder, page, translated: str) -> None:
    """Extend a directly copied source page and append only a text layer."""

    import fitz

    old_rect = page.rect
    width = float(old_rect.width)
    old_height = float(old_rect.height)
    translation_height = builder._estimated_text_height(
        translated,
        width - 52,
    )
    gap = 16.0
    extra = gap + translation_height

    media = page.mediabox
    page.set_mediabox(
        fitz.Rect(
            float(media.x0),
            float(media.y0),
            float(media.x1),
            float(media.y1) + extra,
        )
    )
    new_rect = page.rect
    translation_rect = fitz.Rect(
        26.0,
        old_height + gap,
        max(27.0, float(new_rect.width) - 26.0),
        max(old_height + gap + 20.0, float(new_rect.height) - 20.0),
    )
    _insert_compact_translation_text(
        builder,
        page,
        translation_rect,
        translated,
    )


def _append_original_page_compact(
    builder,
    out_doc,
    source_doc,
    source_index: int,
    page_number: int,
    *,
    final: bool,
) -> None:
    import fitz

    source_page = source_doc[source_index]
    translated = builder._translated_text(page_number)

    if int(getattr(source_page, "rotation", 0) or 0) % 360 == 0:
        # Keeping final=0 between calls preserves PyMuPDF's graft map, so
        # source fonts and images shared across pages remain shared instead of
        # being copied repeatedly into the translated book.
        out_doc.insert_pdf(
            source_doc,
            from_page=source_index,
            to_page=source_index,
            links=True,
            annots=True,
            final=1 if final else 0,
        )
        page = out_doc[out_doc.page_count - 1]
        _extend_copied_page_with_translation(builder, page, translated)
        return

    # Conservative fallback for unusual rotated pages.
    source_rect = source_page.rect
    width = float(source_rect.width)
    source_height = float(source_rect.height)
    translation_height = builder._estimated_text_height(translated, width - 52)
    gap = 16.0
    output_height = source_height + gap + translation_height
    page = out_doc.new_page(width=width, height=output_height)
    page.show_pdf_page(
        fitz.Rect(0, 0, width, source_height),
        source_doc,
        source_index,
    )
    builder._insert_html(
        page,
        fitz.Rect(
            26,
            source_height + gap,
            width - 26,
            output_height - 22,
        ),
        builder._translation_html(translated),
    )


def _compact_translation_pdf_build(
    self,
    *,
    start_page: int,
    total_pages: int,
    layout: str,
    part_pages: int = 0,
    progress=None,
):
    import fitz

    from .translation_pdf import (
        LAYOUT_ORIGINAL_BILINGUAL,
        LAYOUT_TEXT_BILINGUAL,
        LAYOUT_TRANSLATED_ONLY,
    )

    valid_layouts = {
        LAYOUT_ORIGINAL_BILINGUAL,
        LAYOUT_TEXT_BILINGUAL,
        LAYOUT_TRANSLATED_ONLY,
    }
    layout = layout if layout in valid_layouts else LAYOUT_ORIGINAL_BILINGUAL
    part_pages = max(0, int(part_pages or 0))
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
    _remove_old_parts(parts_root)

    selected_total = total_pages - start_page + 1
    source_doc = fitz.open(self.source_pdf)
    out_doc = fitz.open()
    try:
        direct_pages = [
            page_number
            for page_number in range(start_page, total_pages + 1)
            if int(getattr(source_doc[page_number - 1], "rotation", 0) or 0) % 360 == 0
        ]
        last_direct = direct_pages[-1] if direct_pages else None

        for offset, page_number in enumerate(
            range(start_page, total_pages + 1),
            start=1,
        ):
            source_index = page_number - 1
            if source_index < 0 or source_index >= source_doc.page_count:
                raise RuntimeError(f"源PDF不存在第 {page_number} 页")

            if layout == LAYOUT_ORIGINAL_BILINGUAL:
                _append_original_page_compact(
                    self,
                    out_doc,
                    source_doc,
                    source_index,
                    page_number,
                    final=(page_number == last_direct),
                )
            elif layout == LAYOUT_TEXT_BILINGUAL:
                self._append_text_bilingual_page(
                    out_doc,
                    source_doc,
                    source_index,
                    page_number,
                )
            else:
                self._append_translated_only_page(
                    out_doc,
                    source_doc,
                    source_index,
                    page_number,
                )

            if progress:
                progress(
                    offset,
                    selected_total,
                    f"正在生成紧凑译本：第 {page_number}/{total_pages} 页",
                )

        if progress:
            progress(
                selected_total,
                selected_total,
                "页面完成，正在一次压缩写入完整PDF；不会重新渲染原页。",
            )
        _atomic_pdf_save(out_doc, complete_path, compact=True)
    finally:
        out_doc.close()
        source_doc.close()

    source_size = int(Path(self.source_pdf).stat().st_size)
    output_size = int(complete_path.stat().st_size)
    ratio = (output_size / source_size) if source_size > 0 else 0.0

    part_paths: list[Path] = []
    if part_pages > 0:
        parts_root.mkdir(parents=True, exist_ok=True)
        complete = fitz.open(complete_path)
        try:
            total_output_pages = complete.page_count
            part_total = max(
                1,
                (total_output_pages + part_pages - 1) // part_pages,
            )
            for part_index, start_index in enumerate(
                range(0, total_output_pages, part_pages),
                start=1,
            ):
                end_index = min(
                    total_output_pages - 1,
                    start_index + part_pages - 1,
                )
                first_source_page = start_page + start_index
                last_source_page = start_page + end_index
                part_path = parts_root / (
                    f"第{part_index:03d}册_"
                    f"{first_source_page:04d}-{last_source_page:04d}.pdf"
                )
                part_doc = fitz.open()
                try:
                    part_doc.insert_pdf(
                        complete,
                        from_page=start_index,
                        to_page=end_index,
                    )
                    _atomic_pdf_save(part_doc, part_path, compact=True)
                finally:
                    part_doc.close()
                part_paths.append(part_path)
                if progress:
                    progress(
                        selected_total,
                        selected_total,
                        f"正在生成可选分册 {part_index}/{part_total}",
                    )
        finally:
            complete.close()

    if progress:
        size_message = (
            f"完整PDF {_human_size(output_size)}；原PDF {_human_size(source_size)}"
        )
        if source_size:
            size_message += f"；体积比 {ratio:.2f}×"
        if source_size >= 8 * 1024 * 1024 and ratio > 1.50:
            size_message += "；⚠ 体积超过发布目标，请检查源PDF特殊字体/旋转页/附件"
        if part_paths:
            extra = sum(
                int(path.stat().st_size)
                for path in part_paths
                if path.is_file()
            )
            size_message += (
                f"；另生成 {len(part_paths)} 个分册，额外占用 {_human_size(extra)}"
            )
        else:
            size_message += "；未生成重复分册"
        progress(selected_total, selected_total, size_message)

    return complete_path, tuple(part_paths)


def _rewrite_no_split_progress(progress):
    if progress is None:
        return None

    def callback(done, total, message):
        text = str(message)
        text = text.replace(
            "翻译完成，正在生成整书PDF与分册PDF……",
            "翻译完成，正在生成紧凑完整PDF……",
        )
        text = text.replace(
            "整本翻译与PDF分册已完成。",
            "整本翻译与紧凑PDF已完成。",
        )
        progress(done, total, text)

    return callback


def _repair_no_split_checkpoint(translator, pdf_path: Path, target_language: str) -> None:
    """Keep persisted metadata truthful despite the legacy internal clamp."""

    try:
        from .pdf_parser import sha256_file

        source = Path(pdf_path).resolve()
        digest = sha256_file(source)
        _, _, _, checkpoint, _ = translator._book_paths(
            source,
            digest,
            target_language,
        )
        payload = translator._read_json(checkpoint)
        if payload and int(payload.get("part_pages", 0) or 0) != 0:
            payload["part_pages"] = 0
            translator._write_json(checkpoint, payload)
    except Exception:
        pass


def install() -> None:
    """Make translated PDFs storage-efficient by default."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .translation_pdf import TranslationPDFBuilder
    from .translator import PDFTranslator

    TranslationPDFBuilder.build = _compact_translation_pdf_build

    original_build_deliverables = PDFTranslator._build_deliverables

    def _build_deliverables(self, *args, **kwargs):
        if bool(getattr(self, "_phoenix_no_split", False)):
            kwargs["part_pages"] = 0
        return original_build_deliverables(self, *args, **kwargs)

    PDFTranslator._build_deliverables = _build_deliverables

    original_translate_book = PDFTranslator.translate_book

    def translate_book(self, pdf_path: Path, **kwargs):
        requested = kwargs.get("part_pages", None)
        no_split = requested is None
        if requested is not None:
            try:
                no_split = int(requested) <= 0
            except (TypeError, ValueError):
                no_split = True

        target_language = str(kwargs.get("target_language", "中文"))
        original_progress = kwargs.get("progress")
        if no_split:
            # translator.py historically clamps part_pages to >=1. Feed a
            # harmless compatibility value internally; the actual deliverable
            # layer receives zero through the instance flag above.
            kwargs["part_pages"] = 1
            kwargs["progress"] = _rewrite_no_split_progress(original_progress)

        self._phoenix_no_split = bool(no_split)
        try:
            result = original_translate_book(self, pdf_path, **kwargs)
            if no_split and hasattr(result, "part_pages"):
                try:
                    result = replace(result, part_pages=0)
                except Exception:
                    pass
            return result
        finally:
            if no_split:
                _repair_no_split_checkpoint(
                    self,
                    Path(pdf_path),
                    target_language,
                )
            self._phoenix_no_split = False

    PDFTranslator.translate_book = translate_book
