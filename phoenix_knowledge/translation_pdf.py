from __future__ import annotations

import html
import json
import math
import os
import re
from pathlib import Path


LAYOUT_ORIGINAL_BILINGUAL = "original_bilingual"
LAYOUT_TEXT_BILINGUAL = "text_bilingual"
LAYOUT_TRANSLATED_ONLY = "translated_only"
LAYOUT_SOURCE_TRANSLATED = "source_translated"
PDF_SIZE_SLACK_BYTES = int(2.5 * 1024 * 1024)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def pdf_size_target(layout: str, source_size: int) -> tuple[float, int]:
    """Return the release ratio and hard byte ceiling for one complete PDF."""

    source_size = max(0, int(source_size))
    if str(layout) == LAYOUT_SOURCE_TRANSLATED:
        ratio = 1.18
    elif str(layout) == LAYOUT_TRANSLATED_ONLY:
        ratio = 1.30
    else:
        ratio = 1.50
    allowed = max(
        int(source_size * ratio),
        source_size + PDF_SIZE_SLACK_BYTES,
    )
    return ratio, allowed


class TranslationPDFBuilder:
    """Build compact translation PDFs without rasterizing normal source pages.

    Storage policy:
    - original-page bilingual output copies the source PDF page objects directly;
    - only a lightweight CJK text layer is appended to normal pages;
    - text-only layouts use built-in PDF/CJK fonts instead of embedding a large
      external font on every document;
    - the complete PDF is written once with lossless compression;
    - split volumes are opt-in because they duplicate the book on disk.
    """

    def __init__(self, source_pdf: Path, pages_root: Path, output_root: Path):
        self.source_pdf = Path(source_pdf)
        self.pages_root = Path(pages_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _translation_html(
        text: str,
        *,
        title: str = "中文译文",
    ) -> str:
        escaped = html.escape((text or "").strip()).replace("\n", "<br>")
        return (
            f"<h3>{html.escape(title)}</h3>"
            f"<div class='translation'>{escaped}</div>"
        )

    @staticmethod
    def _source_html(text: str) -> str:
        escaped = html.escape((text or "").strip()).replace("\n", "<br>")
        return (
            "<h3>English / 原文</h3>"
            f"<div class='source'>{escaped}</div>"
        )

    @staticmethod
    def _css(font_size: float = 10.5) -> str:
        return (
            "body{font-family:sans-serif;line-height:1.45;}"
            f".translation{{font-size:{font_size}pt;}}"
            f".source{{font-size:{max(8.5, font_size - 1)}pt;}}"
            "h3{font-size:11pt;margin:0 0 8pt 0;}"
        )

    @staticmethod
    def _estimated_text_height(
        text: str,
        width: float,
        font_size: float = 10.5,
    ) -> float:
        text = text or ""
        usable_units = max(12.0, width / max(font_size, 1.0) * 0.90)
        logical_lines = 0
        for paragraph in text.splitlines() or [text]:
            if not paragraph:
                logical_lines += 1
                continue
            cjk = len(_CJK_RE.findall(paragraph))
            other = max(0, len(paragraph) - cjk)
            visual_units = cjk + other * 0.55
            logical_lines += max(
                1,
                math.ceil(visual_units / usable_units),
            )
        return max(
            120.0,
            logical_lines * font_size * 1.72 + 72.0,
        )

    @staticmethod
    def _insert_html(
        page,
        rect,
        markup: str,
        *,
        font_size: float = 10.5,
    ) -> None:
        try:
            result = page.insert_htmlbox(
                rect,
                markup,
                css=TranslationPDFBuilder._css(font_size),
                scale_low=0.75,
            )
        except TypeError:
            result = page.insert_htmlbox(
                rect,
                markup,
                css=TranslationPDFBuilder._css(font_size),
            )
        try:
            spare = (
                float(result[0])
                if isinstance(result, tuple)
                else float(result)
            )
        except Exception:
            return
        if spare < -0.01:
            raise RuntimeError(
                "译本PDF页面文字发生排版溢出，Phoenix已阻止输出被截断的成品。"
            )

    @staticmethod
    def _insert_compact_text(
        page,
        rect,
        text: str,
        *,
        title: str | None = None,
        font_size: float = 10.5,
        min_font_size: float = 5.5,
        cjk: bool = True,
        rotate: int = 0,
    ) -> None:
        """Insert selectable text with a built-in font and no external TTF."""

        import fitz

        payload = (text or "").strip()
        if title:
            payload = f"{title}\n{payload}" if payload else str(title)
        if not payload:
            return

        size = max(float(min_font_size), float(font_size))
        while size >= float(min_font_size) - 1e-6:
            try:
                spare = page.insert_textbox(
                    rect,
                    payload,
                    fontsize=size,
                    fontname="china-s" if cjk else "helv",
                    align=fitz.TEXT_ALIGN_LEFT,
                    lineheight=1.16,
                    rotate=int(rotate) % 360,
                    overlay=True,
                )
            except TypeError:
                spare = page.insert_textbox(
                    rect,
                    payload,
                    fontsize=size,
                    fontname="china-s" if cjk else "helv",
                    align=fitz.TEXT_ALIGN_LEFT,
                    rotate=int(rotate) % 360,
                    overlay=True,
                )
            if float(spare) >= -0.01:
                return
            size -= 0.5

        raise RuntimeError(
            "译本PDF页面文字发生排版溢出，Phoenix已阻止输出被截断的成品。"
        )

    def _translated_text(self, page_number: int) -> str:
        page_file = self.pages_root / f"{int(page_number):06d}.txt"
        if not page_file.is_file():
            raise RuntimeError(
                f"无法生成PDF：第 {page_number} 页尚未完成翻译"
            )
        return page_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

    @staticmethod
    def _copy_document_metadata(source_doc, out_doc) -> None:
        try:
            out_doc.set_metadata(source_doc.metadata or {})
        except Exception:
            pass

    @staticmethod
    def _copy_toc(
        source_doc,
        out_doc,
        *,
        start_page: int,
        total_pages: int,
    ) -> None:
        try:
            toc = source_doc.get_toc(simple=True) or []
        except Exception:
            return
        adjusted: list[list] = []
        for item in toc:
            if len(item) < 3:
                continue
            level, title, page_number = item[:3]
            try:
                page_number = int(page_number)
            except (TypeError, ValueError):
                continue
            if start_page <= page_number <= total_pages:
                adjusted.append(
                    [
                        int(level),
                        str(title),
                        page_number - start_page + 1,
                    ]
                )
        if adjusted:
            try:
                out_doc.set_toc(adjusted)
            except Exception:
                pass

    def _extend_copied_original_page(
        self,
        page,
        translated: str,
    ) -> None:
        """Append a selectable translation strip without rebuilding the page.

        The MediaBox is extended on the side that corresponds to the visual
        bottom for rotations 0/90/180/270. Existing content streams, images,
        vector drawings, fonts, links and annotations remain on the copied page.
        """

        import fitz

        rotation = int(getattr(page, "rotation", 0) or 0) % 360
        media = page.mediabox
        media_width = float(media.width)
        media_height = float(media.height)
        visual_width = float(page.rect.width)
        visual_height = float(page.rect.height)

        translation_height = self._estimated_text_height(
            translated,
            max(120.0, visual_width - 52.0),
        )
        gap = 14.0
        extra = gap + translation_height

        if rotation == 0:
            page.set_mediabox(
                fitz.Rect(
                    float(media.x0),
                    float(media.y0),
                    float(media.x1),
                    float(media.y1) + extra,
                )
            )
            target = fitz.Rect(
                26.0,
                visual_height + gap,
                max(27.0, visual_width - 26.0),
                max(visual_height + gap + 20.0, visual_height + extra - 18.0),
            )
        elif rotation == 90:
            page.set_mediabox(
                fitz.Rect(
                    float(media.x0),
                    float(media.y0),
                    float(media.x1) + extra,
                    float(media.y1),
                )
            )
            target = fitz.Rect(
                media_width + gap,
                26.0,
                media_width + extra - 18.0,
                max(27.0, media_height - 26.0),
            )
        elif rotation == 180:
            page.set_mediabox(
                fitz.Rect(
                    float(media.x0),
                    float(media.y0) - extra,
                    float(media.x1),
                    float(media.y1),
                )
            )
            target = fitz.Rect(
                26.0,
                gap,
                max(27.0, media_width - 26.0),
                extra - 18.0,
            )
        elif rotation == 270:
            page.set_mediabox(
                fitz.Rect(
                    float(media.x0) - extra,
                    float(media.y0),
                    float(media.x1),
                    float(media.y1),
                )
            )
            target = fitz.Rect(
                gap,
                26.0,
                extra - 18.0,
                max(27.0, media_height - 26.0),
            )
        else:
            raise RuntimeError(f"不支持的PDF页面旋转角度：{rotation}")

        self._insert_compact_text(
            page,
            target,
            translated,
            title="中文译文",
            font_size=10.5,
            min_font_size=6.0,
            cjk=True,
            rotate=rotation,
        )

    def _append_original_bilingual_page(
        self,
        out_doc,
        source_doc,
        page_index: int,
        page_number: int,
        *,
        final: bool = True,
    ) -> None:
        """Directly graft the source page, then append only a text stream."""

        translated = self._translated_text(page_number)

        # final=0 keeps PyMuPDF's graft map alive across page copies so shared
        # source images/fonts stay shared. The last direct copy closes the map.
        out_doc.insert_pdf(
            source_doc,
            from_page=page_index,
            to_page=page_index,
            links=True,
            annots=True,
            final=1 if final else 0,
        )
        self._extend_copied_original_page(
            out_doc[out_doc.page_count - 1],
            translated,
        )

    def _append_text_bilingual_page(
        self,
        out_doc,
        source_doc,
        page_index: int,
        page_number: int,
    ) -> None:
        import fitz

        source_page = source_doc[page_index]
        width = max(595.0, float(source_page.rect.width))
        source_text = source_page.get_text("text").strip()
        translated = self._translated_text(page_number)
        source_height = self._estimated_text_height(
            source_text,
            width - 52.0,
            font_size=9.5,
        )
        translation_height = self._estimated_text_height(
            translated,
            width - 52.0,
        )
        page_height = source_height + translation_height + 72.0
        page = out_doc.new_page(width=width, height=page_height)

        self._insert_compact_text(
            page,
            fitz.Rect(
                26.0,
                22.0,
                width - 26.0,
                22.0 + source_height,
            ),
            source_text,
            title="English / 原文",
            font_size=9.5,
            min_font_size=6.0,
            cjk=False,
        )
        self._insert_compact_text(
            page,
            fitz.Rect(
                26.0,
                36.0 + source_height,
                width - 26.0,
                page_height - 18.0,
            ),
            translated,
            title="中文译文",
            font_size=10.5,
            min_font_size=6.0,
            cjk=True,
        )

    def _append_translated_only_page(
        self,
        out_doc,
        source_doc,
        page_index: int,
        page_number: int,
    ) -> None:
        import fitz

        source_page = source_doc[page_index]
        width = max(595.0, float(source_page.rect.width))
        translated = self._translated_text(page_number)
        translation_height = self._estimated_text_height(
            translated,
            width - 52.0,
        )
        page_height = translation_height + 48.0
        page = out_doc.new_page(width=width, height=page_height)
        self._insert_compact_text(
            page,
            fitz.Rect(
                26.0,
                22.0,
                width - 26.0,
                page_height - 16.0,
            ),
            translated,
            title=f"第 {page_number} 页 · 中文译文",
            font_size=10.5,
            min_font_size=6.0,
            cjk=True,
        )

    @staticmethod
    def _atomic_save(
        doc,
        path: Path,
        *,
        stream_merge: bool = False,
    ) -> None:
        """Write once, losslessly, with a compression mode matched to content.

        ``garbage=4`` is used only for outputs known to contain no page images.
        It is very effective for repetitive textbook text streams, but scanning
        large image streams is intentionally avoided because that was the cause
        of the old whole-book final-save stall.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.stem + ".tmp" + path.suffix)
        temp.unlink(missing_ok=True)
        garbage = 4 if stream_merge else 2
        try:
            try:
                doc.save(
                    str(temp),
                    garbage=garbage,
                    deflate=True,
                    deflate_images=True,
                    deflate_fonts=True,
                    use_objstms=1,
                )
            except TypeError:
                doc.save(
                    str(temp),
                    garbage=garbage,
                    deflate=True,
                )
            os.replace(temp, path)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _selected_pages_have_images(
        source_doc,
        *,
        start_page: int,
        total_pages: int,
    ) -> bool:
        for page_index in range(start_page - 1, total_pages):
            try:
                if source_doc[page_index].get_images(full=True):
                    return True
            except Exception:
                # Unknown / malformed page resources are treated conservatively.
                return True
        return False

    @staticmethod
    def _clear_parts(parts_root: Path) -> None:
        if not parts_root.exists():
            return
        for old in parts_root.glob("第*.pdf"):
            old.unlink(missing_ok=True)
        try:
            if not any(parts_root.iterdir()):
                parts_root.rmdir()
        except OSError:
            pass

    @staticmethod
    def _size_target(layout: str, source_size: int) -> tuple[float, int]:
        return pdf_size_target(layout, source_size)

    def _write_size_report(
        self,
        *,
        layout: str,
        complete_path: Path,
        part_paths: tuple[Path, ...],
        source_has_images: bool,
        stream_merge: bool,
    ) -> dict:
        source_size = int(self.source_pdf.stat().st_size)
        output_size = int(complete_path.stat().st_size)
        target_ratio, target_bytes = self._size_target(layout, source_size)
        actual_ratio = (
            float(output_size) / float(source_size)
            if source_size > 0
            else 0.0
        )
        report = {
            "layout": layout,
            "source_bytes": source_size,
            "output_bytes": output_size,
            "ratio": round(actual_ratio, 4),
            "target_ratio": target_ratio,
            "allowed_bytes": target_bytes,
            # Compatibility alias retained for existing report consumers.
            "target_bytes": target_bytes,
            "storage_target_passed": bool(
                source_size <= 0 or output_size <= target_bytes
            ),
            "source_has_images": bool(source_has_images),
            "text_stream_merge": bool(stream_merge),
            "split_volumes": len(part_paths),
        }
        report_path = self.output_root / "PDF体积报告.json"
        temp = report_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(report_path)
        return report

    def build(
        self,
        *,
        start_page: int,
        total_pages: int,
        layout: str = LAYOUT_ORIGINAL_BILINGUAL,
        part_pages: int = 0,
        progress=None,
    ) -> tuple[Path, tuple[Path, ...]]:
        import fitz

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
        self._clear_parts(parts_root)

        source_doc = fitz.open(self.source_pdf)
        out_doc = fitz.open()
        try:
            if total_pages > source_doc.page_count:
                raise RuntimeError(
                    f"源PDF只有 {source_doc.page_count} 页，无法输出到第 {total_pages} 页"
                )
            self._copy_document_metadata(source_doc, out_doc)
            selected_total = total_pages - start_page + 1
            source_has_images = self._selected_pages_have_images(
                source_doc,
                start_page=start_page,
                total_pages=total_pages,
            )

            last_direct = (
                total_pages
                if layout == LAYOUT_ORIGINAL_BILINGUAL
                else None
            )

            for offset, page_number in enumerate(
                range(start_page, total_pages + 1),
                start=1,
            ):
                source_index = page_number - 1
                if layout == LAYOUT_ORIGINAL_BILINGUAL:
                    self._append_original_bilingual_page(
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

            self._copy_toc(
                source_doc,
                out_doc,
                start_page=start_page,
                total_pages=total_pages,
            )
            if progress:
                progress(
                    selected_total,
                    selected_total,
                    "页面完成，正在进行最终一次无损压缩；原页不会重新渲染。",
                )
            if complete_path.exists():
                complete_path.unlink()
            stream_merge = (
                layout != LAYOUT_ORIGINAL_BILINGUAL
                or not source_has_images
            )
            self._atomic_save(
                out_doc,
                complete_path,
                stream_merge=stream_merge,
            )
        finally:
            out_doc.close()
            source_doc.close()

        part_paths: list[Path] = []
        if part_pages > 0:
            parts_root.mkdir(parents=True, exist_ok=True)
            complete = fitz.open(complete_path)
            try:
                total_output_pages = complete.page_count
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
                        self._atomic_save(
                            part_doc,
                            part_path,
                            stream_merge=False,
                        )
                    finally:
                        part_doc.close()
                    part_paths.append(part_path)
            finally:
                complete.close()

        part_tuple = tuple(part_paths)
        report = self._write_size_report(
            layout=layout,
            complete_path=complete_path,
            part_paths=part_tuple,
            source_has_images=source_has_images,
            stream_merge=stream_merge,
        )
        if progress:
            source_mb = report["source_bytes"] / (1024 * 1024)
            output_mb = report["output_bytes"] / (1024 * 1024)
            message = (
                f"PDF完成：原文件 {source_mb:.1f}MB，译本 {output_mb:.1f}MB，"
                f"体积比 {report['ratio']:.2f}×"
            )
            if report["storage_target_passed"]:
                message += f"；通过 ≤{report['target_ratio']:.2f}× 发布体积目标"
            else:
                message += (
                    f"；⚠ 超过 ≤{report['target_ratio']:.2f}× 发布体积目标，"
                    "应检查异常字体/旋转页/新增资源"
                )
            if not part_tuple:
                message += "；未生成重复分册"
            progress(
                total_pages - start_page + 1,
                total_pages - start_page + 1,
                message,
            )

        return complete_path, part_tuple
