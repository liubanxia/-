from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

LAYOUT_SOURCE_TRANSLATED = "source_translated"
_INSTALLED = False

_SENTENCE_RE = re.compile(r".+?(?:[。！？；!?;](?:\s+|$)|\n+|$)", re.S)
_NUMERIC_ONLY_RE = re.compile(r"^[\s\dIVXLCDMivxlcdm./,:;()\[\]{}%+\-–—]+$")


def _clean_units(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    paragraphs = [
        item.strip()
        for item in re.split(r"\n{2,}", text)
        if item.strip()
    ]
    units: list[str] = []
    for paragraph in paragraphs or [text]:
        found = [
            match.group(0).strip()
            for match in _SENTENCE_RE.finditer(paragraph)
            if match.group(0).strip()
        ]
        units.extend(found or [paragraph])
    return units


def _preserve_source_block(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if _NUMERIC_ONLY_RE.fullmatch(compact):
        return True
    lower = compact.lower()
    if lower.startswith(("http://", "https://", "doi:", "pmid:", "pmcid:")):
        return True
    return False


def _source_blocks(page) -> list[dict]:
    try:
        payload = page.get_text("dict") or {}
    except Exception:
        return []

    blocks: list[dict] = []
    for block in payload.get("blocks") or []:
        if int(block.get("type", 0) or 0) != 0:
            continue
        lines = block.get("lines") or []
        if not lines:
            continue
        text_lines: list[str] = []
        sizes: list[float] = []
        for line in lines:
            spans = line.get("spans") or []
            line_text = "".join(str(span.get("text") or "") for span in spans)
            if line_text.strip():
                text_lines.append(line_text.strip())
            for span in spans:
                try:
                    value = float(span.get("size") or 0.0)
                    if value > 0:
                        sizes.append(value)
                except Exception:
                    pass
        text = "\n".join(text_lines).strip()
        if not text or _preserve_source_block(text):
            continue
        try:
            bbox = tuple(float(value) for value in block.get("bbox", ()))
            if len(bbox) != 4:
                continue
            x0, y0, x1, y1 = bbox
            if x1 - x0 < 8 or y1 - y0 < 4:
                continue
        except Exception:
            continue
        if sizes:
            sizes.sort()
            preferred = sizes[len(sizes) // 2]
        else:
            preferred = 10.0
        blocks.append(
            {
                "bbox": bbox,
                "text": text,
                "font_size": max(5.5, min(14.0, float(preferred))),
                "weight": max(4, len(re.sub(r"\s+", "", text))),
            }
        )
    return blocks


def _allocate_translation(translated: str, blocks: list[dict]) -> list[str]:
    if not blocks:
        return []
    units = _clean_units(translated)
    if not units:
        return [""] * len(blocks)
    if len(blocks) == 1:
        return ["\n".join(units)]

    total_weight = max(1, sum(int(item["weight"]) for item in blocks))
    total_chars = max(1, sum(len(item) for item in units))
    assignments: list[str] = []
    cursor = 0
    consumed_chars = 0
    consumed_weight = 0

    for block_index, block in enumerate(blocks):
        remaining_blocks = len(blocks) - block_index
        if remaining_blocks == 1:
            assignments.append("\n".join(units[cursor:]).strip())
            break

        consumed_weight += int(block["weight"])
        target_chars = int(round(total_chars * consumed_weight / total_weight))
        picked: list[str] = []
        while cursor < len(units):
            units_left_after = len(units) - (cursor + 1)
            blocks_left_after = remaining_blocks - 1
            if picked and consumed_chars >= target_chars:
                break
            if units_left_after < blocks_left_after and picked:
                break
            unit = units[cursor]
            picked.append(unit)
            consumed_chars += len(unit)
            cursor += 1
            if units_left_after < blocks_left_after:
                break
        assignments.append("\n".join(picked).strip())

    while len(assignments) < len(blocks):
        assignments.append("")
    return assignments


def _fit_textbox(page, rect, text: str, preferred_size: float) -> bool:
    import fitz

    text = (text or "").strip()
    if not text:
        return True
    start = max(5.5, min(12.5, float(preferred_size)))
    size = start
    while size >= 4.5:
        try:
            spare = page.insert_textbox(
                rect,
                text,
                fontsize=size,
                fontname="china-s",
                align=fitz.TEXT_ALIGN_LEFT,
                lineheight=1.08,
                overlay=True,
            )
            if float(spare) >= -0.01:
                return True
        except Exception:
            return False
        size -= 0.5
    return False


def _append_footer(page, translated: str) -> None:
    import fitz

    translated = (translated or "").strip()
    if not translated:
        return
    if int(getattr(page, "rotation", 0) or 0) % 360 != 0:
        # Rotated specialty pages are rare. Keep page orientation intact and
        # place a compact white translation panel inside the lower quarter.
        rect = page.rect
        target = fitz.Rect(
            24,
            max(24, float(rect.height) * 0.72),
            max(25, float(rect.width) - 24),
            max(25, float(rect.height) - 18),
        )
        page.draw_rect(
            target,
            color=None,
            fill=(1, 1, 1),
            fill_opacity=0.92,
            overlay=True,
        )
        if not _fit_textbox(page, target + (8, 8, -8, -8), translated, 8.5):
            raise RuntimeError("旋转页译文无法完整排入原页面。")
        return

    old_height = float(page.rect.height)
    width = float(page.rect.width)
    visual_units = sum(1.0 if "\u3400" <= ch <= "\u9fff" else 0.55 for ch in translated)
    chars_per_line = max(24.0, (width - 48.0) / 8.2)
    lines = max(2, math.ceil(visual_units / chars_per_line))
    extra = max(70.0, min(420.0, lines * 11.0 + 34.0))
    media = page.mediabox
    page.set_mediabox(
        fitz.Rect(
            float(media.x0),
            float(media.y0),
            float(media.x1),
            float(media.y1) + extra,
        )
    )
    target = fitz.Rect(24, old_height + 12, width - 24, old_height + extra - 12)
    if not _fit_textbox(page, target, translated, 8.5):
        raise RuntimeError("译文过长，无法在保留完整内容的前提下生成紧凑PDF。")


def _replace_page_text(page, translated: str) -> tuple[int, bool]:
    import fitz

    if int(getattr(page, "rotation", 0) or 0) % 360 != 0:
        _append_footer(page, translated)
        return 0, True

    blocks = _source_blocks(page)
    if not blocks:
        _append_footer(page, translated)
        return 0, True

    assignments = _allocate_translation(translated, blocks)
    for block in blocks:
        page.add_redact_annot(
            fitz.Rect(*block["bbox"]),
            fill=False,
            cross_out=False,
        )
    # Critical: remove text only. Images and vector drawings remain untouched.
    page.apply_redactions(images=0, graphics=0, text=0)

    overflow: list[str] = []
    inserted = 0
    for block, text in zip(blocks, assignments):
        if not text:
            continue
        rect = fitz.Rect(*block["bbox"])
        if _fit_textbox(page, rect, text, float(block["font_size"])):
            inserted += 1
        else:
            overflow.append(text)
    if overflow:
        _append_footer(page, "\n\n".join(overflow))
    return inserted, bool(overflow)


def _copy_toc(source_doc, output_doc, start_page: int, total_pages: int) -> None:
    try:
        toc = source_doc.get_toc(simple=True) or []
        adjusted = []
        for item in toc:
            if len(item) < 3:
                continue
            level, title, page_number = item[:3]
            try:
                page_number = int(page_number)
            except Exception:
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
            output_doc.set_toc(adjusted)
    except Exception:
        pass


def _atomic_save(doc, path: Path, source_size: int) -> tuple[int, bool]:
    import fitz

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    optimized = path.with_name(path.stem + ".opt.tmp" + path.suffix)
    temp.unlink(missing_ok=True)
    optimized.unlink(missing_ok=True)
    kwargs = {
        "garbage": 2,
        "deflate": True,
        "deflate_images": True,
        "deflate_fonts": True,
        "use_objstms": 1,
    }
    try:
        try:
            doc.save(str(temp), **kwargs)
        except TypeError:
            doc.save(str(temp), garbage=2, deflate=True)

        budget = max(
            int(source_size * 1.18),
            int(source_size + 2.5 * 1024 * 1024),
        )
        optimized_pass = False
        if source_size > 0 and temp.stat().st_size > budget:
            # Only if the first lossless write exceeds the storage target do we
            # pay for duplicate-object merging. Never use garbage=4: scanning
            # all large image streams caused the old 383-page final-save stall.
            retry = fitz.open(temp)
            try:
                retry.save(
                    str(optimized),
                    garbage=3,
                    deflate=True,
                    deflate_images=True,
                    deflate_fonts=True,
                    use_objstms=1,
                )
            except TypeError:
                retry.save(str(optimized), garbage=3, deflate=True)
            finally:
                retry.close()
            if optimized.is_file() and optimized.stat().st_size < temp.stat().st_size:
                os.replace(optimized, temp)
                optimized_pass = True
            else:
                optimized.unlink(missing_ok=True)

        os.replace(temp, path)
        return int(path.stat().st_size), optimized_pass
    except Exception:
        temp.unlink(missing_ok=True)
        optimized.unlink(missing_ok=True)
        raise


def _build_source_translated(
    builder,
    *,
    start_page: int,
    total_pages: int,
    part_pages: int = 0,
    progress=None,
):
    import fitz

    start_page = max(1, int(start_page))
    total_pages = max(start_page, int(total_pages))
    part_pages = max(0, int(part_pages or 0))
    source = Path(builder.source_pdf)
    complete_path = builder.output_root / "完整图文中文译本_原版布局.pdf"
    parts_root = builder.output_root / "PDF分册"

    if parts_root.exists():
        for old in parts_root.glob("第*.pdf"):
            old.unlink(missing_ok=True)
        try:
            if not any(parts_root.iterdir()):
                parts_root.rmdir()
        except OSError:
            pass

    source_doc = fitz.open(source)
    out_doc = fitz.open()
    selected_total = total_pages - start_page + 1
    overflow_pages = 0
    native_pages = 0
    try:
        if total_pages > source_doc.page_count:
            raise RuntimeError(
                f"源PDF只有 {source_doc.page_count} 页，无法输出到第 {total_pages} 页"
            )
        try:
            out_doc.set_metadata(source_doc.metadata or {})
        except Exception:
            pass
        out_doc.insert_pdf(
            source_doc,
            from_page=start_page - 1,
            to_page=total_pages - 1,
            links=True,
            annots=True,
        )
        _copy_toc(source_doc, out_doc, start_page, total_pages)

        for offset in range(selected_total):
            page_number = start_page + offset
            translated = builder._translated_text(page_number)
            inserted, overflow = _replace_page_text(out_doc[offset], translated)
            if inserted:
                native_pages += 1
            if overflow:
                overflow_pages += 1
            if progress:
                progress(
                    offset + 1,
                    selected_total,
                    f"正在保留原图并替换文字层：第 {page_number}/{total_pages} 页",
                )

        if progress:
            progress(
                selected_total,
                selected_total,
                "图文页面完成，正在进行一次无损压缩；原图不会重新编码降质。",
            )
        source_size = int(source.stat().st_size)
        output_size, optimized_pass = _atomic_save(
            out_doc,
            complete_path,
            source_size,
        )
    finally:
        out_doc.close()
        source_doc.close()

    part_paths: list[Path] = []
    if part_pages > 0:
        parts_root.mkdir(parents=True, exist_ok=True)
        complete = fitz.open(complete_path)
        try:
            for part_index, start_index in enumerate(
                range(0, complete.page_count, part_pages),
                start=1,
            ):
                end_index = min(
                    complete.page_count - 1,
                    start_index + part_pages - 1,
                )
                part_path = parts_root / (
                    f"第{part_index:03d}册_"
                    f"{start_page + start_index:04d}-{start_page + end_index:04d}.pdf"
                )
                part_doc = fitz.open()
                try:
                    part_doc.insert_pdf(
                        complete,
                        from_page=start_index,
                        to_page=end_index,
                    )
                    _atomic_save(
                        part_doc,
                        part_path,
                        max(1, int(complete_path.stat().st_size)),
                    )
                finally:
                    part_doc.close()
                part_paths.append(part_path)
        finally:
            complete.close()

    source_size = int(source.stat().st_size)
    output_size = int(complete_path.stat().st_size)
    ratio = output_size / source_size if source_size else 0.0
    report = {
        "mode": LAYOUT_SOURCE_TRANSLATED,
        "source": str(source),
        "output": str(complete_path),
        "source_bytes": source_size,
        "output_bytes": output_size,
        "ratio": round(ratio, 4),
        "pages": selected_total,
        "pages_replaced_in_place": native_pages,
        "pages_with_footer_overflow_or_scan_fallback": overflow_pages,
        "lossless_second_pass": bool(optimized_pass),
        "split_volumes": len(part_paths),
    }
    try:
        report_path = builder.output_root / "PDF体积报告.json"
        temp = report_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp.replace(report_path)
    except Exception:
        pass

    if progress:
        message = (
            f"图文中文译本完成：{output_size / (1024 * 1024):.1f}MB；"
            f"原PDF {source_size / (1024 * 1024):.1f}MB"
        )
        if source_size:
            message += f"；体积比 {ratio:.2f}×"
        if overflow_pages:
            message += f"；{overflow_pages}页因扫描/复杂排版使用页内或页尾紧凑译文区"
        if not part_paths:
            message += "；未生成重复分册"
        progress(selected_total, selected_total, message)

    return complete_path, tuple(part_paths)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translator as translator_module
    from .translation_pdf import TranslationPDFBuilder
    from .translator import PDFTranslator

    previous_build = TranslationPDFBuilder.build
    previous_normalize = translator_module._normalize_layout
    previous_translate_book = PDFTranslator.translate_book

    def normalize_layout(value):
        if value == LAYOUT_SOURCE_TRANSLATED:
            return LAYOUT_SOURCE_TRANSLATED
        return previous_normalize(value)

    def build(
        self,
        *,
        start_page: int,
        total_pages: int,
        layout: str,
        part_pages: int = 0,
        progress=None,
    ):
        if layout == LAYOUT_SOURCE_TRANSLATED:
            return _build_source_translated(
                self,
                start_page=start_page,
                total_pages=total_pages,
                part_pages=part_pages,
                progress=progress,
            )
        return previous_build(
            self,
            start_page=start_page,
            total_pages=total_pages,
            layout=layout,
            part_pages=part_pages,
            progress=progress,
        )

    def translate_book(self, pdf_path: Path, **kwargs):
        kwargs.setdefault("output_layout", LAYOUT_SOURCE_TRANSLATED)
        return previous_translate_book(self, pdf_path, **kwargs)

    translator_module._normalize_layout = normalize_layout
    TranslationPDFBuilder.build = build
    PDFTranslator.translate_book = translate_book
