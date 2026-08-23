from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .pdf_parser import pdf_page_count, sha256_file
from .translator import (
    EXPORT_RICH,
    PDFTranslator,
    TranslationResult,
)
from .translation_pdf import LAYOUT_ORIGINAL_BILINGUAL


_INSTALLED = False


def _is_translation_output(path: Path | None, output_root: Path) -> bool:
    if path is None:
        return False
    try:
        path.resolve().relative_to(output_root.resolve())
        return True
    except Exception:
        return False


def _rebuild_result(
    translator: PDFTranslator,
    pdf_path: Path,
    *,
    target_language: str,
    requested_start_page: int,
    previous: TranslationResult | None = None,
) -> TranslationResult | None:
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        return None

    digest = sha256_file(pdf_path)
    (
        book_root,
        pages_root,
        _audit_root,
        checkpoint_path,
        _final_output,
    ) = translator._book_paths(
        pdf_path,
        digest,
        target_language,
    )
    state = translator._read_json(checkpoint_path)
    start_page = translator._resolve_resume_start_page(
        state,
        max(1, int(requested_start_page)),
        force_restart=False,
    )
    total_pages = int(pdf_page_count(pdf_path))
    if total_pages <= 0 or start_page > total_pages:
        return None

    missing = [
        page
        for page in range(start_page, total_pages + 1)
        if not (pages_root / f"{page:06d}.txt").is_file()
        or (pages_root / f"{page:06d}.txt").stat().st_size <= 0
    ]
    if missing:
        # This is a genuinely unfinished translation, not an output-assembly
        # failure. Let the normal resume path translate the missing pages.
        return None

    # A warning checkpoint is not a deliverable.  Return to the normal
    # translation path so the failed pages are automatically retried instead
    # of assembling a PDF containing low-confidence text.
    if int(state.get("warning_pages", 0) or 0) > 0:
        return None

    # Old checkpoints predate selectable output formats. Rebuild those in the
    # original rich-text form; new checkpoints preserve the user's selected
    # PDF/text format and bilingual layout.
    output_layout = str(
        state.get("output_layout", LAYOUT_ORIGINAL_BILINGUAL)
    )
    export_format = str(state.get("export_format", EXPORT_RICH))
    part_pages = max(1, int(state.get("part_pages", 50) or 50))

    output_paths, image_count = translator._build_deliverables(
        pdf_path,
        book_root,
        pages_root,
        start_page,
        total_pages,
        target_language,
        output_layout=output_layout,
        export_format=export_format,
        part_pages=part_pages,
        progress=None,
    )
    if not output_paths or not output_paths[0].is_file():
        return None

    warning_pages = int(state.get("warning_pages", 0) or 0)
    state.update(
        {
            "status": "completed",
            "last_completed_page": total_pages,
            "output_path": str(output_paths[0]),
            "output_paths": [str(path) for path in output_paths],
            "image_count": int(image_count),
        }
    )
    translator._write_json(checkpoint_path, state)

    if previous is not None:
        return replace(
            previous,
            output_path=output_paths[0],
            output_paths=tuple(output_paths),
            image_count=int(image_count),
            paused=False,
        )

    selected_total = total_pages - start_page + 1
    formal_names = getattr(translator.engine, "formal_backend_names", None)
    backends = (
        formal_names(target_language)
        if callable(formal_names)
        else translator.engine.available_backends()
    )
    return TranslationResult(
        output_path=output_paths[0],
        source_path=pdf_path,
        start_page=start_page,
        total_pages=total_pages,
        target_language=target_language,
        pages_done=selected_total,
        resumed_pages=selected_total,
        warning_pages=warning_pages,
        available_backends=tuple(backends),
        output_paths=tuple(output_paths),
        image_count=int(image_count),
        paused=False,
        smart_level="smart2",
        output_layout=output_layout,
        export_format=export_format,
        part_pages=part_pages,
    )


def install() -> None:
    """Guarantee completed translations have readable final deliverables."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original = PDFTranslator.translate_book

    def translate_book(self: PDFTranslator, pdf_path: Path, **kwargs):
        target_language = str(kwargs.get("target_language", "中文"))
        requested_start_page = max(1, int(kwargs.get("start_page", 1)))
        source = Path(pdf_path)

        try:
            result = original(self, source, **kwargs)
        except FileNotFoundError as exc:
            missing = (
                Path(exc.filename)
                if getattr(exc, "filename", None)
                else None
            )
            if not _is_translation_output(missing, self.output_root):
                raise
            recovered = _rebuild_result(
                self,
                source,
                target_language=target_language,
                requested_start_page=requested_start_page,
            )
            if recovered is None:
                raise
            return recovered

        # Paused tasks intentionally do not have final assembled deliverables.
        # The page checkpoints are the valid product state until resume.
        if bool(getattr(result, "paused", False)):
            return result

        output_path = Path(result.output_path)
        if output_path.is_file():
            return result

        recovered = _rebuild_result(
            self,
            Path(result.source_path),
            target_language=str(result.target_language),
            requested_start_page=int(result.start_page),
            previous=result,
        )
        if recovered is not None:
            return recovered

        raise FileNotFoundError(
            f"翻译任务返回完成状态，但完整译本不存在：{output_path}。"
            "逐页checkpoint仍保留；再次执行‘开始/继续翻译’会从缺失页继续。"
        )

    PDFTranslator.translate_book = translate_book
