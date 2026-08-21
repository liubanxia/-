from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import replace
from pathlib import Path

from .translation_output_validation import (
    CONTRACT_VERSION,
    TranslationOutputError,
    assert_stable_against_previous,
    build_input_signature,
    validate_deliverables,
    validate_pdf,
    write_integrity_report,
)


LAYOUT_SOURCE_TRANSLATED = "source_translated"
_CORE_TRANSLATE_BOOK = None
_CORE_BUILD_DELIVERABLES = None
_CORE_PDF_BUILD = None
_CORE_NORMALIZE_LAYOUT = None
_CAPTURED = False
_INSTALLED = False


def capture_core() -> None:
    """Capture unwrapped translation entry points before legacy installers run."""

    global _CORE_TRANSLATE_BOOK
    global _CORE_BUILD_DELIVERABLES
    global _CORE_PDF_BUILD
    global _CORE_NORMALIZE_LAYOUT
    global _CAPTURED

    if _CAPTURED:
        return
    from . import translator as translator_module
    from .translation_pdf import TranslationPDFBuilder
    from .translator import PDFTranslator

    _CORE_TRANSLATE_BOOK = PDFTranslator.translate_book
    _CORE_BUILD_DELIVERABLES = PDFTranslator._build_deliverables
    _CORE_PDF_BUILD = TranslationPDFBuilder.build
    _CORE_NORMALIZE_LAYOUT = translator_module._normalize_layout
    _CAPTURED = True


def _remove_tree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _swap_parts_directory(staged_root: Path, real_root: Path) -> tuple[Path, ...]:
    staged_parts = staged_root / "PDF分册"
    real_parts = real_root / "PDF分册"
    incoming = real_root / ".PDF分册.new"
    backup = real_root / ".PDF分册.old"
    _remove_tree(incoming)
    _remove_tree(backup)

    if not staged_parts.exists():
        _remove_tree(real_parts)
        return ()

    os.replace(staged_parts, incoming)
    try:
        if real_parts.exists():
            os.replace(real_parts, backup)
        os.replace(incoming, real_parts)
        _remove_tree(backup)
    except Exception:
        if backup.exists() and not real_parts.exists():
            try:
                os.replace(backup, real_parts)
            except Exception:
                pass
        raise

    return tuple(sorted(real_parts.glob("第*.pdf")))


def _promote_auxiliary_reports(staged_root: Path, real_root: Path, real_complete: Path) -> None:
    for name in ("PDF体积报告.json",):
        staged = staged_root / name
        if not staged.is_file():
            continue
        target = real_root / name
        if name == "PDF体积报告.json":
            try:
                payload = json.loads(staged.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    if "output" in payload:
                        payload["output"] = str(real_complete)
                    temp = target.with_name(target.name + ".tmp")
                    temp.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temp, target)
                    continue
            except Exception:
                pass
        os.replace(staged, target)


def _required_staging_bytes(
    *,
    source_size: int,
    layout: str,
    part_pages: int,
) -> int:
    # Transactional publishing intentionally keeps the last known-good output
    # until the new file passes validation, so reserve the expected new output
    # plus a safety margin. Explicit split volumes roughly duplicate that peak.
    bilingual = str(layout) in {"original_bilingual", "text_bilingual"}
    ratio = 1.70 if bilingual else 1.45
    required = int(max(0, int(source_size)) * ratio) + 32 * 1024 * 1024
    if int(part_pages or 0) > 0:
        required += int(max(0, int(source_size)) * ratio)
    return max(32 * 1024 * 1024, required)


def _assert_staging_space(
    root: Path,
    *,
    source_size: int,
    layout: str,
    part_pages: int,
) -> None:
    required = _required_staging_bytes(
        source_size=source_size,
        layout=layout,
        part_pages=part_pages,
    )
    free = int(shutil.disk_usage(Path(root)).free)
    if free < required:
        raise TranslationOutputError(
            "PDF发布前磁盘空间预检失败："
            f"当前可用 {free / (1024 ** 3):.2f}GB，"
            f"本次安全构建至少需要约 {required / (1024 ** 3):.2f}GB。"
            "Phoenix未开始生成临时PDF，上一份稳定成品保持不变。"
        )


def _stable_pdf_build(
    self,
    *,
    start_page: int,
    total_pages: int,
    layout: str,
    part_pages: int = 0,
    progress=None,
):
    """Build into a staging directory, validate, then atomically publish."""

    if _CORE_PDF_BUILD is None:
        raise RuntimeError("翻译稳定性核心尚未捕获原始PDF构建器。")

    from .translation_pdf import LAYOUT_ORIGINAL_BILINGUAL

    start_page = max(1, int(start_page))
    total_pages = max(start_page, int(total_pages))
    part_pages = max(0, int(part_pages or 0))
    selected_total = total_pages - start_page + 1
    real_root = Path(self.output_root)
    real_root.mkdir(parents=True, exist_ok=True)
    _assert_staging_space(
        real_root,
        source_size=int(Path(self.source_pdf).stat().st_size),
        layout=str(layout),
        part_pages=part_pages,
    )
    stage_root = real_root.parent / f".pxpdf-{uuid.uuid4().hex[:10]}"
    _remove_tree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    original_output_root = self.output_root
    try:
        self.output_root = stage_root
        try:
            if str(layout) == LAYOUT_SOURCE_TRANSLATED:
                from .translation_layout_compact import _build_source_translated

                staged_complete, staged_parts = _build_source_translated(
                    self,
                    start_page=start_page,
                    total_pages=total_pages,
                    part_pages=part_pages,
                    progress=progress,
                )
            else:
                staged_complete, staged_parts = _CORE_PDF_BUILD(
                    self,
                    start_page=start_page,
                    total_pages=total_pages,
                    layout=layout,
                    part_pages=part_pages,
                    progress=progress,
                )
        finally:
            self.output_root = original_output_root

        staged_complete = Path(staged_complete)
        staged_parts = tuple(Path(path) for path in staged_parts)
        preserve_images = str(layout) in {
            LAYOUT_SOURCE_TRANSLATED,
            LAYOUT_ORIGINAL_BILINGUAL,
        }
        pdf_report = validate_pdf(
            staged_complete,
            expected_pages=selected_total,
            pages_root=Path(self.pages_root),
            start_page=start_page,
            source_pdf=Path(self.source_pdf) if preserve_images else None,
            preserve_source_images=preserve_images,
            minimum_translation_coverage=0.62,
        )
        validate_deliverables(
            (staged_complete, *staged_parts),
            expected_complete_pages=selected_total,
        )
        signature = build_input_signature(
            source_pdf=Path(self.source_pdf),
            pages_root=Path(self.pages_root),
            start_page=start_page,
            total_pages=total_pages,
            layout=str(layout),
        )
        integrity_path = real_root / "PDF完整性报告.json"
        assert_stable_against_previous(
            integrity_path,
            signature=signature,
            current_structure_sha256=str(pdf_report["structure_sha256"]),
        )

        real_part_paths = _swap_parts_directory(stage_root, real_root)
        real_complete = real_root / staged_complete.name
        os.replace(staged_complete, real_complete)
        _promote_auxiliary_reports(stage_root, real_root, real_complete)

        final_pdf_report = validate_pdf(
            real_complete,
            expected_pages=selected_total,
            pages_root=Path(self.pages_root),
            start_page=start_page,
            source_pdf=Path(self.source_pdf) if preserve_images else None,
            preserve_source_images=preserve_images,
            minimum_translation_coverage=0.62,
        )
        final_delivery_report = validate_deliverables(
            (real_complete, *real_part_paths),
            expected_complete_pages=selected_total,
        )
        final_pdf_report["path"] = str(real_complete)
        write_integrity_report(
            integrity_path,
            signature=signature,
            pdf_report=final_pdf_report,
            delivery_report=final_delivery_report,
        )
        return real_complete, real_part_paths
    finally:
        self.output_root = original_output_root
        _remove_tree(stage_root)


def _normalize_layout_stable(value):
    if value == LAYOUT_SOURCE_TRANSLATED:
        return LAYOUT_SOURCE_TRANSLATED
    if _CORE_NORMALIZE_LAYOUT is None:
        return value
    return _CORE_NORMALIZE_LAYOUT(value)


def _stable_build_deliverables(self, *args, **kwargs):
    if _CORE_BUILD_DELIVERABLES is None:
        raise RuntimeError("翻译稳定性核心尚未捕获交付构建器。")
    if bool(getattr(self, "_phoenix_no_split", False)):
        kwargs["part_pages"] = 0
    outputs, image_count = _CORE_BUILD_DELIVERABLES(self, *args, **kwargs)
    expected_pages = int(args[4]) - int(args[3]) + 1 if len(args) >= 5 else None
    if expected_pages is None:
        start_page = int(kwargs.get("start_page", 1))
        total_pages = int(kwargs.get("total_pages", start_page))
        expected_pages = total_pages - start_page + 1
    validate_deliverables(
        outputs,
        expected_complete_pages=max(1, int(expected_pages)),
    )
    return outputs, image_count


def _checkpoint_paths(translator, pdf_path: Path, target_language: str):
    from .pdf_parser import sha256_file

    source = Path(pdf_path).resolve()
    digest = sha256_file(source)
    return translator._book_paths(source, digest, target_language)


def _invalidate_unstable_resume_pages(
    translator,
    pdf_path: Path,
    target_language: str,
    *,
    retry_warning_pages: bool,
) -> int:
    try:
        _, pages_root, audit_root, _, _ = _checkpoint_paths(
            translator,
            pdf_path,
            target_language,
        )
    except Exception:
        return 0

    removed = 0
    for audit_file in audit_root.glob("*.json"):
        try:
            payload = translator._read_json(audit_file)
            parts = payload.get("parts") or ()
            hard_failure = any(
                str(part.get("backend", "")).strip() == "failed_all"
                for part in parts
            )
            has_warning = int(payload.get("warning_count", 0) or 0) > 0
            if not hard_failure and not (retry_warning_pages and has_warning):
                continue
            page_file = pages_root / f"{int(audit_file.stem):06d}.txt"
            if page_file.is_file():
                page_file.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue
    return removed


def _rewrite_progress_for_no_split(progress):
    if progress is None:
        return None

    def callback(done, total, message):
        text = str(message)
        text = text.replace(
            "翻译完成，正在生成整书PDF与分册PDF……",
            "翻译完成，正在生成并验收完整PDF……",
        )
        text = text.replace(
            "整本翻译与PDF分册已完成。",
            "整本翻译与完整PDF验收已完成。",
        )
        progress(done, total, text)

    return callback


def _repair_checkpoint_part_pages_only(
    translator,
    source_path: Path,
    target_language: str,
) -> None:
    try:
        _, _, _, checkpoint, _ = _checkpoint_paths(
            translator,
            Path(source_path),
            str(target_language),
        )
        state = translator._read_json(checkpoint)
        if state and int(state.get("part_pages", 0) or 0) != 0:
            state["part_pages"] = 0
            translator._write_json(checkpoint, state)
    except Exception:
        pass


def _repair_checkpoint_after_validation(
    translator,
    result,
    *,
    no_split: bool,
    delivery_report: dict,
) -> None:
    try:
        _, _, _, checkpoint, _ = _checkpoint_paths(
            translator,
            Path(result.source_path),
            str(result.target_language),
        )
        state = translator._read_json(checkpoint)
        if not state:
            return
        if no_split:
            state["part_pages"] = 0
        state["delivery_validated"] = True
        state["delivery_contract"] = CONTRACT_VERSION
        state["delivery_manifest_sha256"] = str(
            delivery_report.get("manifest_sha256", "")
        )
        state.pop("error", None)
        translator._write_json(checkpoint, state)
    except Exception:
        pass


def _recover_completed_checkpoints(
    translator,
    result,
    *,
    no_split: bool,
):
    try:
        from .translation_recovery import _rebuild_result

        recovered = _rebuild_result(
            translator,
            Path(result.source_path),
            target_language=str(result.target_language),
            requested_start_page=int(result.start_page),
            previous=result,
        )
        if recovered is not None and no_split:
            try:
                recovered = replace(recovered, part_pages=0)
            except Exception:
                pass
        return recovered
    except Exception:
        return None


def _stable_translate_book(self, pdf_path: Path, **kwargs):
    if _CORE_TRANSLATE_BOOK is None:
        raise RuntimeError("翻译稳定性核心尚未捕获原始翻译入口。")

    source = Path(pdf_path)
    target_language = str(kwargs.get("target_language", "中文"))
    kwargs.setdefault("output_layout", LAYOUT_SOURCE_TRANSLATED)
    retry_warning_pages = bool(kwargs.get("retry_warning_pages", False))

    requested_part_pages = kwargs.get("part_pages", None)
    if requested_part_pages is None:
        no_split = True
    else:
        try:
            no_split = int(requested_part_pages) <= 0
        except (TypeError, ValueError):
            no_split = True

    removed = _invalidate_unstable_resume_pages(
        self,
        source,
        target_language,
        retry_warning_pages=retry_warning_pages,
    )
    original_progress = kwargs.get("progress")
    if removed and original_progress:
        original_progress(
            0,
            1,
            f"稳定性预检移除 {removed} 个硬失败/待重试页，将重新生成这些页。",
        )

    if no_split:
        # Core translator historically clamps this value to >=1. The final
        # deliverable hook below converts it back to zero without adding a
        # second translate_book wrapper.
        kwargs["part_pages"] = 1
        kwargs["progress"] = _rewrite_progress_for_no_split(original_progress)

    self._phoenix_no_split = bool(no_split)
    try:
        result = _CORE_TRANSLATE_BOOK(self, source, **kwargs)
        if bool(getattr(result, "paused", False)):
            if no_split:
                _repair_checkpoint_part_pages_only(
                    self,
                    Path(result.source_path),
                    str(result.target_language),
                )
                try:
                    result = replace(result, part_pages=0)
                except Exception:
                    pass
            return result

        expected_pages = int(result.total_pages) - int(result.start_page) + 1
        output_paths = tuple(Path(path) for path in result.output_paths)
        if not output_paths and Path(result.output_path).is_file():
            output_paths = (Path(result.output_path),)

        try:
            delivery_report = validate_deliverables(
                output_paths,
                expected_complete_pages=expected_pages,
            )
        except Exception:
            recovered = _recover_completed_checkpoints(
                self,
                result,
                no_split=no_split,
            )
            if recovered is None:
                raise
            result = recovered
            output_paths = tuple(Path(path) for path in result.output_paths)
            delivery_report = validate_deliverables(
                output_paths,
                expected_complete_pages=expected_pages,
            )

        if no_split:
            try:
                result = replace(result, part_pages=0)
            except Exception:
                pass
        _repair_checkpoint_after_validation(
            self,
            result,
            no_split=no_split,
            delivery_report=delivery_report,
        )
        return result
    except TranslationOutputError:
        raise
    finally:
        self._phoenix_no_split = False


def install_final() -> None:
    """Collapse the legacy patch stack into one deterministic runtime contract."""

    global _INSTALLED
    if _INSTALLED:
        return
    if not _CAPTURED:
        capture_core()

    from . import translator as translator_module
    from .translation_pdf import TranslationPDFBuilder
    from .translator import PDFTranslator

    translator_module._normalize_layout = _normalize_layout_stable
    TranslationPDFBuilder.build = _stable_pdf_build
    PDFTranslator._build_deliverables = _stable_build_deliverables
    PDFTranslator.translate_book = _stable_translate_book

    TranslationPDFBuilder._phoenix_stability_contract = CONTRACT_VERSION
    PDFTranslator._phoenix_stability_contract = CONTRACT_VERSION
    PDFTranslator._phoenix_translation_wrapper_depth = 1
    _INSTALLED = True
