from __future__ import annotations

import importlib
import os
from pathlib import Path

_INSTALLED = False


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_module_available(name: str) -> bool:
    """READY means the runtime can actually import, not merely find a folder."""
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _model_dir_ready(path: Path) -> bool:
    path = Path(path)
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


def _embedding_readiness(self) -> dict:
    chunks = int(self.db.count_chunks())
    try:
        vectors = len(self.db.iter_embeddings(self.model_name))
    except Exception:
        vectors = 0
    model_ready = _model_dir_ready(self.model_path)
    runtime_ready = _runtime_module_available("sentence_transformers")
    missing = max(0, chunks - vectors)
    ready = bool(model_ready and runtime_ready and missing == 0)
    if not model_ready:
        state, label = "model_missing", "语义模型未下载"
    elif not runtime_ready:
        state, label = "runtime_missing", "语义组件缺失或加载失败"
    elif chunks == 0:
        state, label = "ready", "语义检索就绪（资料库为空）"
    elif missing > 0:
        state, label = "index_incomplete", f"语义索引 {vectors}/{chunks}"
    else:
        state, label = "ready", f"语义索引 {vectors}/{chunks} READY"
    return {
        "state": state,
        "label": label,
        "ready": ready,
        "model_ready": model_ready,
        "runtime_ready": runtime_ready,
        "chunks": chunks,
        "vectors": vectors,
        "missing": missing,
        "device": self.device if model_ready and runtime_ready else "unavailable",
    }


def _commercial_release(paths) -> bool:
    try:
        from .licensing import product_mode_enabled
        if product_mode_enabled(paths.project_root):
            return True
    except Exception:
        pass
    return _truthy("PHOENIX_COMMERCIAL_RELEASE")


def _audit_has_hard_failure(payload: dict) -> bool:
    return any(
        str(part.get("backend", "")).strip() == "failed_all"
        for part in (payload.get("parts") or ())
    )


def _prepare_hard_failed_pages(translator, pdf_path: Path, target_language: str) -> int:
    try:
        from .pdf_parser import sha256_file
        source = Path(pdf_path).resolve()
        if not source.is_file() or source.suffix.lower() != ".pdf":
            return 0
        digest = sha256_file(source)
        _, pages_root, audit_root, _, _ = translator._book_paths(
            source,
            digest,
            target_language,
        )
    except Exception:
        return 0

    removed = 0
    for audit_file in audit_root.glob("*.json"):
        try:
            payload = translator._read_json(audit_file)
            if not _audit_has_hard_failure(payload):
                continue
            page_file = pages_root / f"{int(audit_file.stem):06d}.txt"
            if page_file.is_file():
                page_file.unlink(missing_ok=True)
                removed += 1
        except Exception:
            continue
    return removed


def _atomic_pdf_save(doc, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    temp.unlink(missing_ok=True)
    try:
        doc.save(str(temp), garbage=0, deflate=True)
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _safe_translation_pdf_build(
    self,
    *,
    start_page: int,
    total_pages: int,
    layout: str,
    part_pages: int = 50,
    progress=None,
):
    import fitz
    from .translation_pdf import (
        LAYOUT_ORIGINAL_BILINGUAL,
        LAYOUT_TEXT_BILINGUAL,
        LAYOUT_TRANSLATED_ONLY,
    )

    layout = (
        layout
        if layout in {
            LAYOUT_ORIGINAL_BILINGUAL,
            LAYOUT_TEXT_BILINGUAL,
            LAYOUT_TRANSLATED_ONLY,
        }
        else LAYOUT_ORIGINAL_BILINGUAL
    )
    part_pages = max(1, int(part_pages))
    start_page = max(1, int(start_page))
    total_pages = max(start_page, int(total_pages))
    complete_name = (
        "完整双语译本_原页在上中文在下.pdf"
        if layout == LAYOUT_ORIGINAL_BILINGUAL
        else "完整双语译本_英文在上中文在下.pdf"
        if layout == LAYOUT_TEXT_BILINGUAL
        else "完整中文译本.pdf"
    )
    complete_path = self.output_root / complete_name
    parts_root = self.output_root / "PDF分册"
    parts_root.mkdir(parents=True, exist_ok=True)
    for old in parts_root.glob("第*.pdf"):
        old.unlink(missing_ok=True)

    selected_total = total_pages - start_page + 1
    source_doc = fitz.open(self.source_pdf)
    out_doc = fitz.open()
    try:
        for offset, page_number in enumerate(
            range(start_page, total_pages + 1),
            start=1,
        ):
            source_index = page_number - 1
            if source_index < 0 or source_index >= source_doc.page_count:
                raise RuntimeError(f"源PDF不存在第 {page_number} 页")
            if layout == LAYOUT_ORIGINAL_BILINGUAL:
                self._append_original_bilingual_page(
                    out_doc, source_doc, source_index, page_number
                )
            elif layout == LAYOUT_TEXT_BILINGUAL:
                self._append_text_bilingual_page(
                    out_doc, source_doc, source_index, page_number
                )
            else:
                self._append_translated_only_page(
                    out_doc, source_doc, source_index, page_number
                )
            if progress:
                progress(
                    offset,
                    selected_total,
                    f"正在生成PDF页面：第 {page_number}/{total_pages} 页",
                )
        if progress:
            progress(
                selected_total,
                selected_total,
                "页面已生成，正在压缩写入完整PDF；此阶段不会重新翻译。",
            )
        _atomic_pdf_save(out_doc, complete_path)
    finally:
        out_doc.close()
        source_doc.close()

    complete = fitz.open(complete_path)
    part_paths: list[Path] = []
    try:
        total_output_pages = complete.page_count
        part_total = max(1, (total_output_pages + part_pages - 1) // part_pages)
        for part_index, start_index in enumerate(
            range(0, total_output_pages, part_pages), start=1
        ):
            end_index = min(total_output_pages - 1, start_index + part_pages - 1)
            first_source_page = start_page + start_index
            last_source_page = start_page + end_index
            part_path = parts_root / (
                f"第{part_index:03d}册_{first_source_page:04d}-{last_source_page:04d}.pdf"
            )
            part_doc = fitz.open()
            try:
                part_doc.insert_pdf(
                    complete,
                    from_page=start_index,
                    to_page=end_index,
                )
                if progress:
                    progress(
                        selected_total,
                        selected_total,
                        f"正在写入PDF分册 {part_index}/{part_total}："
                        f"{first_source_page}-{last_source_page} 页",
                    )
                _atomic_pdf_save(part_doc, part_path)
            finally:
                part_doc.close()
            part_paths.append(part_path)
    finally:
        complete.close()

    if progress:
        progress(
            selected_total,
            selected_total,
            f"PDF成品写入完成：完整PDF + {len(part_paths)} 册分册。",
        )
    return complete_path, tuple(part_paths)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .retrieval import EmbeddingEngine
    from .translation_models import MultiModelTranslationEngine
    from .translation_pdf import TranslationPDFBuilder
    from .translator import PDFTranslator
    from .workbench import MedicalKnowledgeWorkbench

    if not hasattr(EmbeddingEngine, "readiness"):
        EmbeddingEngine.readiness = _embedding_readiness

    original_status = MedicalKnowledgeWorkbench.status

    def status(self):
        payload = original_status(self)
        semantic = self.retriever.embeddings.readiness()
        payload.update(
            {
                "semantic_ready": semantic["ready"],
                "semantic_state": semantic["state"],
                "semantic_label": semantic["label"],
                "embedding_available": semantic["ready"],
                "embedding_model_ready": semantic["model_ready"],
                "embedding_runtime_available": semantic["runtime_ready"],
                "embedding_vectors": semantic["vectors"],
                "embedding_missing": semantic["missing"],
                "embedding_chunks": semantic["chunks"],
                "embedding_device": semantic["device"],
                "commercial_release": _commercial_release(self.paths),
            }
        )
        return payload

    MedicalKnowledgeWorkbench.status = status

    original_ingest = MedicalKnowledgeWorkbench.ingest

    def ingest(self, path: Path, **kwargs):
        defer_embeddings = bool(kwargs.pop("_defer_embeddings", False))
        result = original_ingest(self, path, **kwargs)
        try:
            self.retriever.embeddings._invalidate_vector_index()
        except Exception:
            pass

        if defer_embeddings:
            return result

        progress = kwargs.get("progress")
        semantic = self.retriever.embeddings.readiness()
        if (
            semantic["model_ready"]
            and semantic["runtime_ready"]
            and semantic["missing"] > 0
        ):
            if progress:
                progress(
                    0,
                    max(1, semantic["missing"]),
                    "资料索引完成，正在自动补齐语义向量……",
                )
            missing_total = max(1, int(semantic["missing"]))

            def vector_progress(done, _total, message):
                if progress:
                    progress(
                        min(max(0, int(done)), missing_total),
                        missing_total,
                        str(message),
                    )

            self.retriever.embeddings.build_missing(
                progress=vector_progress if progress else None
            )
        return result

    MedicalKnowledgeWorkbench.ingest = ingest

    original_available_backends = MultiModelTranslationEngine.available_backends
    original_active_backends = MultiModelTranslationEngine.active_backends

    def available_backends(self):
        names = list(original_available_backends(self))
        return [
            name
            for name in names
            if not (_commercial_release(self.paths) and name == self.nllb.name)
        ]

    def active_backends(self, target_language="中文", smart_level="smart1"):
        backends = list(
            original_active_backends(self, target_language, smart_level)
        )
        return [
            backend
            for backend in backends
            if not (
                _commercial_release(self.paths)
                and getattr(backend, "name", "") == self.nllb.name
            )
        ]

    MultiModelTranslationEngine.available_backends = available_backends
    MultiModelTranslationEngine.active_backends = active_backends

    original_translate_book = PDFTranslator.translate_book

    def translate_book(self, pdf_path: Path, **kwargs):
        target_language = str(kwargs.get("target_language", "中文"))
        hard_failed = _prepare_hard_failed_pages(
            self, Path(pdf_path), target_language
        )
        if hard_failed and kwargs.get("progress"):
            kwargs["progress"](
                0,
                1,
                f"检测到 {hard_failed} 个硬失败页，将自动重新翻译，不再错误跳过。",
            )
        return original_translate_book(self, pdf_path, **kwargs)

    PDFTranslator.translate_book = translate_book
    TranslationPDFBuilder.build = _safe_translation_pdf_build
