from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .output_contracts import (
    CONTRACT_VERSION as OUTPUT_CONTRACT_VERSION,
    OutputContractError,
    sha256_file,
    transactional_export_path,
    validate_export_bundle,
    validate_text_file,
)

WORKBENCH_CONTRACT_VERSION = 2

_CAPTURED = False
_INSTALLED = False
_CORE_INIT = None
_CORE_STATUS = None
_CORE_INGEST = None
_CORE_ASK = None


def capture_core() -> None:
    """Capture unwrapped public Workbench methods before legacy installers run."""
    global _CAPTURED, _CORE_INIT, _CORE_STATUS, _CORE_INGEST, _CORE_ASK
    if _CAPTURED:
        return
    from .workbench import MedicalKnowledgeWorkbench
    _CORE_INIT = MedicalKnowledgeWorkbench.__init__
    _CORE_STATUS = MedicalKnowledgeWorkbench.status
    _CORE_INGEST = MedicalKnowledgeWorkbench.ingest
    _CORE_ASK = MedicalKnowledgeWorkbench.ask
    _CAPTURED = True


def _atomic_copy_verified(source: Path, target: Path) -> Path:
    source = Path(source).resolve()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_file(source)
    temp = target.with_name(f".pxcopy-{uuid.uuid4().hex[:10]}-{target.name}")
    temp.unlink(missing_ok=True)
    try:
        shutil.copy2(source, temp)
        if not temp.is_file() or temp.stat().st_size != source.stat().st_size:
            raise OutputContractError(f"资料库临时副本大小异常：{temp}")
        if sha256_file(temp) != expected:
            raise OutputContractError(f"资料库临时副本SHA256校验失败：{temp}")
        os.replace(temp, target)
        if sha256_file(target) != expected:
            raise OutputContractError(f"资料库正式副本SHA256校验失败：{target}")
        return target
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _stable_ingestor_class():
    from .product_document_ingest import ProductDocumentIngestor

    class StableProductDocumentIngestor(ProductDocumentIngestor):
        _phoenix_output_contract = OUTPUT_CONTRACT_VERSION

        def _library_copy(self, source: Path) -> Path:
            source = Path(source).resolve()
            root = Path(self.paths.source_root)
            root.mkdir(parents=True, exist_ok=True)
            direct = root / source.name
            if direct.resolve() == source:
                return source

            source_digest = sha256_file(source)
            stem, suffix = source.stem, source.suffix
            candidates: list[Path] = []
            if direct.is_file():
                candidates.append(direct)
            try:
                candidates.extend(
                    path for path in sorted(root.glob(f"{stem}_*{suffix}"))
                    if path.is_file()
                )
            except OSError:
                pass
            for candidate in candidates:
                try:
                    if sha256_file(candidate) == source_digest:
                        return candidate
                except OSError:
                    continue

            if not direct.exists():
                target = direct
            else:
                counter = 2
                while True:
                    target = root / f"{stem}_{counter}{suffix}"
                    if not target.exists():
                        break
                    counter += 1
            return _atomic_copy_verified(source, target)

    return StableProductDocumentIngestor


def _sqlite_quick_check(workbench) -> None:
    with workbench.db._lock:
        row = workbench.db._conn.execute("PRAGMA quick_check").fetchone()
    result = str(row[0] if row else "").strip().lower()
    if result != "ok":
        raise RuntimeError(
            "知识库SQLite完整性检查未通过，Phoenix已阻止继续写入。"
            f" quick_check={result or 'unknown'}"
        )


def _cleanup_old_temp_dirs(workbench, age_seconds: int = 6 * 3600) -> int:
    now = time.time()
    roots = {
        Path(workbench.paths.evidence_root),
        Path(workbench.paths.source_root),
        Path(workbench.paths.runtime_root),
    }
    removed = 0
    prefixes = (".pxexport-", ".pxnotes-", ".pxpdf-")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            items = list(root.iterdir())
        except OSError:
            continue
        for item in items:
            if not item.is_dir() or not item.name.startswith(prefixes):
                continue
            try:
                if now - item.stat().st_mtime < age_seconds:
                    continue
            except OSError:
                continue
            shutil.rmtree(item, ignore_errors=True)
            removed += 1
    return removed


def _commercial_release(workbench) -> bool:
    try:
        from .release_hardening import _commercial_release as checker
        return bool(checker(workbench.paths))
    except Exception:
        return False


def _runtime_flags(workbench) -> dict:
    try:
        from .release_runtime_hardening import (
            local_generation_runtime_ready,
            local_seq2seq_runtime_ready,
        )
        generation_runtime = bool(local_generation_runtime_ready())
        seq2seq_runtime = bool(local_seq2seq_runtime_ready())
    except Exception:
        generation_runtime = False
        seq2seq_runtime = False
    try:
        semantic = workbench.retriever.embeddings.readiness()
    except Exception as exc:
        chunks = int(workbench.db.count_chunks())
        semantic = {
            "state": "error",
            "label": f"语义状态读取失败: {type(exc).__name__}: {exc}",
            "ready": False,
            "model_ready": False,
            "runtime_ready": False,
            "chunks": chunks,
            "vectors": 0,
            "missing": chunks,
            "device": "unavailable",
        }
    return {
        "semantic": semantic,
        "generation_runtime": generation_runtime,
        "seq2seq_runtime": seq2seq_runtime,
    }


def _method_contract(method) -> int:
    try:
        return int(getattr(method, "_phoenix_workbench_contract", 0) or 0)
    except Exception:
        return 0


def architecture_status(workbench=None) -> dict:
    from .translation_pdf import TranslationPDFBuilder
    from .translator import PDFTranslator
    from .workbench import MedicalKnowledgeWorkbench

    names = (
        "__init__", "status", "ingest", "ask", "organize", "resume_task",
        "translate_book", "organize_txt", "organize_txt_file",
    )
    method_modules = {}
    broken = []
    for name in names:
        method = getattr(MedicalKnowledgeWorkbench, name, None)
        method_modules[name] = getattr(method, "__module__", "")
        if not callable(method):
            broken.append(f"{name}:missing")
        elif _method_contract(method) != WORKBENCH_CONTRACT_VERSION:
            broken.append(f"{name}:contract")

    wb_depth = int(
        getattr(MedicalKnowledgeWorkbench, "_phoenix_workbench_wrapper_depth", 0)
        or 0
    )
    if wb_depth != 1:
        broken.append(f"workbench_wrapper_depth={wb_depth}")
    tr_depth = int(getattr(PDFTranslator, "_phoenix_translation_wrapper_depth", 0) or 0)
    if tr_depth != 1:
        broken.append(f"translation_wrapper_depth={tr_depth}")
    if not getattr(PDFTranslator, "_phoenix_stability_contract", None):
        broken.append("translation_contract_missing")
    if not getattr(TranslationPDFBuilder, "_phoenix_stability_contract", None):
        broken.append("pdf_builder_contract_missing")
    if workbench is not None:
        ingestor = getattr(workbench, "ingestor", None)
        if getattr(type(ingestor), "_phoenix_output_contract", 0) != OUTPUT_CONTRACT_VERSION:
            broken.append("stable_ingestor_missing")

    fingerprint_payload = {
        "workbench_contract": WORKBENCH_CONTRACT_VERSION,
        "output_contract": OUTPUT_CONTRACT_VERSION,
        "workbench_depth": wb_depth,
        "translation_depth": tr_depth,
        "methods": method_modules,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "ready": not broken,
        "broken": broken,
        "fingerprint": fingerprint,
        "workbench_wrapper_depth": wb_depth,
        "translation_wrapper_depth": tr_depth,
        "method_modules": method_modules,
    }


def _stable_init(self, *args, **kwargs):
    if _CORE_INIT is None:
        raise RuntimeError("Workbench稳定性核心尚未捕获原始初始化入口。")
    _CORE_INIT(self, *args, **kwargs)
    self._phoenix_output_lock = threading.RLock()
    StableIngestor = _stable_ingestor_class()
    self.ingestor = StableIngestor(self.db, self.paths)
    _sqlite_quick_check(self)
    try:
        from .release_portability import rebase_stale_document_paths
        rebased, unresolved = rebase_stale_document_paths(self)
    except Exception:
        rebased, unresolved = 0, 0
    self._portable_paths_rebased = int(rebased)
    self._portable_paths_unresolved = int(unresolved)
    self._phoenix_old_temp_removed = _cleanup_old_temp_dirs(self)


def _stable_status(self) -> dict:
    if _CORE_STATUS is None:
        raise RuntimeError("Workbench稳定性核心尚未捕获原始状态入口。")
    payload = dict(_CORE_STATUS(self))
    runtime = _runtime_flags(self)
    semantic = runtime["semantic"]
    architecture = architecture_status(self)
    try:
        smart1 = bool(self.llm.available("fast"))
    except Exception:
        smart1 = False
    try:
        smart2 = bool(self.llm.available("deep"))
    except Exception:
        smart2 = False
    payload.update({
        "semantic_ready": bool(semantic.get("ready")),
        "semantic_state": str(semantic.get("state", "")),
        "semantic_label": str(semantic.get("label", "")),
        "embedding_available": bool(semantic.get("ready")),
        "embedding_model_ready": bool(semantic.get("model_ready")),
        "embedding_runtime_available": bool(semantic.get("runtime_ready")),
        "embedding_vectors": int(semantic.get("vectors", 0) or 0),
        "embedding_missing": int(semantic.get("missing", 0) or 0),
        "embedding_chunks": int(semantic.get("chunks", 0) or 0),
        "embedding_device": str(semantic.get("device", "unavailable")),
        "generator_runtime_available": runtime["generation_runtime"],
        "translation_seq2seq_runtime_available": runtime["seq2seq_runtime"],
        "generator_fast_ready": smart1,
        "generator_deep_ready": smart2,
        "generator_fast_active_model": self.llm.active_model_name("fast"),
        "generator_deep_active_model": self.llm.active_model_name("deep"),
        "commercial_release": _commercial_release(self),
        "document_paths_rebased": int(getattr(self, "_portable_paths_rebased", 0) or 0),
        "document_paths_unresolved": int(getattr(self, "_portable_paths_unresolved", 0) or 0),
        "workbench_contract": WORKBENCH_CONTRACT_VERSION,
        "output_contract": OUTPUT_CONTRACT_VERSION,
        "architecture_ready": bool(architecture["ready"]),
        "architecture_broken": list(architecture["broken"]),
        "architecture_fingerprint": architecture["fingerprint"],
        "workbench_wrapper_depth": architecture["workbench_wrapper_depth"],
        "translation_wrapper_depth": architecture["translation_wrapper_depth"],
    })
    return payload


def _validate_ingest_result(self, result):
    stored = Path(result.copied_to_library)
    if not stored.is_file() or stored.stat().st_size <= 0:
        raise OutputContractError(f"导入返回成功，但资料库副本不存在/为空：{stored}")
    row = self.db.get_document(int(result.document_id))
    if row is None:
        raise OutputContractError(
            f"导入返回成功，但数据库没有 document_id={result.document_id}"
        )
    expected = str(row["sha256"] or "").strip().lower()
    actual = sha256_file(stored).lower()
    if expected and expected != actual:
        raise OutputContractError(f"导入资料SHA256与数据库不一致：{stored}")
    total = int(result.pages_total)
    indexed = int(result.pages_indexed)
    if total <= 0 or indexed < 0 or indexed > total:
        raise OutputContractError(
            f"导入页数状态异常：indexed={indexed}, total={total}"
        )
    return result


def _stable_ingest(self, path: Path, **kwargs):
    if _CORE_INGEST is None:
        raise RuntimeError("Workbench稳定性核心尚未捕获原始导入入口。")
    defer_embeddings = bool(kwargs.pop("_defer_embeddings", False))
    result = _CORE_INGEST(self, Path(path), **kwargs)
    _validate_ingest_result(self, result)
    try:
        self.retriever.embeddings._invalidate_vector_index()
    except Exception:
        pass
    if defer_embeddings:
        return result
    progress = kwargs.get("progress")
    try:
        semantic = self.retriever.embeddings.readiness()
        if (
            semantic.get("model_ready")
            and semantic.get("runtime_ready")
            and int(semantic.get("missing", 0) or 0) > 0
        ):
            missing = max(1, int(semantic.get("missing", 0) or 0))
            if progress:
                progress(0, missing, "资料导入完成，正在补齐语义向量……")
            def callback(done, _total, message):
                if progress:
                    progress(min(max(0, int(done)), missing), missing, str(message))
            self.retriever.embeddings.build_missing(
                progress=callback if progress else None
            )
    except Exception as exc:
        extra = f"语义向量补齐失败，已保留关键词检索：{type(exc).__name__}: {exc}"
        result.warning = (
            f"{result.warning}；{extra}" if getattr(result, "warning", "") else extra
        )
    return result


def _unload_llm(self) -> None:
    try:
        self.llm.unload()
    except Exception:
        pass


def _unload_embeddings(self) -> None:
    try:
        self.retriever.embeddings.unload_model()
    except Exception:
        pass


def _stable_ask(self, query: str, **kwargs):
    if _CORE_ASK is None:
        raise RuntimeError("Workbench稳定性核心尚未捕获原始问答入口。")
    if kwargs.get("use_embeddings", True):
        _unload_llm(self)
    try:
        result = _CORE_ASK(self, query, **kwargs)
        if not str(getattr(result, "text", "") or "").strip():
            raise RuntimeError("问答返回空文本")
        return result
    except Exception as exc:
        fallback = _CORE_ASK(
            self,
            query,
            limit=int(kwargs.get("limit", 18) or 18),
            use_embeddings=False,
            deep=False,
        )
        text = str(getattr(fallback, "text", "") or "").strip()
        if not text:
            raise
        from .answerer import AnswerResult
        return AnswerResult(
            text=(
                "【降级保护】语义/智能阶段发生异常，Phoenix已自动退回"
                "本地关键词证据，不使用未验证生成结果。\n"
                f"原因：{type(exc).__name__}: {exc}\n\n{text}"
            ),
            evidence=list(getattr(fallback, "evidence", []) or []),
            mode="degraded_evidence_only",
        )


def _mark_task_failed(self, task_id: int, exc: Exception) -> None:
    if int(task_id or 0) <= 0:
        return
    try:
        self.db.update_task(
            int(task_id),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception:
        pass


def _strict_export(self, output: Path, task_id: int, *, title: str | None = None):
    self.last_export_bundle = None
    self.last_export_error = ""
    try:
        validate_text_file(Path(output), label="联合整理正文")
        with self._phoenix_output_lock:
            bundle = transactional_export_path(
                self.exporter,
                Path(output),
                title=title,
            )
        validate_export_bundle(bundle)
        self.last_export_bundle = bundle
        return bundle
    except Exception as exc:
        self.last_export_bundle = None
        self.last_export_error = f"{type(exc).__name__}: {exc}"
        _mark_task_failed(self, task_id, exc)
        if isinstance(exc, OutputContractError):
            raise
        raise OutputContractError(
            "联合整理正文已生成，但多格式成品未全部通过完整性验收；"
            f"任务未被标记为完成。原因：{type(exc).__name__}: {exc}"
        ) from exc


def _stable_organize(self, title: str, instruction: str, **kwargs):
    output, task_id = self.organizer.organize(title, instruction, **kwargs)
    _strict_export(
        self,
        Path(output),
        int(task_id or 0),
        title=title or Path(output).stem,
    )
    return output, task_id


def _stable_resume_task(self, task_id: int, **kwargs):
    _unload_embeddings(self)
    output, resumed_id = self.organizer.resume(int(task_id), **kwargs)
    _strict_export(self, Path(output), int(resumed_id or task_id))
    return output, resumed_id


def _stable_translate_book(self, path: Path, **kwargs):
    _unload_embeddings(self)
    result = self.translator.translate_book(Path(path), **kwargs)
    if bool(getattr(result, "paused", False)):
        return result
    from .translation_output_validation import validate_deliverables
    outputs = tuple(Path(p) for p in (getattr(result, "output_paths", ()) or ()))
    if not outputs:
        outputs = (Path(result.output_path),)
    expected = int(result.total_pages) - int(result.start_page) + 1
    validate_deliverables(outputs, expected_complete_pages=max(1, expected))
    return result


def _begin_file_publish(staged: Path, final: Path) -> Path | None:
    staged = Path(staged)
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    incoming = final.with_name(final.name + ".incoming")
    backup = final.with_name(final.name + ".backup")
    incoming.unlink(missing_ok=True)
    backup.unlink(missing_ok=True)
    os.replace(staged, incoming)
    old = final.is_file()
    if old:
        os.replace(final, backup)
    try:
        os.replace(incoming, final)
    except Exception:
        if old and backup.is_file() and not final.exists():
            try:
                os.replace(backup, final)
            except Exception:
                pass
        raise
    return backup if old else None


def _rollback_file(final: Path, backup: Path | None) -> None:
    final = Path(final)
    failed = final.with_name(final.name + ".failed")
    failed.unlink(missing_ok=True)
    try:
        if final.exists():
            os.replace(final, failed)
    except Exception:
        pass
    try:
        if backup is not None and backup.exists():
            os.replace(backup, final)
    finally:
        failed.unlink(missing_ok=True)


def _notes_transaction(self, method_name: str, *args, **kwargs):
    _unload_embeddings(self)
    real_root = Path(self.notes.output_root)
    real_root.mkdir(parents=True, exist_ok=True)
    stage_root = real_root / f".pxnotes-{uuid.uuid4().hex[:10]}"
    stage_root.mkdir(parents=True, exist_ok=True)
    original_root = self.notes.output_root
    backup = None
    final = None
    did_publish = False
    try:
        self.notes.output_root = stage_root
        method = getattr(self.notes, method_name)
        staged_result = method(*args, **kwargs)
        staged = Path(staged_result.output_path)
        staged_report = validate_text_file(staged, label="笔记成品")
        final = real_root / staged.name
        backup = _begin_file_publish(staged, final)
        did_publish = True
        final_report = validate_text_file(final, label="正式笔记成品")
        if final_report["sha256"] != staged_report["sha256"]:
            raise OutputContractError("笔记发布后SHA256变化，已阻止损坏成品。")
        if backup is not None:
            backup.unlink(missing_ok=True)
            backup = None
        did_publish = False
        return replace(staged_result, output_path=final)
    except Exception:
        if final is not None and did_publish:
            _rollback_file(final, backup)
            backup = None
            did_publish = False
        raise
    finally:
        self.notes.output_root = original_root
        shutil.rmtree(stage_root, ignore_errors=True)
        if backup is not None:
            backup.unlink(missing_ok=True)


def _stable_organize_txt(self, source_text: str, **kwargs):
    return _notes_transaction(self, "organize", source_text, **kwargs)


def _stable_organize_txt_file(self, path: Path, **kwargs):
    return _notes_transaction(self, "organize_file", Path(path), **kwargs)


def _tag(method):
    method._phoenix_workbench_contract = WORKBENCH_CONTRACT_VERSION
    return method


def install_final() -> None:
    """Replace the historical public wrapper stack with one explicit contract."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not _CAPTURED:
        capture_core()
    from .workbench import MedicalKnowledgeWorkbench
    MedicalKnowledgeWorkbench.__init__ = _tag(_stable_init)
    MedicalKnowledgeWorkbench.status = _tag(_stable_status)
    MedicalKnowledgeWorkbench.ingest = _tag(_stable_ingest)
    MedicalKnowledgeWorkbench.ask = _tag(_stable_ask)
    MedicalKnowledgeWorkbench.organize = _tag(_stable_organize)
    MedicalKnowledgeWorkbench.resume_task = _tag(_stable_resume_task)
    MedicalKnowledgeWorkbench.translate_book = _tag(_stable_translate_book)
    MedicalKnowledgeWorkbench.organize_txt = _tag(_stable_organize_txt)
    MedicalKnowledgeWorkbench.organize_txt_file = _tag(_stable_organize_txt_file)
    MedicalKnowledgeWorkbench._phoenix_workbench_contract = WORKBENCH_CONTRACT_VERSION
    MedicalKnowledgeWorkbench._phoenix_workbench_wrapper_depth = 1
    _INSTALLED = True
