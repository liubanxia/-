from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
import uuid
from pathlib import Path


_CITATION_RE = re.compile(r"\[S\d+\]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _ok(label: str, detail: str = "") -> None:
    print(
        f"PASS  {label}" + (f" | {detail}" if detail else ""),
        flush=True,
    )


def _fail(label: str, detail: str) -> None:
    print(f"FAIL  {label} | {detail}", flush=True)


def _force_local_acceptance() -> None:
    """Acceptance is local by default and never inherits cloud authorization."""

    os.environ["PHOENIX_KNOWLEDGE_ACCELERATOR"] = "auto"
    os.environ["PHOENIX_KNOWLEDGE_ALLOW_REMOTE"] = "0"
    for name in (
        "PHOENIX_KNOWLEDGE_REMOTE_API_KEY",
        "PHOENIX_KNOWLEDGE_REMOTE_URL",
        "PHOENIX_KNOWLEDGE_REMOTE_MODEL_FAST",
        "PHOENIX_KNOWLEDGE_REMOTE_MODEL_DEEP",
    ):
        os.environ.pop(name, None)


def _make_acceptance_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    texts = (
        "CT demonstrates no pleural effusion and a 12 mm lesion in the right kidney.",
        "MRI demonstrates a small lesion in the left hepatic lobe without restricted diffusion.",
    )
    for text in texts:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(
            fitz.Rect(48, 60, 545, 760),
            text,
            fontsize=12,
        )
    doc.save(str(path))
    doc.close()


def _validate_evidence_pages(evidence) -> None:
    import fitz

    pdf_pages: dict[str, int] = {}
    for item in evidence:
        path = Path(item.path)
        if not path.is_file():
            raise RuntimeError(
                f"引用源文件不存在：{item.title} -> {path}"
            )
        if path.suffix.lower() != ".pdf":
            continue
        key = str(path.resolve())
        if key not in pdf_pages:
            doc = fitz.open(str(path))
            try:
                pdf_pages[key] = int(doc.page_count)
            finally:
                doc.close()
        max_page = pdf_pages[key]
        if int(item.page) < 1 or int(item.page) > max_page:
            raise RuntimeError(
                f"引用页码越界：{item.title} page={item.page}，"
                f"实际PDF={max_page}页"
            )


def _sandbox_paths(source_workbench, root: Path):
    from phoenix_knowledge.config import WorkbenchPaths

    root = Path(root)
    runtime = root / "runtime"
    paths = WorkbenchPaths(
        project_root=source_workbench.paths.project_root,
        source_root=source_workbench.paths.source_root,
        runtime_root=runtime,
        evidence_root=root / "evidence",
        model_root=source_workbench.paths.model_root,
        database=runtime / "knowledge.sqlite3",
        structure_root=runtime / "structure",
    ).ensure()

    backup = sqlite3.connect(paths.database)
    try:
        with source_workbench.db._lock:
            source_workbench.db._conn.backup(backup)
    finally:
        backup.close()
    return paths


def _license_detail(project_root: Path) -> str:
    from phoenix_knowledge.licensing import LicenseManager

    status = LicenseManager(project_root).status()
    if status.product_mode and not status.valid:
        raise RuntimeError(status.message)
    if status.product_mode:
        return (
            f"正式版授权有效 / edition={status.edition} / "
            f"license={status.license_id or '-'}"
        )
    return "Development 模式；正式打包后仍必须通过离线授权门"


def _gui_contract_smoke(sandbox_paths) -> str:
    """Instantiate the same enhanced GUI stack as app.py without user data writes."""

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget
    from phoenix_knowledge import MedicalKnowledgeWorkbench
    from phoenix_knowledge import gui as gui_module
    from phoenix_knowledge.compute_gui import install as install_compute_gui
    from phoenix_knowledge.document_gui import install as install_document_gui
    from phoenix_knowledge.gui_enhancements import (
        install as install_gui_enhancements,
    )

    install_gui_enhancements(gui_module)
    install_compute_gui(gui_module)
    install_document_gui(gui_module)

    original_factory = gui_module.MedicalKnowledgeWorkbench
    gui_module.MedicalKnowledgeWorkbench = (
        lambda: MedicalKnowledgeWorkbench(sandbox_paths)
    )
    app = QApplication.instance() or QApplication([])
    window = None
    try:
        window = gui_module.WorkbenchWindow()
        tabs = window.centralWidget()
        if not isinstance(tabs, QTabWidget):
            raise RuntimeError("主窗口中央区域不是产品页签容器")
        names = [tabs.tabText(i) for i in range(tabs.count())]
        required = (
            "医学资料库",
            "资料问答",
            "多资料/论文联合整理",
            "整本书翻译",
            "笔记整理",
        )
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError(
                "产品页签缺失：" + ", ".join(missing)
            )
        if not hasattr(window, "compute_status_label"):
            raise RuntimeError("发布版算力状态组件未接入")
        status_text = window._status_text()
        if not status_text.strip():
            raise RuntimeError("主窗口运行状态为空")
        return " / ".join(names)
    finally:
        gui_module.MedicalKnowledgeWorkbench = original_factory
        if window is not None:
            try:
                window.close()
            except Exception:
                try:
                    window.workbench.close()
                except Exception:
                    pass
            window.deleteLater()
        app.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phoenix 正式上线真实平台验收"
    )
    parser.add_argument(
        "--query",
        default="肿瘤影像学中CT有什么作用？",
        help="用于真实资料检索、问答和联合整理的验收问题",
    )
    parser.add_argument(
        "--skip-organize",
        action="store_true",
        help="只在需要快速复测时跳过多资料联合整理",
    )
    args = parser.parse_args()

    _force_local_acceptance()
    os.environ["PHOENIX_KNOWLEDGE_DEEP_QA"] = "1"
    os.environ["PHOENIX_KNOWLEDGE_LLM_PROFILE"] = "fast"

    from phoenix_knowledge import MedicalKnowledgeWorkbench
    from phoenix_knowledge.translation_pdf import LAYOUT_TRANSLATED_ONLY
    from phoenix_knowledge.translator import EXPORT_PDF

    failures: list[str] = []

    with tempfile.TemporaryDirectory(
        prefix="phoenix_release_acceptance_"
    ) as td:
        root = Path(td)
        source_workbench = MedicalKnowledgeWorkbench()
        try:
            try:
                detail = _license_detail(
                    source_workbench.paths.project_root
                )
                _ok("产品授权门", detail)
            except Exception as exc:
                failures.append(f"产品授权门: {exc}")
                _fail("产品授权门", str(exc))

            source_status = source_workbench.status()
            unresolved = int(
                source_status.get(
                    "document_paths_unresolved",
                    0,
                )
                or 0
            )
            if unresolved:
                failures.append(
                    f"SSD资料路径: 尚有 {unresolved} 份来源无法定位"
                )
                _fail(
                    "SSD资料路径",
                    f"尚有 {unresolved} 份来源无法定位",
                )
            else:
                _ok(
                    "SSD资料路径",
                    "全部来源可定位；跨盘符自动重定位已完成",
                )

            sandbox_paths = _sandbox_paths(
                source_workbench,
                root,
            )
        finally:
            source_workbench.close()

        workbench = MedicalKnowledgeWorkbench(sandbox_paths)
        try:
            status = workbench.status()
            print(
                "========== Phoenix 真实平台上线验收 ==========",
                flush=True,
            )
            print(
                "说明：本次验收使用真实资料库的SQLite安全副本；"
                "不会写入正式任务、译本或整理结果。",
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "documents": status["documents"],
                        "chunks": status["chunks"],
                        "semantic_label": status.get(
                            "semantic_label"
                        ),
                        "compute": workbench.llm.compute_status(),
                        "translation_backends": status.get(
                            "translation_backends"
                        ),
                        "commercial_release": status.get(
                            "commercial_release"
                        ),
                        "document_paths_unresolved": status.get(
                            "document_paths_unresolved"
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

            try:
                if (
                    int(status["documents"]) <= 0
                    or int(status["chunks"]) <= 0
                ):
                    raise RuntimeError("当前资料库为空")
                if not status.get("semantic_ready"):
                    raise RuntimeError(
                        str(
                            status.get("semantic_label")
                            or "语义检索未就绪"
                        )
                    )
                _ok(
                    "资料库与语义索引",
                    f"{status['documents']}份 / "
                    f"{status['chunks']}块 / "
                    f"{status.get('embedding_vectors', 0)}向量",
                )
            except Exception as exc:
                failures.append(
                    f"资料库与语义索引: {exc}"
                )
                _fail("资料库与语义索引", str(exc))

            hits = []
            try:
                hits = workbench.retriever.search(
                    args.query,
                    limit=8,
                    use_embeddings=True,
                )
                if not hits:
                    raise RuntimeError(
                        "中文问题未召回任何真实资料"
                    )
                _validate_evidence_pages(hits)
                _ok(
                    "中文跨语言检索",
                    f"HITS={len(hits)} / TOP={hits[0].citation} "
                    f"{hits[0].title} 第{hits[0].page}页",
                )
            except Exception as exc:
                failures.append(f"中文跨语言检索: {exc}")
                _fail("中文跨语言检索", str(exc))

            try:
                if not workbench.llm.available("fast"):
                    raise RuntimeError("智能1本地模型未就绪")
                answer = workbench.ask(args.query, deep=True)
                if not answer.evidence:
                    raise RuntimeError("问答没有证据")
                if answer.mode != "grounded_generation":
                    raise RuntimeError(
                        "智能问答未形成有效引用答案："
                        f"mode={answer.mode}"
                    )
                if not _CITATION_RE.search(answer.text):
                    raise RuntimeError(
                        "问答输出没有[S编号]引用"
                    )
                _validate_evidence_pages(answer.evidence)
                _ok(
                    "真实资料智能问答",
                    f"mode={answer.mode} / "
                    f"evidence={len(answer.evidence)}",
                )
            except Exception as exc:
                failures.append(
                    f"真实资料智能问答: {exc}"
                )
                _fail("真实资料智能问答", str(exc))

            try:
                decision = workbench.translator.engine.translate(
                    "CT demonstrates no pleural effusion and a 12 mm lesion in the right kidney.",
                    "中文",
                    smart_level="smart2",
                )
                if not decision.quality.ok:
                    raise RuntimeError(
                        "医学翻译质量门未通过："
                        + "; ".join(
                            decision.quality.reasons
                        )
                    )
                translated = decision.text
                if (
                    "12" not in translated
                    or "mm" not in translated
                ):
                    raise RuntimeError("12 mm 未完整保留")
                if len(_CJK_RE.findall(translated)) < 4:
                    raise RuntimeError("翻译输出中文不足")
                _ok(
                    "医学翻译与安全校验",
                    f"{decision.backend} / "
                    f"score={decision.quality.score:.2f} / "
                    f"{translated}",
                )
            except Exception as exc:
                failures.append(
                    f"医学翻译与安全校验: {exc}"
                )
                _fail("医学翻译与安全校验", str(exc))

            try:
                import fitz

                synthetic = root / "Phoenix_上线验收.pdf"
                _make_acceptance_pdf(synthetic)
                result = workbench.translate_book(
                    synthetic,
                    start_page=1,
                    target_language="中文",
                    smart_level="smart2",
                    output_layout=LAYOUT_TRANSLATED_ONLY,
                    export_format=EXPORT_PDF,
                    part_pages=50,
                    progress=lambda done, total, msg: print(
                        f"TRANSLATE {done}/{total} {msg}",
                        flush=True,
                    ),
                )
                outputs = tuple(
                    result.output_paths
                    or (result.output_path,)
                )
                if not outputs or not all(
                    Path(p).is_file()
                    and Path(p).stat().st_size > 0
                    for p in outputs
                ):
                    raise RuntimeError(
                        "整本翻译没有生成完整有效PDF成品"
                    )
                complete = fitz.open(str(outputs[0]))
                try:
                    if complete.page_count != 2:
                        raise RuntimeError(
                            "完整译本页数异常："
                            f"{complete.page_count}"
                        )
                    pdf_text = "\n".join(
                        page.get_text("text")
                        for page in complete
                    )
                finally:
                    complete.close()
                if (
                    "12" not in pdf_text
                    or len(_CJK_RE.findall(pdf_text)) < 4
                ):
                    raise RuntimeError(
                        "PDF成品未保留完整中文译文/关键数字"
                    )
                _ok(
                    "整本PDF翻译与成品导出",
                    f"outputs={len(outputs)} / "
                    f"warning_pages={result.warning_pages}",
                )
            except Exception as exc:
                failures.append(
                    f"整本PDF翻译与成品导出: {exc}"
                )
                _fail(
                    "整本PDF翻译与成品导出",
                    str(exc),
                )

            if not args.skip_organize:
                try:
                    title = (
                        "Phoenix正式上线多资料验收_"
                        + uuid.uuid4().hex[:8]
                    )
                    output, task_id = workbench.organize(
                        title,
                        "只根据全部已导入资料回答并整理："
                        f"{args.query}。要求保留来源编号、原始数字、"
                        "检查技术和鉴别点；不得补充资料外事实。",
                        candidate_limit=32,
                        batch_size=8,
                        progress=lambda done, total, msg: print(
                            f"ORGANIZE {done}/{total} {msg}",
                            flush=True,
                        ),
                    )
                    output = Path(output)
                    if (
                        not output.is_file()
                        or output.stat().st_size <= 0
                    ):
                        raise RuntimeError(
                            "联合整理正文没有生成"
                        )
                    text = output.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                    if (
                        "当前导入资料中未找到明确依据"
                        not in text
                        and not _CITATION_RE.search(text)
                    ):
                        raise RuntimeError(
                            "联合整理输出没有来源编号"
                        )
                    bundle = workbench.last_export_bundle
                    if bundle is None:
                        raise RuntimeError(
                            "联合整理正文完成，但PDF/DOCX/"
                            "Markdown/TXT输出包未生成："
                            + str(workbench.last_export_error)
                        )
                    missing_outputs = [
                        str(p)
                        for p in bundle.output_paths
                        if not Path(p).is_file()
                        or Path(p).stat().st_size <= 0
                    ]
                    if missing_outputs:
                        raise RuntimeError(
                            "多格式输出缺失："
                            + ", ".join(missing_outputs)
                        )
                    _ok(
                        "多资料联合整理与多格式输出",
                        f"task={task_id} / PDF+DOCX+MD+TXT",
                    )
                except Exception as exc:
                    failures.append(
                        "多资料联合整理与多格式输出: "
                        f"{exc}"
                    )
                    _fail(
                        "多资料联合整理与多格式输出",
                        str(exc),
                    )

            try:
                gui_detail = _gui_contract_smoke(
                    sandbox_paths
                )
                _ok("GUI主窗口与发布增强接线", gui_detail)
            except Exception as exc:
                failures.append(
                    f"GUI主窗口与发布增强接线: {exc}"
                )
                _fail(
                    "GUI主窗口与发布增强接线",
                    str(exc),
                )

        finally:
            workbench.close()

    print("========================================", flush=True)
    if failures:
        print(
            f"PHOENIX_RELEASE_ACCEPTANCE=FAIL ({len(failures)})",
            flush=True,
        )
        for item in failures:
            print("- " + item, flush=True)
        return 1

    print("PHOENIX_RELEASE_ACCEPTANCE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
