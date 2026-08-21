from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Check:
    name: str
    state: str
    detail: str
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "PASS"


def _workbench_root() -> Path:
    return Path(__file__).resolve().parent


def _project_root() -> Path:
    return _workbench_root().parents[1]


def _expected_python() -> Path:
    return _project_root() / "02_开发环境" / "python.exe"


def _human_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024.0
    return f"{size:.1f}TB"


def _record(
    checks: list[Check],
    name: str,
    fn: Callable[[], str],
    *,
    fix: str = "",
) -> None:
    try:
        checks.append(Check(name, "PASS", str(fn() or "OK"), fix))
    except Exception as exc:
        checks.append(Check(name, "FAIL", f"{type(exc).__name__}: {exc}", fix))


def _python_check() -> str:
    expected = _expected_python()
    current = Path(sys.executable).resolve()
    if expected.is_file() and current != expected.resolve():
        raise RuntimeError(f"当前解释器={current}；项目解释器={expected.resolve()}")
    return f"{sys.version.split()[0]} / {current}"


def _dependency_check() -> str:
    required = {
        "PyMuPDF": "fitz",
        "PySide6": "PySide6",
        "numpy": "numpy",
        "Pillow": "PIL",
        "python-docx": "docx",
    }
    missing = []
    for package, module in required.items():
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(package)
    if missing:
        raise RuntimeError("缺少：" + ", ".join(missing))
    return "核心运行依赖齐全"


def _write_check() -> str:
    roots = [_workbench_root(), _project_root() / "08_运行时"]
    checked = []
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".phoenix_write_probe_{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            if probe.read_text(encoding="utf-8") != "ok":
                raise RuntimeError(f"写入后读回异常：{root}")
        finally:
            probe.unlink(missing_ok=True)
        checked.append(str(root))
    return "可写：" + " | ".join(checked)


def _disk_check() -> str:
    free = int(shutil.disk_usage(_project_root()).free)
    minimum = 8 * 1024**3
    if free < minimum:
        raise RuntimeError(
            f"SSD剩余 {_human_bytes(free)}，低于安全线 {_human_bytes(minimum)}"
        )
    return f"SSD剩余 {_human_bytes(free)}"


def _database_check() -> str:
    from phoenix_knowledge.config import get_paths
    database = Path(get_paths().database)
    if not database.exists():
        return f"数据库尚未创建：{database}"
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute("PRAGMA quick_check").fetchone()
    finally:
        connection.close()
    result = str(row[0] if row else "").strip().lower()
    if result != "ok":
        raise RuntimeError(f"SQLite quick_check={result or 'unknown'}")
    return f"SQLite=OK / {_human_bytes(database.stat().st_size)}"


def _pdf_engine_check() -> str:
    import fitz
    from phoenix_knowledge.translation_pdf import (
        LAYOUT_TRANSLATED_ONLY,
        TranslationPDFBuilder,
    )
    with tempfile.TemporaryDirectory(prefix="phoenix_onsite_pdf_") as td:
        root = Path(td)
        source = root / "source.pdf"
        pages = root / "pages"
        outputs = root / "outputs"
        pages.mkdir()
        doc = fitz.open()
        try:
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 80), "CT demonstrates a 12 mm lesion.", fontsize=12)
            doc.save(source)
        finally:
            doc.close()
        (pages / "000001.txt").write_text("CT显示12 mm病灶。", encoding="utf-8")
        complete, parts = TranslationPDFBuilder(source, pages, outputs).build(
            start_page=1,
            total_pages=1,
            layout=LAYOUT_TRANSLATED_ONLY,
            part_pages=0,
        )
        if parts:
            raise RuntimeError("默认总检不应生成分册")
        check = fitz.open(complete)
        try:
            if check.page_count != 1:
                raise RuntimeError(f"页数={check.page_count}")
            text = check[0].get_text("text")
            if "12" not in text:
                raise RuntimeError("文字层缺少关键数字12")
        finally:
            check.close()
    return "PDF构建/保存/reopen/文字层=OK"


def _bundle_contract_check() -> str:
    from phoenix_knowledge.output_contracts import (
        transactional_export_path,
        validate_export_bundle,
    )
    from phoenix_knowledge.rich_export import MultiFormatExporter
    with tempfile.TemporaryDirectory(prefix="phoenix_bundle_check_") as td:
        root = Path(td)
        source = root / "source.md"
        source.write_text("# Phoenix稳定性\n\nCT病灶12 mm。[S1]\n", encoding="utf-8")
        exporter = MultiFormatExporter(root / "outputs")
        first = transactional_export_path(exporter, source, title="稳定性验收")
        first_report = validate_export_bundle(first)
        second = transactional_export_path(exporter, source, title="稳定性验收")
        second_report = validate_export_bundle(second)
        if first_report["structure_sha256"] != second_report["structure_sha256"]:
            raise RuntimeError("同一输入重复导出的结构摘要不稳定")
    return "PDF+DOCX+Markdown+TXT事务发布/重复稳定性=OK"


def _sandbox_paths(root: Path):
    from phoenix_knowledge.config import WorkbenchPaths
    return WorkbenchPaths(
        project_root=root,
        source_root=root / "sources",
        runtime_root=root / "runtime",
        evidence_root=root / "evidence",
        model_root=root / "models",
        database=root / "runtime" / "knowledge.sqlite3",
        structure_root=root / "runtime" / "structure",
    ).ensure()


def _public_function_smoke_check() -> str:
    from phoenix_knowledge import MedicalKnowledgeWorkbench
    from phoenix_knowledge.output_contracts import validate_export_bundle
    from phoenix_knowledge.workbench_stability_core import architecture_status
    with tempfile.TemporaryDirectory(prefix="phoenix_public_smoke_") as td:
        root = Path(td)
        source = root / "smoke.txt"
        source.write_text(
            "肺结节CT征象为边缘毛刺，病灶直径12 mm。"
            "这是Phoenix离线稳定性验收资料。",
            encoding="utf-8",
        )
        wb = MedicalKnowledgeWorkbench(_sandbox_paths(root))
        try:
            imported = wb.ingest(source, _defer_embeddings=True)
            if not Path(imported.copied_to_library).is_file():
                raise RuntimeError("导入没有库内副本")
            answer = wb.ask("肺结节12 mm", use_embeddings=False, deep=False)
            if not str(answer.text).strip() or not answer.evidence:
                raise RuntimeError("关键词问答没有真实证据输出")
            output, _task_id = wb.organize(
                "肺结节稳定性",
                "整理肺结节CT征象和病灶大小",
                candidate_limit=12,
                batch_size=4,
            )
            if not Path(output).is_file():
                raise RuntimeError("联合整理正文未生成")
            if wb.last_export_bundle is None:
                raise RuntimeError("联合整理返回成功但多格式成品不存在")
            validate_export_bundle(wb.last_export_bundle)
            note = wb.organize_txt("CT提示右肺结节12 mm。", title="上线总检笔记")
            if not note.output_path.is_file():
                raise RuntimeError("笔记整理没有成品")
            architecture = architecture_status(wb)
            if not architecture["ready"]:
                raise RuntimeError(
                    "功能运行后架构被改写：" + ", ".join(architecture["broken"])
                )
        finally:
            wb.close()
    return "导入+问答+联合整理+PDF/DOCX/MD/TXT+笔记=OK"


def _workbench_status_check() -> tuple[str, dict]:
    from phoenix_knowledge import MedicalKnowledgeWorkbench
    wb = MedicalKnowledgeWorkbench()
    try:
        status = wb.status()
        capabilities = {
            "documents": int(status.get("documents", 0) or 0),
            "chunks": int(status.get("chunks", 0) or 0),
            "semantic_ready": bool(status.get("semantic_ready", False)),
            "semantic_label": str(status.get("semantic_label", "")),
            "translation_backends": list(status.get("translation_backends") or []),
            "document_paths_unresolved": int(
                status.get("document_paths_unresolved", 0) or 0
            ),
            "smart1": bool(wb.llm.available("fast")),
            "smart2": bool(wb.llm.available("deep")),
            "architecture_ready": bool(status.get("architecture_ready", False)),
            "architecture_broken": list(status.get("architecture_broken") or []),
            "architecture_fingerprint": str(
                status.get("architecture_fingerprint", "")
            ),
            "workbench_contract": int(status.get("workbench_contract", 0) or 0),
            "output_contract": int(status.get("output_contract", 0) or 0),
        }
        if capabilities["document_paths_unresolved"]:
            raise RuntimeError(
                f"{capabilities['document_paths_unresolved']}份资料路径无法定位"
            )
        if not capabilities["architecture_ready"]:
            raise RuntimeError(
                "架构契约损坏：" + ", ".join(capabilities["architecture_broken"])
            )
        return (
            f"资料={capabilities['documents']} / 知识块={capabilities['chunks']} / "
            f"架构={capabilities['workbench_contract']} / "
            f"输出={capabilities['output_contract']}",
            capabilities,
        )
    finally:
        wb.close()


def _gui_check() -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget
    from phoenix_knowledge import gui as gui_module
    from phoenix_knowledge.compute_gui import install as install_compute_gui
    from phoenix_knowledge.document_gui import install as install_document_gui
    from phoenix_knowledge.gui_enhancements import install as install_gui_enhancements
    from phoenix_knowledge.translation_layout_compact import LAYOUT_SOURCE_TRANSLATED
    from phoenix_knowledge.translation_storage_gui import (
        install as install_translation_storage_gui,
    )
    from phoenix_knowledge.workbench_stability_core import architecture_status
    install_gui_enhancements(gui_module)
    install_compute_gui(gui_module)
    install_document_gui(gui_module)
    install_translation_storage_gui(gui_module)
    app = QApplication.instance() or QApplication([])
    window = gui_module.WorkbenchWindow()
    try:
        tabs = window.centralWidget()
        if not isinstance(tabs, QTabWidget):
            raise RuntimeError("GUI主容器异常")
        names = [tabs.tabText(i) for i in range(tabs.count())]
        required = (
            "医学资料库", "资料问答", "多资料/论文联合整理", "整本书翻译", "笔记整理",
        )
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError("缺少页签：" + ", ".join(missing))
        for method in ("start_organize", "resume_organize", "start_translation", "ask_question"):
            if not callable(getattr(window, method, None)):
                raise RuntimeError(f"GUI入口丢失：{method}")
        if int(window.translation_part_pages.value()) != 0:
            raise RuntimeError("翻译GUI默认又开始生成重复分册")
        if str(window.translation_layout_combo.currentData()) != LAYOUT_SOURCE_TRANSLATED:
            raise RuntimeError("翻译GUI默认版式被其他补丁覆盖")
        architecture = architecture_status(window.workbench)
        if not architecture["ready"]:
            raise RuntimeError(
                "GUI加载后核心架构被覆盖：" + ", ".join(architecture["broken"])
            )
        return "GUI=" + " / ".join(names)
    finally:
        try:
            window.close()
        finally:
            try:
                window.workbench.close()
            except Exception:
                pass
        app.processEvents()


def _classify(checks: list[Check], capabilities: dict) -> tuple[str, list[str]]:
    failed = {item.name: item for item in checks if not item.ok}
    causes: list[str] = []
    hard = {
        "Python解释器", "核心依赖", "项目写入", "PDF输出引擎",
        "多格式输出契约", "公共功能实跑", "工作台能力", "GUI框架",
    }
    if hard.intersection(failed):
        causes.append(
            "核心运行/输出/架构契约未通过；不要逐个按钮试，先修共同根因。"
        )
    if "SSD空间" in failed:
        causes.append("SSD空间不足可能造成大任务最终阶段不输出。")
    if "数据库" in failed:
        causes.append("数据库异常会同时影响导入、检索、问答和整理。")
    if capabilities:
        if not capabilities.get("translation_backends"):
            causes.append("没有可用翻译后端，整本翻译无法真正执行。")
        if not capabilities.get("smart1") and not capabilities.get("smart2"):
            causes.append("智能1/智能2均未就绪；智能问答/整理会使用证据降级路径。")
        if not capabilities.get("semantic_ready"):
            causes.append("语义索引未就绪；关键词证据仍可用，语义召回会降级。")
        if capabilities.get("architecture_ready") is False:
            causes.append("Workbench公共入口架构契约已被破坏。")
    if hard.intersection(failed) or "SSD空间" in failed or "数据库" in failed:
        return "BLOCKED", causes
    full_ai = bool(
        capabilities
        and capabilities.get("translation_backends")
        and (capabilities.get("smart1") or capabilities.get("smart2"))
    )
    if failed or not full_ai or (
        capabilities and not capabilities.get("semantic_ready")
    ):
        return "DEGRADED", causes
    return "READY", causes


def _report_path() -> Path:
    root = _project_root() / "08_运行时" / "上线总检"
    root.mkdir(parents=True, exist_ok=True)
    return root / "latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phoenix 上线前一键总检（实跑输出+架构契约）"
    )
    parser.add_argument("--no-gui", action="store_true", help="不实例化GUI")
    args = parser.parse_args()
    os.environ.setdefault("PHOENIX_KNOWLEDGE_ALLOW_REMOTE", "0")
    checks: list[Check] = []
    capabilities: dict = {}
    _record(checks, "Python解释器", _python_check, fix="必须使用SSD自带 02_开发环境/python.exe。")
    _record(checks, "核心依赖", _dependency_check, fix="运行 runtime_preflight.py --repair。")
    _record(checks, "项目写入", _write_check, fix="检查SSD只读、权限或文件系统。")
    _record(checks, "SSD空间", _disk_check, fix="至少保留8GB；大教材建议更多。")
    _record(checks, "数据库", _database_check, fix="异常时先恢复数据库备份，不要继续写入。")
    _record(checks, "PDF输出引擎", _pdf_engine_check, fix="失败时不要开始真实整本翻译。")
    _record(
        checks, "多格式输出契约", _bundle_contract_check,
        fix="失败表示PDF/DOCX/MD/TXT事务发布链不可靠。",
    )
    _record(
        checks, "公共功能实跑", _public_function_smoke_check,
        fix="失败时按此项错误修共同入口，不要逐个按钮试。",
    )
    try:
        detail, capabilities = _workbench_status_check()
        checks.append(Check("工作台能力", "PASS", detail))
    except Exception as exc:
        checks.append(Check(
            "工作台能力", "FAIL", f"{type(exc).__name__}: {exc}",
            "检查架构契约、模型路径和资料路径。",
        ))
    if not args.no_gui:
        _record(
            checks, "GUI框架", _gui_check,
            fix="GUI失败时先修导入/补丁覆盖，不做正式任务。",
        )
    state, causes = _classify(checks, capabilities)
    payload = {
        "state": state,
        "python": sys.executable,
        "project_root": str(_project_root()),
        "checks": [asdict(item) for item in checks],
        "capabilities": capabilities,
        "root_causes": causes,
    }
    report = _report_path()
    temp = report.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, report)
    print("=" * 72, flush=True)
    print(f"PHOENIX_ONSITE_PREFLIGHT={state}", flush=True)
    print(f"PROJECT={_project_root()}", flush=True)
    print(f"PYTHON={sys.executable}", flush=True)
    print("-" * 72, flush=True)
    for item in checks:
        print(f"{'PASS' if item.ok else 'FAIL':4}  {item.name} | {item.detail}", flush=True)
        if not item.ok and item.fix:
            print(f"      处理：{item.fix}", flush=True)
    if capabilities:
        print("-" * 72, flush=True)
        print(
            "CAPABILITIES "
            f"smart1={capabilities.get('smart1')} "
            f"smart2={capabilities.get('smart2')} "
            f"semantic={capabilities.get('semantic_ready')} "
            f"translation={bool(capabilities.get('translation_backends'))} "
            f"architecture={capabilities.get('architecture_ready')}",
            flush=True,
        )
        print(
            "ARCHITECTURE_FINGERPRINT="
            f"{capabilities.get('architecture_fingerprint', '')}",
            flush=True,
        )
    if causes:
        print("-" * 72, flush=True)
        print("共同根因/降级原因：", flush=True)
        for item in causes:
            print(f"- {item}", flush=True)
    print(f"REPORT={report}", flush=True)
    print("=" * 72, flush=True)
    return 0 if state == "READY" else 2 if state == "DEGRADED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
