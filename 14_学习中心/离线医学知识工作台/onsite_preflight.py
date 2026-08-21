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


def _record(checks: list[Check], name: str, fn: Callable[[], str], *, fix: str = "") -> None:
    try:
        detail = str(fn() or "OK")
        checks.append(Check(name, "PASS", detail, fix))
    except Exception as exc:
        checks.append(Check(name, "FAIL", f"{type(exc).__name__}: {exc}", fix))


def _python_check() -> str:
    expected = _expected_python()
    current = Path(sys.executable).resolve()
    if expected.is_file():
        try:
            expected_resolved = expected.resolve()
        except Exception:
            expected_resolved = expected
        if current != expected_resolved:
            raise RuntimeError(f"当前解释器={current}；项目解释器={expected_resolved}")
    return f"{sys.version.split()[0]} / {current}"


def _dependency_check() -> str:
    required = {
        "PyMuPDF": "fitz",
        "PySide6": "PySide6",
        "numpy": "numpy",
        "Pillow": "PIL",
        "python-docx": "docx",
    }
    missing: list[str] = []
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
    tested: list[str] = []
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".phoenix_write_probe_{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            if probe.read_text(encoding="utf-8") != "ok":
                raise RuntimeError("读回失败")
        finally:
            probe.unlink(missing_ok=True)
        tested.append(str(root))
    return "可写：" + " | ".join(tested)


def _disk_check() -> str:
    usage = shutil.disk_usage(_project_root())
    free = int(usage.free)
    minimum = 8 * 1024**3
    if free < minimum:
        raise RuntimeError(f"SSD剩余 {_human_bytes(free)}，低于上线安全线 {_human_bytes(minimum)}")
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
    result = str(row[0] if row else "").strip()
    if result.lower() != "ok":
        raise RuntimeError(f"SQLite quick_check={result or 'unknown'}")
    return f"SQLite=OK / {_human_bytes(database.stat().st_size)}"


def _pdf_engine_check() -> str:
    import fitz
    from phoenix_knowledge.translation_pdf import LAYOUT_TRANSLATED_ONLY, TranslationPDFBuilder

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
            raise RuntimeError("默认自检不应生成分册")
        if not complete.is_file() or complete.stat().st_size <= 0:
            raise RuntimeError("PDF没有输出")
        check = fitz.open(complete)
        try:
            if check.page_count != 1:
                raise RuntimeError(f"页数={check.page_count}")
            text = check[0].get_text("text")
            if "12" not in text:
                raise RuntimeError("输出文字层缺少关键数字12")
        finally:
            check.close()
    return "PDF构建/保存/reopen/文字层=OK"


def _workbench_status_check() -> tuple[str, dict]:
    from phoenix_knowledge import MedicalKnowledgeWorkbench

    workbench = MedicalKnowledgeWorkbench()
    try:
        status = workbench.status()
        payload = {
            "documents": int(status.get("documents", 0) or 0),
            "chunks": int(status.get("chunks", 0) or 0),
            "semantic_ready": bool(status.get("semantic_ready", False)),
            "semantic_label": str(status.get("semantic_label", "")),
            "translation_backends": list(status.get("translation_backends") or []),
            "document_paths_unresolved": int(status.get("document_paths_unresolved", 0) or 0),
            "smart1": bool(workbench.llm.available("fast")),
            "smart2": bool(workbench.llm.available("deep")),
        }
        if payload["document_paths_unresolved"]:
            raise RuntimeError(f"{payload['document_paths_unresolved']}份资料路径无法定位")
        return (
            f"资料={payload['documents']} / 知识块={payload['chunks']} / 翻译后端={len(payload['translation_backends'])}",
            payload,
        )
    finally:
        workbench.close()


def _gui_check() -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget
    from phoenix_knowledge import gui as gui_module
    from phoenix_knowledge.compute_gui import install as install_compute_gui
    from phoenix_knowledge.document_gui import install as install_document_gui
    from phoenix_knowledge.gui_enhancements import install as install_gui_enhancements
    from phoenix_knowledge.translation_storage_gui import install as install_translation_storage_gui

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
        required = ("医学资料库", "资料问答", "多资料/论文联合整理", "整本书翻译", "笔记整理")
        missing = [name for name in required if name not in names]
        if missing:
            raise RuntimeError("缺少页签：" + ", ".join(missing))
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
    root_causes: list[str] = []
    hard = {"Python解释器", "核心依赖", "项目写入", "PDF输出引擎", "GUI框架"}

    if hard.intersection(failed):
        root_causes.append("基础运行环境未通过，先修基础环境，不要逐个功能试。")
    if "SSD空间" in failed:
        root_causes.append("SSD空间不足会造成翻译/导出最后阶段不输出。")
    if "数据库" in failed:
        root_causes.append("知识库数据库异常，问答/检索/整理会一起受影响。")

    if capabilities:
        if not capabilities.get("translation_backends"):
            root_causes.append("没有可用翻译后端，整本翻译无法真正执行。")
        if not capabilities.get("smart1") and not capabilities.get("smart2"):
            root_causes.append("智能1/智能2均未就绪，智能问答与高质量整理会降级。")
        if not capabilities.get("semantic_ready"):
            root_causes.append("语义索引未就绪；快速证据仍可能可用，但跨语言/语义召回会降级。")

    if hard.intersection(failed) or "SSD空间" in failed or "数据库" in failed:
        return "BLOCKED", root_causes

    full_ai = bool(capabilities and capabilities.get("translation_backends") and (capabilities.get("smart1") or capabilities.get("smart2")))
    if failed or not full_ai or (capabilities and not capabilities.get("semantic_ready")):
        return "DEGRADED", root_causes
    return "READY", root_causes


def _report_path() -> Path:
    target = _project_root() / "08_运行时" / "上线总检"
    target.mkdir(parents=True, exist_ok=True)
    return target / "latest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phoenix 上线前一键总检")
    parser.add_argument("--no-gui", action="store_true", help="不实例化GUI，仅做运行与输出链检查")
    args = parser.parse_args()

    os.environ.setdefault("PHOENIX_KNOWLEDGE_ALLOW_REMOTE", "0")
    checks: list[Check] = []
    capabilities: dict = {}

    _record(checks, "Python解释器", _python_check, fix="必须使用SSD项目自带 02_开发环境/python.exe 启动。")
    _record(checks, "核心依赖", _dependency_check, fix="运行 runtime_preflight.py --repair 修复缺失依赖。")
    _record(checks, "项目写入", _write_check, fix="检查SSD只读、权限或文件系统错误。")
    _record(checks, "SSD空间", _disk_check, fix="至少保留8GB临时安全空间；大教材建议保留原PDF大小2倍以上。")
    _record(checks, "数据库", _database_check, fix="数据库损坏时先恢复 runtime/db_backups，不要继续写入。")
    _record(checks, "PDF输出引擎", _pdf_engine_check, fix="此项失败时不要开始真实整本翻译。")

    try:
        detail, capabilities = _workbench_status_check()
        checks.append(Check("工作台能力", "PASS", detail))
    except Exception as exc:
        checks.append(Check("工作台能力", "FAIL", f"{type(exc).__name__}: {exc}", "检查模型路径、资料路径和工作台初始化日志。"))

    if not args.no_gui:
        _record(checks, "GUI框架", _gui_check, fix="GUI失败时使用命令行总检结果定位依赖/导入错误。")

    state, root_causes = _classify(checks, capabilities)
    payload = {
        "state": state,
        "python": sys.executable,
        "project_root": str(_project_root()),
        "checks": [asdict(item) for item in checks],
        "capabilities": capabilities,
        "root_causes": root_causes,
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
        mark = "PASS" if item.ok else "FAIL"
        print(f"{mark:4}  {item.name} | {item.detail}", flush=True)
        if not item.ok and item.fix:
            print(f"      处理：{item.fix}", flush=True)
    if capabilities:
        print("-" * 72, flush=True)
        print(
            "CAPABILITIES "
            f"smart1={capabilities.get('smart1')} "
            f"smart2={capabilities.get('smart2')} "
            f"semantic={capabilities.get('semantic_ready')} "
            f"translation={bool(capabilities.get('translation_backends'))}",
            flush=True,
        )
    if root_causes:
        print("-" * 72, flush=True)
        print("共同根因/降级原因：", flush=True)
        for item in root_causes:
            print(f"- {item}", flush=True)
    print(f"REPORT={report}", flush=True)
    print("=" * 72, flush=True)

    return 0 if state == "READY" else 2 if state == "DEGRADED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
