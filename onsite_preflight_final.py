from __future__ import annotations

import os

import onsite_preflight as _base


def _production_gui_check() -> str:
    """Exercise the exact GUI composition used by app.py.

    This check exists because a hand-written approximation of the installer
    stack can pass while the real app ends up with different wrapped methods.
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication, QTabWidget
    from phoenix_knowledge import gui as gui_module
    from phoenix_knowledge.gui_bootstrap import (
        GUI_CONTRACT_VERSION,
        GUI_INSTALL_ORDER,
        install_gui_stack,
    )
    from phoenix_knowledge.translation_layout_compact import LAYOUT_SOURCE_TRANSLATED
    from phoenix_knowledge.workbench_stability_core import architecture_status

    applied = install_gui_stack(gui_module, strict=True)
    cls = gui_module.WorkbenchWindow

    if tuple(applied) != GUI_INSTALL_ORDER:
        raise RuntimeError("正式GUI安装顺序异常")
    if int(getattr(cls, "__phoenix_gui_contract__", 0) or 0) != GUI_CONTRACT_VERSION:
        raise RuntimeError("正式GUI契约版本未生效")
    if int(getattr(cls, "__phoenix_release_gui_hardening__", 0) or 0) < 2:
        raise RuntimeError("最终GUI长任务保护层未生效")

    guarded = (
        "add_pdfs",
        "add_documents",
        "ask_question",
        "build_embeddings",
        "start_organize",
        "resume_organize",
        "start_translation",
        "start_notes_organize",
    )
    for method_name in guarded:
        method = getattr(cls, method_name, None)
        if not callable(method):
            raise RuntimeError(f"GUI入口丢失：{method_name}")
        if not bool(getattr(method, "__phoenix_busy_guard__", False)):
            raise RuntimeError(
                f"GUI入口缺少最终任务保护：{method_name}；"
                "这会导致长任务期间按钮静默无响应"
            )

    app = QApplication.instance() or QApplication([])
    window = cls()
    try:
        tabs = window.centralWidget()
        if not isinstance(tabs, QTabWidget):
            raise RuntimeError("GUI主容器异常")
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
            raise RuntimeError("缺少页签：" + ", ".join(missing))
        if int(window.translation_part_pages.value()) != 0:
            raise RuntimeError("翻译GUI默认又开始生成重复分册")
        if str(window.translation_layout_combo.currentData()) != LAYOUT_SOURCE_TRANSLATED:
            raise RuntimeError("翻译GUI默认版式被其他补丁覆盖")
        architecture = architecture_status(window.workbench)
        if not architecture["ready"]:
            raise RuntimeError(
                "GUI加载后核心架构被覆盖：" + ", ".join(architecture["broken"])
            )
        return (
            "GUI_PRODUCTION_BOOTSTRAP="
            + " -> ".join(applied)
            + " | TABS="
            + " / ".join(names)
            + " | BUSY_GUARDS=PASS"
        )
    finally:
        try:
            window.close()
        finally:
            try:
                window.workbench.close()
            except Exception:
                pass
        app.processEvents()


_base._gui_check = _production_gui_check


def main() -> int:
    return int(_base.main())


if __name__ == "__main__":
    raise SystemExit(main())
