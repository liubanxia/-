from __future__ import annotations

import os

import real_acceptance as _base


def _production_gui_contract_smoke(sandbox_paths) -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication, QTabWidget
    from phoenix_knowledge import MedicalKnowledgeWorkbench
    from phoenix_knowledge import gui as gui_module
    from phoenix_knowledge.gui_bootstrap import (
        GUI_CONTRACT_VERSION,
        GUI_INSTALL_ORDER,
        install_gui_stack,
    )

    applied = install_gui_stack(gui_module, strict=True)
    cls = gui_module.WorkbenchWindow
    if tuple(applied) != GUI_INSTALL_ORDER:
        raise RuntimeError("真实验收GUI安装顺序与正式软件不一致")
    if int(getattr(cls, "__phoenix_gui_contract__", 0) or 0) != GUI_CONTRACT_VERSION:
        raise RuntimeError("真实验收未加载正式GUI契约")
    if int(getattr(cls, "__phoenix_release_gui_hardening__", 0) or 0) < 2:
        raise RuntimeError("真实验收未加载最终长任务保护层")

    for method_name in (
        "add_pdfs",
        "add_documents",
        "ask_question",
        "build_embeddings",
        "start_organize",
        "resume_organize",
        "start_translation",
        "start_notes_organize",
    ):
        method = getattr(cls, method_name, None)
        if not callable(method) or not bool(
            getattr(method, "__phoenix_busy_guard__", False)
        ):
            raise RuntimeError(
                f"正式GUI任务入口未受保护：{method_name}"
            )

    original_factory = gui_module.MedicalKnowledgeWorkbench
    gui_module.MedicalKnowledgeWorkbench = (
        lambda: MedicalKnowledgeWorkbench(sandbox_paths)
    )
    app = QApplication.instance() or QApplication([])
    window = None
    try:
        window = cls()
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
            raise RuntimeError("产品页签缺失：" + ", ".join(missing))
        if not hasattr(window, "compute_status_label"):
            raise RuntimeError("发布版算力状态组件未接入")
        status_text = window._status_text()
        if not status_text.strip():
            raise RuntimeError("主窗口运行状态为空")
        return (
            " / ".join(names)
            + " | bootstrap="
            + " -> ".join(applied)
            + " | busy_guards=PASS"
        )
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


_base._gui_contract_smoke = _production_gui_contract_smoke


def main() -> int:
    return int(_base.main())


if __name__ == "__main__":
    raise SystemExit(main())
