from __future__ import annotations

import importlib
from collections.abc import Callable


GUI_CONTRACT_VERSION = 5
GUI_INSTALL_ORDER = (
    "gui_enhancements",
    "compute_gui",
    "translation_api_consent_gui",
    "document_gui",
    "translation_storage_gui",
    "translation_drag_drop_gui",
    "release_gui_hardening",
    "release_gui_truth",
)


class GUIBootstrapError(RuntimeError):
    pass


def install_gui_stack(
    gui_module,
    *,
    strict: bool = True,
    reporter: Callable[[str], None] | None = None,
    module_loader=None,
) -> tuple[str, ...]:
    """Install the complete GUI stack once, in the only supported order.

    Compatibility/product extensions run first. Release hardening and visible
    truth run last so no later installer can silently replace their guarded
    entry points. Both the real app and onsite preflight use this function;
    tests must never maintain a second handwritten installer sequence.
    """

    loader = module_loader or importlib.import_module
    applied: list[str] = []
    failures: list[str] = []

    for module_name in GUI_INSTALL_ORDER:
        try:
            module = loader(f"{__package__}.{module_name}")
            installer = getattr(module, "install")
            installer(gui_module)
            applied.append(module_name)
        except Exception as exc:
            detail = f"{module_name}: {type(exc).__name__}: {exc}"
            failures.append(detail)
            if reporter is not None:
                reporter("GUI_BOOTSTRAP_WARNING=" + detail)

    cls = gui_module.WorkbenchWindow
    cls.__phoenix_gui_contract__ = GUI_CONTRACT_VERSION
    cls.__phoenix_gui_install_order__ = tuple(applied)
    cls.__phoenix_gui_bootstrap_failures__ = tuple(failures)

    if failures and strict:
        raise GUIBootstrapError(
            "GUI安装链不完整，拒绝启动半成品界面: " + " | ".join(failures)
        )
    if tuple(applied) != GUI_INSTALL_ORDER and strict:
        raise GUIBootstrapError(
            "GUI安装顺序异常: " + " -> ".join(applied)
        )
    return tuple(applied)
