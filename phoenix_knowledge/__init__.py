from __future__ import annotations

"""Phoenix medical knowledge platform package.

Importing a low-level module must not silently install the production translation
wrapper stack. Only connection-lifecycle safety is enabled at package import.
The full runtime is bootstrapped lazily when the public Workbench or GUI is used.
"""

import importlib

from .sqlite_runtime import install_translation_sqlite_safety

# This is deliberately semantic-neutral. It only guarantees that short-lived
# translation SQLite connections commit/rollback and then close on every path.
install_translation_sqlite_safety()


def bootstrap_runtime() -> tuple[str, ...]:
    # Configure stdout/stderr before any installer emits Chinese diagnostics.
    # This prevents locale-specific Windows encoders (for example cp1252) from
    # aborting application startup with UnicodeEncodeError.
    from .console_runtime import configure_console_text
    from .runtime_bootstrap import bootstrap_runtime as _bootstrap

    configure_console_text()
    return _bootstrap()


def runtime_bootstrapped() -> bool:
    from .runtime_bootstrap import runtime_bootstrapped as _state

    return bool(_state())


def runtime_install_order() -> tuple[str, ...]:
    from .runtime_bootstrap import runtime_install_order as _order

    return tuple(_order())


def __getattr__(name: str):
    if name == "MedicalKnowledgeWorkbench":
        bootstrap_runtime()
        from .workbench import MedicalKnowledgeWorkbench

        return MedicalKnowledgeWorkbench

    if name == "TranslationLearningPool":
        from .translation_learning_pool import TranslationLearningPool

        return TranslationLearningPool

    if name == "TranslationCorrectionSample":
        from .translation_learning_pool import TranslationCorrectionSample

        return TranslationCorrectionSample

    if name == "TranslationLearningCollector":
        from .translation_learning_collector import TranslationLearningCollector

        return TranslationLearningCollector

    if name == "TranslationLearningRecord":
        from .translation_learning_collector import TranslationLearningRecord

        return TranslationLearningRecord

    # `from phoenix_knowledge import gui` is a public application entry path.
    # Ensure translation/workbench production contracts are installed before the
    # GUI module captures or constructs Workbench objects.
    if name == "gui":
        bootstrap_runtime()
        module = importlib.import_module(f"{__name__}.gui")
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MedicalKnowledgeWorkbench",
    "TranslationLearningPool",
    "TranslationCorrectionSample",
    "TranslationLearningCollector",
    "TranslationLearningRecord",
    "bootstrap_runtime",
    "runtime_bootstrapped",
    "runtime_install_order",
]
