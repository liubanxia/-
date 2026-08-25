from __future__ import annotations

"""Explicit Phoenix production-runtime bootstrap.

Low-level module imports stay side-effect-light so tests, migrations and
maintenance tools can exercise their real semantics. Production monkey-patches
are installed only when the application/workbench/GUI requests the full runtime.

A failed bootstrap is process-fatal by design. Installers mutate class/function
objects, so retrying after a partial failure could stack wrappers over a mixed
runtime. The caller must restart the process after correcting the cause.
"""

import os
import threading

_LOCK = threading.RLock()
_BOOTSTRAPPED = False
_BOOTSTRAPPING = False
_BOOTSTRAP_FAILURE: str | None = None
_INSTALL_ORDER: tuple[str, ...] = ()


class RuntimeBootstrapError(RuntimeError):
    pass


def runtime_bootstrapped() -> bool:
    return bool(_BOOTSTRAPPED)


def runtime_install_order() -> tuple[str, ...]:
    return tuple(_INSTALL_ORDER)


def runtime_bootstrap_failure() -> str | None:
    return _BOOTSTRAP_FAILURE


def _refuse_after_previous_failure() -> None:
    if _BOOTSTRAP_FAILURE is None:
        return
    raise RuntimeBootstrapError(
        "Phoenix生产运行时此前已部分安装失败；为避免重复monkey-patch，"
        "当前进程禁止重试。请修复原因后重新启动程序。"
        f" 首次错误={_BOOTSTRAP_FAILURE}"
    )


def bootstrap_runtime() -> tuple[str, ...]:
    global _BOOTSTRAPPED, _BOOTSTRAPPING, _BOOTSTRAP_FAILURE, _INSTALL_ORDER

    if _BOOTSTRAPPED:
        return tuple(_INSTALL_ORDER)
    _refuse_after_previous_failure()

    with _LOCK:
        if _BOOTSTRAPPED:
            return tuple(_INSTALL_ORDER)
        _refuse_after_previous_failure()
        if _BOOTSTRAPPING:
            raise RuntimeBootstrapError("Phoenix生产运行时发生递归启动。")
        _BOOTSTRAPPING = True

        applied: list[str] = []
        try:
            # Semantic-neutral SQLite lifecycle hardening is always safe and must
            # precede any learning/memory runtime installation.
            from .sqlite_runtime import install_translation_sqlite_safety

            install_translation_sqlite_safety()
            applied.append("sqlite_runtime")

            # Capture pristine public entry points before compatibility/release
            # installers start replacing them.
            from .translation_stability_core import (
                capture_core as capture_translation_core,
                install_final as install_translation_stability_core,
            )
            from .workbench_stability_core import (
                capture_core as capture_workbench_core,
                install_final as install_workbench_stability_core,
            )

            capture_workbench_core()
            capture_translation_core()
            applied.extend(("workbench_core_capture", "translation_core_capture"))

            from .translation_recovery import install as install_translation_recovery
            from .scholarly_pubmed import install as install_scholarly_pubmed
            from .scholarly_product_hardening import install as install_scholarly_product_hardening
            from .translation_semantics import install as install_translation_semantics
            from .translation_short_chinese import install as install_translation_short_chinese
            from .release_hardening import install as install_release_hardening
            from .release_memory_hardening import install as install_release_memory_hardening
            from .release_runtime_hardening import install as install_release_runtime_hardening
            from .release_portability import install as install_release_portability
            from .translation_storage_hardening import install as install_translation_storage_hardening
            from .translation_layout_compact import install as install_translation_layout_compact
            from .provider_hub import install as install_provider_hub
            from .provider_hub_compat import install as install_provider_hub_compat
            from .provider_hub_v2 import install as install_provider_hub_v2
            from .token_efficiency_hardening import install as install_token_efficiency_hardening

            early = (
                ("translation_recovery", install_translation_recovery),
                ("scholarly_pubmed", install_scholarly_pubmed),
                ("scholarly_product_hardening", install_scholarly_product_hardening),
                ("translation_semantics", install_translation_semantics),
                ("translation_short_chinese", install_translation_short_chinese),
                ("release_hardening", install_release_hardening),
                ("release_memory_hardening", install_release_memory_hardening),
                ("release_runtime_hardening", install_release_runtime_hardening),
                ("release_portability", install_release_portability),
                ("translation_storage_hardening", install_translation_storage_hardening),
                ("translation_layout_compact", install_translation_layout_compact),
                ("provider_hub", install_provider_hub),
                ("provider_hub_compat", install_provider_hub_compat),
                ("provider_hub_v2", install_provider_hub_v2),
                ("token_efficiency_hardening", install_token_efficiency_hardening),
            )
            for name, installer in early:
                installer()
                applied.append(name)

            # Collapse historical public wrapper stacks before the current
            # translation runtime is layered on top.
            install_translation_stability_core()
            applied.append("translation_stability_core")
            install_workbench_stability_core()
            applied.append("workbench_stability_core")

            from .translation_refusal_guard import install as install_translation_refusal_guard
            from .translation_local_first_release import install as install_translation_local_first_release
            from .translation_quality_first_release import install as install_translation_quality_first_release
            from .translation_dual_route_release import install as install_translation_dual_route_release
            from .medical_terminology_core import install as install_medical_terminology_core
            from .translation_model3_audit_acceleration import install as install_translation_model3_audit_acceleration
            from .translation_portable_model3_runtime import install as install_translation_portable_model3_runtime
            from .translation_portable_local_runtime import install as install_translation_portable_local_runtime
            from .translation_ssd_storage_runtime_v2 import install as install_translation_ssd_storage_runtime_v2
            from .translation_survival_memory import install as install_translation_survival_memory
            from .translation_maturity_runtime_v2 import install as install_translation_maturity_runtime_v2
            from .translation_api_value_runtime_v2 import install as install_translation_api_value_runtime_v2
            from .translation_blank_student_runtime_v2 import install as install_translation_blank_student_runtime_v2
            from .translation_document_postprocess_v2 import install as install_translation_document_postprocess_v2

            current = (
                ("translation_refusal_guard", install_translation_refusal_guard),
                ("translation_local_first_release", install_translation_local_first_release),
                ("translation_quality_first_release", install_translation_quality_first_release),
                ("translation_dual_route_release", install_translation_dual_route_release),
                ("medical_terminology_core", install_medical_terminology_core),
                ("translation_model3_audit_acceleration", install_translation_model3_audit_acceleration),
                ("translation_portable_model3_runtime", install_translation_portable_model3_runtime),
                ("translation_portable_local_runtime", install_translation_portable_local_runtime),
                ("translation_ssd_storage_runtime_v2", install_translation_ssd_storage_runtime_v2),
                ("translation_survival_memory", install_translation_survival_memory),
                ("translation_maturity_runtime_v2", install_translation_maturity_runtime_v2),
                ("translation_api_value_runtime_v2", install_translation_api_value_runtime_v2),
                ("translation_blank_student_runtime_v2", install_translation_blank_student_runtime_v2),
                ("translation_document_postprocess_v2", install_translation_document_postprocess_v2),
            )
            for name, installer in current:
                installer()
                applied.append(name)

            # Historical exhaustive cascade remains an explicit developer-only
            # comparison path and is never silently enabled in production.
            experimental = os.environ.get(
                "PHOENIX_EXPERIMENTAL_TRANSLATION_CASCADE",
                "",
            ).strip().lower() in {"1", "true", "yes", "on"}
            if experimental:
                from .hybrid_translation_policy import install as install_hybrid_translation_policy
                from .hymt_cascade_policy import install as install_hymt_cascade_policy
                from .translation_cascade_v2 import install as install_translation_cascade_v2
                from .translation_model3_inventory import install as install_translation_model3_inventory
                from .translation_review_integration import install as install_translation_review_integration

                optional = (
                    ("hybrid_translation_policy", install_hybrid_translation_policy),
                    ("hymt_cascade_policy", install_hymt_cascade_policy),
                    ("translation_cascade_v2", install_translation_cascade_v2),
                    ("translation_model3_inventory", install_translation_model3_inventory),
                    ("translation_review_integration", install_translation_review_integration),
                )
                for name, installer in optional:
                    installer()
                    applied.append(name)

            _INSTALL_ORDER = tuple(applied)
            _BOOTSTRAPPED = True
            return tuple(_INSTALL_ORDER)
        except BaseException as exc:
            # Preserve what already ran for diagnosis, but never retry this
            # partially-mutated process. A clean process restart is the only safe
            # recovery from a failed global patch installation.
            _INSTALL_ORDER = tuple(applied)
            _BOOTSTRAP_FAILURE = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeBootstrapError(
                "Phoenix生产运行时安装失败，当前进程已锁定为不可重试；"
                "请修复后重新启动。"
                f" 已完成={','.join(applied) or 'none'}；"
                f" 错误={_BOOTSTRAP_FAILURE}"
            ) from exc
        finally:
            _BOOTSTRAPPING = False
