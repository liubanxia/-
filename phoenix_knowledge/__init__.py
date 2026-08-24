import os

from .workbench import MedicalKnowledgeWorkbench
from .translation_learning_pool import TranslationLearningPool, TranslationCorrectionSample
from .translation_learning_collector import TranslationLearningCollector, TranslationLearningRecord
from .translation_stability_core import (
    capture_core as _capture_translation_core,
    install_final as _install_translation_stability_core,
)
from .workbench_stability_core import (
    capture_core as _capture_workbench_core,
    install_final as _install_workbench_stability_core,
)

_capture_workbench_core()
_capture_translation_core()

from .translation_recovery import install as _install_translation_recovery
from .scholarly_pubmed import install as _install_scholarly_pubmed
from .scholarly_product_hardening import install as _install_scholarly_product_hardening
from .translation_semantics import install as _install_translation_semantics
from .translation_short_chinese import install as _install_translation_short_chinese
from .release_hardening import install as _install_release_hardening
from .release_memory_hardening import install as _install_release_memory_hardening
from .release_runtime_hardening import install as _install_release_runtime_hardening
from .release_portability import install as _install_release_portability
from .translation_storage_hardening import install as _install_translation_storage_hardening
from .translation_layout_compact import install as _install_translation_layout_compact
from .provider_hub import install as _install_provider_hub
from .provider_hub_compat import install as _install_provider_hub_compat
from .provider_hub_v2 import install as _install_provider_hub_v2
from .token_efficiency_hardening import install as _install_token_efficiency_hardening
from .hybrid_translation_policy import install as _install_hybrid_translation_policy
from .hymt_cascade_policy import install as _install_hymt_cascade_policy
from .translation_cascade_v2 import install as _install_translation_cascade_v2
from .translation_model3_inventory import install as _install_translation_model3_inventory
from .translation_review_integration import install as _install_translation_review_integration
from .translation_refusal_guard import install as _install_translation_refusal_guard
from .translation_local_first_release import install as _install_translation_local_first_release
from .translation_quality_first_release import install as _install_translation_quality_first_release
from .translation_dual_route_release import install as _install_translation_dual_route_release
from .medical_terminology_core import install as _install_medical_terminology_core
from .translation_model3_audit_acceleration import install as _install_translation_model3_audit_acceleration
from .translation_portable_model3_runtime import install as _install_translation_portable_model3_runtime
from .translation_portable_local_runtime import install as _install_translation_portable_local_runtime
from .translation_ssd_storage import install as _install_translation_ssd_storage
from .translation_survival_memory import install as _install_translation_survival_memory

_install_translation_recovery()
_install_scholarly_pubmed()
_install_scholarly_product_hardening()
_install_translation_semantics()
_install_translation_short_chinese()
_install_release_hardening()
_install_release_memory_hardening()
_install_release_runtime_hardening()
_install_release_portability()
_install_translation_storage_hardening()
_install_translation_layout_compact()
_install_provider_hub()
_install_provider_hub_compat()
_install_provider_hub_v2()
_install_token_efficiency_hardening()

_install_translation_stability_core()
_install_workbench_stability_core()
_install_translation_refusal_guard()
_install_translation_local_first_release()
_install_translation_quality_first_release()
_install_translation_dual_route_release()
_install_medical_terminology_core()
_install_translation_model3_audit_acceleration()
_install_translation_portable_model3_runtime()
_install_translation_portable_local_runtime()
_install_translation_ssd_storage()
_install_translation_survival_memory()

# Production translation is context/terminology driven. Model1 prepares the
# initial medical translation, failed/weak rows escalate to HY-MT model2 with
# English terminology support, local Qwen model3 performs a full source/context
# audit but normally emits only PASS or exact PATCH edits. If that compact audit
# cannot be applied safely, the proven full model3 refiner runs before API is
# allowed. All local stages are hardware-adaptive: compatible CUDA is preferred,
# while older/incompatible/failed CUDA automatically falls back to CPU without
# changing the model order, prompts or quality gates. No GPU model is required.
#
# Before API is spent, Phoenix now also has an offline survival layer: exact
# translation memory, conservative high-frequency medical sentence rules,
# guarded similar-memory drafts for model3, and an optional CPU-only ONNX
# emergency translator. Accepted results are written back to translation memory.
# If every model and API route is unavailable, the English source is preserved
# and queued for later translation instead of publishing refusal boilerplate.
# API corrections are stored as learning candidates; model weights are never
# changed online.
#
# The old exhaustive page-review stack remains opt-in for developer comparison
# only. It is intentionally outside the normal release path because whole-page
# model3 regeneration is too expensive for routine translation.
_experimental_cascade = os.environ.get(
    "PHOENIX_EXPERIMENTAL_TRANSLATION_CASCADE",
    "",
).strip().lower() in {"1", "true", "yes", "on"}
if _experimental_cascade:
    _install_hybrid_translation_policy()
    _install_hymt_cascade_policy()
    _install_translation_cascade_v2()
    _install_translation_model3_inventory()
    _install_translation_review_integration()

__all__ = [
    "MedicalKnowledgeWorkbench",
    "TranslationLearningPool",
    "TranslationCorrectionSample",
    "TranslationLearningCollector",
    "TranslationLearningRecord",
]
