from .workbench import MedicalKnowledgeWorkbench
from .translation_stability_core import (
    capture_core as _capture_translation_core,
    install_final as _install_translation_stability_core,
)
from .workbench_stability_core import (
    capture_core as _capture_workbench_core,
    install_final as _install_workbench_stability_core,
)

# Capture the unwrapped public cores before historical compatibility installers
# can wrap them. Final installers below deliberately replace those wrapper
# chains with one deterministic release contract.
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

# Translation owns its lower-level PDF contract first. Workbench installs last
# so every user-facing public operation sees the final translation/runtime
# topology rather than a partially patched class.
_install_translation_stability_core()
_install_workbench_stability_core()

# Base local-first policy keeps translation functional when the external API is
# unavailable. The HY-MT cascade is installed last so the final production order
# becomes: lightweight local model -> HY-MT local refinement -> Smart2 API.
_install_hybrid_translation_policy()
_install_hymt_cascade_policy()

__all__ = ["MedicalKnowledgeWorkbench"]
