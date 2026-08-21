from .workbench import MedicalKnowledgeWorkbench
from .translation_stability_core import (
    capture_core as _capture_translation_core,
    install_final as _install_translation_stability_core,
)

# Capture the unwrapped core before any legacy hardening installer can wrap the
# translation entry points. The final stability install below replaces the
# historical wrapper chain with one deterministic runtime contract.
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
_install_translation_stability_core()

__all__ = ["MedicalKnowledgeWorkbench"]
