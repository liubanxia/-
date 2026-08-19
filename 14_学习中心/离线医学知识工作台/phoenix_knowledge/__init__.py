from .workbench import MedicalKnowledgeWorkbench
from .translation_recovery import install as _install_translation_recovery

_install_translation_recovery()

__all__ = ["MedicalKnowledgeWorkbench"]
