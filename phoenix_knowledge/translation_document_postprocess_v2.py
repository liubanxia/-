from __future__ import annotations

"""Single post-translation hook for learning-only side effects.

PDFTranslator and OfficeDocumentTranslator remain publication-focused. Book
maturity accounting and blank-student shadow training run only after the stable
Workbench translation contract has returned a successful non-paused result.
Learning failures are non-fatal and can never invalidate an already-validated
formal deliverable.
"""

from pathlib import Path

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import translation_blank_student as student
    from . import translation_learning_maturity_gate as maturity
    from . import translation_survival_memory as survival
    from .pdf_parser import sha256_file
    from .workbench import MedicalKnowledgeWorkbench
    from .workbench_stability_core import WORKBENCH_CONTRACT_VERSION

    old_translate_book = MedicalKnowledgeWorkbench.translate_book

    def translate_book(self, path: Path, **kwargs):
        result = old_translate_book(self, path, **kwargs)
        if bool(getattr(result, "paused", False)):
            return result

        source = Path(path)

        # Maturity counts distinct successfully completed PDF books only. This
        # is bookkeeping after formal output validation, never a translation
        # decision path.
        if source.suffix.lower() == ".pdf":
            try:
                memory = survival._memory_for_engine(self.translator.engine)
                tracker = maturity._tracker_for_memory(memory)
                if source.is_file():
                    tracker.record_completed_book(
                        sha256_file(source),
                        str(source),
                    )
                    maturity._report(self.translator.engine, tracker.stats())
            except Exception as exc:
                print(
                    "[Phoenix][学习成熟度] 完成书籍计数失败，但不影响已验收译文："
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        # Shadow training is document-bounded and runs only after the deliverable
        # is complete. The student can never self-promote into production.
        student._train_after_document(getattr(self, "paths", None))
        return result

    translate_book._phoenix_workbench_contract = WORKBENCH_CONTRACT_VERSION
    translate_book._phoenix_document_postprocess_v2 = True
    # Architecture fingerprint intentionally still identifies this as the stable
    # Workbench contract layer; we are extending its post-success hook, not
    # replacing the translation/publication semantics.
    translate_book.__module__ = "phoenix_knowledge.workbench_stability_core"
    MedicalKnowledgeWorkbench.translate_book = translate_book

    # Current v3 installers intentionally change routing internals, but the last
    # production step must re-assert licensing/runtime policy and the one stable
    # PDF publication boundary. Keeping this here makes document-learning hooks
    # the final semantic extension instead of another public translation wrapper.
    from .translation_model1_policy_v2 import install as install_model1_policy_v2
    from .translation_final_contract_v2 import install as install_final_contract_v2

    install_model1_policy_v2()
    install_final_contract_v2()

    _INSTALLED = True
