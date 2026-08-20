from __future__ import annotations

_INSTALLED = False


def _unload_embeddings(workbench_or_retriever) -> None:
    try:
        retriever = getattr(
            workbench_or_retriever,
            "retriever",
            workbench_or_retriever,
        )
        embeddings = getattr(retriever, "embeddings", None)
        if embeddings is not None:
            embeddings.unload_model()
    except Exception:
        pass


def _unload_llm(target) -> None:
    try:
        llm = getattr(target, "llm", None)
        if llm is not None:
            llm.unload()
    except Exception:
        pass


def install() -> None:
    """Keep embedding and generator weights from competing for small VRAM.

    Phoenix frequently runs on 8 GB GPUs. Retrieval needs the embedding model;
    grounded synthesis then needs Qwen. The two stages do not need to coexist.
    """

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .organizer import DeepOrganizer
    from .workbench import MedicalKnowledgeWorkbench

    original_retrieve = DeepOrganizer._retrieve_evidence

    def retrieve_evidence(self, *args, **kwargs):
        # A generator left resident by an earlier QA/translation can otherwise
        # collide with the embedding model during broad multi-document search.
        _unload_llm(self)
        try:
            return original_retrieve(self, *args, **kwargs)
        finally:
            # Retrieval evidence is already materialized in RAM/SQLite. Release
            # embedding weights before the first Qwen batch is generated.
            _unload_embeddings(self.retriever)

    DeepOrganizer._retrieve_evidence = retrieve_evidence

    original_ask = MedicalKnowledgeWorkbench.ask

    def ask(self, query: str, **kwargs):
        if kwargs.get("use_embeddings", True):
            # Repeated intelligent questions may leave Qwen resident. Free it
            # before semantic query encoding, then Answerer releases embeddings
            # before loading Qwen for grounded synthesis.
            _unload_llm(self)
        return original_ask(self, query, **kwargs)

    MedicalKnowledgeWorkbench.ask = ask

    original_resume = MedicalKnowledgeWorkbench.resume_task

    def resume_task(self, task_id: int, **kwargs):
        # Resume uses saved evidence and can jump directly into generation.
        _unload_embeddings(self)
        return original_resume(self, task_id, **kwargs)

    MedicalKnowledgeWorkbench.resume_task = resume_task

    original_translate = MedicalKnowledgeWorkbench.translate_book

    def translate_book(self, path, **kwargs):
        _unload_embeddings(self)
        return original_translate(self, path, **kwargs)

    MedicalKnowledgeWorkbench.translate_book = translate_book

    original_notes = MedicalKnowledgeWorkbench.organize_txt

    def organize_txt(self, source_text: str, **kwargs):
        _unload_embeddings(self)
        return original_notes(self, source_text, **kwargs)

    MedicalKnowledgeWorkbench.organize_txt = organize_txt

    original_notes_file = MedicalKnowledgeWorkbench.organize_txt_file

    def organize_txt_file(self, path, **kwargs):
        _unload_embeddings(self)
        return original_notes_file(self, path, **kwargs)

    MedicalKnowledgeWorkbench.organize_txt_file = organize_txt_file
