from __future__ import annotations

from pathlib import Path

from .answerer import KnowledgeAnswerer
from .config import WorkbenchPaths, get_paths
from .db import KnowledgeDB
from .ingest import LibraryIngestor
from .llm import LocalLLM
from .notes import TXTNotesOrganizer
from .organizer import DeepOrganizer
from .retrieval import Retriever
from .translator import PDFTranslator


class MedicalKnowledgeWorkbench:
    def __init__(self, paths: WorkbenchPaths | None = None):
        self.paths = paths or get_paths()
        self.db = KnowledgeDB(self.paths.database)
        self.ingestor = LibraryIngestor(self.db, self.paths)
        self.retriever = Retriever(self.db, self.paths)
        self.llm = LocalLLM(self.paths)
        self.answerer = KnowledgeAnswerer(
            self.retriever,
            self.llm,
        )
        self.organizer = DeepOrganizer(
            self.db,
            self.retriever,
            self.llm,
            self.paths.evidence_root,
        )
        self.translator = PDFTranslator(
            self.paths,
            self.llm,
        )
        self.notes = TXTNotesOrganizer(
            self.paths,
            self.llm,
        )

    def close(self):
        self.db.close()

    def status(self) -> dict:
        docs = self.db.list_documents()
        tasks = self.db.list_tasks(limit=20)
        return {
            "project_root": str(self.paths.project_root),
            "source_root": str(self.paths.source_root),
            "database": str(self.paths.database),
            "evidence_root": str(self.paths.evidence_root),
            "model_root": str(self.paths.model_root),
            "documents": len(docs),
            "chunks": self.db.count_chunks(),
            "llm_backend": self.llm.backend(),
            "embedding_available": self.retriever.embeddings.available(),
            "translation_backends": self.translator.engine.available_backends(),
            "recent_tasks": [dict(row) for row in tasks],
        }

    def latest_resumable_task(self):
        for row in self.db.list_tasks(limit=100):
            if (
                str(row["kind"]) == "deep_organize"
                and str(row["status"])
                in {"queued", "running", "failed", "paused"}
            ):
                return row
        return None

    def ingest(self, path: Path, **kwargs):
        return self.ingestor.ingest_pdf(path, **kwargs)

    def ask(self, query: str, **kwargs):
        return self.answerer.ask(query, **kwargs)

    def organize(self, title: str, instruction: str, **kwargs):
        return self.organizer.organize(
            title,
            instruction,
            **kwargs,
        )

    def resume_task(self, task_id: int, **kwargs):
        return self.organizer.resume(
            int(task_id),
            **kwargs,
        )

    def translate_book(self, path: Path, **kwargs):
        return self.translator.translate_book(
            Path(path),
            **kwargs,
        )

    def organize_txt(self, source_text: str, **kwargs):
        return self.notes.organize(
            source_text,
            **kwargs,
        )

    def organize_txt_file(self, path: Path, **kwargs):
        return self.notes.organize_file(
            Path(path),
            **kwargs,
        )
