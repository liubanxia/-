from __future__ import annotations

from pathlib import Path

from .answerer import KnowledgeAnswerer
from .config import WorkbenchPaths, get_paths
from .db import KnowledgeDB
from .product_document_ingest import ProductDocumentIngestor, SUPPORTED_EXTENSIONS
from .llm_safe import LocalLLM
from .notes import TXTNotesOrganizer
from .document_organizer import MultiDocumentOrganizer
from .retrieval import Retriever
from .rich_export import MultiFormatExporter
from .translator import PDFTranslator


class MedicalKnowledgeWorkbench:
    def __init__(self, paths: WorkbenchPaths | None = None):
        self.paths = paths or get_paths()
        self.db = KnowledgeDB(self.paths.database)
        self.ingestor = ProductDocumentIngestor(self.db, self.paths)
        self.retriever = Retriever(self.db, self.paths)
        self.llm = LocalLLM(self.paths)
        self.answerer = KnowledgeAnswerer(self.retriever, self.llm)
        self.organizer = MultiDocumentOrganizer(
            self.db,
            self.retriever,
            self.llm,
            self.paths.evidence_root,
            self.paths.runtime_root,
        )
        self.translator = PDFTranslator(self.paths, self.llm)
        self.notes = TXTNotesOrganizer(self.paths, self.llm)
        self.exporter = MultiFormatExporter(
            self.paths.evidence_root / "多格式输出"
        )
        self.last_export_bundle = None
        self.last_export_error = ""

    def close(self):
        try:
            self.translator.engine.unload()
        except Exception:
            pass
        try:
            self.retriever.embeddings.unload_model()
        except Exception:
            pass
        try:
            self.llm.unload()
        except Exception:
            pass
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
            "supported_input_formats": sorted(SUPPORTED_EXTENSIONS),
            "legacy_ppt": self.ingestor.legacy_ppt_status(),
            "organized_output_formats": ["pdf", "docx", "md", "txt"],
            "llm_backend": self.llm.backend(),
            "generator_fast": self.llm.active_model_name("fast"),
            "generator_deep": self.llm.active_model_name("deep"),
            "embedding_available": self.retriever.embeddings.available(),
            "embedding_device": self.retriever.embeddings.device,
            "translation_backends": self.translator.engine.available_backends(),
            "translation_qwen_review_default": False,
            "response_pipeline": [
                "lexical_immediate",
                "semantic_vector",
                "optional_generator",
            ],
            "recent_tasks": [dict(row) for row in tasks],
            "last_export_error": self.last_export_error,
        }

    def latest_resumable_task(self):
        for row in self.db.list_tasks(limit=100):
            if (
                str(row["kind"]) == "deep_organize"
                and str(row["status"]) in {"queued", "running", "failed", "paused"}
            ):
                return row
        return None

    def ingest(self, path: Path, **kwargs):
        return self.ingestor.ingest(Path(path), **kwargs)

    def ask(self, query: str, **kwargs):
        return self.answerer.ask(query, **kwargs)

    def organize(self, title: str, instruction: str, **kwargs):
        output, task_id = self.organizer.organize(title, instruction, **kwargs)
        self.last_export_bundle = None
        self.last_export_error = ""
        try:
            self.last_export_bundle = self.exporter.export_path(
                Path(output),
                title=title or Path(output).stem,
            )
        except Exception as exc:
            self.last_export_error = f"{type(exc).__name__}: {exc}"
        return output, task_id

    def resume_task(self, task_id: int, **kwargs):
        output, resumed_id = self.organizer.resume(int(task_id), **kwargs)
        self.last_export_bundle = None
        self.last_export_error = ""
        try:
            self.last_export_bundle = self.exporter.export_path(Path(output))
        except Exception as exc:
            self.last_export_error = f"{type(exc).__name__}: {exc}"
        return output, resumed_id

    def translate_book(self, path: Path, **kwargs):
        return self.translator.translate_book(Path(path), **kwargs)

    def organize_txt(self, source_text: str, **kwargs):
        return self.notes.organize(source_text, **kwargs)

    def organize_txt_file(self, path: Path, **kwargs):
        return self.notes.organize_file(Path(path), **kwargs)
