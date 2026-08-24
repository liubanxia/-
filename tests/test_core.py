from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenix_knowledge.answerer import KnowledgeAnswerer
from phoenix_knowledge.chunker import chunk_text
from phoenix_knowledge.config import WorkbenchPaths, resolve_model_dir
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.organizer import DeepOrganizer
from phoenix_knowledge.retrieval import Evidence, Retriever, query_variants


class FakeLLM:
    def __init__(self, text: str, available: bool = True):
        self.text = text
        self._available = available

    def available(self):
        return self._available

    def backend(self):
        return "fake" if self._available else "evidence_only"

    def generate(self, prompt: str, max_new_tokens: int = 1200):
        return self.text


class FailAfterOneLLM:
    def __init__(self):
        self.calls = 0

    def available(self):
        return True

    def generate(self, prompt: str, max_new_tokens: int = 1200):
        self.calls += 1
        if self.calls == 1:
            return "第一批已整理。[S1]"
        raise RuntimeError("simulated interruption")


class StaticRetriever:
    def __init__(self, evidence):
        self.evidence = evidence

    def search(
        self,
        query: str,
        limit: int = 20,
        use_embeddings: bool = True,
    ):
        return self.evidence[:limit]

    def search_diverse(
        self,
        query: str,
        limit: int = 200,
        use_embeddings: bool = True,
    ):
        return self.evidence[:limit]


class CoreTest(unittest.TestCase):
    def test_chunking_is_bounded_and_nonempty(self):
        text = (
            "肺结节影像学征象。" * 300
        ) + "\n\n" + (
            "鉴别诊断。" * 300
        )
        chunks = chunk_text(
            text,
            max_chars=500,
            overlap_chars=50,
        )
        self.assertGreater(len(chunks), 2)
        self.assertTrue(
            all(0 < len(chunk) <= 500 for chunk in chunks)
        )

    def test_db_tracks_page_source_and_search(self):
        with tempfile.TemporaryDirectory() as td:
            db = KnowledgeDB(
                Path(td) / "k.sqlite3"
            )
            try:
                doc_id = db.upsert_document(
                    Path(td) / "book.pdf",
                    "abc",
                    "胸部CT",
                    10,
                )
                ids = db.replace_page_chunks(
                    doc_id,
                    7,
                    ["磨玻璃结节可见分叶及胸膜牵拉。"],
                )
                rows = db.search_lexical(
                    "磨玻璃",
                    limit=5,
                )
                self.assertEqual(len(ids), 1)
                self.assertTrue(rows)
                self.assertEqual(
                    int(rows[0]["page"]),
                    7,
                )
                self.assertIn(
                    "胸部CT",
                    rows[0]["title"],
                )
            finally:
                db.close()

    def test_chinese_natural_query_has_keyword_fallbacks(self):
        variants = query_variants(
            "请整理所有PDF中关于肺磨玻璃结节的恶性征象和鉴别诊断"
        )
        self.assertIn("肺磨玻璃结节", variants)
        self.assertIn("恶性征象", variants)
        self.assertIn("鉴别诊断", variants)

    def test_retrieval_works_without_embedding_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = root / "runtime"
            model_root = root / "models"
            paths = WorkbenchPaths(
                project_root=root,
                source_root=root / "pdf",
                runtime_root=runtime,
                evidence_root=root / "evidence",
                model_root=model_root,
                database=runtime / "knowledge.sqlite3",
                structure_root=runtime / "docling",
            ).ensure()
            db = KnowledgeDB(paths.database)
            try:
                doc_id = db.upsert_document(
                    root / "book.pdf",
                    "abc",
                    "胸部CT",
                    1,
                )
                db.replace_page_chunks(
                    doc_id,
                    1,
                    ["肺磨玻璃结节可出现分叶征、毛刺征，并需进行鉴别诊断。"],
                )
                hits = Retriever(db, paths).search(
                    "请整理所有PDF中关于肺磨玻璃结节的恶性征象和鉴别诊断",
                    limit=5,
                    use_embeddings=True,
                )
                self.assertTrue(hits)
                self.assertEqual(hits[0].page, 1)
            finally:
                db.close()

    def test_generated_answer_without_valid_source_is_blocked(self):
        evidence = [
            Evidence(
                chunk_id=12,
                source_key="D1:P4:C0",
                title="测试书",
                path="book.pdf",
                page=4,
                text="肺结节可有分叶征。",
                score=1.0,
            )
        ]
        answerer = KnowledgeAnswerer(
            StaticRetriever(evidence),
            FakeLLM("肺结节可有分叶征。"),
        )
        result = answerer.ask(
            "肺结节征象",
            deep=True,
        )
        self.assertEqual(
            result.mode,
            "grounding_blocked",
        )
        self.assertIn("已阻止", result.text)
        self.assertIn("[S12]", result.text)

    def test_generated_answer_with_valid_source_is_kept(self):
        evidence = [
            Evidence(
                chunk_id=12,
                source_key="D1:P4:C0",
                title="测试书",
                path="book.pdf",
                page=4,
                text="肺结节可有分叶征。",
                score=1.0,
            )
        ]
        answerer = KnowledgeAnswerer(
            StaticRetriever(evidence),
            FakeLLM(
                "肺结节可见分叶征。[S12]"
            ),
        )
        result = answerer.ask(
            "肺结节征象",
            deep=True,
        )
        self.assertEqual(
            result.mode,
            "grounded_generation",
        )
        self.assertIn(
            "测试书，第4页",
            result.text,
        )

    def test_organizer_checkpoint_and_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = KnowledgeDB(
                root / "knowledge.sqlite3"
            )
            evidence = [
                Evidence(
                    chunk_id=i,
                    source_key=f"D1:P{i}:C0",
                    title="测试书",
                    path="book.pdf",
                    page=i,
                    text=f"证据{i}",
                    score=1.0,
                )
                for i in range(1, 5)
            ]
            llm = FakeLLM(
                "整理内容 [S1] [S2] [S3] [S4]"
            )
            organizer = DeepOrganizer(
                db,
                StaticRetriever(evidence),
                llm,
                root / "outputs",
            )
            try:
                output, task_id = organizer.organize(
                    "测试专题",
                    "整理测试证据",
                    candidate_limit=4,
                    batch_size=2,
                )
                self.assertTrue(output.exists())
                task = db.get_task(task_id)
                self.assertEqual(
                    task["status"],
                    "completed",
                )
                self.assertEqual(
                    int(task["progress"]),
                    2,
                )
            finally:
                db.close()

    def test_failed_organizer_resumes_from_saved_batch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = KnowledgeDB(root / "knowledge.sqlite3")
            try:
                document_id = db.upsert_document(
                    root / "book.pdf",
                    "abc",
                    "测试书",
                    1,
                )
                chunk_ids = db.replace_page_chunks(
                    document_id,
                    1,
                    ["第一条证据", "第二条证据"],
                )
                rows = db.fetch_chunks(chunk_ids)
                evidence = [
                    Evidence(
                        chunk_id=int(row["id"]),
                        source_key=str(row["source_key"]),
                        title=str(row["title"]),
                        path=str(row["path"]),
                        page=int(row["page"]),
                        text=str(row["text"]),
                        score=1.0,
                    )
                    for row in rows
                ]
                organizer = DeepOrganizer(
                    db,
                    StaticRetriever(evidence),
                    FailAfterOneLLM(),
                    root / "outputs",
                )

                with self.assertRaises(RuntimeError):
                    organizer.organize(
                        "可恢复专题",
                        "整理证据",
                        candidate_limit=2,
                        batch_size=1,
                    )

                task = db.list_tasks(limit=1)[0]
                self.assertEqual(task["status"], "failed")
                self.assertEqual(int(task["progress"]), 1)

                organizer.llm = FakeLLM(
                    "恢复整理 [S1] [S2]"
                )
                output, task_id = organizer.resume(
                    int(task["id"])
                )
                resumed = db.get_task(task_id)
                self.assertTrue(output.exists())
                self.assertEqual(resumed["status"], "completed")
                self.assertEqual(int(resumed["progress"]), 2)
            finally:
                db.close()

    def test_modelscope_cache_pointer_is_resolved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_root = root / "models"
            target = model_root / "Qwen3.5-4B"
            actual = root / "modelscope_cache" / "snapshot"
            target.mkdir(parents=True)
            actual.mkdir(parents=True)
            (actual / "config.json").write_text("{}", encoding="utf-8")
            (target / "MODELSCOPE_CACHE_PATH.txt").write_text(
                str(actual),
                encoding="utf-8",
            )

            resolved = resolve_model_dir(
                model_root,
                "Qwen3.5-4B",
            )
            self.assertEqual(
                resolved,
                actual.resolve(),
            )

    def test_loopback_policy_rejects_external_url(self):
        from phoenix_knowledge.llm import LocalLLM

        self.assertTrue(
            LocalLLM._is_loopback_url(
                "http://127.0.0.1:8080/v1/chat/completions"
            )
        )
        self.assertTrue(
            LocalLLM._is_loopback_url(
                "http://localhost:8080/v1/chat/completions"
            )
        )
        self.assertFalse(
            LocalLLM._is_loopback_url(
                "https://example.com/v1/chat/completions"
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
