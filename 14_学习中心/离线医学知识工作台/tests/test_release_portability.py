from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from phoenix_knowledge.config import (
    model_dir_ready,
    resolve_model_dir,
)
from phoenix_knowledge.db import KnowledgeDB
from phoenix_knowledge.release_portability import (
    rebase_stale_document_paths,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleasePortabilityTests(unittest.TestCase):
    def test_pointer_only_folder_is_not_a_ready_model(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "Qwen3.5-2B"
            target.mkdir()
            (target / "MODELSCOPE_CACHE_PATH.txt").write_text(
                "X:/stale/cache/model",
                encoding="utf-8",
            )
            self.assertFalse(model_dir_ready(target))

    def test_stale_modelscope_pointer_rebases_inside_current_ssd_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_root = root / "04_AI模型" / "知识工作台"
            target = model_root / "Qwen3.5-2B"
            target.mkdir(parents=True)

            current_snapshot = (
                model_root
                / "_modelscope_cache"
                / "Qwen"
                / "Qwen3.5-2B"
            )
            current_snapshot.mkdir(parents=True)
            (current_snapshot / "config.json").write_text(
                "{}",
                encoding="utf-8",
            )
            (current_snapshot / "model.safetensors").write_bytes(
                b"real-weight-placeholder"
            )

            stale = (
                root
                / "old_mount"
                / "_modelscope_cache"
                / "Qwen"
                / "Qwen3.5-2B"
            )
            (target / "MODELSCOPE_CACHE_PATH.txt").write_text(
                str(stale),
                encoding="utf-8",
            )

            resolved = resolve_model_dir(
                model_root,
                "Qwen3.5-2B",
            )
            self.assertEqual(
                resolved.resolve(),
                current_snapshot.resolve(),
            )
            rewritten = Path(
                (target / "MODELSCOPE_CACHE_PATH.txt")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                rewritten.resolve(),
                current_snapshot.resolve(),
            )

    def test_stale_document_path_rebases_only_after_sha_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "14_学习中心" / "PDF资料"
            runtime = root / "runtime"
            source_root.mkdir(parents=True)
            current = source_root / "book.pdf"
            current.write_bytes(b"same-medical-source")

            db = KnowledgeDB(runtime / "knowledge.sqlite3")
            try:
                stale = root / "old_drive" / "book.pdf"
                doc_id = db.upsert_document(
                    stale,
                    _sha(current),
                    "book",
                    1,
                )
                fake = SimpleNamespace(
                    db=db,
                    paths=SimpleNamespace(
                        source_root=source_root,
                        project_root=root,
                    ),
                )
                rebased, unresolved = rebase_stale_document_paths(
                    fake
                )
                self.assertEqual(rebased, 1)
                self.assertEqual(unresolved, 0)
                row = db.get_document(doc_id)
                self.assertEqual(
                    Path(row["path"]).resolve(),
                    current.resolve(),
                )
            finally:
                db.close()

    def test_wrong_same_name_file_is_never_rebased(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_root = root / "14_学习中心" / "PDF资料"
            runtime = root / "runtime"
            source_root.mkdir(parents=True)
            current = source_root / "book.pdf"
            current.write_bytes(b"different-file")

            db = KnowledgeDB(runtime / "knowledge.sqlite3")
            try:
                stale = root / "old_drive" / "book.pdf"
                doc_id = db.upsert_document(
                    stale,
                    hashlib.sha256(b"expected-source").hexdigest(),
                    "book",
                    1,
                )
                fake = SimpleNamespace(
                    db=db,
                    paths=SimpleNamespace(
                        source_root=source_root,
                        project_root=root,
                    ),
                )
                rebased, unresolved = rebase_stale_document_paths(
                    fake
                )
                self.assertEqual(rebased, 0)
                self.assertEqual(unresolved, 1)
                self.assertNotEqual(
                    Path(db.get_document(doc_id)["path"]).resolve(),
                    current.resolve(),
                )
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
