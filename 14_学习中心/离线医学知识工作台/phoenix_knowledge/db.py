from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.fts_enabled = False
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL,
                    title TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    indexed_pages INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'new',
                    warning TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    source_key TEXT NOT NULL UNIQUE,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(document_id, page, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    model_name TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(chunk_id, model_name)
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checkpoint_json TEXT NOT NULL DEFAULT '{}',
                    progress INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS outputs (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    query TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_page
                    ON chunks(document_id, page, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_embeddings_model
                    ON embeddings(model_name, chunk_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                    ON tasks(status, updated_at);
                """
            )
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                    "USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')"
                )
                self.fts_enabled = True
            except sqlite3.OperationalError:
                self.fts_enabled = False

    def upsert_document(self, path: Path, sha256: str, title: str, page_count: int) -> int:
        path = str(Path(path).resolve())
        now = _now()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id, sha256 FROM documents WHERE path=?", (path,)
            ).fetchone()
            if row is None:
                cur = self._conn.execute(
                    """
                    INSERT INTO documents(
                        path, sha256, title, page_count, indexed_pages,
                        status, warning, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, 'indexing', '', ?, ?)
                    """,
                    (path, sha256, title, int(page_count), now, now),
                )
                return int(cur.lastrowid)

            document_id = int(row["id"])
            if row["sha256"] != sha256:
                chunk_ids = [
                    int(r[0])
                    for r in self._conn.execute(
                        "SELECT id FROM chunks WHERE document_id=?", (document_id,)
                    ).fetchall()
                ]
                if self.fts_enabled and chunk_ids:
                    self._conn.executemany(
                        "DELETE FROM chunks_fts WHERE chunk_id=?",
                        [(chunk_id,) for chunk_id in chunk_ids],
                    )
                self._conn.execute(
                    "DELETE FROM chunks WHERE document_id=?", (document_id,)
                )
                self._conn.execute(
                    """
                    UPDATE documents
                    SET sha256=?, title=?, page_count=?, indexed_pages=0,
                        status='indexing', warning='', updated_at=?
                    WHERE id=?
                    """,
                    (sha256, title, int(page_count), now, document_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE documents
                    SET title=?, page_count=?, status='indexing', updated_at=?
                    WHERE id=?
                    """,
                    (title, int(page_count), now, document_id),
                )
            return document_id

    def page_is_indexed(self, document_id: int, page: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM chunks WHERE document_id=? AND page=? LIMIT 1",
                (int(document_id), int(page)),
            ).fetchone()
            return row is not None

    def replace_page_chunks(self, document_id: int, page: int, chunks: Sequence[str]) -> list[int]:
        now = _now()
        page = int(page)
        with self._lock, self._conn:
            old_ids = [
                int(r[0])
                for r in self._conn.execute(
                    "SELECT id FROM chunks WHERE document_id=? AND page=?",
                    (int(document_id), page),
                ).fetchall()
            ]
            if self.fts_enabled and old_ids:
                self._conn.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id=?",
                    [(chunk_id,) for chunk_id in old_ids],
                )
            self._conn.execute(
                "DELETE FROM chunks WHERE document_id=? AND page=?",
                (int(document_id), page),
            )

            inserted: list[int] = []
            for index, text in enumerate(chunks):
                source_key = f"D{document_id}:P{page}:C{index}"
                cur = self._conn.execute(
                    """
                    INSERT INTO chunks(
                        document_id, page, chunk_index, source_key, text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (int(document_id), page, index, source_key, text, now),
                )
                chunk_id = int(cur.lastrowid)
                inserted.append(chunk_id)
                if self.fts_enabled:
                    self._conn.execute(
                        "INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)",
                        (chunk_id, text),
                    )

            indexed_pages = self._conn.execute(
                "SELECT COUNT(DISTINCT page) FROM chunks WHERE document_id=?",
                (int(document_id),),
            ).fetchone()[0]
            self._conn.execute(
                "UPDATE documents SET indexed_pages=?, updated_at=? WHERE id=?",
                (int(indexed_pages), now, int(document_id)),
            )
            return inserted

    def mark_document(self, document_id: int, status: str, warning: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE documents SET status=?, warning=?, updated_at=? WHERE id=?",
                (status, warning, _now(), int(document_id)),
            )

    def get_document(self, document_id: int):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM documents WHERE id=?", (int(document_id),)
            ).fetchone()

    def list_documents(self):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM documents ORDER BY updated_at DESC, id DESC"
            ).fetchall()

    def count_chunks(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def search_lexical(self, query: str, limit: int = 20):
        query = (query or "").strip()
        if not query:
            return []
        with self._lock:
            if self.fts_enabled:
                try:
                    rows = self._conn.execute(
                        """
                        SELECT c.id, c.source_key, c.page, c.text,
                               d.title, d.path, bm25(chunks_fts) AS rank
                        FROM chunks_fts
                        JOIN chunks c ON c.id = chunks_fts.chunk_id
                        JOIN documents d ON d.id = c.document_id
                        WHERE chunks_fts MATCH ?
                        ORDER BY rank ASC
                        LIMIT ?
                        """,
                        (query, int(limit)),
                    ).fetchall()
                    if rows:
                        return rows
                except sqlite3.OperationalError:
                    pass

            like = f"%{query}%"
            return self._conn.execute(
                """
                SELECT c.id, c.source_key, c.page, c.text,
                       d.title, d.path, 0.0 AS rank
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.text LIKE ? OR d.title LIKE ?
                ORDER BY c.id DESC
                LIMIT ?
                """,
                (like, like, int(limit)),
            ).fetchall()

    def fetch_chunks(self, chunk_ids: Iterable[int]):
        ids = [int(x) for x in chunk_ids]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT c.id, c.source_key, c.page, c.text,
                       d.title, d.path
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                """,
                ids,
            ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def missing_embedding_chunks(self, model_name: str, limit: int = 1000):
        with self._lock:
            return self._conn.execute(
                """
                SELECT c.id, c.text
                FROM chunks c
                LEFT JOIN embeddings e
                  ON e.chunk_id=c.id AND e.model_name=?
                WHERE e.chunk_id IS NULL
                ORDER BY c.id
                LIMIT ?
                """,
                (model_name, int(limit)),
            ).fetchall()

    def store_embeddings(self, model_name: str, items: Sequence[tuple[int, int, bytes]]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings(chunk_id, model_name, dim, vector)
                VALUES (?, ?, ?, ?)
                """,
                [(int(cid), model_name, int(dim), blob) for cid, dim, blob in items],
            )

    def iter_embeddings(self, model_name: str):
        with self._lock:
            return self._conn.execute(
                "SELECT chunk_id, dim, vector FROM embeddings WHERE model_name=?",
                (model_name,),
            ).fetchall()

    def create_task(self, kind: str, payload: dict, total: int = 0) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO tasks(
                    kind, status, payload_json, checkpoint_json,
                    progress, total, error, created_at, updated_at
                ) VALUES (?, 'queued', ?, '{}', 0, ?, '', ?, ?)
                """,
                (kind, json.dumps(payload, ensure_ascii=False), int(total), now, now),
            )
            return int(cur.lastrowid)

    def update_task(
        self,
        task_id: int,
        *,
        status: str | None = None,
        checkpoint: dict | None = None,
        progress: int | None = None,
        total: int | None = None,
        error: str | None = None,
    ) -> None:
        fields = ["updated_at=?"]
        values: list[object] = [_now()]
        if status is not None:
            fields.append("status=?")
            values.append(status)
        if checkpoint is not None:
            fields.append("checkpoint_json=?")
            values.append(json.dumps(checkpoint, ensure_ascii=False))
        if progress is not None:
            fields.append("progress=?")
            values.append(int(progress))
        if total is not None:
            fields.append("total=?")
            values.append(int(total))
        if error is not None:
            fields.append("error=?")
            values.append(error)
        values.append(int(task_id))
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id=?",
                values,
            )

    def get_task(self, task_id: int):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM tasks WHERE id=?", (int(task_id),)
            ).fetchone()

    def list_tasks(self, limit: int = 100):
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM tasks ORDER BY updated_at DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()

    def record_output(self, title: str, query: str, path: Path) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO outputs(title, query, path, created_at) VALUES (?, ?, ?, ?)",
                (title, query, str(path), _now()),
            )
