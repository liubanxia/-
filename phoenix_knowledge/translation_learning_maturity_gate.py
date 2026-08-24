from __future__ import annotations

"""Delay adaptive translation-memory reuse until the corpus is mature.

Phoenix collects high-quality translations from the first document, but learned
memory is not allowed to influence production translation until BOTH gates pass:
1) at least 10 distinct completed PDF books;
2) at least 1000 verified translation-memory rows.

The thresholds are configurable for testing/deployment, but the conservative
release defaults remain 10 books + 1000 verified rows. Static terminology rules,
the normal model1->model2->model3 chain, emergency CPU translation, and API
fallback continue to work before maturity. Failure of this tracker must never
block translation.
"""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_INSTALLED = False


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(int(minimum), value)


def minimum_books() -> int:
    return _int_env("PHOENIX_MEMORY_MIN_BOOKS", 10, 10)


def minimum_verified_entries() -> int:
    return _int_env("PHOENIX_MEMORY_MIN_VERIFIED", 1000, 100)


def is_mature(
    book_count: int,
    verified_entries: int,
    *,
    min_books: int = 10,
    min_verified_entries: int = 1000,
) -> bool:
    return (
        int(book_count) >= max(10, int(min_books))
        and int(verified_entries) >= max(100, int(min_verified_entries))
    )


@dataclass(frozen=True)
class MaturityStats:
    completed_books: int
    verified_entries: int
    min_books: int
    min_verified_entries: int

    @property
    def mature(self) -> bool:
        return is_mature(
            self.completed_books,
            self.verified_entries,
            min_books=self.min_books,
            min_verified_entries=self.min_verified_entries,
        )


class TranslationLearningMaturity:
    def __init__(self, memory_path: str | Path):
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.memory_path), timeout=15)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_learning_documents (
                    document_hash TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                )
                """
            )

    def record_completed_book(self, document_hash: str, source_path: str) -> None:
        key = str(document_hash or "").strip()
        if not key:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO translation_learning_documents(
                    document_hash, source_path, completed_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(document_hash) DO UPDATE SET
                    source_path=excluded.source_path,
                    completed_at=excluded.completed_at
                """,
                (key, str(source_path or ""), now),
            )

    def stats(self) -> MaturityStats:
        books = 0
        verified = 0
        with self._connect() as db:
            try:
                row = db.execute(
                    "SELECT COUNT(*) FROM translation_learning_documents"
                ).fetchone()
                books = int(row[0] if row else 0)
            except sqlite3.Error:
                books = 0
            try:
                row = db.execute(
                    """
                    SELECT COUNT(*) FROM translation_memory
                    WHERE verified_level >= 1 AND quality_score >= 0.62
                    """
                ).fetchone()
                verified = int(row[0] if row else 0)
            except sqlite3.Error:
                verified = 0
        return MaturityStats(
            completed_books=books,
            verified_entries=verified,
            min_books=minimum_books(),
            min_verified_entries=minimum_verified_entries(),
        )


def _tracker_for_memory(memory) -> TranslationLearningMaturity:
    tracker = getattr(memory, "_phoenix_maturity_tracker", None)
    if tracker is None:
        tracker = TranslationLearningMaturity(memory.path)
        memory._phoenix_maturity_tracker = tracker
    return tracker


def _report(engine, stats: MaturityStats) -> None:
    state = (
        stats.mature,
        stats.completed_books,
        stats.verified_entries,
        stats.min_books,
        stats.min_verified_entries,
    )
    if getattr(engine, "_phoenix_maturity_report", None) == state:
        return
    engine._phoenix_maturity_report = state
    if stats.mature:
        print(
            "[Phoenix][学习成熟度] 已成熟："
            f"完成书籍={stats.completed_books}，已验证译文={stats.verified_entries}；"
            "启用精确翻译记忆与相似句模型3草稿。",
            flush=True,
        )
    else:
        print(
            "[Phoenix][学习成熟度] 收集中："
            f"完成书籍={stats.completed_books}/{stats.min_books}，"
            f"已验证译文={stats.verified_entries}/{stats.min_verified_entries}；"
            "翻译记忆只收集、不介入生产翻译。",
            flush=True,
        )


def _memory_is_mature(memory, engine=None) -> bool:
    try:
        stats = _tracker_for_memory(memory).stats()
        if engine is not None:
            _report(engine, stats)
        return bool(stats.mature)
    except Exception as exc:
        # Safety default: if maturity state cannot be proven, do not reuse
        # learned memory. Translation itself continues through the normal chain.
        if engine is not None and not bool(
            getattr(engine, "_phoenix_maturity_error_reported", False)
        ):
            engine._phoenix_maturity_error_reported = True
            print(
                "[Phoenix][学习成熟度] 状态读取失败，记忆复用保持关闭；"
                f"正常翻译链继续运行：{type(exc).__name__}: {exc}",
                flush=True,
            )
        return False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translation_survival_memory as survival
    from .pdf_parser import sha256_file
    from .translator import PDFTranslator

    memory_cls = survival.TranslationMemory
    old_exact = memory_cls.lookup_exact
    old_similar = memory_cls.lookup_similar

    def lookup_exact(self, source: str, target_language: str):
        if not _memory_is_mature(self):
            return None
        return old_exact(self, source, target_language)

    def lookup_similar(self, source: str, target_language: str, **kwargs):
        if not _memory_is_mature(self):
            return None
        return old_similar(self, source, target_language, **kwargs)

    memory_cls.lookup_exact = lookup_exact
    memory_cls.lookup_similar = lookup_similar

    # Report maturity from the actual engine even while memory reuse is dormant.
    old_try = survival._try_exact_or_rule

    def try_exact_or_rule(engine, source: str, target: str):
        try:
            memory = survival._memory_for_engine(engine)
            _report(engine, _tracker_for_memory(memory).stats())
        except Exception:
            pass
        return old_try(engine, source, target)

    survival._try_exact_or_rule = try_exact_or_rule

    # Count only distinct, successfully completed PDF books. A paused or failed
    # translation never advances the gate. SHA-256 prevents renaming the same
    # book from being counted twice.
    old_book = PDFTranslator.translate_book

    def translate_book(self, *args, **kwargs):
        result = old_book(self, *args, **kwargs)
        try:
            if not bool(getattr(result, "paused", False)):
                source_path = Path(getattr(result, "source_path", ""))
                if source_path.is_file():
                    memory = survival._memory_for_engine(self.engine)
                    tracker = _tracker_for_memory(memory)
                    tracker.record_completed_book(
                        sha256_file(source_path),
                        str(source_path),
                    )
                    _report(self.engine, tracker.stats())
        except Exception as exc:
            print(
                "[Phoenix][学习成熟度] 完成书籍计数失败，但不影响译文："
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return result

    PDFTranslator.translate_book = translate_book

    print(
        "[Phoenix][学习成熟度] 安全门已启用：从第1本开始只收集；"
        f"至少完成{minimum_books()}本PDF且累计{minimum_verified_entries()}条已验证译文后，"
        "学习记忆才允许介入生产翻译。",
        flush=True,
    )
