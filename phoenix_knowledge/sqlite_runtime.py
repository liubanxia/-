from __future__ import annotations

"""Centralized transient SQLite lifecycle for Phoenix translation subsystems.

Python's sqlite3.Connection context manager commits or rolls back transactions,
but it does not close the connection. On Windows this leaves database files
locked and breaks temporary-directory cleanup. Translation/learning subsystems
use short-lived connections, so every session must explicitly close.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_INSTALLED = False
_ALLOWED_SYNCHRONOUS = {"OFF", "NORMAL", "FULL", "EXTRA"}


@contextmanager
def sqlite_session(
    path: str | Path,
    *,
    timeout: float = 15.0,
    wal: bool = True,
    synchronous: str = "NORMAL",
    row_factory=None,
) -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(str(path), timeout=float(timeout))
    if row_factory is not None:
        db.row_factory = row_factory

    sync = str(synchronous or "NORMAL").upper()
    if sync not in _ALLOWED_SYNCHRONOUS:
        sync = "NORMAL"

    try:
        if wal:
            db.execute("PRAGMA journal_mode=WAL")
        db.execute(f"PRAGMA synchronous={sync}")
        yield db
        db.commit()
    except BaseException:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    finally:
        db.close()


def _connection_method(path_attribute: str):
    @contextmanager
    def _connect(self):
        path = getattr(self, path_attribute)
        with sqlite_session(path) as db:
            yield db

    _connect.__name__ = "_connect"
    _connect.__qualname__ = f"PhoenixSQLiteSafety.{path_attribute}._connect"
    _connect._phoenix_explicit_close = True
    return _connect


def install_translation_sqlite_safety() -> None:
    """Apply one lifecycle contract to every short-lived translation database.

    This installer is intentionally semantic-neutral: it changes only connection
    ownership/cleanup. It may safely run during package import, unlike production
    translation policy installers.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    from .translation_api_value_ledger import APIValueLedger
    from .translation_blank_student import BlankStudentStore
    from .translation_learning_maturity_gate import TranslationLearningMaturity
    from .translation_survival_memory import TranslationMemory

    TranslationMemory._connect = _connection_method("path")
    TranslationLearningMaturity._connect = _connection_method("memory_path")
    APIValueLedger._connect = _connection_method("path")
    BlankStudentStore._connect = _connection_method("db_path")

    _INSTALLED = True
