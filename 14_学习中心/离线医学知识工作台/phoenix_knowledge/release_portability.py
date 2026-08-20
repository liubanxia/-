from __future__ import annotations

import hashlib
from pathlib import Path

_INSTALLED = False


def _sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _stable_tail(stored: str) -> tuple[str, ...]:
    normalized = str(stored or "").replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part)
    for marker in ("14_学习中心", "15_证据中心"):
        if marker in parts:
            return parts[parts.index(marker) :]
    return ()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _candidate_paths(workbench, stored: str) -> list[Path]:
    source_root = Path(workbench.paths.source_root)
    project_root = Path(workbench.paths.project_root)
    filename = Path(str(stored).replace("\\", "/")).name
    candidates: list[Path] = []

    if filename:
        candidates.append(source_root / filename)

    tail = _stable_tail(stored)
    if tail:
        candidates.append(project_root.joinpath(*tail))

    # Normal product imports are copied under source_root. Search only this
    # Phoenix-owned library and cap matches so a stale path never causes an
    # unbounded disk crawl.
    if filename and source_root.is_dir():
        try:
            for index, candidate in enumerate(
                source_root.rglob(filename),
                start=1,
            ):
                candidates.append(candidate)
                if index >= 32:
                    break
        except OSError:
            pass

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _path_matches(path: Path, expected_sha: str) -> bool:
    try:
        if not path.is_file():
            return False
        return (
            not expected_sha
            or _sha256(path).lower() == expected_sha.lower()
        )
    except OSError:
        return False


def rebase_stale_document_paths(workbench) -> tuple[int, int]:
    """Rebase D:/G: document paths only after SHA-256 identity verification.

    Files already inside the current Phoenix source library are trusted by path
    and are not re-hashed on every startup. Only stale/out-of-library paths pay
    the SHA cost. Chunk IDs and embeddings remain untouched.
    """

    rebased = 0
    unresolved = 0
    source_root = Path(workbench.paths.source_root)

    for row in workbench.db.list_documents():
        stored = str(row["path"] or "").strip()
        expected_sha = str(row["sha256"] or "").strip().lower()
        if not stored:
            unresolved += 1
            continue

        stored_path = Path(stored)
        try:
            if stored_path.is_file() and _inside(
                stored_path,
                source_root,
            ):
                continue
        except OSError:
            pass

        # A path outside the current SSD library might be an old D:/G: mount or
        # an external import. Trust it only after content identity verification.
        if _path_matches(stored_path, expected_sha):
            continue

        matched: Path | None = None
        for candidate in _candidate_paths(workbench, stored):
            if not _path_matches(candidate, expected_sha):
                continue
            try:
                matched = candidate.resolve()
            except OSError:
                matched = candidate
            break

        if matched is None:
            unresolved += 1
            continue

        with workbench.db._lock, workbench.db._conn:
            conflict = workbench.db._conn.execute(
                "SELECT id, sha256 FROM documents WHERE path=? AND id<>?",
                (str(matched), int(row["id"])),
            ).fetchone()
            if conflict is not None:
                # Do not merge/delete medical knowledge records implicitly.
                unresolved += 1
                continue
            workbench.db._conn.execute(
                "UPDATE documents SET path=? WHERE id=?",
                (str(matched), int(row["id"])),
            )
        rebased += 1

    return rebased, unresolved


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .workbench import MedicalKnowledgeWorkbench

    original_init = MedicalKnowledgeWorkbench.__init__
    original_status = MedicalKnowledgeWorkbench.status

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            rebased, unresolved = rebase_stale_document_paths(self)
        except Exception:
            rebased, unresolved = 0, 0
        self._portable_paths_rebased = int(rebased)
        self._portable_paths_unresolved = int(unresolved)

    def status(self):
        payload = original_status(self)
        payload.update(
            {
                "document_paths_rebased": int(
                    getattr(self, "_portable_paths_rebased", 0)
                ),
                "document_paths_unresolved": int(
                    getattr(self, "_portable_paths_unresolved", 0)
                ),
            }
        )
        return payload

    MedicalKnowledgeWorkbench.__init__ = __init__
    MedicalKnowledgeWorkbench.status = status
