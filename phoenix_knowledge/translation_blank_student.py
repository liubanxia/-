from __future__ import annotations

"""Tiny from-scratch shadow learner for Phoenix medical translation.

The student observes every accepted English->Chinese translation from the first
file onward. It never participates in production translation and therefore can
learn continuously without risking current PDF/PPT output. Neural optimization
is delayed until a document finishes and is bounded by a short time budget.

No tokenizer or pretrained model is required: UTF-8 bytes are the vocabulary.
The default seed network is intentionally tiny; larger capacity profiles can be
selected later and trained from the same accumulated corpus.
"""

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_INSTALLED = False
_ROOT_NAME = "blank_student"
_DB_NAME = "blank_student.sqlite3"
PAD_ID = 256
BOS_ID = 257
EOS_ID = 258
VOCAB_SIZE = 259
PRODUCTION_ELIGIBLE = False


@dataclass(frozen=True)
class CapacityProfile:
    name: str
    embedding_dim: int
    hidden_dim: int
    layers: int
    max_source_bytes: int
    max_target_bytes: int
    batch_size: int
    max_steps_per_document: int


CAPACITY_PROFILES: dict[str, CapacityProfile] = {
    "seed": CapacityProfile("seed", 32, 48, 1, 128, 160, 4, 6),
    "small": CapacityProfile("small", 64, 128, 1, 320, 384, 6, 8),
    "medium": CapacityProfile("medium", 128, 384, 2, 640, 768, 8, 10),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def capacity_profile(name: str | None = None) -> CapacityProfile:
    selected = str(
        name or os.environ.get("PHOENIX_BLANK_STUDENT_CAPACITY", "seed")
    ).strip().lower()
    return CAPACITY_PROFILES.get(selected, CAPACITY_PROFILES["seed"])


def parameter_count(profile: str | CapacityProfile = "seed") -> int:
    """Pure arithmetic so package startup never imports torch."""
    p = profile if isinstance(profile, CapacityProfile) else capacity_profile(profile)
    e, h = int(p.embedding_dim), int(p.hidden_dim)
    total = 2 * VOCAB_SIZE * e + h * VOCAB_SIZE + VOCAB_SIZE
    input_size = e
    for _ in range(int(p.layers)):
        # One encoder GRU + one decoder GRU. Each PyTorch GRU layer has
        # weight_ih, weight_hh and two biases, each bias sized 3*hidden.
        per_gru = 3 * h * input_size + 3 * h * h + 6 * h
        total += 2 * per_gru
        input_size = h
    return int(total)


def _training_budget_seconds() -> float:
    raw = os.environ.get("PHOENIX_BLANK_STUDENT_TRAIN_SECONDS", "").strip()
    try:
        value = float(raw) if raw else 2.0
    except (TypeError, ValueError):
        value = 2.0
    return max(0.2, min(10.0, value))


def _sample_key(source: str, target: str) -> str:
    source_norm = " ".join(str(source or "").split()).casefold()
    payload = source_norm + "\n" + str(target or "").strip()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _teacher_rank(backend: str) -> int:
    value = str(backend or "").lower()
    if any(
        token in value
        for token in (
            "translation_fallback",
            "remote_server",
            "deepseek",
            "openai",
            "gemini",
            "_api",
            "api_",
        )
    ):
        return 5
    if "qwen_local_medical_model3" in value or "quality_final" in value:
        return 4
    if "deterministic_medical" in value:
        return 4
    if "hymt" in value or "model2" in value:
        return 3
    if "translation_memory_exact" in value:
        return 3
    if "model1" in value or "marian" in value or "nllb" in value:
        return 2
    return 1


@dataclass(frozen=True)
class StudentStats:
    samples: int
    exposures: int
    trained_samples: int
    train_steps: int
    capacity: str
    checkpoint_exists: bool
    shadow_only: bool = True


class BlankStudentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / _DB_NAME
        self.models_root = self.root / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.db_path), timeout=15)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS student_samples (
                    sample_key TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    teacher_backend TEXT NOT NULL,
                    teacher_rank INTEGER NOT NULL DEFAULT 1,
                    quality_score REAL NOT NULL DEFAULT 0,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_student_samples_rank
                    ON student_samples(teacher_rank DESC, quality_score DESC, updated_at DESC);

                CREATE TABLE IF NOT EXISTS student_training_progress (
                    profile TEXT NOT NULL,
                    sample_key TEXT NOT NULL,
                    train_count INTEGER NOT NULL DEFAULT 0,
                    last_loss REAL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(profile, sample_key)
                );
                """
            )

    def observe(
        self,
        source: str,
        target: str,
        target_language: str,
        *,
        teacher_backend: str,
        quality_score: float,
    ) -> bool:
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target or source == target:
            return False
        key = _sample_key(source, target)
        now = _utc_now()
        rank = _teacher_rank(teacher_backend)
        with self._connect() as db:
            existed = db.execute(
                "SELECT 1 FROM student_samples WHERE sample_key=?", (key,)
            ).fetchone() is not None
            db.execute(
                """
                INSERT INTO student_samples(
                    sample_key, source_text, target_text, target_language,
                    teacher_backend, teacher_rank, quality_score,
                    seen_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(sample_key) DO UPDATE SET
                    seen_count=student_samples.seen_count+1,
                    teacher_backend=CASE
                        WHEN excluded.teacher_rank >= student_samples.teacher_rank
                        THEN excluded.teacher_backend ELSE student_samples.teacher_backend END,
                    teacher_rank=MAX(student_samples.teacher_rank, excluded.teacher_rank),
                    quality_score=MAX(student_samples.quality_score, excluded.quality_score),
                    updated_at=excluded.updated_at
                """,
                (
                    key,
                    source,
                    target,
                    str(target_language or "中文").strip(),
                    str(teacher_backend or "unknown"),
                    rank,
                    float(quality_score),
                    now,
                    now,
                ),
            )
        return not existed

    def training_rows(self, profile: CapacityProfile, limit: int = 256) -> list[tuple]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT s.sample_key, s.source_text, s.target_text,
                       s.teacher_rank, s.quality_score,
                       COALESCE(p.train_count, 0)
                FROM student_samples AS s
                LEFT JOIN student_training_progress AS p
                  ON p.profile=? AND p.sample_key=s.sample_key
                WHERE s.target_language IN ('中文','简体中文','Chinese','zh','zh-cn','zh-CN')
                  AND s.quality_score >= 0.62
                ORDER BY COALESCE(p.train_count, 0) ASC,
                         s.teacher_rank DESC,
                         s.quality_score DESC,
                         s.updated_at DESC
                LIMIT ?
                """,
                (profile.name, max(16, int(limit))),
            ).fetchall()
        eligible: list[tuple] = []
        for row in rows:
            source_bytes = str(row[1]).encode("utf-8", errors="ignore")
            target_bytes = str(row[2]).encode("utf-8", errors="ignore")
            if not source_bytes or not target_bytes:
                continue
            # Long samples remain in the corpus and become trainable after a
            # later capacity increase instead of being destructively truncated.
            if len(source_bytes) > profile.max_source_bytes:
                continue
            if len(target_bytes) > profile.max_target_bytes:
                continue
            eligible.append(row)
        return eligible

    def mark_trained(self, profile: str, sample_keys: list[str], loss: float) -> None:
        now = _utc_now()
        with self._connect() as db:
            for key in sample_keys:
                db.execute(
                    """
                    INSERT INTO student_training_progress(
                        profile, sample_key, train_count, last_loss, updated_at
                    ) VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(profile, sample_key) DO UPDATE SET
                        train_count=student_training_progress.train_count+1,
                        last_loss=excluded.last_loss,
                        updated_at=excluded.updated_at
                    """,
                    (profile, key, float(loss), now),
                )

    def checkpoint_path(self, profile: str) -> Path:
        return self.models_root / f"student_{profile}.pt"

    def stats(self, profile: str | None = None) -> StudentStats:
        name = capacity_profile(profile).name
        with self._connect() as db:
            observed = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(seen_count), 0) FROM student_samples"
            ).fetchone()
            trained = db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(train_count), 0)
                FROM student_training_progress WHERE profile=?
                """,
                (name,),
            ).fetchone()
        return StudentStats(
            samples=int(observed[0] if observed else 0),
            exposures=int(observed[1] if observed else 0),
            trained_samples=int(trained[0] if trained else 0),
            train_steps=int(trained[1] if trained else 0),
            capacity=name,
            checkpoint_exists=self.checkpoint_path(name).is_file(),
            shadow_only=True,
        )


def _encode_bytes(text: str, max_bytes: int) -> list[int]:
    raw = str(text or "").encode("utf-8", errors="ignore")[:max_bytes]
    return [BOS_ID, *raw, EOS_ID]


def _pad_batch(torch, rows: list[list[int]]):
    width = max(len(row) for row in rows)
    tensor = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
    for index, row in enumerate(rows):
        tensor[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return tensor


def _build_model(profile: CapacityProfile):
    import torch.nn as nn

    class TinyByteSeq2Seq(nn.Module):
        def __init__(self):
            super().__init__()
            self.src_embedding = nn.Embedding(
                VOCAB_SIZE, profile.embedding_dim, padding_idx=PAD_ID
            )
            self.tgt_embedding = nn.Embedding(
                VOCAB_SIZE, profile.embedding_dim, padding_idx=PAD_ID
            )
            self.encoder = nn.GRU(
                profile.embedding_dim,
                profile.hidden_dim,
                num_layers=profile.layers,
                batch_first=True,
            )
            self.decoder = nn.GRU(
                profile.embedding_dim,
                profile.hidden_dim,
                num_layers=profile.layers,
                batch_first=True,
            )
            self.output = nn.Linear(profile.hidden_dim, VOCAB_SIZE)

        def forward(self, source_ids, decoder_input_ids):
            _, hidden = self.encoder(self.src_embedding(source_ids))
            decoded, _ = self.decoder(self.tgt_embedding(decoder_input_ids), hidden)
            return self.output(decoded)

    return TinyByteSeq2Seq()


class BlankStudentTrainer:
    def __init__(self, store: BlankStudentStore, profile: CapacityProfile | None = None):
        self.store = store
        self.profile = profile or capacity_profile()
        self.path = store.checkpoint_path(self.profile.name)

    def _load_or_create(self):
        import torch

        model = _build_model(self.profile).to("cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
        if self.path.is_file():
            try:
                payload = torch.load(str(self.path), map_location="cpu")
                if isinstance(payload, dict) and payload.get("profile") == self.profile.name:
                    model.load_state_dict(payload["model"])
                    if isinstance(payload.get("optimizer"), dict):
                        optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                print(
                    "[Phoenix][空白学生] checkpoint载入失败，当前容量重新初始化："
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        return torch, model, optimizer

    def _save(self, torch, model, optimizer) -> None:
        temp = self.path.with_suffix(".pt.tmp")
        torch.save(
            {
                "format": 1,
                "profile": self.profile.name,
                "created_from_scratch": True,
                "production_eligible": False,
                "parameters": parameter_count(self.profile),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            str(temp),
        )
        os.replace(temp, self.path)

    def train_budget(self, seconds: float | None = None) -> dict[str, Any]:
        rows = self.store.training_rows(self.profile)
        if len(rows) < 2:
            return {"trained": False, "steps": 0, "reason": "insufficient_short_samples"}
        try:
            torch, model, optimizer = self._load_or_create()
        except Exception as exc:
            return {
                "trained": False,
                "steps": 0,
                "reason": f"torch_unavailable:{type(exc).__name__}:{exc}",
            }

        model.train()
        deadline = time.perf_counter() + float(seconds or _training_budget_seconds())
        steps = 0
        cursor = 0
        losses: list[float] = []
        while (
            time.perf_counter() < deadline
            and steps < self.profile.max_steps_per_document
        ):
            batch = rows[cursor : cursor + self.profile.batch_size]
            if len(batch) < 2:
                cursor = 0
                batch = rows[: self.profile.batch_size]
            if len(batch) < 2:
                break
            cursor = (cursor + len(batch)) % len(rows)

            source_ids = _pad_batch(
                torch,
                [
                    _encode_bytes(str(row[1]), self.profile.max_source_bytes)
                    for row in batch
                ],
            )
            target_ids = _pad_batch(
                torch,
                [
                    _encode_bytes(str(row[2]), self.profile.max_target_bytes)
                    for row in batch
                ],
            )
            decoder_in = target_ids[:, :-1]
            labels = target_ids[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            logits = model(source_ids, decoder_in)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
                ignore_index=PAD_ID,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            self.store.mark_trained(
                self.profile.name,
                [str(row[0]) for row in batch],
                loss_value,
            )
            steps += 1

        if steps:
            try:
                self._save(torch, model, optimizer)
            except Exception as exc:
                return {
                    "trained": False,
                    "steps": steps,
                    "reason": f"save_failed:{type(exc).__name__}:{exc}",
                }
        return {
            "trained": bool(steps),
            "steps": steps,
            "last_loss": losses[-1] if losses else None,
            "reason": "ok" if steps else "time_budget",
        }


def _student_root(paths=None) -> Path:
    from .translation_ssd_storage import translation_storage_root

    root = translation_storage_root(paths) / "人工修订与学习" / _ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _store_for_paths(paths=None) -> BlankStudentStore:
    return BlankStudentStore(_student_root(paths))


def _observe_decision(engine, source: str, target_language: str, decision) -> None:
    try:
        if decision is None or bool(getattr(decision, "needs_review", True)):
            return
        quality = getattr(decision, "quality", None)
        if quality is None or not bool(getattr(quality, "ok", False)):
            return
        backend = str(getattr(decision, "backend", "") or "")
        if backend.startswith(("offline_pending_", "failed_", "blocked_")):
            return
        source = str(source or "").strip()
        text = str(getattr(decision, "text", "") or "").strip()
        if not source or not text or source == text:
            return
        try:
            from .translation_refusal_guard import looks_like_model_refusal

            if looks_like_model_refusal(text):
                return
        except Exception:
            pass

        store = _store_for_paths(getattr(engine, "paths", None))
        inserted = store.observe(
            source,
            text,
            target_language,
            teacher_backend=backend,
            quality_score=float(getattr(quality, "score", 0.0) or 0.0),
        )
        if inserted:
            stats = store.stats()
            if stats.samples in {1, 10, 100, 1000} or stats.samples % 5000 == 0:
                print(
                    "[Phoenix][空白学生] 全程学习样本已收集："
                    f"{stats.samples}条 | profile={stats.capacity} | 仅影子学习。",
                    flush=True,
                )
    except Exception as exc:
        print(
            "[Phoenix][空白学生] 样本收集失败但正式翻译继续："
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def _train_after_document(paths=None) -> None:
    try:
        store = _store_for_paths(paths)
        trainer = BlankStudentTrainer(store)
        result = trainer.train_budget()
        if not result.get("trained"):
            return
        stats = store.stats(trainer.profile.name)
        try:
            checkpoint_mb = (
                store.checkpoint_path(trainer.profile.name).stat().st_size
                / 1024
                / 1024
            )
        except OSError:
            checkpoint_mb = 0.0
        print(
            "[Phoenix][空白学生] 文档结束影子训练："
            f"steps={result.get('steps', 0)} | samples={stats.samples} | "
            f"profile={stats.capacity} | checkpoint={checkpoint_mb:.2f}MB；"
            "仍禁止参与正式译文。",
            flush=True,
        )
    except Exception as exc:
        print(
            "[Phoenix][空白学生] 影子训练跳过但不影响译文："
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


def status(paths=None) -> StudentStats:
    return _store_for_paths(paths).stats()


def expert_admission_allowed(*_args, **_kwargs) -> bool:
    """This learning module can never promote itself into production."""
    return False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translation_survival_memory as survival
    from .office_translation import OfficeDocumentTranslator
    from .translator import PDFTranslator

    old_store = survival._store_decision

    def store_decision(engine, source: str, target: str, decision) -> None:
        old_store(engine, source, target, decision)
        _observe_decision(engine, source, target, decision)

    survival._store_decision = store_decision

    # Backprop is not performed per sentence/page. It runs only after an entire
    # document returns successfully, with a strict time cap, so weak machines
    # remain usable and translation output is already complete before learning.
    old_pdf_book = PDFTranslator.translate_book

    def pdf_book(self, *args, **kwargs):
        result = old_pdf_book(self, *args, **kwargs)
        if not bool(getattr(result, "paused", False)):
            _train_after_document(getattr(self, "paths", None))
        return result

    PDFTranslator.translate_book = pdf_book

    old_office = OfficeDocumentTranslator.translate_document

    def office_document(self, *args, **kwargs):
        result = old_office(self, *args, **kwargs)
        if not bool(getattr(result, "paused", False)):
            _train_after_document(getattr(self, "paths", None))
        return result

    OfficeDocumentTranslator.translate_document = office_document

    params = parameter_count("seed")
    print(
        "[Phoenix][空白学生] 已启用：随机初始化Byte-GRU从第1份资料开始全程收集/影子学习；"
        f"seed参数={params:,}，原始FP32权重约{params * 4 / 1024 / 1024:.2f}MB。"
        "不参与正式翻译；以后可切换small/medium并复用全部历史语料。",
        flush=True,
    )
