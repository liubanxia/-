from __future__ import annotations

"""Zero-model survival and API-saving translation layer.

Goals:
- Reuse accepted translations before spending compute or API tokens.
- Resolve a small set of deterministic, high-frequency medical sentences offline.
- Use a CPU-only ONNX emergency translator when the normal local stack cannot
  produce an acceptable draft.
- Use high-similarity translation memory only as a *draft* for model3, never as
  an automatic final translation.
- If every model and API route is unavailable, preserve the English source and
  queue it for later translation instead of publishing refusal text.
"""

import hashlib
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

_INSTALLED = False
_MEMORY_FILE = "translation_memory.sqlite3"
_EMERGENCY_FOLDER = "Emergency-Translator-ONNX"
_MIN_SIMILARITY_FOR_MODEL3_DRAFT = 0.92

_NEGATION_RE = re.compile(
    r"\b(?:no|not|without|absent|absence|negative|neither|nor|cannot|can't|exclude|"
    r"unlikely|free of|lack of|lacks|lacking)\b",
    re.I,
)
_LATERALITY_RE = re.compile(r"\b(?:left|right|bilateral|unilateral)\b", re.I)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*(?:mm|cm|mL|ml|mg|g|kg|HU|%|mmHg|Gy|mSv))?", re.I)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9/+.-]{1,12}\b")

_FIXED_SENTENCES = {
    "no acute cardiopulmonary abnormality": "未见急性心肺异常。",
    "no acute osseous abnormality": "未见急性骨性异常。",
    "no acute intracranial abnormality": "未见急性颅内异常。",
    "no evidence of acute intracranial hemorrhage": "未见急性颅内出血证据。",
    "no intracranial hemorrhage": "未见颅内出血。",
    "no focal consolidation": "未见局灶性肺实变。",
    "no pleural effusion or pneumothorax": "未见胸腔积液或气胸。",
    "no pneumothorax": "未见气胸。",
    "no pleural effusion": "未见胸腔积液。",
    "no significant interval change": "与前次相比未见明显变化。",
    "no significant change": "未见明显变化。",
    "clinical correlation is recommended": "建议结合临床。",
    "follow-up is recommended": "建议随访。",
    "further evaluation is recommended": "建议进一步评估。",
    "correlate clinically": "请结合临床。",
}

_SLOT_TERMS = {
    "acute intracranial hemorrhage": "急性颅内出血",
    "intracranial hemorrhage": "颅内出血",
    "acute infarction": "急性梗死",
    "acute ischemic infarction": "急性缺血性梗死",
    "acute pulmonary embolism": "急性肺栓塞",
    "pulmonary embolism": "肺栓塞",
    "pleural effusion": "胸腔积液",
    "pneumothorax": "气胸",
    "focal consolidation": "局灶性肺实变",
    "hydronephrosis": "肾积水",
    "nephrolithiasis": "肾结石",
    "bowel obstruction": "肠梗阻",
    "free intraperitoneal air": "腹腔游离气体",
    "lymphadenopathy": "淋巴结肿大",
    "osseous injury": "骨性损伤",
    "acute fracture": "急性骨折",
    "metastatic disease": "转移性疾病",
}

_NO_EVIDENCE_RE = re.compile(r"^(?:there is\s+)?no evidence of\s+(.+?)[.!?]?$", re.I)
_NO_IDENTIFIED_RE = re.compile(r"^no\s+(.+?)\s+(?:is|are)\s+(?:identified|seen|demonstrated)[.!?]?$", re.I)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_source(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"[\s.;,:!?]+$", "", value)
    return value.casefold()


def _source_hash(source: str, target_language: str) -> str:
    payload = f"{normalize_source(source)}\n{str(target_language or '').strip().casefold()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _signature(pattern: re.Pattern, text: str) -> tuple[str, ...]:
    return tuple(sorted(match.group(0).casefold() for match in pattern.finditer(str(text or ""))))


def _safety_signature(text: str) -> tuple[tuple[str, ...], ...]:
    return (
        _signature(_NEGATION_RE, text),
        _signature(_LATERALITY_RE, text),
        _signature(_NUMBER_RE, text),
        tuple(sorted(_ACRONYM_RE.findall(str(text or "")))),
    )


@dataclass(frozen=True)
class MemoryHit:
    source: str
    translation: str
    backend: str
    quality_score: float
    similarity: float = 1.0


class TranslationMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=15)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS translation_memory (
                    source_hash TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_norm TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    quality_score REAL NOT NULL DEFAULT 0,
                    verified_level INTEGER NOT NULL DEFAULT 1,
                    use_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_hash, target_language)
                );
                CREATE INDEX IF NOT EXISTS idx_translation_memory_norm
                    ON translation_memory(target_language, source_norm);

                CREATE TABLE IF NOT EXISTS pending_translation (
                    source_hash TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_hash, target_language)
                );
                """
            )

    def store(
        self,
        source: str,
        translation: str,
        target_language: str,
        *,
        backend: str,
        quality_score: float,
        verified_level: int = 1,
    ) -> None:
        source = str(source or "").strip()
        translation = str(translation or "").strip()
        target = str(target_language or "中文").strip()
        if not source or not translation:
            return
        key = _source_hash(source, target)
        now = _utc_now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO translation_memory(
                    source_hash, target_language, source_norm, source_text,
                    target_text, backend, quality_score, verified_level,
                    use_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(source_hash, target_language) DO UPDATE SET
                    source_text=excluded.source_text,
                    target_text=CASE
                        WHEN excluded.quality_score >= translation_memory.quality_score
                        THEN excluded.target_text ELSE translation_memory.target_text END,
                    backend=CASE
                        WHEN excluded.quality_score >= translation_memory.quality_score
                        THEN excluded.backend ELSE translation_memory.backend END,
                    quality_score=MAX(translation_memory.quality_score, excluded.quality_score),
                    verified_level=MAX(translation_memory.verified_level, excluded.verified_level),
                    updated_at=excluded.updated_at
                """,
                (
                    key,
                    target,
                    normalize_source(source),
                    source,
                    translation,
                    str(backend or "unknown"),
                    float(quality_score),
                    int(verified_level),
                    now,
                    now,
                ),
            )

    def lookup_exact(self, source: str, target_language: str) -> MemoryHit | None:
        target = str(target_language or "中文").strip()
        key = _source_hash(source, target)
        with self._connect() as db:
            row = db.execute(
                """
                SELECT source_text, target_text, backend, quality_score
                FROM translation_memory
                WHERE source_hash=? AND target_language=? AND verified_level>=1
                """,
                (key, target),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                """
                UPDATE translation_memory
                SET use_count=use_count+1, updated_at=?
                WHERE source_hash=? AND target_language=?
                """,
                (_utc_now(), key, target),
            )
        return MemoryHit(str(row[0]), str(row[1]), str(row[2]), float(row[3]), 1.0)

    def lookup_similar(
        self,
        source: str,
        target_language: str,
        *,
        threshold: float = _MIN_SIMILARITY_FOR_MODEL3_DRAFT,
        limit: int = 250,
    ) -> MemoryHit | None:
        """Return a guarded fuzzy-memory hint; never publish this directly."""

        norm = normalize_source(source)
        if len(norm) < 12:
            return None
        target = str(target_language or "中文").strip()
        low, high = max(1, int(len(norm) * 0.70)), int(len(norm) * 1.30) + 8
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT source_text, source_norm, target_text, backend, quality_score
                FROM translation_memory
                WHERE target_language=? AND verified_level>=1
                  AND LENGTH(source_norm) BETWEEN ? AND ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (target, low, high, int(limit)),
            ).fetchall()

        source_sig = _safety_signature(source)
        best: MemoryHit | None = None
        for row in rows:
            candidate_source = str(row[0])
            if _safety_signature(candidate_source) != source_sig:
                continue
            ratio = SequenceMatcher(None, norm, str(row[1])).ratio()
            if ratio < threshold:
                continue
            if best is None or ratio > best.similarity:
                best = MemoryHit(
                    source=candidate_source,
                    translation=str(row[2]),
                    backend=str(row[3]),
                    quality_score=float(row[4]),
                    similarity=float(ratio),
                )
        return best

    def enqueue_pending(self, source: str, target_language: str, reason: str) -> None:
        source = str(source or "").strip()
        if not source:
            return
        target = str(target_language or "中文").strip()
        key = _source_hash(source, target)
        now = _utc_now()
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO pending_translation(
                    source_hash, target_language, source_text, reason,
                    attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(source_hash, target_language) DO UPDATE SET
                    reason=excluded.reason,
                    attempts=pending_translation.attempts+1,
                    updated_at=excluded.updated_at
                """,
                (key, target, source, str(reason or "offline"), now, now),
            )


def _memory_for_engine(engine) -> TranslationMemory:
    memory = getattr(engine, "_phoenix_translation_memory", None)
    if memory is not None:
        return memory
    from .translation_ssd_storage import translation_storage_root

    root = translation_storage_root(engine.paths) / "历史翻译数据"
    memory = TranslationMemory(root / _MEMORY_FILE)
    engine._phoenix_translation_memory = memory
    return memory


def _slot_translation(text: str) -> str | None:
    raw = normalize_source(text)
    if raw in _SLOT_TERMS:
        return _SLOT_TERMS[raw]

    try:
        from .medical_terminology_core import ABBREVIATION_SENSES, PHRASE_ALIASES

        sense = PHRASE_ALIASES.get(raw)
        if sense is not None:
            return str(sense.chinese)

        key = str(text or "").strip()
        senses = ABBREVIATION_SENSES.get(key) or ABBREVIATION_SENSES.get(key.upper())
        if senses and len(senses) == 1:
            return f"{senses[0].chinese}（{key}）"
    except Exception:
        pass
    return None


def deterministic_medical_translation(source: str) -> str | None:
    """Resolve only conservative whole-sentence patterns."""

    norm = normalize_source(source)
    if not norm:
        return None
    fixed = _FIXED_SENTENCES.get(norm)
    if fixed:
        return fixed

    match = _NO_EVIDENCE_RE.match(str(source or "").strip())
    if match:
        translated = _slot_translation(match.group(1).strip(" ."))
        if translated:
            return f"未见{translated}证据。"

    match = _NO_IDENTIFIED_RE.match(str(source or "").strip())
    if match:
        translated = _slot_translation(match.group(1).strip(" ."))
        if translated:
            return f"未见{translated}。"
    return None


class EmergencyONNXTranslator:
    """Optional CPU-only fallback. No model weights are bundled in the repo."""

    name = "emergency_onnx_cpu"

    def __init__(self, paths):
        self.model_path = Path(paths.model_root) / _EMERGENCY_FOLDER
        self._tokenizer = None
        self._model = None

    def available(self) -> bool:
        try:
            if not self.model_path.is_dir():
                return False
            has_onnx = any(self.model_path.rglob("*.onnx"))
            has_tokenizer = any(
                (self.model_path / name).is_file()
                for name in ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "spiece.model")
            )
            return bool(has_onnx and has_tokenizer)
        except OSError:
            return False

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if not self.available():
            raise RuntimeError(f"应急ONNX翻译模型未安装: {self.model_path}")

        from transformers import AutoTokenizer
        try:
            from optimum.onnxruntime import ORTModelForSeq2SeqLM
        except Exception as exc:
            raise RuntimeError(
                "应急ONNX模型存在，但当前环境缺少 optimum[onnxruntime]"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            use_fast=True,
        )
        kwargs = {"provider": "CPUExecutionProvider"}
        try:
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                **kwargs,
            )
        except TypeError:
            self._model = ORTModelForSeq2SeqLM.from_pretrained(
                str(self.model_path),
                **kwargs,
            )

    def translate(self, source: str, target_language: str = "中文") -> str:
        if str(target_language or "中文").strip().lower() not in {
            "中文", "简体中文", "chinese", "zh", "zh-cn"
        }:
            raise RuntimeError("应急ONNX层当前只处理英文→中文")
        self._load()
        text = str(source or "").strip()
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        )
        output = self._model.generate(
            **inputs,
            num_beams=2,
            do_sample=False,
            max_new_tokens=max(128, min(900, int(len(text) * 0.75) + 192)),
        )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()


def _emergency_backend(engine) -> EmergencyONNXTranslator:
    backend = getattr(engine, "_phoenix_emergency_onnx", None)
    if backend is None:
        backend = EmergencyONNXTranslator(engine.paths)
        engine._phoenix_emergency_onnx = backend
    return backend


def _decision_from_text(engine, source: str, text: str, target: str, backend: str):
    from .translation_models import TranslationAttempt, TranslationDecision
    from .translation_refusal_guard import looks_like_model_refusal

    value = str(text or "").strip()
    if not value or looks_like_model_refusal(value):
        return None
    report = engine.validator.validate(source, value, target)
    if not report.ok:
        return None
    attempt = TranslationAttempt(backend=backend, text=value, quality=report)
    return TranslationDecision(
        text=value,
        backend=backend,
        quality=report,
        needs_review=False,
        attempts=(attempt,),
    )


def _store_decision(engine, source: str, target: str, decision) -> None:
    try:
        if decision is None or bool(getattr(decision, "needs_review", True)):
            return
        quality = getattr(decision, "quality", None)
        if quality is None or not bool(getattr(quality, "ok", False)):
            return
        from .translation_refusal_guard import looks_like_model_refusal
        if looks_like_model_refusal(str(getattr(decision, "text", "") or "")):
            return
        _memory_for_engine(engine).store(
            source,
            decision.text,
            target,
            backend=str(getattr(decision, "backend", "unknown")),
            quality_score=float(getattr(quality, "score", 0.0)),
            verified_level=1,
        )
    except Exception as exc:
        print(
            f"[Phoenix][翻译记忆] 写入失败: {type(exc).__name__}: {exc}",
            flush=True,
        )


def _try_exact_or_rule(engine, source: str, target: str):
    try:
        hit = _memory_for_engine(engine).lookup_exact(source, target)
        if hit is not None:
            decision = _decision_from_text(
                engine,
                source,
                hit.translation,
                target,
                f"translation_memory_exact:{hit.backend}",
            )
            if decision is not None:
                print("[Phoenix][翻译记忆] 精确命中，0模型/0 API复用。", flush=True)
                return decision
    except Exception:
        pass

    text = deterministic_medical_translation(source)
    if text:
        decision = _decision_from_text(
            engine,
            source,
            text,
            target,
            "deterministic_medical_sentence",
        )
        if decision is not None:
            print("[Phoenix][零模型层] 高频医学句型直接命中。", flush=True)
            _store_decision(engine, source, target, decision)
            return decision
    return None


def _try_emergency(engine, source: str, target: str):
    backend = _emergency_backend(engine)
    if not backend.available():
        return None
    try:
        text = backend.translate(source, target)
        decision = _decision_from_text(engine, source, text, target, backend.name)
        if decision is not None:
            print("[Phoenix][零模型层] CPU应急ONNX翻译通过质量门。", flush=True)
            _store_decision(engine, source, target, decision)
        return decision
    except Exception as exc:
        print(
            f"[Phoenix][零模型层] CPU应急ONNX不可用: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _local_model_available(engine) -> bool:
    for name in ("marian", "nllb"):
        backend = getattr(engine, name, None)
        try:
            if backend is not None and backend.available():
                return True
        except Exception:
            pass
    try:
        from . import hymt_cascade_policy as hymt
        if hymt._model2_available(engine):
            return True
    except Exception:
        pass
    try:
        from . import translation_cascade_v2 as cascade
        if cascade._model3_available(engine):
            return True
    except Exception:
        pass
    return False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translation_cascade_v2 as cascade
    from .translation_models import TranslationAttempt, TranslationDecision

    old_translate = cascade._translate
    old_run_local = cascade._run_local_cascade
    old_accept = cascade._local_draft_accepted
    old_api = cascade._api_polish_local_draft

    def run_local(engine, source, target, attempts, errors):
        draft, stage = old_run_local(engine, source, target, attempts, errors)
        if draft is not None:
            return draft, stage

        # If normal local translation failed completely, a guarded similar
        # memory item may seed model3. It is never accepted directly.
        try:
            hit = _memory_for_engine(engine).lookup_similar(source, target)
            if hit is not None and cascade._model3_available(engine):
                backend = cascade._model3(engine)
                repaired = backend.refine(source, hit.translation, target)
                attempt = TranslationAttempt(
                    backend=f"{backend.name}|memory_seeded",
                    text=str(repaired or "").strip(),
                    quality=engine.validator.validate(source, repaired, target),
                )
                attempts.append(attempt)
                if attempt.quality.ok:
                    print(
                        f"[Phoenix][翻译记忆] 相似句 {hit.similarity:.0%} 仅作模型3草稿，"
                        "本地终审通过，避免API。",
                        flush=True,
                    )
                    return attempt, "memory_similar_model3"
        except Exception as exc:
            errors.append(f"memory-model3-seed: {type(exc).__name__}: {exc}")

        emergency = _try_emergency(engine, source, target)
        if emergency is not None:
            attempt = emergency.attempts[0]
            attempts.append(attempt)
            return attempt, "emergency_cpu"
        return None, stage

    def accepted(local_draft, local_stage):
        if local_stage in {"memory_similar_model3", "emergency_cpu"}:
            return bool(local_draft.quality.ok and float(local_draft.quality.score) >= 0.62)
        return old_accept(local_draft, local_stage)

    def api(engine, source, local_draft, local_stage, target, attempts, errors):
        # One last CPU-only attempt before spending API tokens.
        if local_stage != "emergency_cpu":
            emergency = _try_emergency(engine, source, target)
            if emergency is not None:
                item = emergency.attempts[0]
                attempts.append(item)
                return TranslationDecision(
                    text=item.text,
                    backend=item.backend,
                    quality=item.quality,
                    needs_review=False,
                    attempts=tuple(attempts),
                )

        result = old_api(
            engine,
            source,
            local_draft,
            local_stage,
            target,
            attempts,
            errors,
        )
        if result is not None:
            _store_decision(engine, source, target, result)
        return result

    def translate(engine, source: str, target_language: str = "中文", *, smart_level: str = "smart1"):
        source = str(source or "").strip()
        target = str(target_language or "中文").strip()

        precomputed = _try_exact_or_rule(engine, source, target)
        if precomputed is not None:
            return precomputed

        # If none of the normal local models can even start, try the CPU
        # emergency layer before entering a source-only API path.
        if not _local_model_available(engine):
            emergency = _try_emergency(engine, source, target)
            if emergency is not None:
                return emergency

        try:
            result = old_translate(
                engine,
                source,
                target,
                smart_level=smart_level,
            )
            _store_decision(engine, source, target, result)
            return result
        except RuntimeError as exc:
            # Offline survival: preserve source, queue for later, never emit a
            # refusal template as if it were a translation.
            try:
                _memory_for_engine(engine).enqueue_pending(source, target, str(exc))
            except Exception:
                pass
            quality = engine.validator.validate(source, source, target)
            attempt = TranslationAttempt(
                backend="offline_pending_source_preserved",
                text=source,
                quality=quality,
                errors=(str(exc),),
            )
            print(
                "[Phoenix][零模型层] 本地模型/API均不可用：已保留英文原文并加入待翻译队列。",
                flush=True,
            )
            return TranslationDecision(
                text=source,
                backend=attempt.backend,
                quality=quality,
                needs_review=True,
                attempts=(attempt,),
            )

    cascade._run_local_cascade = run_local
    cascade._local_draft_accepted = accepted
    cascade._api_polish_local_draft = api
    cascade._translate = translate

    print(
        "[Phoenix][零模型生存层] 已启用：精确翻译记忆→高频医学句型→"
        "相似句仅供模型3纠错→CPU应急ONNX→API；全部不可用则保留原文并排队。",
        flush=True,
    )
