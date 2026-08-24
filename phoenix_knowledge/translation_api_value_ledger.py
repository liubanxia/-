from __future__ import annotations

"""Turn every paid API translation into reusable Phoenix learning assets.

This layer NEVER makes an extra API request.  It only observes translation
results that already reached the API fallback and records what Phoenix paid for:
source, local attempts, API correction, likely error class, terminology/pattern
candidates, estimated token volume, and later reuse counts.

Nothing collected here is promoted directly into model weights or expert rules.
All records remain candidate-only until the separate maturity/review pipeline
allows them to be used.
"""

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_INSTALLED = False
_DB_NAME = "api_value.sqlite3"

_NEG_EN = re.compile(r"\b(?:no|not|without|absent|negative|cannot|can't|unlikely|exclude)\b", re.I)
_NEG_ZH = re.compile(r"(?:未见|无|不|不能|无法|否认|阴性|排除|不支持)")
_LATERALITY_EN = re.compile(r"\b(?:left|right|bilateral|unilateral)\b", re.I)
_NUMBER_UNIT = re.compile(r"[-+]?\d+(?:\.\d+)?\s*(?:%|mmHg|cmH2O|mm|cm|mL|ml|mg|kg|g|HU|Gy|mGy|mSv|Sv|°C)?", re.I)
_PATTERN_HINTS = (
    (re.compile(r"\bno evidence of\s+(.+)", re.I), "no_evidence_of_X"),
    (re.compile(r"\bcannot exclude\s+(.+)", re.I), "cannot_exclude_X"),
    (re.compile(r"\bconsistent with\s+(.+)", re.I), "consistent_with_X"),
    (re.compile(r"\bsuggestive of\s+(.+)", re.I), "suggestive_of_X"),
    (re.compile(r"\bcompatible with\s+(.+)", re.I), "compatible_with_X"),
    (re.compile(r"\bcompared with\s+(.+)", re.I), "compared_with_X"),
    (re.compile(r"\bcorrelate clinically\b", re.I), "correlate_clinically"),
    (re.compile(r"\bfollow[- ]?up is recommended\b", re.I), "follow_up_recommended"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _source_key(source: str, target: str) -> str:
    value = " ".join(str(source or "").split()).casefold()
    return _hash_text(value + "\n" + str(target or "中文").strip().casefold())


def _estimate_tokens(text: str) -> int:
    """Conservative local estimate only; never presented as provider billing."""

    value = str(text or "")
    if not value:
        return 0
    cjk = sum(1 for ch in value if "\u3400" <= ch <= "\u9fff")
    other = max(0, len(value) - cjk)
    return max(1, int(cjk / 1.35 + other / 4.0))


def _optional_price(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
        return value if value >= 0 else None
    except (TypeError, ValueError):
        return None


def _estimated_cost(input_tokens: int, output_tokens: int) -> float | None:
    """Return cost only when operator explicitly supplied current provider prices."""

    input_price = _optional_price("PHOENIX_API_INPUT_PRICE_PER_1M")
    output_price = _optional_price("PHOENIX_API_OUTPUT_PRICE_PER_1M")
    if input_price is None or output_price is None:
        return None
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


def _is_api_backend(name: str) -> bool:
    value = str(name or "").strip().lower()
    if not value:
        return False
    if value.startswith(("translation_memory_", "emergency_onnx", "deterministic_", "offline_pending_")):
        return False
    return any(token in value for token in (
        "qwen35_medical_translation",
        "remote_server",
        "deepseek",
        "openai",
        "gemini",
        "api_",
        "_api",
    ))


def _quality_reasons(attempt) -> tuple[str, ...]:
    quality = getattr(attempt, "quality", None)
    values = getattr(quality, "reasons", ()) if quality is not None else ()
    return tuple(str(x) for x in values if str(x).strip())


def _best_local_before(attempts: list, api_index: int):
    local = [
        item for item in attempts[:api_index]
        if not _is_api_backend(getattr(item, "backend", ""))
        and str(getattr(item, "text", "") or "").strip()
    ]
    if not local:
        return None
    return max(
        local,
        key=lambda item: float(getattr(getattr(item, "quality", None), "score", 0.0) or 0.0),
    )


def _classify_errors(source: str, local_text: str, api_text: str, reasons: tuple[str, ...]) -> tuple[str, ...]:
    categories: list[str] = []
    reason_text = "；".join(reasons)
    mapping = (
        ("数字", "number_unit"),
        ("单位", "number_unit"),
        ("正负号", "number_unit"),
        ("缩写", "acronym"),
        ("过短", "omission"),
        ("漏译", "omission"),
        ("英文残留", "untranslated_text"),
        ("拒答", "refusal"),
    )
    for needle, label in mapping:
        if needle in reason_text and label not in categories:
            categories.append(label)

    source_has_neg = bool(_NEG_EN.search(source))
    if source_has_neg and local_text:
        if not _NEG_ZH.search(local_text) and _NEG_ZH.search(api_text):
            categories.append("negation")

    if _LATERALITY_EN.search(source):
        categories.append("laterality_check")

    source_numbers = tuple(_NUMBER_UNIT.findall(source))
    if source_numbers and local_text and api_text and local_text != api_text:
        if "number_unit" not in categories:
            categories.append("number_unit_check")

    if not categories:
        categories.append("semantic_or_terminology")
    return tuple(dict.fromkeys(categories))


def _terminology_assets(source: str) -> tuple[dict, ...]:
    try:
        from .medical_terminology_core import find_core_terms

        rows: list[dict] = []
        for hit in find_core_terms(source, limit=32):
            senses = [
                {"english": str(s.english), "chinese": str(s.chinese), "domain": str(s.domain)}
                for s in tuple(hit.senses)[:6]
            ]
            rows.append({
                "term": str(hit.display),
                "kind": str(hit.kind),
                "ambiguous": len(senses) > 1,
                "senses": senses,
            })
        return tuple(rows)
    except Exception:
        return ()


def _pattern_assets(source: str) -> tuple[str, ...]:
    found: list[str] = []
    for regex, label in _PATTERN_HINTS:
        if regex.search(str(source or "")):
            found.append(label)
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class APIValueStats:
    calls: int
    accepted_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    reusable_assets: int
    exact_reuses: int
    similar_assists: int
    configured_estimated_cost: float | None


class APIValueLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=15)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_translation_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    local_backend TEXT NOT NULL,
                    local_text TEXT NOT NULL,
                    api_backend TEXT NOT NULL,
                    api_text TEXT NOT NULL,
                    quality_ok INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    failure_reasons_json TEXT NOT NULL,
                    error_categories_json TEXT NOT NULL,
                    terminology_json TEXT NOT NULL,
                    pattern_candidates_json TEXT NOT NULL,
                    estimated_input_tokens INTEGER NOT NULL,
                    estimated_output_tokens INTEGER NOT NULL,
                    estimated_cost REAL,
                    cost_currency TEXT NOT NULL,
                    training_status TEXT NOT NULL DEFAULT 'candidate_only',
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    expert_eligible INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_api_calls_source
                    ON api_translation_calls(source_key, target_language);

                CREATE TABLE IF NOT EXISTS api_learning_assets (
                    source_key TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    final_text TEXT NOT NULL,
                    api_backend TEXT NOT NULL,
                    error_categories_json TEXT NOT NULL,
                    terminology_json TEXT NOT NULL,
                    pattern_candidates_json TEXT NOT NULL,
                    successful_calls INTEGER NOT NULL DEFAULT 1,
                    exact_reuse_count INTEGER NOT NULL DEFAULT 0,
                    similar_assist_count INTEGER NOT NULL DEFAULT 0,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    training_status TEXT NOT NULL DEFAULT 'candidate_only',
                    expert_eligible INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(source_key, target_language)
                );
                """
            )

    def record_attempt(
        self,
        *,
        source: str,
        target: str,
        local_attempt,
        api_attempt,
    ) -> None:
        api_text = str(getattr(api_attempt, "text", "") or "").strip()
        api_backend = str(getattr(api_attempt, "backend", "") or "")
        quality = getattr(api_attempt, "quality", None)
        quality_ok = bool(getattr(quality, "ok", False))
        quality_score = float(getattr(quality, "score", 0.0) or 0.0)
        local_text = str(getattr(local_attempt, "text", "") or "").strip() if local_attempt else ""
        local_backend = str(getattr(local_attempt, "backend", "") or "") if local_attempt else ""
        reasons = _quality_reasons(local_attempt) if local_attempt is not None else ()
        categories = _classify_errors(source, local_text, api_text, reasons)
        terminology = _terminology_assets(source)
        patterns = _pattern_assets(source)

        context_for_estimate = source + "\n" + local_text + "\n" + "；".join(reasons)
        input_tokens = _estimate_tokens(context_for_estimate)
        output_tokens = _estimate_tokens(api_text)
        cost = _estimated_cost(input_tokens, output_tokens)
        currency = os.environ.get("PHOENIX_API_COST_CURRENCY", "").strip()
        now = _utc_now()
        key = _source_key(source, target)

        with self._connect() as db:
            db.execute(
                """
                INSERT INTO api_translation_calls(
                    created_at, source_key, target_language, source_text,
                    local_backend, local_text, api_backend, api_text,
                    quality_ok, quality_score, failure_reasons_json,
                    error_categories_json, terminology_json, pattern_candidates_json,
                    estimated_input_tokens, estimated_output_tokens, estimated_cost,
                    cost_currency, training_status, reviewed, expert_eligible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'candidate_only', 0, 0)
                """,
                (
                    now, key, target, source, local_backend, local_text,
                    api_backend, api_text, int(quality_ok), quality_score,
                    json.dumps(reasons, ensure_ascii=False),
                    json.dumps(categories, ensure_ascii=False),
                    json.dumps(terminology, ensure_ascii=False),
                    json.dumps(patterns, ensure_ascii=False),
                    input_tokens, output_tokens, cost, currency,
                ),
            )

            # Only successful, quality-gated API output becomes a reusable asset.
            if quality_ok and api_text:
                db.execute(
                    """
                    INSERT INTO api_learning_assets(
                        source_key, target_language, source_text, final_text,
                        api_backend, error_categories_json, terminology_json,
                        pattern_candidates_json, successful_calls,
                        exact_reuse_count, similar_assist_count, first_seen_at,
                        last_seen_at, reviewed, training_status, expert_eligible
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 0, 0, ?, ?, 0,
                              'candidate_only', 0)
                    ON CONFLICT(source_key, target_language) DO UPDATE SET
                        final_text=excluded.final_text,
                        api_backend=excluded.api_backend,
                        error_categories_json=excluded.error_categories_json,
                        terminology_json=excluded.terminology_json,
                        pattern_candidates_json=excluded.pattern_candidates_json,
                        successful_calls=api_learning_assets.successful_calls+1,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (
                        key, target, source, api_text, api_backend,
                        json.dumps(categories, ensure_ascii=False),
                        json.dumps(terminology, ensure_ascii=False),
                        json.dumps(patterns, ensure_ascii=False),
                        now, now,
                    ),
                )

    def mark_reuse(self, source: str, target: str, *, similar: bool = False) -> None:
        key = _source_key(source, target)
        column = "similar_assist_count" if similar else "exact_reuse_count"
        with self._connect() as db:
            db.execute(
                f"UPDATE api_learning_assets SET {column}={column}+1, last_seen_at=? "
                "WHERE source_key=? AND target_language=?",
                (_utc_now(), key, target),
            )

    def stats(self) -> APIValueStats:
        with self._connect() as db:
            row = db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(quality_ok),0),
                       COALESCE(SUM(estimated_input_tokens),0),
                       COALESCE(SUM(estimated_output_tokens),0),
                       SUM(estimated_cost)
                FROM api_translation_calls
                """
            ).fetchone() or (0, 0, 0, 0, None)
            assets = db.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(exact_reuse_count),0),
                       COALESCE(SUM(similar_assist_count),0)
                FROM api_learning_assets
                """
            ).fetchone() or (0, 0, 0)
        return APIValueStats(
            calls=int(row[0]),
            accepted_calls=int(row[1]),
            estimated_input_tokens=int(row[2]),
            estimated_output_tokens=int(row[3]),
            reusable_assets=int(assets[0]),
            exact_reuses=int(assets[1]),
            similar_assists=int(assets[2]),
            configured_estimated_cost=(float(row[4]) if row[4] is not None else None),
        )


def _ledger_for_engine(engine) -> APIValueLedger:
    ledger = getattr(engine, "_phoenix_api_value_ledger", None)
    if ledger is not None:
        return ledger
    from .translation_ssd_storage import translation_storage_root

    root = translation_storage_root(engine.paths) / "人工修订与学习"
    ledger = APIValueLedger(root / _DB_NAME)
    engine._phoenix_api_value_ledger = ledger
    return ledger


def _record_decision(engine, source: str, target: str, decision) -> None:
    attempts = list(getattr(decision, "attempts", ()) or ())
    if not attempts:
        return
    ledger = _ledger_for_engine(engine)
    for index, attempt in enumerate(attempts):
        if not _is_api_backend(getattr(attempt, "backend", "")):
            continue
        local = _best_local_before(attempts, index)
        try:
            ledger.record_attempt(
                source=source,
                target=target,
                local_attempt=local,
                api_attempt=attempt,
            )
        except Exception as exc:
            print(
                f"[Phoenix][API价值账本] 记录失败但不影响翻译: {type(exc).__name__}: {exc}",
                flush=True,
            )


def _report(engine) -> None:
    try:
        stats = _ledger_for_engine(engine).stats()
    except Exception:
        return
    state = (
        stats.calls, stats.accepted_calls, stats.reusable_assets,
        stats.exact_reuses, stats.similar_assists,
    )
    if getattr(engine, "_phoenix_api_value_report", None) == state:
        return
    engine._phoenix_api_value_report = state
    money = ""
    if stats.configured_estimated_cost is not None:
        currency = os.environ.get("PHOENIX_API_COST_CURRENCY", "").strip()
        money = f" | 配置价格估算={stats.configured_estimated_cost:.4f}{currency}"
    print(
        "[Phoenix][API价值账本] "
        f"API尝试={stats.calls} | 质量通过={stats.accepted_calls} | "
        f"可复用资产={stats.reusable_assets} | 精确复用={stats.exact_reuses} | "
        f"相似辅助={stats.similar_assists}{money}；"
        "API经验仅为候选，不自动晋级专家。",
        flush=True,
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from . import translation_cascade_v2 as cascade
    from . import translation_survival_memory as survival

    old_translate = cascade._translate

    def translate(engine, source: str, target_language: str = "中文", *, smart_level: str = "smart1"):
        result = old_translate(
            engine,
            source,
            target_language,
            smart_level=smart_level,
        )
        try:
            _record_decision(engine, str(source or ""), str(target_language or "中文"), result)
            _report(engine)
        except Exception as exc:
            print(
                f"[Phoenix][API价值账本] 旁路分析失败但不影响译文: {type(exc).__name__}: {exc}",
                flush=True,
            )
        return result

    cascade._translate = translate

    # After the separate 10-book maturity gate allows translation memory to
    # participate, count how often API-paid knowledge saves future work.
    memory_cls = survival.TranslationMemory
    old_exact = memory_cls.lookup_exact
    old_similar = memory_cls.lookup_similar

    def lookup_exact(self, source: str, target_language: str):
        hit = old_exact(self, source, target_language)
        if hit is not None and _is_api_backend(getattr(hit, "backend", "")):
            try:
                # API ledger lives beside the memory database on the same SSD.
                ledger_path = self.path.parent.parent / "人工修订与学习" / _DB_NAME
                APIValueLedger(ledger_path).mark_reuse(source, target_language, similar=False)
            except Exception:
                pass
        return hit

    def lookup_similar(self, source: str, target_language: str, **kwargs):
        hit = old_similar(self, source, target_language, **kwargs)
        if hit is not None and _is_api_backend(getattr(hit, "backend", "")):
            try:
                ledger_path = self.path.parent.parent / "人工修订与学习" / _DB_NAME
                APIValueLedger(ledger_path).mark_reuse(source, target_language, similar=True)
            except Exception:
                pass
        return hit

    memory_cls.lookup_exact = lookup_exact
    memory_cls.lookup_similar = lookup_similar

    print(
        "[Phoenix][API价值账本] 已启用：不增加任何API调用；每次已发生的API纠错都会沉淀"
        "错误分类、术语/句型候选、模型1/2/3对照、token估算和后续复用收益。"
        "所有资产默认candidate_only，不能自动训练或晋级专家。",
        flush=True,
    )
