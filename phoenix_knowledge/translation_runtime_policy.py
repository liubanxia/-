from __future__ import annotations

"""Translation runtime controls.

Central policy layer for reducing unnecessary LLM calls during PDF translation.
The translator can import this module before dispatching chunks.
"""

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class TranslationRuntimeDecision:
    smart_level: str
    use_cache: bool
    chunk_cache_key: str


def normalize_task_level(is_medical: bool, requested: str | None = None) -> str:
    if is_medical:
        return "smart2"
    return "smart2" if (requested or "").lower() in {"smart2", "quality", "deep"} else "smart1"


def build_chunk_cache_key(
    source_text: str,
    target_language: str,
    smart_level: str,
) -> str:
    payload = "|".join([
        target_language,
        smart_level,
        source_text.strip(),
    ])
    return sha256(payload.encode("utf-8")).hexdigest()


def make_runtime_decision(
    source_text: str,
    target_language: str,
    *,
    is_medical: bool = False,
    requested_level: str | None = None,
) -> TranslationRuntimeDecision:
    level = normalize_task_level(is_medical, requested_level)
    return TranslationRuntimeDecision(
        smart_level=level,
        use_cache=True,
        chunk_cache_key=build_chunk_cache_key(
            source_text,
            target_language,
            level,
        ),
    )
