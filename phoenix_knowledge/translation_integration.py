from __future__ import annotations

"""Translation pipeline integration helpers.

This module turns document- and chunk-level translation requests into one
normalized runtime contract before a backend is invoked.
"""

from dataclasses import dataclass
from pathlib import Path

from .medical_translation_policy import preferred_translation_level
from .translation_runtime_policy import (
    build_chunk_cache_key,
    make_runtime_decision,
)


@dataclass(frozen=True)
class TranslationRuntime:
    page_number: int
    smart_level: str
    use_cache: bool
    chunk_cache_key: str


def build_translation_runtime(
    source_text: str,
    page_number: int,
    requested_level: str | None = "smart1",
    target_language: str = "中文",
) -> TranslationRuntime:
    """Build the per-chunk runtime contract used by the adapter.

    The caller's already-resolved Smart1/Smart2 choice remains authoritative
    here. Document classification belongs to resolve_translation_request;
    chunk dispatch must not silently promote Smart1 into Smart2.
    """

    text = str(source_text or "")
    language = str(target_language or "中文").strip() or "中文"
    page = max(1, int(page_number))
    decision = make_runtime_decision(
        text,
        language,
        is_medical=False,
        requested_level=requested_level,
    )
    return TranslationRuntime(
        page_number=page,
        smart_level=decision.smart_level,
        use_cache=decision.use_cache,
        chunk_cache_key=decision.chunk_cache_key,
    )


def resolve_translation_request(
    document_path: str | Path,
    requested_level: str | None = None,
    target_language: str = "中文",
) -> dict:
    path = Path(document_path)
    level = preferred_translation_level(path.name, requested_level)
    runtime = build_translation_runtime(
        source_text=str(path),
        page_number=1,
        requested_level=level,
        target_language=target_language,
    )
    return {
        "document": str(path),
        "smart_level": runtime.smart_level,
        "cache_key": build_chunk_cache_key(
            str(path),
            target_language,
            runtime.smart_level,
        ),
    }
