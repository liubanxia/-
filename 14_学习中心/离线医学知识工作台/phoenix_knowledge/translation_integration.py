from __future__ import annotations

"""Translation pipeline integration helpers.

This module provides a single routing entry for callers that need to choose
translation quality before invoking the translation engine.
"""

from pathlib import Path

from .medical_translation_policy import preferred_translation_level
from .translation_runtime_policy import build_translation_cache_key


def resolve_translation_request(
    document_path: str | Path,
    requested_level: str | None = None,
) -> dict:
    path = Path(document_path)
    level = preferred_translation_level(path.name, requested_level)
    return {
        "document": str(path),
        "smart_level": level,
        "cache_key": build_translation_cache_key(
            str(path),
            level,
        ),
    }
