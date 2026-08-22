from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from .translation_integration import TranslationRuntime, build_translation_runtime


@dataclass
class TranslationRuntimeAdapter:
    """Route translation chunks through one normalized runtime contract."""

    cache_limit: int = 256
    _cache: OrderedDict[tuple[int, str], Any] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.cache_limit = max(0, int(self.cache_limit))

    def prepare(
        self,
        source_text: str,
        page_number: int,
        requested_level: str = "smart1",
        target_language: str = "中文",
    ) -> TranslationRuntime:
        return build_translation_runtime(
            source_text=source_text,
            page_number=page_number,
            requested_level=requested_level,
            target_language=target_language,
        )

    def translate(
        self,
        engine,
        source_text: str,
        target_language: str = "中文",
        *,
        page_number: int,
        requested_level: str = "smart1",
    ):
        runtime = self.prepare(
            source_text=source_text,
            page_number=page_number,
            requested_level=requested_level,
            target_language=target_language,
        )
        cache_key = (id(engine), runtime.chunk_cache_key)
        if runtime.use_cache and cache_key in self._cache:
            decision = self._cache.pop(cache_key)
            self._cache[cache_key] = decision
            return decision

        decision = engine.translate(
            source_text,
            target_language,
            smart_level=runtime.smart_level,
        )
        if (
            runtime.use_cache
            and self.cache_limit > 0
            and not bool(getattr(decision, "needs_review", False))
        ):
            self._cache[cache_key] = decision
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_limit:
                self._cache.popitem(last=False)
        return decision

    def clear(self) -> None:
        self._cache.clear()


def prepare_translation_runtime(
    source_text: str,
    page_number: int,
    requested_level: str = "smart1",
    target_language: str = "中文",
) -> TranslationRuntime:
    return TranslationRuntimeAdapter(cache_limit=0).prepare(
        source_text=source_text,
        page_number=page_number,
        requested_level=requested_level,
        target_language=target_language,
    )
