from __future__ import annotations

"""Reduce redundant remote translation calls without weakening publication safety.

This runtime is deliberately conservative:
- exact remote responses are cached only for the current process;
- refusal-like responses are never cached;
- duplicate source strings inside one batch are translated once;
- PDF/Office remote batches are moderately enlarged to reduce prompt overhead;
- remote source/batch prompts keep the same medical invariants in a compact form;
- a second Smart2 correction is kept for structural/safety failures, while a
  purely non-safety repeat reuses the first correction instead of paying for an
  equivalent second request.

Persistent production-memory maturity rules remain untouched.
"""

from collections import OrderedDict
import hashlib
import json
import os
import re
from threading import RLock

_INSTALLED = False
_CACHE_LIMIT = 512
_LOCK = RLock()

_HARD_RETRY_MARKERS = (
    "数字",
    "单位",
    "正负号",
    "侧别",
    "否定",
    "缩写",
    "未翻译",
    "漏译",
    "英文残留",
    "过短",
    "过长",
    "拒答",
)
_PURE_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9+./_-]{1,15}$")


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _remote_translation_enabled() -> bool:
    if not _flag("PHOENIX_TRANSLATION_ALLOW_API_FALLBACK", False):
        return False
    if not _flag("PHOENIX_KNOWLEDGE_ALLOW_REMOTE", False):
        return False
    mode = os.environ.get("PHOENIX_KNOWLEDGE_ACCELERATOR", "").strip().lower()
    return mode in {"remote", "api", "external", "cloud"}


def _remote_backend(backend) -> bool:
    if not _remote_translation_enabled():
        return False
    llm = getattr(backend, "llm", None)
    if llm is None:
        return False
    try:
        return str(llm.backend("translation") or "").strip().lower() == "remote_server"
    except Exception:
        try:
            compute = getattr(llm, "compute", None)
            requested = getattr(compute, "requested_mode", None)
            return callable(requested) and str(requested() or "").strip().lower() == "remote"
        except Exception:
            return False


def _provider_signature(backend) -> str:
    llm = getattr(backend, "llm", None)
    compute = getattr(llm, "compute", None)
    provider = "remote"
    model = ""
    try:
        label = getattr(compute, "provider_id", None)
        if callable(label):
            provider = str(label() or provider)
    except Exception:
        pass
    try:
        remote_model = getattr(compute, "remote_model", None)
        if callable(remote_model):
            model = str(remote_model("translation") or "")
    except Exception:
        pass
    return f"{provider}|{model}"


def _glossary_signature(backend, source: str) -> str:
    try:
        text = str(backend.glossary_prompt(source) or "")
    except Exception:
        text = ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _key(*parts: str) -> str:
    payload = "\n\x1f\n".join(str(part or "").strip() for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache(backend) -> OrderedDict[str, str]:
    value = getattr(backend, "_phoenix_api_efficiency_cache", None)
    if not isinstance(value, OrderedDict):
        value = OrderedDict()
        backend._phoenix_api_efficiency_cache = value
    return value


def _stats(backend) -> dict[str, int]:
    value = getattr(backend, "_phoenix_api_efficiency_stats", None)
    if not isinstance(value, dict):
        value = {
            "remote_calls": 0,
            "cache_hits": 0,
            "deduplicated_segments": 0,
            "retry_calls_avoided": 0,
        }
        backend._phoenix_api_efficiency_stats = value
    return value


def api_efficiency_stats(backend) -> dict[str, int]:
    return dict(_stats(backend))


def _cache_get(backend, key: str) -> str | None:
    with _LOCK:
        store = _cache(backend)
        if key not in store:
            return None
        value = store.pop(key)
        store[key] = value
        _stats(backend)["cache_hits"] += 1
        return value


def _cache_put(backend, key: str, value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    try:
        from .translation_refusal_guard import looks_like_model_refusal

        if looks_like_model_refusal(text):
            return
    except Exception:
        pass
    with _LOCK:
        store = _cache(backend)
        store[key] = text
        store.move_to_end(key)
        while len(store) > _CACHE_LIMIT:
            store.popitem(last=False)


def _retry_is_safety_critical(reasons) -> bool:
    text = "；".join(str(item or "") for item in (reasons or ()))
    return any(marker in text for marker in _HARD_RETRY_MARKERS)


def _safe_batch_for_cache(values: dict[str, str]) -> bool:
    if not values:
        return False
    try:
        from .translation_refusal_guard import looks_like_model_refusal

        return all(
            str(value or "").strip() and not looks_like_model_refusal(value)
            for value in values.values()
        )
    except Exception:
        return all(str(value or "").strip() for value in values.values())


def _install_backend_cache() -> None:
    from .translation_models import (
        QwenMedicalTranslationBackend,
        translation_output_budget,
    )

    old_translate = QwenMedicalTranslationBackend.translate
    old_translate_segments = QwenMedicalTranslationBackend.translate_segments
    old_retry = QwenMedicalTranslationBackend.retry_translation

    def translate(
        self,
        text: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ) -> str:
        if not _remote_backend(self):
            return old_translate(
                self,
                text,
                target_language,
                smart_level=smart_level,
            )
        source = str(text or "").strip()
        cache_key = _key(
            "translate",
            _provider_signature(self),
            target_language,
            smart_level,
            _glossary_signature(self, source),
            source,
        )
        cached = _cache_get(self, cache_key)
        if cached is not None:
            return cached

        glossary = str(self.glossary_prompt(source) or "").strip()
        glossary_section = f"\n固定缩写表：\n{glossary}\n" if glossary else ""
        prompt = (
            f"医学教材精译：把英文逐句完整译成{target_language}。使用规范医学术语；"
            "严格保留数字、单位、正负号、侧别、否定、分级、诊断确定性、医学缩写、"
            "图表/参考文献编号；作者、期刊、DOI、URL不改写。缩写按固定表，独立缩写写成"
            "“规范中文（缩写）”。OCR/原文确实损坏才标[原文不清]。禁止总结、删减、扩写、"
            "解释或添加原文没有的信息，只输出译文。"
            f"{glossary_section}\n原文：\n{source}"
        )
        result = self.llm.generate(
            prompt,
            max_new_tokens=translation_output_budget(source, smart_level),
            profile="translation",
        ).strip()
        _stats(self)["remote_calls"] += 1
        _cache_put(self, cache_key, result)
        return result

    def translate_segments(
        self,
        sources: list[str],
        target_language: str = "中文",
    ) -> dict[str, str]:
        if not _remote_backend(self):
            return old_translate_segments(self, sources, target_language)

        values = [str(source or "").strip() for source in sources]
        joined = "\n".join(values)
        cache_key = _key(
            "batch",
            _provider_signature(self),
            target_language,
            _glossary_signature(self, joined),
            json.dumps(values, ensure_ascii=False, separators=(",", ":")),
        )
        cached = _cache_get(self, cache_key)
        if cached is not None:
            try:
                payload = json.loads(cached)
                if isinstance(payload, dict):
                    return {str(k): str(v) for k, v in payload.items()}
            except Exception:
                pass

        rows = [
            {"id": f"S{index:04d}", "text": source}
            for index, source in enumerate(values, start=1)
        ]
        source_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        glossary = str(self.glossary_prompt(joined) or "").strip()
        glossary_section = f"\n固定缩写表：\n{glossary}\n" if glossary else ""
        prompt = (
            f"医学文献批量精译：逐项把JSON里的text完整译成{target_language}；id、数量、顺序不变。"
            "使用规范医学术语；严格保留数字、单位、正负号、侧别、否定、诊断确定性、医学缩写、"
            "图表/参考文献编号；禁止总结、删减、扩写或合并条目。相邻项只可用于消歧。"
            "只输出JSON数组，每项仅含id和translation。"
            f"{glossary_section}\n输入JSON：\n{source_json}"
        )
        max_tokens = max(
            640,
            min(
                3200,
                translation_output_budget(joined, "smart2") + 36 * len(rows),
            ),
        )
        raw = self.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
            profile="translation",
        ).strip()
        _stats(self)["remote_calls"] += 1
        try:
            result = self._parse_segment_json(raw)
        except Exception:
            repair_prompt = (
                "仅把下列输出修复成合法JSON数组；不得重新翻译、总结或添加内容；"
                "保留全部id，每项只能含id和translation。\n"
                f"输入id：{source_json}\n待修复：{raw}\n只输出JSON数组。"
            )
            repaired = self.llm.generate(
                repair_prompt,
                max_new_tokens=max_tokens,
                profile="translation",
            ).strip()
            _stats(self)["remote_calls"] += 1
            result = self._parse_segment_json(repaired)

        expected = {f"S{index:04d}" for index in range(1, len(rows) + 1)}
        if set(result) == expected and _safe_batch_for_cache(result):
            _cache_put(
                self,
                cache_key,
                json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            )
        return result

    def retry_translation(
        self,
        source: str,
        draft: str,
        reasons: tuple[str, ...],
        target_language: str = "中文",
    ) -> str:
        if not _remote_backend(self):
            return old_retry(self, source, draft, reasons, target_language)

        source_text = str(source or "").strip()
        retry_key = _key(
            "retry",
            _provider_signature(self),
            target_language,
            _glossary_signature(self, source_text),
            source_text,
            str(draft or "").strip(),
            json.dumps([str(x) for x in (reasons or ())], ensure_ascii=False),
        )
        cached = _cache_get(self, retry_key)
        if cached is not None:
            return cached

        source_key = _key(
            "retry-source",
            _provider_signature(self),
            target_language,
            _glossary_signature(self, source_text),
            source_text,
        )
        attempts = getattr(self, "_phoenix_api_retry_attempts", None)
        if not isinstance(attempts, dict):
            attempts = {}
            self._phoenix_api_retry_attempts = attempts
        previous = attempts.get(source_key)
        count = int(previous.get("count", 0)) if isinstance(previous, dict) else 0
        previous_text = str(previous.get("text", "")) if isinstance(previous, dict) else ""

        # The normal cascade already allows at most two corrections. Keep the
        # second call for medical/structural safety failures, but avoid paying
        # for a second generic pass that has no new safety signal.
        if count >= 1 and previous_text and not _retry_is_safety_critical(reasons):
            _stats(self)["retry_calls_avoided"] += 1
            return previous_text
        if count >= 2 and previous_text:
            _stats(self)["retry_calls_avoided"] += 1
            return previous_text

        result = old_retry(self, source_text, draft, reasons, target_language)
        _stats(self)["remote_calls"] += 1
        attempts[source_key] = {"count": count + 1, "text": str(result or "").strip()}
        _cache_put(self, retry_key, result)
        return result

    translate._phoenix_api_efficiency_v2 = True
    translate_segments._phoenix_api_efficiency_v2 = True
    retry_translation._phoenix_api_efficiency_v2 = True
    QwenMedicalTranslationBackend.translate = translate
    QwenMedicalTranslationBackend.translate_segments = translate_segments
    QwenMedicalTranslationBackend.retry_translation = retry_translation


def _dedupe_key(source: str) -> str | None:
    value = " ".join(str(source or "").split())
    # Do not merge context-sensitive short labels/acronyms.
    if len(value) < 24 or _PURE_ACRONYM_RE.fullmatch(value):
        return None
    return value


def _install_segment_deduplication() -> None:
    from .translation_models import MultiModelTranslationEngine, _normalize_smart_level

    old_segments = MultiModelTranslationEngine.translate_segments

    def translate_segments(
        self,
        sources: list[str],
        target_language: str = "中文",
        *,
        smart_level: str = "smart2",
    ):
        values = [str(value or "").strip() for value in sources]
        if (
            not values
            or _normalize_smart_level(smart_level) != "smart2"
            or not _remote_translation_enabled()
        ):
            return old_segments(
                self,
                values,
                target_language,
                smart_level=smart_level,
            )

        unique: list[str] = []
        positions: list[int] = []
        seen: dict[str, int] = {}
        duplicates = 0
        for value in values:
            key = _dedupe_key(value)
            if key is None or key not in seen:
                index = len(unique)
                unique.append(value)
                positions.append(index)
                if key is not None:
                    seen[key] = index
            else:
                positions.append(seen[key])
                duplicates += 1

        if duplicates == 0:
            return old_segments(
                self,
                values,
                target_language,
                smart_level=smart_level,
            )
        decisions = tuple(
            old_segments(
                self,
                unique,
                target_language,
                smart_level=smart_level,
            )
        )
        if len(decisions) != len(unique):
            # Fail safe: never guess output alignment.
            return old_segments(
                self,
                values,
                target_language,
                smart_level=smart_level,
            )
        try:
            _stats(self.qwen)["deduplicated_segments"] += duplicates
        except Exception:
            pass
        return tuple(decisions[index] for index in positions)

    translate_segments._phoenix_api_efficiency_v2 = True
    MultiModelTranslationEngine.translate_segments = translate_segments


def _install_remote_batch_sizes() -> None:
    from . import office_translation as office
    from . import translator

    old_office_batches = office._segment_batches
    old_pdf_chunk_chars = translator._translation_chunk_chars

    def segment_batches(
        segments,
        *,
        max_chars: int = 2600,
        max_segments: int = 24,
    ):
        explicit_chars = os.environ.get("PHOENIX_OFFICE_API_BATCH_CHARS", "").strip()
        explicit_segments = os.environ.get("PHOENIX_OFFICE_API_BATCH_SEGMENTS", "").strip()
        if _remote_translation_enabled() and max_chars == 2600 and max_segments == 24:
            try:
                max_chars = int(explicit_chars) if explicit_chars else 5200
            except (TypeError, ValueError):
                max_chars = 5200
            try:
                max_segments = int(explicit_segments) if explicit_segments else 40
            except (TypeError, ValueError):
                max_segments = 40
            max_chars = max(2600, min(7000, max_chars))
            max_segments = max(24, min(64, max_segments))
        return old_office_batches(
            segments,
            max_chars=max_chars,
            max_segments=max_segments,
        )

    def pdf_chunk_chars() -> int:
        if os.environ.get("PHOENIX_TRANSLATION_CHUNK_CHARS", "").strip():
            return old_pdf_chunk_chars()
        if _remote_translation_enabled():
            # ~6.8k characters remains comfortably below modern API context
            # limits while reducing prompt repetition on dense textbook pages.
            return 6800
        return old_pdf_chunk_chars()

    office._segment_batches = segment_batches
    translator._translation_chunk_chars = pdf_chunk_chars


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_backend_cache()
    _install_segment_deduplication()
    _install_remote_batch_sizes()
    _INSTALLED = True
    print(
        "[Phoenix][API节流] v2已启用：会话精确缓存、重复段去重、远程批次放大、"
        "精简等价医学提示、非安全型二次纠错节流；医学质量门与10本/1000条成熟度门保持不变。",
        flush=True,
    )
