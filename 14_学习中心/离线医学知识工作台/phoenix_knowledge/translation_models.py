from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import WorkbenchPaths, resolve_model_dir
from .llm import LocalLLM


_MEASUREMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<sign>[+\-\u2212]?)\s*"
    r"(?P<number>\d+(?:[.,]\d+)*)"
    r"(?:\s*(?P<unit>%|mmHg|cmH2O|mm|cm|mL|ml|mg|kg|g|HU|kV|mAs|mGy|Gy|mSv|Sv|°C))?",
    re.I,
)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,10}\b")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_TRANSLATABLE_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'-]{3,}\b")
_SIMPLIFIED_TARGETS = {"中文", "简体中文", "Chinese", "zh", "zh-CN"}
_CHINESE_VALIDATION_TARGETS = _SIMPLIFIED_TARGETS | {
    "繁体中文",
    "Traditional Chinese",
    "zh-TW",
    "zh-HK",
}

FORMAL_TRANSLATION_CONTRACT_VERSION = 1
LEGACY_PREVIEW_BACKEND_NAMES = frozenset({
    "marian_en_zh",
    "nllb_600m_en_zh",
})


def _cuda_is_usable() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major) >= 5
    except Exception:
        return False


def _numeric_tokens(text: str) -> list[str]:
    result: list[str] = []
    for match in _MEASUREMENT_RE.finditer(text or ""):
        sign = (match.group("sign") or "").replace("\u2212", "-")
        number = (match.group("number") or "").replace(",", "")
        unit = (match.group("unit") or "").lower()
        result.append(f"{sign}{number}{unit}")
    return result


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _normalize_smart_level(level: str | None) -> str:
    raw = (level or "smart1").strip().lower()
    if raw in {"smart2", "2", "deep", "quality", "max"}:
        return "smart2"
    return "smart1"


def translation_output_budget(
    text: str,
    smart_level: str = "smart2",
) -> int:
    """Bound output tokens to the source size instead of always reserving 2600."""

    chars = len(str(text or "").strip())
    ceiling = (
        2600
        if _normalize_smart_level(smart_level) == "smart2"
        else 1800
    )
    return max(512, min(ceiling, int(chars * 0.72) + 384))


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationAttempt:
    backend: str
    text: str
    quality: QualityReport
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationDecision:
    text: str
    backend: str
    quality: QualityReport
    needs_review: bool
    attempts: tuple[TranslationAttempt, ...] = field(default_factory=tuple)


class TranslationValidator:
    """Structural guardrail for medical translation.

    Numeric values, signs and measurement units are treated as safety-critical:
    a mismatch prevents automatic acceptance even if the surrounding prose is
    fluent. This protects values such as ``-20 HU`` and ``12 mm`` from silently
    becoming ``20 HU`` or ``12 cm``.
    """

    def validate(self, source: str, translated: str, target_language: str = "中文") -> QualityReport:
        source = (source or "").strip()
        translated = (translated or "").strip()
        reasons: list[str] = []
        score = 1.0
        hard_numeric_failure = False

        if not translated:
            return QualityReport(False, 0.0, ("空译文",))

        if len(source) >= 80:
            ratio = len(translated) / max(len(source), 1)
            if ratio < 0.18:
                reasons.append("译文明显过短，疑似漏译")
                score -= 0.45
            elif ratio > 4.5:
                reasons.append("译文明显过长，疑似扩写或重复")
                score -= 0.3

        source_tokens = Counter(_numeric_tokens(source))
        if source_tokens:
            translated_tokens = Counter(_numeric_tokens(translated))
            kept = sum((source_tokens & translated_tokens).values())
            total = sum(source_tokens.values())
            coverage = kept / max(total, 1)
            if coverage < 1.0:
                hard_numeric_failure = True
                reasons.append(
                    f"数字/单位/正负号未完整保留({coverage:.0%})"
                )
                score = min(score - 0.35, 0.55)

        acronyms = list(dict.fromkeys(_ACRONYM_RE.findall(source)))
        if acronyms:
            kept = sum(1 for token in acronyms if token in translated)
            coverage = kept / len(acronyms)
            if coverage < 0.65:
                reasons.append(f"医学缩写保留率偏低({coverage:.0%})")
                score -= 0.2

        if target_language in _CHINESE_VALIDATION_TARGETS:
            cjk_count = len(_CJK_RE.findall(translated))
            latin_count = len(_LATIN_RE.findall(translated))
            source_latin = len(_LATIN_RE.findall(source))
            translatable_words = [
                word
                for word in _TRANSLATABLE_WORD_RE.findall(source)
                if not word.isupper()
            ]
            if translatable_words and cjk_count == 0:
                reasons.append("短文本疑似未翻译")
                score -= 0.55
            if source_latin >= 30 and cjk_count < 8:
                reasons.append("中文字符过少，疑似未翻译")
                score -= 0.55
            if source_latin >= 80 and latin_count > cjk_count * 7 and cjk_count < 30:
                reasons.append("英文残留比例过高")
                score -= 0.25

        if len(translated) >= 160:
            tail = translated[-80:]
            if tail and translated.count(tail) >= 3:
                reasons.append("检测到重复输出")
                score -= 0.25

        score = max(0.0, min(1.0, score))
        return QualityReport(
            (not hard_numeric_failure) and score >= 0.62,
            score,
            tuple(reasons),
        )


class _Seq2SeqBackend:
    name = "seq2seq"
    folder = ""

    def __init__(self, paths: WorkbenchPaths):
        self.paths = paths
        self.model_path = resolve_model_dir(paths.model_root, self.folder)
        self._tokenizer = None
        self._model = None
        self._device = None

    def available(self) -> bool:
        return self.model_path.exists() and any(self.model_path.iterdir())

    def _device_name(self) -> str:
        return "cuda:0" if _cuda_is_usable() else "cpu"

    def unload(self) -> None:
        self._tokenizer = None
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


class MarianEnZhBackend(_Seq2SeqBackend):
    name = "marian_en_zh"
    folder = "opus-mt-en-zh"

    def _load(self):
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"Marian翻译模型未下载: {self.model_path}")
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._device = self._device_name()
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), local_files_only=True).to(self._device)
        self._model.eval()
        if self._device == "cpu":
            self._model.to(dtype=torch.float32)

    def translate(self, text: str) -> str:
        import torch
        self._load()
        prefix = ""
        try:
            vocab = self._tokenizer.get_vocab()
            if ">>cmn_Hans<<" in vocab:
                prefix = ">>cmn_Hans<< "
            elif ">>zho_Hans<<" in vocab:
                prefix = ">>zho_Hans<< "
        except Exception:
            pass
        inputs = self._tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=480)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=700,
                renormalize_logits=True,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()


class NLLBEnZhBackend(_Seq2SeqBackend):
    name = "nllb_600m_en_zh"
    folder = "NLLB-200-distilled-600M"

    def _load(self):
        if self._model is not None:
            return
        if not self.available():
            raise RuntimeError(f"NLLB翻译模型未下载: {self.model_path}")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._device = self._device_name()
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, src_lang="eng_Latn"
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), local_files_only=True).to(self._device)
        self._model.eval()

    def translate(self, text: str) -> str:
        import torch
        self._load()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=900)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        forced_bos = self._tokenizer.convert_tokens_to_ids("zho_Hans")
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=4,
                max_new_tokens=900,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True).strip()


class QwenMedicalTranslationBackend:
    name = "qwen35_medical_translation"

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def available(self, smart_level: str = "smart1") -> bool:
        profile = (
            "translation"
            if _normalize_smart_level(smart_level) == "smart2"
            else "fast"
        )
        return self.llm.available(profile)

    def translate(
        self,
        text: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ) -> str:
        level = _normalize_smart_level(smart_level)
        profile = "translation" if level == "smart2" else "fast"
        max_tokens = translation_output_budget(text, level)
        prompt = f"""你是 Phoenix 医学教材精译器。把下面英文医学原文完整、准确地翻译成{target_language}。

这是医学教材正文，不是摘要任务。必须做到：
- 逐段完整翻译，不总结、不删减、不扩写，不加入原文没有的医学知识。
- 先准确理解句义，再使用自然、规范的中文医学教材语序，禁止生硬逐词直译。
- 疾病、解剖、影像学征象、检查技术、药物、分级和病理术语使用规范医学中文。
- 首次出现且中文可能歧义的专业名词，可采用“规范中文（English）”；之后使用规范中文。
- 所有数字、单位、百分比、HU、分级、剂量、图号、表号、公式、参考文献编号必须保留。
- CT/MRI/X线/超声/心电等专业缩写按医学惯例保留，不擅自改写。
- 标题、项目符号、编号、表格式行尽量维持原来的层级和顺序。
- 句子损坏、OCR错误或语义无法确定时标记“[原文不清]”，不得猜测。
- 译文应达到医生直接阅读教材的中文可读性，不输出解释、评语、翻译过程或模型信息。

原文：
{text}
"""
        return self.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
            profile=profile,
        ).strip()

    @staticmethod
    def _parse_segment_json(raw: str) -> dict[str, str]:
        text = str(raw or "").strip()
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            raise ValueError("批量译文不是JSON数组")
        payload = json.loads(text[start:end + 1])
        if not isinstance(payload, list):
            raise ValueError("批量译文JSON顶层必须为数组")
        result: dict[str, str] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            segment_id = str(row.get("id", "")).strip()
            translated = row.get("translation")
            if not segment_id or not isinstance(translated, str):
                continue
            result[segment_id] = translated.strip()
        return result

    def translate_segments(
        self,
        sources: list[str],
        target_language: str = "中文",
    ) -> dict[str, str]:
        rows = [
            {"id": f"S{index:04d}", "text": str(source)}
            for index, source in enumerate(sources, start=1)
        ]
        joined = "\n".join(str(source) for source in sources)
        max_tokens = max(
            640,
            min(
                3200,
                translation_output_budget(joined, "smart2") + 36 * len(rows),
            ),
        )
        source_json = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
        prompt = f"""你是 Phoenix 医学课件和论文精译器。把JSON数组中每个 text 完整、准确地翻译成{target_language}。

要求：
- 保持每个 id 不变、条目数量和顺序不变；只翻译 text 的内容。
- 使用规范医学术语，不总结、不删减、不扩写，不加入原文没有的知识。
- 数字、单位、百分比、HU、剂量、图表编号、参考文献编号和医学缩写必须保留。
- 否定/肯定、左右侧、可能/明确、增高/降低、增大/缩小、稳定/进展等关系必须保持。
- 相邻条目属于同一页或同一论文段落组，可利用上下文消除歧义，但不得合并条目。
- 只输出合法JSON数组，格式严格为 [{{"id":"S0001","translation":"译文"}}]；不得输出Markdown代码围栏或说明。

输入JSON：
{source_json}
"""
        raw = self.llm.generate(
            prompt,
            max_new_tokens=max_tokens,
            profile="translation",
        ).strip()
        try:
            return self._parse_segment_json(raw)
        except Exception:
            repair_prompt = f"""把下面批量医学译文修复为合法JSON数组。
不得重新总结或添加内容；必须保留输入 id，字段只能是 id 和 translation。
目标语言：{target_language}
原始输入：{source_json}
待修复输出：{raw}
只输出JSON数组。"""
            repaired = self.llm.generate(
                repair_prompt,
                max_new_tokens=max_tokens,
                profile="translation",
            ).strip()
            return self._parse_segment_json(repaired)

    def retry_translation(
        self,
        source: str,
        draft: str,
        reasons: tuple[str, ...],
        target_language: str = "中文",
    ) -> str:
        reason_text = "；".join(reasons) or "批量结果缺失或结构校验失败"
        prompt = f"""你是 Phoenix 医学翻译纠错器。只修正下面这一条译文，输出修正后的{target_language}译文，不要解释。

硬性要求：完整翻译，不总结、不扩写；数字、单位、正负号、缩写、否定、侧别、诊断确定性和方向关系必须与原文一致。
校验失败原因：{reason_text}

原文：
{source}

上一版译文：
{draft or '[缺失]'}
"""
        return self.llm.generate(
            prompt,
            max_new_tokens=translation_output_budget(source, "smart2"),
            # Keep the quality-model route, but never enable general-purpose
            # chain-of-thought for a deterministic translation correction.
            # Provider Hub maps ``translation`` to the Smart2 model while
            # explicitly disabling billable reasoning/thinking tokens.
            profile="translation",
        ).strip()


class MultiModelTranslationEngine:
    """Translation with an explicit preview/medical-quality boundary.

    Marian/NLLB remain as dormant compatibility backends for old preview
    checkpoints only. Smart2 is the sole formal medical route and uses only the
    quality model; it never silently falls back to those inaccurate preview
    models. Every candidate remains subject to numeric, unit, acronym, and
    medical-semantic validation before automatic acceptance.
    """

    _phoenix_formal_translation_contract = FORMAL_TRANSLATION_CONTRACT_VERSION

    def __init__(self, paths: WorkbenchPaths, llm: LocalLLM):
        self.paths = paths
        self.validator = TranslationValidator()
        self.marian = MarianEnZhBackend(paths)
        self.nllb = NLLBEnZhBackend(paths)
        self.qwen = QwenMedicalTranslationBackend(llm)

    @staticmethod
    def _backend_available(backend, smart_level: str | None = None) -> bool:
        if smart_level is not None:
            try:
                return bool(backend.available(smart_level))
            except TypeError:
                pass
        try:
            return bool(backend.available())
        except Exception:
            return False

    def _real_smart_backend(self) -> bool:
        return isinstance(self.qwen, QwenMedicalTranslationBackend)

    def available_backends(self) -> list[str]:
        """Return installed backend inventory, including dormant legacy models."""

        result: list[str] = []
        if self._backend_available(self.marian):
            result.append(self.marian.name)
        if self._backend_available(self.nllb):
            result.append(self.nllb.name)
        if self._real_smart_backend():
            if (
                self._backend_available(self.qwen, "smart1")
                or self._backend_available(self.qwen, "smart2")
            ):
                result.append(self.qwen.name)
        elif self._backend_available(self.qwen):
            result.append(self.qwen.name)
        return result

    def formal_backend_names(self, target_language: str = "中文") -> list[str]:
        """Return only backends permitted to create a formal medical document."""

        try:
            active = list(self.active_backends(target_language, "smart2"))
        except Exception:
            active = []
        names = [
            str(getattr(backend, "name", "") or "").strip()
            for backend in active
        ]
        names = [
            name for name in names
            if name and name not in LEGACY_PREVIEW_BACKEND_NAMES
        ]
        if not names and active:
            # Test/provider adapters may expose anonymous active objects. Keep
            # their declared inventory while applying the same legacy denylist.
            names = [
                str(name)
                for name in self.available_backends()
                if str(name) not in LEGACY_PREVIEW_BACKEND_NAMES
            ]
        return list(dict.fromkeys(names))

    def preview_backend_names(self, target_language: str = "中文") -> list[str]:
        """Expose legacy preview inventory separately from formal readiness."""

        try:
            active = self.active_backends(target_language, "smart1")
        except Exception:
            active = []
        return list(dict.fromkeys(
            str(getattr(backend, "name", "") or "").strip()
            for backend in active
            if str(getattr(backend, "name", "") or "").strip()
            in LEGACY_PREVIEW_BACKEND_NAMES
        ))

    def active_backends(
        self,
        target_language: str = "中文",
        smart_level: str = "smart1",
    ) -> list[object]:
        level = _normalize_smart_level(smart_level)
        smart_backend_available = False
        if self._real_smart_backend():
            smart_backend_available = self._backend_available(self.qwen, level)
        elif _flag("PHOENIX_TRANSLATION_QWEN_REVIEW", default=False):
            smart_backend_available = self._backend_available(self.qwen)

        dedicated: list[object] = []
        if target_language in _SIMPLIFIED_TARGETS:
            if self._backend_available(self.marian):
                dedicated.append(self.marian)
            if self._backend_available(self.nllb):
                dedicated.append(self.nllb)

        if level == "smart2":
            return [self.qwen] if smart_backend_available else []

        return dedicated

    def translate(
        self,
        source: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ) -> TranslationDecision:
        attempts: list[TranslationAttempt] = []
        best: TranslationAttempt | None = None
        backend_errors: list[str] = []
        level = _normalize_smart_level(smart_level)
        backends = self.active_backends(target_language, level)

        for backend in backends:
            try:
                if isinstance(backend, QwenMedicalTranslationBackend):
                    text = backend.translate(
                        source,
                        target_language,
                        smart_level=level,
                    )
                else:
                    text = backend.translate(source)
                quality = self.validator.validate(source, text, target_language)
                attempt = TranslationAttempt(
                    backend=backend.name,
                    text=text,
                    quality=quality,
                )
                attempts.append(attempt)
                if best is None or attempt.quality.score > best.quality.score:
                    best = attempt

                if quality.ok:
                    return TranslationDecision(
                        text=text,
                        backend=backend.name,
                        quality=quality,
                        needs_review=False,
                        attempts=tuple(attempts),
                    )
                if isinstance(backend, QwenMedicalTranslationBackend) and level == "smart2":
                    try:
                        corrected = backend.retry_translation(
                            source,
                            text,
                            quality.reasons,
                            target_language,
                        )
                        corrected_quality = self.validator.validate(
                            source,
                            corrected,
                            target_language,
                        )
                        corrected_attempt = TranslationAttempt(
                            backend=f"{backend.name}_quality_retry",
                            text=corrected,
                            quality=corrected_quality,
                        )
                        attempts.append(corrected_attempt)
                        if best is None or corrected_quality.score > best.quality.score:
                            best = corrected_attempt
                        if corrected_quality.ok:
                            return TranslationDecision(
                                text=corrected,
                                backend=corrected_attempt.backend,
                                quality=corrected_quality,
                                needs_review=False,
                                attempts=tuple(attempts),
                            )
                    except Exception as exc:
                        backend_errors.append(
                            f"{backend.name}_quality_retry: {type(exc).__name__}: {exc}"
                        )
            except Exception as exc:
                backend_errors.append(f"{backend.name}: {type(exc).__name__}: {exc}")

        if best is not None:
            return TranslationDecision(
                text=best.text,
                backend=best.backend,
                quality=best.quality,
                needs_review=True,
                attempts=tuple(attempts),
            )

        if not backends:
            if target_language in _SIMPLIFIED_TARGETS:
                raise RuntimeError("没有可用的本地智能翻译或英译中后端。")
            raise RuntimeError(
                f"目标语言“{target_language}”当前没有可用的本地智能翻译能力。"
            )
        raise RuntimeError("所有翻译后端均执行失败: " + " | ".join(backend_errors))

    def translate_segments(
        self,
        sources: list[str],
        target_language: str = "中文",
        *,
        smart_level: str = "smart2",
    ) -> tuple[TranslationDecision, ...]:
        """Translate one slide/paragraph unit in one normal model call.

        The quality model handles the common batch. Only missing or
        semantically invalid rows are retried individually with the same
        no-reasoning translation profile, keeping token use proportional to
        actual failures.
        """

        values = [str(source or "").strip() for source in sources]
        if not values:
            return ()
        level = _normalize_smart_level(smart_level)
        if level != "smart2":
            return tuple(
                self.translate(
                    source,
                    target_language,
                    smart_level=level,
                )
                for source in values
            )
        if not self._backend_available(self.qwen, "smart2"):
            raise RuntimeError("医学精译质量模型未就绪。")

        batch = self.qwen.translate_segments(values, target_language)
        decisions: list[TranslationDecision] = []
        for index, source in enumerate(values, start=1):
            segment_id = f"S{index:04d}"
            draft = str(batch.get(segment_id, "") or "").strip()
            quality = self.validator.validate(source, draft, target_language)
            first = TranslationAttempt(
                backend=f"{self.qwen.name}_batch",
                text=draft,
                quality=quality,
                errors=() if draft else ("批量译文缺失",),
            )
            if quality.ok:
                decisions.append(
                    TranslationDecision(
                        text=draft,
                        backend=first.backend,
                        quality=quality,
                        needs_review=False,
                        attempts=(first,),
                    )
                )
                continue

            attempts = [first]
            retry_error = ""
            try:
                corrected = self.qwen.retry_translation(
                    source,
                    draft,
                    quality.reasons,
                    target_language,
                )
                corrected_quality = self.validator.validate(
                    source,
                    corrected,
                    target_language,
                )
                retry = TranslationAttempt(
                    backend=f"{self.qwen.name}_quality_retry",
                    text=corrected,
                    quality=corrected_quality,
                )
                attempts.append(retry)
            except Exception as exc:
                retry_error = f"{type(exc).__name__}: {exc}"
                retry = None

            candidates = [attempt for attempt in attempts if attempt.text]
            best = max(candidates, key=lambda item: item.quality.score) if candidates else first
            if retry is None and retry_error:
                best = TranslationAttempt(
                    backend=best.backend,
                    text=best.text,
                    quality=best.quality,
                    errors=tuple((*best.errors, retry_error)),
                )
                attempts[-1] = best
            decisions.append(
                TranslationDecision(
                    text=best.text or source,
                    backend=best.backend,
                    quality=best.quality,
                    needs_review=not best.quality.ok,
                    attempts=tuple(attempts),
                )
            )
        return tuple(decisions)

    def unload(self) -> None:
        self.marian.unload()
        self.nllb.unload()
