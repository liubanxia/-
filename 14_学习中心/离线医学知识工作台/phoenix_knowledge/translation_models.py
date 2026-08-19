from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import WorkbenchPaths, resolve_model_dir
from .llm import LocalLLM


_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%|mm|cm|mL|ml|mg|g|kg|HU|kV|mAs)?", re.I)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,10}\b")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SIMPLIFIED_TARGETS = {"中文", "简体中文", "Chinese", "zh", "zh-CN"}


def _cuda_is_usable() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
        return int(major) >= 5
    except Exception:
        return False


def _normalize_number(token: str) -> str:
    return token.replace(",", "").lower()


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

    This validator is deliberately conservative: it checks for omissions,
    corrupted numbers/units, lost acronyms and obviously untranslated output.
    Natural Chinese medical prose is produced by the intelligent translation
    backend instead of trying to infer semantic quality from heuristics alone.
    """

    def validate(self, source: str, translated: str, target_language: str = "中文") -> QualityReport:
        source = (source or "").strip()
        translated = (translated or "").strip()
        reasons: list[str] = []
        score = 1.0
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

        source_numbers = [_normalize_number(x) for x in _NUMBER_RE.findall(source)]
        if source_numbers:
            translated_numbers = {_normalize_number(x) for x in _NUMBER_RE.findall(translated)}
            kept = sum(1 for token in source_numbers if token in translated_numbers)
            coverage = kept / len(source_numbers)
            if coverage < 0.8:
                reasons.append(f"数字/单位保留率偏低({coverage:.0%})")
                score -= 0.35

        acronyms = list(dict.fromkeys(_ACRONYM_RE.findall(source)))
        if acronyms:
            kept = sum(1 for token in acronyms if token in translated)
            coverage = kept / len(acronyms)
            if coverage < 0.65:
                reasons.append(f"医学缩写保留率偏低({coverage:.0%})")
                score -= 0.2

        if target_language in _SIMPLIFIED_TARGETS:
            cjk_count = len(_CJK_RE.findall(translated))
            latin_count = len(_LATIN_RE.findall(translated))
            source_latin = len(_LATIN_RE.findall(source))
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
        return QualityReport(score >= 0.62, score, tuple(reasons))


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
        profile = "deep" if _normalize_smart_level(smart_level) == "smart2" else "fast"
        return self.llm.available(profile)

    def translate(
        self,
        text: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ) -> str:
        level = _normalize_smart_level(smart_level)
        profile = "deep" if level == "smart2" else "fast"
        max_tokens = 2600 if level == "smart2" else 2100
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


class MultiModelTranslationEngine:
    """Offline medical translation with intelligent-first quality routing.

    For Simplified Chinese, intelligent translation is now the preferred final
    output whenever a local generator is available. Marian / NLLB remain as
    fast fallback engines when the intelligent backend is unavailable or
    produces a structurally invalid result. This matches the product goal:
    readable medical Chinese first, raw machine translation only as fallback.
    """

    def __init__(self, paths: WorkbenchPaths, llm: LocalLLM):
        self.paths = paths
        self.validator = TranslationValidator()
        self.marian = MarianEnZhBackend(paths)
        self.nllb = NLLBEnZhBackend(paths)
        self.qwen = QwenMedicalTranslationBackend(llm)

    def available_backends(self) -> list[str]:
        result = []
        if self.qwen.available("smart1") or self.qwen.available("smart2"):
            result.append(self.qwen.name)
        if self.marian.available():
            result.append(self.marian.name)
        if self.nllb.available():
            result.append(self.nllb.name)
        return result

    def active_backends(
        self,
        target_language: str = "中文",
        smart_level: str = "smart1",
    ) -> list[object]:
        level = _normalize_smart_level(smart_level)
        result: list[object] = []

        if self.qwen.available(level):
            result.append(self.qwen)

        if target_language in _SIMPLIFIED_TARGETS:
            if self.marian.available():
                result.append(self.marian)
            if self.nllb.available():
                result.append(self.nllb)

        return result

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

                # Intelligent translation is preferred even when a minor
                # structural warning remains. Only a severe validator failure
                # should fall through to the raw seq2seq fallback engines.
                if isinstance(backend, QwenMedicalTranslationBackend):
                    if quality.ok or quality.score >= 0.45:
                        return TranslationDecision(
                            text=text,
                            backend=backend.name,
                            quality=quality,
                            needs_review=not quality.ok,
                            attempts=tuple(attempts),
                        )
                elif quality.ok:
                    return TranslationDecision(
                        text=text,
                        backend=backend.name,
                        quality=quality,
                        needs_review=False,
                        attempts=tuple(attempts),
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

    def unload(self) -> None:
        self.marian.unload()
        self.nllb.unload()
