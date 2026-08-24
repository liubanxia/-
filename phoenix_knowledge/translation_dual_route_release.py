from __future__ import annotations

import json
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

_INSTALLED = False
_FINAL_TAG = "|quality_final_v2"
_CTX = ContextVar("phoenix_translation_context_v3", default={})
_TERM_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:SUV(?:max|mean|peak)|MinIP|eGFR|mRNA|pH|HFrEF|HFpEF|SARS-CoV-2|"
    r"[A-Z][A-Z0-9]{1,11}(?:[/+.-][A-Z0-9]{1,8})*|[A-Z][0-9](?:WI)?)(?![A-Za-z0-9])|"
    r"(?i:\b(?:[A-Za-z][A-Za-z-]{2,}\s+){0,3}(?:disease|syndrome|artery|vein|stenosis|occlusion|"
    r"hemorrhage|infarction|ischemia|edema|effusion|fracture|lesion|nodule|mass|carcinoma|sarcoma|"
    r"lymphoma|metastasis|enhancement|diffusion|perfusion|imaging|tomography|angiography|sequence|"
    r"signal|intensity|coefficient|fraction|volume|function|ratio|index|score|grade|pressure|fibrosis)\b)"
)


def _clip(text: str, n: int = 1200) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= n else value[-n:]


def _terms(text: str, limit: int = 48) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _TERM_RE.finditer(str(text or "")):
        term = " ".join(match.group(0).strip(" ,;:.()[]{}").split())
        key = term.lower()
        if len(term) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return tuple(out)


def _context() -> str:
    ctx = _CTX.get({}) or {}
    lines: list[str] = []
    if ctx.get("label"):
        lines.append(f"当前位置：{ctx['label']}")
    if ctx.get("previous_source"):
        lines.append(f"上一单元英文：{_clip(ctx['previous_source'])}")
    if ctx.get("previous_translation"):
        lines.append(f"上一单元已确定译文：{_clip(ctx['previous_translation'])}")
    if ctx.get("current_source"):
        lines.append(f"当前同页/同段组：{_clip(ctx['current_source'], 1600)}")
    if ctx.get("terms"):
        lines.append("累计医学英语术语/名词：" + "；".join(tuple(ctx["terms"])[:48]))
    if not lines:
        return ""
    return "\n上下文辅助信息（只用于理解，不得重复翻译）：\n" + "\n".join(lines) + "\n"


def _push(owner, source: str, label: str):
    merged: list[str] = []
    seen: set[str] = set()
    for item in tuple(getattr(owner, "_phoenix_terms", ()) or ()) + _terms(source):
        key = str(item).lower()
        if item and key not in seen:
            seen.add(key)
            merged.append(item)
    owner._phoenix_terms = tuple(merged[:96])
    return _CTX.set({
        "label": label,
        "previous_source": getattr(owner, "_phoenix_previous_source", ""),
        "previous_translation": getattr(owner, "_phoenix_previous_translation", ""),
        "current_source": source,
        "terms": owner._phoenix_terms,
    })


def _pop(owner, token, source: str, translated: str) -> None:
    owner._phoenix_previous_source = _clip(source, 1600)
    owner._phoenix_previous_translation = _clip(translated, 1600)
    try:
        _CTX.reset(token)
    except Exception:
        pass


def _remote_selected(engine) -> bool:
    llm = getattr(getattr(engine, "qwen", None), "llm", None)
    if llm is None:
        return False
    try:
        if str(llm.compute.requested_mode() or "").strip().lower() == "remote":
            return True
    except Exception:
        pass
    try:
        return str(llm.backend("translation") or "").strip().lower() == "remote_server"
    except Exception:
        return False


def _install_api_gate() -> None:
    from . import hybrid_translation_policy as hybrid

    def available(engine) -> bool:
        if not _remote_selected(engine):
            return False
        backend = getattr(engine, "qwen", None)
        check = getattr(engine, "_backend_available", None)
        if backend is None or not callable(check):
            return False
        try:
            return bool(check(backend, "smart2"))
        except TypeError:
            return bool(check(backend))
        except Exception:
            return False

    hybrid._smart_available = available


def _fast_local_ready(engine) -> bool:
    qwen = getattr(engine, "qwen", None)
    llm = getattr(qwen, "llm", None)
    if qwen is None or llm is None:
        return False
    try:
        if str(llm.backend("fast") or "").strip().lower() == "remote_server":
            return False
        return bool(engine._backend_available(qwen, "smart1"))
    except Exception:
        return False


def _model1(engine, source: str, target: str, attempts: list, errors: list[str]):
    from .translation_models import TranslationAttempt, translation_output_budget

    if _fast_local_ready(engine):
        try:
            prompt = f"""你是 Phoenix 本地模型1：医学语境理解与初译器。
先理解上一页/上一段与当前同页/同段组，再完整翻译当前英文原文为{target}。本阶段只是初稿。
禁止总结、删减、扩写、解释或拒答；数字、单位、正负号、侧别、否定、分级、诊断确定性、缩写和图表编号必须保留。
{_context()}
{engine.qwen.glossary_prompt(source)}
当前英文原文：\n{source}\n只输出当前原文译文。"""
            text = engine.qwen.llm.generate(
                prompt,
                max_new_tokens=translation_output_budget(source, "smart1"),
                profile="fast",
            ).strip()
            item = TranslationAttempt(
                backend="model1_context_fast_draft",
                text=text,
                quality=engine.validator.validate(source, text, target),
            )
            attempts.append(item)
            return item
        except Exception as exc:
            errors.append(f"model1-fast: {type(exc).__name__}: {exc}")

    best = None
    for backend in (getattr(engine, "nllb", None), getattr(engine, "marian", None)):
        try:
            if backend is None or not engine._backend_available(backend):
                continue
            text = backend.translate(source)
            item = TranslationAttempt(
                backend=f"model1_draft:{backend.name}",
                text=str(text or "").strip(),
                quality=engine.validator.validate(source, text, target),
            )
            attempts.append(item)
            if best is None or item.quality.score > best.quality.score:
                best = item
        except Exception as exc:
            errors.append(f"model1-local: {type(exc).__name__}: {exc}")
    return best


def _patch_model2() -> None:
    from .hymt_translation_backend import HYMTMedicalTranslationBackend

    def refine(self, source: str, draft: str, target_language: str = "中文") -> str:
        target = self._target_name(target_language)
        source, draft = str(source or "").strip(), str(draft or "").strip()
        prompt = (
            "你是 Phoenix 本地模型2：医学英语术语增强纠错器。模型1初稿未通过质量门。\n"
            "重新对照英文原文，重点识别疾病、解剖、检查技术、影像征象、病理、药物、统计学术语和缩写，并结合上下文统一译法。\n"
            + _context()
            + "本段医学英语术语/名词候选：" + ("；".join(_terms(source)) or "[自动识别]") + "\n\n"
            + f"模型1初稿：\n{draft or '[缺失]'}\n\n英文原文：\n{source}\n\n"
            + f"输出完整准确的{target}译文。不得总结、删减、扩写或解释。"
        )
        return self._generate(prompt, max_new_tokens=int(len(source) * 0.8) + 512)

    HYMTMedicalTranslationBackend.refine = refine


def _patch_model3() -> None:
    from .qwen_local_medical_backend import LocalQwenMedicalBackend

    def prompt(self, source: str, draft: str, target_language: str) -> str:
        system = (
            "你是 Phoenix 本地模型3：医学终审与纠错器。逐句对照英文原文，只修正确有问题的地方。"
            "重点核对医学术语、解剖关系、影像征象、检查技术、病理、药物、统计术语、数字、单位、正负号、侧别、否定、分级、"
            "诊断确定性、缩写和图表编号。正确内容保持不变。禁止总结、删减、扩写、解释、拒答或添加原文没有的信息。只输出最终译文。"
        )
        user = (
            f"目标语言：{target_language}\n" + _context()
            + "本段医学英语术语/名词候选：" + ("；".join(_terms(source)) or "[无额外候选]")
            + f"\n\n英文原文：\n{source}\n\n当前译文：\n{draft}\n\n只输出终审核对后的译文。"
        )
        return self._chat_prompt(system, user)

    LocalQwenMedicalBackend._refine_prompt = prompt


def _patch_api() -> None:
    from .translation_models import QwenMedicalTranslationBackend, translation_output_budget

    def retry(self, source: str, draft: str, reasons: tuple[str, ...], target_language: str = "中文") -> str:
        source = str(source or "").strip()
        reason = "；".join(str(x) for x in reasons if str(x).strip())
        prompt = f"""你是 Phoenix Smart2 医学翻译兜底教师。本地模型3终审仍未通过质量门，请只修正剩余错误。
严格对照英文原文、上下文、术语候选和当前译文。禁止总结、删减、扩写、解释或拒答。
数字、单位、正负号、侧别、否定、分级、诊断确定性、缩写和图表编号必须与原文一致。
失败原因：{reason or '本地终审未通过'}
{_context()}
{self.glossary_prompt(source)}
本段医学英语术语/名词候选：{'；'.join(_terms(source)) or '[无额外候选]'}
英文原文：\n{source}\n当前本地最佳译文：\n{draft or '[缺失]'}\n只输出最终修正版。"""
        return self.llm.generate(
            prompt,
            max_new_tokens=translation_output_budget(source, "smart2"),
            profile="translation",
        ).strip()

    QwenMedicalTranslationBackend.retry_translation = retry


def _install_chain() -> None:
    from . import hymt_cascade_policy as hymt
    from . import translation_cascade_v2 as cascade

    def run(engine, source: str, target: str, attempts: list, errors: list[str]):
        m1 = _model1(engine, source, target, attempts, errors)
        m1_ok = bool(m1 and m1.quality.ok and float(m1.quality.score) >= hymt.MODEL1_ACCEPT_SCORE)
        m2, base = None, m1
        if not m1_ok:
            m2 = hymt._run_model2(engine, source, m1, target, attempts, errors)
            if m2 is not None and (base is None or m2.quality.score >= base.quality.score):
                base = m2
        if base is None or not str(base.text or "").strip():
            return None, "quality_no_draft"
        if not cascade._model3_available(engine):
            return base, "quality_model3_unavailable"
        backend = cascade._model3(engine)
        try:
            text = backend.refine(source, base.text, target)
            final = hymt._quality_attempt(engine, backend.name + _FINAL_TAG, source, text, target)
            attempts.append(final)
            if final.quality.ok and float(final.quality.score) >= cascade.MODEL3_ACCEPT_SCORE:
                return final, "quality_final_model3"
            candidates = [x for x in (m1, m2, final) if x is not None and str(x.text or "").strip()]
            return max(candidates, key=lambda x: float(x.quality.score)), "model3_failed"
        except Exception as exc:
            errors.append(f"Qwen-model3-final: {type(exc).__name__}: {exc}")
            return base, "model3_failed"

    cascade._run_local_cascade = run


def _install_teacher_pool() -> None:
    from . import translation_cascade_v2 as cascade
    original = cascade._api_polish_local_draft

    def wrapped(engine, source, local_draft, local_stage, target, attempts, errors):
        result = original(engine, source, local_draft, local_stage, target, attempts, errors)
        if result is None or not bool(getattr(result.quality, "ok", False)):
            return result
        try:
            ctx = _CTX.get({}) or {}
            root = Path(getattr(engine.paths, "runtime_root", ".")) / "translation_learning"
            root.mkdir(parents=True, exist_ok=True)

            def pick(prefixes):
                value = ""
                for item in getattr(result, "attempts", ()) or attempts:
                    if any(str(item.backend).startswith(p) for p in prefixes) and str(item.text or "").strip():
                        value = item.text.strip()
                return value

            row = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "context": {"label": ctx.get("label", ""), "terms": list(ctx.get("terms", ()))},
                "model1_text": pick(("model1_", "model1_draft:")),
                "model2_text": pick(("hymt15_1p8b",)),
                "model3_text": pick(("qwen_local_medical_model3",)),
                "api_final_text": result.text,
                "final_backend": result.backend,
                "quality_verified": True,
                "reviewed": False,
                "training_status": "candidate_only",
            }
            with (root / "api_teacher_candidates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[Phoenix][学习池] API纠错样本记录失败: {type(exc).__name__}: {exc}", flush=True)
        return result

    cascade._api_polish_local_draft = wrapped


def _install_context_hooks() -> None:
    try:
        from .office_translation import OfficeDocumentTranslator
        old_doc = OfficeDocumentTranslator.translate_document
        old_sources = OfficeDocumentTranslator._translate_sources

        def doc(self, *args, **kwargs):
            self._phoenix_previous_source = self._phoenix_previous_translation = ""
            self._phoenix_terms = ()
            try:
                return old_doc(self, *args, **kwargs)
            finally:
                _CTX.set({})

        def sources(self, values, target):
            current = "\n".join(str(x or "").strip() for x in values if str(x or "").strip())
            token = _push(self, current, "Office当前幻灯片/段落组")
            translated = ""
            try:
                decisions = list(old_sources(self, values, target))
                translated = "\n".join(str(getattr(x, "text", "") or "") for x in decisions)
                return decisions
            finally:
                _pop(self, token, current, translated)

        OfficeDocumentTranslator.translate_document = doc
        OfficeDocumentTranslator._translate_sources = sources
    except Exception as exc:
        print(f"[Phoenix][上下文] Office安装失败: {type(exc).__name__}: {exc}", flush=True)

    try:
        from .translator import PDFTranslator
        old_book = PDFTranslator.translate_book
        old_page = PDFTranslator._translate_page

        def book(self, *args, **kwargs):
            self._phoenix_previous_source = self._phoenix_previous_translation = ""
            self._phoenix_terms = ()
            try:
                return old_book(self, *args, **kwargs)
            finally:
                _CTX.set({})

        def page(self, source, number, target, *, smart_level="smart1", status=None):
            current = str(source or "").strip()
            token = _push(self, current, f"PDF第{int(number)}页")
            translated = ""
            try:
                result = old_page(self, source, number, target, smart_level=smart_level, status=status)
                translated = str(result[0] or "")
                return result
            finally:
                _pop(self, token, current, translated)

        PDFTranslator.translate_book = book
        PDFTranslator._translate_page = page
    except Exception as exc:
        print(f"[Phoenix][上下文] PDF安装失败: {type(exc).__name__}: {exc}", flush=True)


def _report(engine) -> None:
    if getattr(engine, "_phoenix_v3_reported", False):
        return
    engine._phoenix_v3_reported = True
    from . import hybrid_translation_policy as hybrid
    api = "已连接，仅模型3失败时兜底" if hybrid._smart_available(engine) else ("已选择但不可用" if _remote_selected(engine) else "未连接")
    print(
        "[Phoenix][翻译路线] v3：模型1语境初译→失败项HY-MT模型2术语增强→Qwen模型3逐段终审→仍失败才Smart2 API；"
        f"API={api}。API修正进入学习候选池，不在线改权重。",
        flush=True,
    )


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from .translation_models import MultiModelTranslationEngine, _normalize_smart_level
    from . import translation_cascade_v2 as cascade

    _install_api_gate()
    _patch_model2()
    _patch_model3()
    _patch_api()
    _install_chain()
    _install_teacher_pool()
    _install_context_hooks()

    cls = MultiModelTranslationEngine

    def translate(self, source: str, target_language: str = "中文", *, smart_level: str = "smart1"):
        _report(self)
        return cascade._translate(self, source, target_language, smart_level=_normalize_smart_level(smart_level))

    def segments(self, sources: list[str], target_language: str = "中文", *, smart_level: str = "smart2"):
        values = [str(x or "").strip() for x in sources]
        if not values:
            return ()
        _report(self)
        return tuple(
            cascade._translate(self, value, target_language, smart_level=_normalize_smart_level(smart_level))
            for value in values
        )

    cls.translate = translate
    cls.translate_segments = segments

    print(
        "[Phoenix][翻译架构] v3上下文学习链启用：模型1理解/初译；失败项进模型2并同步英语医学名词；"
        "模型3终审核对修正；仍失败才API。统一拒答安全闸继续生效。",
        flush=True,
    )
