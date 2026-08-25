from __future__ import annotations

"""Final production policy for Phoenix v3 model-1 candidates.

The v3 chain deliberately uses Marian/NLLB only as draft generators before
HY-MT/model3 validation. A folder on disk is not sufficient readiness, and
NLLB must never enter a commercial runtime. Adapter tests/plugins that replace
these concrete backends remain usable because runtime checks apply only to the
real Phoenix Seq2Seq classes.
"""

_INSTALLED = False
_ZH_TARGETS = {"中文", "简体中文", "chinese", "zh", "zh-cn"}


def _backends(engine, target_language: str) -> list[object]:
    from .release_hardening import _commercial_release
    from .release_runtime_hardening import local_seq2seq_runtime_ready
    from .translation_models import _Seq2SeqBackend

    target = str(target_language or "").strip().lower()
    if target not in _ZH_TARGETS:
        return []

    result: list[object] = []
    commercial = bool(_commercial_release(engine.paths))
    runtime_ready = bool(local_seq2seq_runtime_ready())
    available = getattr(engine, "_backend_available", None)

    for attr in ("marian", "nllb"):
        backend = getattr(engine, attr, None)
        if backend is None:
            continue
        name = str(getattr(backend, "name", "") or "")
        if commercial and name == "nllb_600m_en_zh":
            continue
        if isinstance(backend, _Seq2SeqBackend) and not runtime_ready:
            continue
        try:
            ready = bool(available(backend)) if callable(available) else bool(backend.available())
        except Exception:
            ready = False
        if ready:
            result.append(backend)
    return result


def _install_acronym_gate() -> None:
    from .medical_acronyms import MedicalAcronymResolver
    from .translation_local_first_release import _api_fallback_allowed

    def llm_available(self) -> bool:
        llm = getattr(self, "llm", None)
        if llm is None:
            return False
        try:
            available = getattr(llm, "available", None)
            if callable(available) and not bool(available("translation")):
                return False
        except TypeError:
            try:
                if callable(available) and not bool(available()):
                    return False
            except Exception:
                return False
        except Exception:
            return False

        backend_fn = getattr(llm, "backend", None)
        if not callable(backend_fn):
            # Lightweight local adapters used by offline tools need not expose
            # the full provider-hub interface.
            return True
        try:
            backend = str(backend_fn("translation") or "").strip().lower()
        except Exception:
            return False
        if backend == "remote_server" and not _api_fallback_allowed():
            return False
        return True

    MedicalAcronymResolver._llm_available = llm_available


def _install_dual_route_model1() -> None:
    from . import translation_dual_route_release as dual
    from .translation_models import TranslationAttempt, translation_output_budget

    def model1(engine, source: str, target: str, attempts: list, errors: list[str]):
        # Prefer the configured local fast LLM as the contextual draft stage.
        if dual._fast_local_ready(engine):
            try:
                prompt = f"""你是 Phoenix 本地模型1：医学语境理解与初译器。
先理解上一页/上一段与当前同页/同段组，再完整翻译当前英文原文为{target}。本阶段只是初稿。
禁止总结、删减、扩写、解释或拒答；数字、单位、正负号、侧别、否定、分级、诊断确定性、缩写和图表编号必须保留。
{dual._context()}
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
        for backend in _backends(engine, target):
            try:
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

    dual._model1 = model1


def _install_inventory_policy() -> None:
    from . import hybrid_translation_policy as hybrid
    from .translation_models import MultiModelTranslationEngine
    from .release_hardening import _commercial_release

    hybrid._local_backends = _backends

    previous_available = MultiModelTranslationEngine.available_backends
    previous_active = MultiModelTranslationEngine.active_backends

    def available_backends(self):
        names = list(previous_available(self))
        if _commercial_release(self.paths):
            names = [name for name in names if str(name) != "nllb_600m_en_zh"]
        return list(dict.fromkeys(str(name) for name in names if str(name)))

    def active_backends(self, target_language="中文", smart_level="smart1"):
        values = list(previous_active(self, target_language, smart_level))
        allowed_model1 = {id(item) for item in _backends(self, target_language)}
        filtered: list[object] = []
        for item in values:
            name = str(getattr(item, "name", "") or "")
            if name in {"marian_en_zh", "nllb_600m_en_zh"} and id(item) not in allowed_model1:
                continue
            filtered.append(item)
        return filtered

    MultiModelTranslationEngine.available_backends = available_backends
    MultiModelTranslationEngine.active_backends = active_backends


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_inventory_policy()
    _install_dual_route_model1()
    _install_acronym_gate()
    _INSTALLED = True
