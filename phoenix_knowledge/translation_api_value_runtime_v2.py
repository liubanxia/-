from __future__ import annotations

"""Production-scoped API value accounting runtime.

The ledger remains observational: it never makes an API request. Reuse counters
are updated only for the production translation-memory instance so low-level
unit tests, migration tools, and diagnostics cannot create side-effect databases
outside their own temporary roots.
"""

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import translation_api_value_ledger as ledger
    from . import translation_cascade_v2 as cascade
    from . import translation_survival_memory as survival

    old_translate = cascade._translate

    def translate(
        engine,
        source: str,
        target_language: str = "中文",
        *,
        smart_level: str = "smart1",
    ):
        result = old_translate(
            engine,
            source,
            target_language,
            smart_level=smart_level,
        )
        try:
            ledger._record_decision(
                engine,
                str(source or ""),
                str(target_language or "中文"),
                result,
            )
            ledger._report(engine)
        except Exception as exc:
            print(
                "[Phoenix][API价值账本] 旁路分析失败但不影响译文: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return result

    translate._phoenix_api_value_runtime_v2 = True
    cascade._translate = translate

    memory_cls = survival.TranslationMemory
    old_exact = memory_cls.lookup_exact
    old_similar = memory_cls.lookup_similar

    def _mark(self, source: str, target_language: str, *, similar: bool) -> None:
        if not bool(getattr(self, "_phoenix_production_memory", False)):
            return
        try:
            ledger_path = self.path.parent.parent / "人工修订与学习" / ledger._DB_NAME
            ledger.APIValueLedger(ledger_path).mark_reuse(
                source,
                target_language,
                similar=similar,
            )
        except Exception:
            pass

    def lookup_exact(self, source: str, target_language: str):
        hit = old_exact(self, source, target_language)
        if hit is not None and ledger._is_api_backend(getattr(hit, "backend", "")):
            _mark(self, source, target_language, similar=False)
        return hit

    lookup_exact._phoenix_api_reuse_v2 = True

    def lookup_similar(self, source: str, target_language: str, **kwargs):
        hit = old_similar(self, source, target_language, **kwargs)
        if hit is not None and ledger._is_api_backend(getattr(hit, "backend", "")):
            _mark(self, source, target_language, similar=True)
        return hit

    lookup_similar._phoenix_api_reuse_v2 = True

    memory_cls.lookup_exact = lookup_exact
    memory_cls.lookup_similar = lookup_similar

    _INSTALLED = True
    print(
        "[Phoenix][API价值账本] v2已启用：只观察已发生API纠错，"
        "复用计数只作用于生产翻译记忆；候选资产不得自动晋级专家。",
        flush=True,
    )
