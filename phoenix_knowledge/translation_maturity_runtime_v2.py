from __future__ import annotations

"""Production-scoped translation-memory maturity gate.

The historical gate replaced TranslationMemory.lookup_exact/lookup_similar for
all instances. That made low-level memory objects unusable in tests, migration,
maintenance, and diagnostics until the production corpus reached 10 books and
1000 verified rows. This runtime keeps the conservative production rule while
leaving ordinary TranslationMemory instances semantically correct.

Completed-book accounting is intentionally handled by the single Workbench
post-translation aggregator, not by wrapping PDFTranslator again.
"""

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from . import translation_learning_maturity_gate as maturity
    from . import translation_survival_memory as survival

    memory_cls = survival.TranslationMemory
    old_exact = memory_cls.lookup_exact
    old_similar = memory_cls.lookup_similar
    old_memory_for_engine = survival._memory_for_engine

    def memory_for_engine(engine):
        memory = old_memory_for_engine(engine)
        memory._phoenix_production_memory = True
        return memory

    memory_for_engine._phoenix_production_memory_factory = True
    survival._memory_for_engine = memory_for_engine

    def lookup_exact(self, source: str, target_language: str):
        if bool(getattr(self, "_phoenix_production_memory", False)):
            if not maturity._memory_is_mature(self):
                return None
        return old_exact(self, source, target_language)

    lookup_exact._phoenix_maturity_gate_v2 = True

    def lookup_similar(self, source: str, target_language: str, **kwargs):
        if bool(getattr(self, "_phoenix_production_memory", False)):
            if not maturity._memory_is_mature(self):
                return None
        return old_similar(self, source, target_language, **kwargs)

    lookup_similar._phoenix_maturity_gate_v2 = True

    memory_cls.lookup_exact = lookup_exact
    memory_cls.lookup_similar = lookup_similar

    # Report production maturity without changing deterministic zero-model rules.
    old_try = survival._try_exact_or_rule

    def try_exact_or_rule(engine, source: str, target: str):
        try:
            memory = survival._memory_for_engine(engine)
            maturity._report(engine, maturity._tracker_for_memory(memory).stats())
        except Exception:
            pass
        return old_try(engine, source, target)

    try_exact_or_rule._phoenix_maturity_reporting_v2 = True
    survival._try_exact_or_rule = try_exact_or_rule

    _INSTALLED = True
    print(
        "[Phoenix][学习成熟度] v2安全门已启用：只限制生产翻译记忆；"
        f"至少完成{maturity.minimum_books()}本PDF且累计"
        f"{maturity.minimum_verified_entries()}条已验证译文后才允许生产复用。",
        flush=True,
    )
