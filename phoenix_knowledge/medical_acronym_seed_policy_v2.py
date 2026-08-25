from __future__ import annotations

"""Keep the curated radiology resolver deterministic while expanding vocabulary.

medical_terminology_core contains broader cross-specialty senses (for example
ADC can also mean adenocarcinoma). Those senses belong in the general
translation context, but must not turn an already-curated one-sense radiology
acronym into an LLM disambiguation call. Existing resolver entries therefore
keep their original senses; the terminology core may still add entirely new
acronyms such as LGE/ECV.
"""

_BASELINE: dict[str, tuple[tuple[str, str], ...]] | None = None
_INSTALLED = False


def capture() -> None:
    global _BASELINE
    if _BASELINE is not None:
        return
    from . import medical_acronyms

    _BASELINE = {
        str(key): tuple(tuple(row) for row in rows)
        for key, rows in medical_acronyms.RADIOLOGY_SEED.items()
    }


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if _BASELINE is None:
        raise RuntimeError("medical acronym seed baseline was not captured before core expansion")

    from . import medical_acronyms

    # Restore only keys that existed in the curated resolver. New terminology
    # entries added by the core remain available to the resolver.
    for key, rows in _BASELINE.items():
        medical_acronyms.RADIOLOGY_SEED[key] = rows

    _INSTALLED = True
