from __future__ import annotations

"""Portable console text configuration for Phoenix runtime logs."""

import sys

_CONFIGURED = False


def configure_console_text() -> None:
    """Prevent locale-specific stdout/stderr encoders from crashing startup.

    Some Windows installations expose cp1252/cp936 streams to Python subprocesses.
    Phoenix emits Chinese diagnostics during bootstrap, so a narrow encoder can
    raise UnicodeEncodeError before the application is even constructed. UTF-8
    with replacement fallback keeps diagnostics non-fatal across locales.
    """

    global _CONFIGURED
    if _CONFIGURED:
        return

    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            try:
                reconfigure(errors="backslashreplace")
            except Exception:
                pass

    _CONFIGURED = True
