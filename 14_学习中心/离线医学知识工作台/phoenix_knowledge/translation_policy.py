from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationPolicy:
    """Runtime policy for balancing quality and token cost.

    Online LLMs should improve quality, not receive an entire book blindly.
    """

    online_enabled: bool = True
    batch_pages: int = 8
    max_input_chars: int = 12000
    max_output_chars: int = 8000
    max_retry_pages: int = 3
    cache_pages: bool = True
    translate_images: bool = True

    def mode(self) -> str:
        return "online_quality" if self.online_enabled else "offline_fallback"

    def should_send_to_llm(self, confidence: float) -> bool:
        return self.online_enabled and confidence < 0.92

    def trim_context(self, text: str) -> str:
        text = str(text or "")
        if len(text) <= self.max_input_chars:
            return text
        return text[: self.max_input_chars]


def default_translation_policy() -> TranslationPolicy:
    return TranslationPolicy()
