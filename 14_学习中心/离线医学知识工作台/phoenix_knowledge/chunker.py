from __future__ import annotations

import re


_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\.])\s+")


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _units(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    result: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= 900:
            result.append(paragraph)
            continue
        parts = [x.strip() for x in _SENTENCE_SPLIT.split(paragraph) if x.strip()]
        if len(parts) <= 1:
            parts = [paragraph[i : i + 800] for i in range(0, len(paragraph), 800)]
        result.extend(parts)
    return result


def chunk_text(text: str, max_chars: int = 1600, overlap_chars: int = 180) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    units = _units(text)
    chunks: list[str] = []
    current = ""

    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            for i in range(0, len(unit), max(1, max_chars - overlap_chars)):
                piece = unit[i : i + max_chars].strip()
                if piece:
                    chunks.append(piece)
            current = ""

        while len(current) > max_chars:
            chunks.append(current[:max_chars].strip())
            current = current[max_chars - overlap_chars :].strip()

    if current:
        chunks.append(current.strip())

    deduped: list[str] = []
    for chunk in chunks:
        if chunk and (not deduped or chunk != deduped[-1]):
            deduped.append(chunk)
    return deduped
