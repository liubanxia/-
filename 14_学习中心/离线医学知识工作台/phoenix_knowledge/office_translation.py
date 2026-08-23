from __future__ import annotations

import hashlib
import html
import json
import os
import posixpath
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from xml.etree import ElementTree

from .translation_models import MultiModelTranslationEngine
from .translator import TranslationResult


ProgressCallback = Callable[[int, int, str], None]
PauseCallback = Callable[[], bool]
PreviewCallback = Callable[[int, str, Path], None]

SUPPORTED_OFFICE_TRANSLATION_EXTENSIONS = {".pptx", ".docx"}
OFFICE_TRANSLATION_CONTRACT_VERSION = 2
OFFICE_OUTPUT_LAYOUT = "source_format"
OFFICE_SIZE_RATIO_DEFAULT = 1.25
OFFICE_SIZE_SLACK_BYTES = 1024 * 1024

_TEXT_NODE_RE = re.compile(
    r"(?P<open><(?P<tag>[A-Za-z_][\w.-]*:(?:t|v)|(?:t|v))\b[^>]*>)"
    r"(?P<body>.*?)"
    r"(?P<close></(?P=tag)\s*>)",
    re.DOTALL,
)
_LATIN_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.I)
_PURE_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9+./_-]{1,15}$")
_PPT_SLIDE_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")


class OfficeTranslationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfficeSegment:
    segment_id: str
    part: str
    node_index: int
    source: str
    leading: str = ""
    trailing: str = ""

    def rendered(self, translated: str) -> str:
        return self.leading + str(translated or "").strip() + self.trailing


@dataclass(frozen=True)
class OfficeUnit:
    number: int
    label: str
    segments: tuple[OfficeSegment, ...]


def _safe_filename(text: str, limit: int = 72) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "_", str(text or "")).strip(" ._")
    return (cleaned or "医学文档")[: max(12, int(limit))]


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _size_ratio_limit() -> float:
    raw = os.environ.get("PHOENIX_OFFICE_TRANSLATION_MAX_RATIO", "").strip()
    try:
        value = float(raw) if raw else OFFICE_SIZE_RATIO_DEFAULT
    except (TypeError, ValueError):
        value = OFFICE_SIZE_RATIO_DEFAULT
    return max(1.05, min(1.50, value))


def _is_chinese_target(target_language: str) -> bool:
    raw = str(target_language or "").strip()
    lower = raw.lower()
    return "中文" in raw or lower in {"chinese", "zh", "zh-cn", "zh-tw"}


def _is_english_target(target_language: str) -> bool:
    raw = str(target_language or "").strip().lower()
    return raw in {"英文", "english", "en", "en-us", "en-gb"}


def _should_translate(text: str, target_language: str) -> bool:
    value = str(text or "").strip()
    if not value or _URL_RE.fullmatch(value):
        return False
    if _is_chinese_target(target_language):
        if not _LATIN_RE.search(value):
            return False
        if _PURE_ACRONYM_RE.fullmatch(value):
            return False
        return True
    if _is_english_target(target_language):
        return bool(_CJK_RE.search(value))
    return bool(_LATIN_RE.search(value) or _CJK_RE.search(value))


def _split_spacing(value: str) -> tuple[str, str, str]:
    leading_match = re.match(r"^\s*", value)
    trailing_match = re.search(r"\s*$", value)
    leading = leading_match.group(0) if leading_match else ""
    trailing = trailing_match.group(0) if trailing_match else ""
    end = len(value) - len(trailing) if trailing else len(value)
    core = value[len(leading):end]
    return leading, core, trailing


def _candidate_parts(names: Iterable[str], suffix: str) -> list[str]:
    result: list[str] = []
    for name in names:
        if suffix == ".pptx":
            accepted = bool(
                _PPT_SLIDE_RE.match(name)
                or re.match(r"^ppt/notesSlides/notesSlide\d+\.xml$", name)
                or re.match(r"^ppt/charts/[^/]+\.xml$", name)
                or re.match(r"^ppt/diagrams/[^/]+\.xml$", name)
            )
        else:
            accepted = bool(
                name == "word/document.xml"
                or re.match(
                    r"^word/(?:header|footer|footnotes|endnotes|comments)\d*\.xml$",
                    name,
                )
                or re.match(r"^word/charts/[^/]+\.xml$", name)
                or re.match(r"^word/diagrams/[^/]+\.xml$", name)
            )
        if accepted:
            result.append(name)
    return result


def _natural_part_key(name: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    )


def _extract_segments(
    archive: zipfile.ZipFile,
    parts: list[str],
    target_language: str,
) -> dict[str, tuple[OfficeSegment, ...]]:
    by_part: dict[str, tuple[OfficeSegment, ...]] = {}
    serial = 0
    for part in sorted(parts, key=_natural_part_key):
        try:
            xml = archive.read(part).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            by_part[part] = ()
            continue
        segments: list[OfficeSegment] = []
        for node_index, match in enumerate(_TEXT_NODE_RE.finditer(xml)):
            body = match.group("body")
            if "<" in body or ">" in body:
                continue
            decoded = html.unescape(body)
            leading, core, trailing = _split_spacing(decoded)
            if not _should_translate(core, target_language):
                continue
            serial += 1
            segments.append(
                OfficeSegment(
                    segment_id=f"S{serial:06d}",
                    part=part,
                    node_index=node_index,
                    source=core,
                    leading=leading,
                    trailing=trailing,
                )
            )
        by_part[part] = tuple(segments)
    return by_part


def _rels_name(part: str) -> str:
    directory = posixpath.dirname(part)
    return posixpath.join(directory, "_rels", posixpath.basename(part) + ".rels")


def _related_candidate_parts(
    archive: zipfile.ZipFile,
    part: str,
    candidates: set[str],
) -> list[str]:
    result: list[str] = []
    queue = [part]
    visited = {part}
    names = set(archive.namelist())
    while queue:
        current = queue.pop(0)
        rels = _rels_name(current)
        if rels not in names:
            continue
        try:
            root = ElementTree.fromstring(archive.read(rels))
        except Exception:
            continue
        for relation in root:
            if str(relation.attrib.get("TargetMode", "")).lower() == "external":
                continue
            target = str(relation.attrib.get("Target", "") or "")
            if not target:
                continue
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(current), target)
            ).lstrip("/")
            if resolved not in candidates or resolved in visited:
                continue
            visited.add(resolved)
            result.append(resolved)
            queue.append(resolved)
    return result


def _ppt_units(
    archive: zipfile.ZipFile,
    parts: list[str],
    by_part: dict[str, tuple[OfficeSegment, ...]],
) -> list[OfficeUnit]:
    fallback_slides = sorted(
        (name for name in parts if _PPT_SLIDE_RE.match(name)),
        key=_natural_part_key,
    )
    slides: list[str] = []
    names = set(archive.namelist())
    if {
        "ppt/presentation.xml",
        "ppt/_rels/presentation.xml.rels",
    }.issubset(names):
        try:
            rel_root = ElementTree.fromstring(
                archive.read("ppt/_rels/presentation.xml.rels")
            )
            rels = {
                str(item.attrib.get("Id", "")): posixpath.normpath(
                    posixpath.join("ppt", str(item.attrib.get("Target", "")))
                ).lstrip("/")
                for item in rel_root
                if item.attrib.get("Id") and item.attrib.get("Target")
            }
            presentation = ElementTree.fromstring(
                archive.read("ppt/presentation.xml")
            )
            for item in presentation.iter():
                if not str(item.tag).endswith("}sldId"):
                    continue
                relationship_id = next(
                    (
                        value
                        for key, value in item.attrib.items()
                        if str(key).endswith("}id")
                    ),
                    "",
                )
                target = rels.get(str(relationship_id), "")
                if target in fallback_slides and target not in slides:
                    slides.append(target)
        except Exception:
            slides = []
    slides.extend(item for item in fallback_slides if item not in slides)
    if not slides:
        raise OfficeTranslationError("PPTX没有可读取的幻灯片XML。")
    candidates = set(parts)
    claimed: set[str] = set()
    unit_parts: list[list[str]] = []
    for slide in slides:
        related = _related_candidate_parts(archive, slide, candidates)
        selected = [slide]
        selected.extend(item for item in related if item not in claimed)
        claimed.update(selected)
        unit_parts.append(selected)

    remaining = [
        part
        for part in sorted(parts, key=_natural_part_key)
        if part not in claimed
    ]
    if remaining:
        unit_parts[-1].extend(remaining)

    units: list[OfficeUnit] = []
    for number, selected in enumerate(unit_parts, start=1):
        segments = tuple(
            segment
            for part in selected
            for segment in by_part.get(part, ())
        )
        units.append(OfficeUnit(number, f"幻灯片 {number}", segments))
    return units


def _docx_units(
    parts: list[str],
    by_part: dict[str, tuple[OfficeSegment, ...]],
    *,
    max_chars: int = 2600,
    max_segments: int = 24,
) -> list[OfficeUnit]:
    ordered = sorted(
        parts,
        key=lambda name: (0 if name == "word/document.xml" else 1, _natural_part_key(name)),
    )
    all_segments = [segment for part in ordered for segment in by_part.get(part, ())]
    if not all_segments:
        return [OfficeUnit(1, "论文段落组 1", ())]

    groups: list[list[OfficeSegment]] = []
    current: list[OfficeSegment] = []
    chars = 0
    for segment in all_segments:
        size = len(segment.source)
        if current and (
            len(current) >= max_segments or chars + size > max_chars
        ):
            groups.append(current)
            current = []
            chars = 0
        current.append(segment)
        chars += size
    if current:
        groups.append(current)
    return [
        OfficeUnit(index, f"论文段落组 {index}", tuple(group))
        for index, group in enumerate(groups, start=1)
    ]


def _segment_batches(
    segments: Iterable[OfficeSegment],
    *,
    max_chars: int = 2600,
    max_segments: int = 24,
) -> list[list[OfficeSegment]]:
    result: list[list[OfficeSegment]] = []
    current: list[OfficeSegment] = []
    chars = 0
    for segment in segments:
        size = len(segment.source)
        if current and (
            len(current) >= max_segments or chars + size > max_chars
        ):
            result.append(current)
            current = []
            chars = 0
        current.append(segment)
        chars += size
    if current:
        result.append(current)
    return result


def _replace_xml_nodes(
    raw: bytes,
    replacements: dict[int, str],
) -> bytes:
    xml = raw.decode("utf-8")
    node_index = -1

    def replace(match: re.Match[str]) -> str:
        nonlocal node_index
        node_index += 1
        value = replacements.get(node_index)
        if value is None:
            return match.group(0)
        escaped = html.escape(value, quote=False)
        return match.group("open") + escaped + match.group("close")

    return _TEXT_NODE_RE.sub(replace, xml).encode("utf-8")


def _media_names(names: Iterable[str], suffix: str) -> list[str]:
    prefix = "ppt/media/" if suffix == ".pptx" else "word/media/"
    return sorted(name for name in names if name.startswith(prefix) and not name.endswith("/"))


def validate_office_package(
    source: Path,
    output: Path,
    *,
    expected_replacements: dict[tuple[str, int], str] | None = None,
    max_ratio: float | None = None,
) -> dict:
    source = Path(source)
    output = Path(output)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_OFFICE_TRANSLATION_EXTENSIONS:
        raise OfficeTranslationError(f"不支持的Office翻译格式：{source.suffix}")
    if output.suffix.lower() != suffix:
        raise OfficeTranslationError("同格式翻译验收失败：输出扩展名与输入不一致。")
    if not source.is_file() or not output.is_file() or output.stat().st_size <= 0:
        raise OfficeTranslationError("同格式翻译验收失败：输入或输出文件不存在/为空。")

    with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(output) as output_zip:
        source_broken = source_zip.testzip()
        output_broken = output_zip.testzip()
        if source_broken:
            raise OfficeTranslationError(f"源Office压缩包损坏：{source_broken}")
        if output_broken:
            raise OfficeTranslationError(f"译文Office压缩包损坏：{output_broken}")
        source_names = source_zip.namelist()
        output_names = output_zip.namelist()
        if Counter(source_names) != Counter(output_names):
            raise OfficeTranslationError("译文Office成员集合与源文件不一致。")

        if suffix == ".docx" and "word/document.xml" not in output_names:
            raise OfficeTranslationError("DOCX译文缺少 word/document.xml。")
        if suffix == ".pptx" and not any(_PPT_SLIDE_RE.match(x) for x in output_names):
            raise OfficeTranslationError("PPTX译文缺少幻灯片XML。")

        media = _media_names(source_names, suffix)
        for name in media:
            if _sha256_bytes(source_zip.read(name)) != _sha256_bytes(output_zip.read(name)):
                raise OfficeTranslationError(f"原媒体资源发生变化：{name}")

        text_parts = set(_candidate_parts(source_names, suffix))
        preserved_members = 0
        for name in source_names:
            if name.endswith("/") or name in text_parts:
                continue
            if _sha256_bytes(source_zip.read(name)) != _sha256_bytes(output_zip.read(name)):
                raise OfficeTranslationError(
                    f"原版式/关系/非文字资源发生变化：{name}"
                )
            preserved_members += 1

        if expected_replacements:
            by_part: dict[str, dict[int, str]] = {}
            for (part, node_index), value in expected_replacements.items():
                by_part.setdefault(part, {})[int(node_index)] = str(value)
            for part, expected in by_part.items():
                try:
                    xml = output_zip.read(part).decode("utf-8")
                except Exception as exc:
                    raise OfficeTranslationError(f"无法复核译文XML：{part}: {exc}") from exc
                actual = {
                    index: html.unescape(match.group("body"))
                    for index, match in enumerate(_TEXT_NODE_RE.finditer(xml))
                }
                for node_index, value in expected.items():
                    if actual.get(node_index) != value:
                        raise OfficeTranslationError(
                            f"译文没有稳定写入 {part} 的文字节点 {node_index}。"
                        )

    source_bytes = int(source.stat().st_size)
    output_bytes = int(output.stat().st_size)
    ratio_limit = float(max_ratio if max_ratio is not None else _size_ratio_limit())
    allowed_bytes = max(
        int(source_bytes * ratio_limit),
        source_bytes + OFFICE_SIZE_SLACK_BYTES,
    )
    if output_bytes > allowed_bytes:
        raise OfficeTranslationError(
            "同格式译文体积超过发布预算："
            f"源文件={source_bytes} bytes，译文={output_bytes} bytes，"
            f"预算≤{ratio_limit:.2f}×（小文件另含1MB固定余量）。"
        )
    return {
        "passed": True,
        "format": suffix.lstrip("."),
        "source": str(source),
        "output": str(output),
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "size_ratio": output_bytes / max(source_bytes, 1),
        "max_ratio": ratio_limit,
        "allowed_bytes": allowed_bytes,
        "member_count": len(source_names),
        "media_count": len(media),
        "media_preserved": True,
        "non_text_members_preserved": preserved_members,
    }


class OfficeDocumentTranslator:
    """Translate PPTX/DOCX text in place while preserving the OOXML package.

    Media, relationships, layout definitions and every non-text member are
    copied byte-for-byte. Translation is checkpointed at a slide/paragraph-unit
    boundary and a preview callback fires immediately after each unit.
    """

    _phoenix_office_translation_contract = OFFICE_TRANSLATION_CONTRACT_VERSION

    def __init__(
        self,
        paths,
        engine: MultiModelTranslationEngine,
    ):
        self.paths = paths
        self.engine = engine
        self.output_root = Path(paths.evidence_root) / "同格式医学翻译"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _task_paths(
        self,
        source: Path,
        digest: str,
        target_language: str,
    ) -> tuple[Path, Path, Path, Path, Path]:
        name = _safe_filename(source.stem, 48)
        language = _safe_filename(target_language, 16)
        root = self.output_root / f"{name}_{digest[:10]}_{language}"
        units = root / "units"
        previews = root / "previews"
        checkpoint = root / "checkpoint.json"
        final = root / f"{_safe_filename(source.stem, 72)}_{language}译本{source.suffix.lower()}"
        units.mkdir(parents=True, exist_ok=True)
        previews.mkdir(parents=True, exist_ok=True)
        return root, units, previews, checkpoint, final

    @staticmethod
    def _clear_checkpoints(units_root: Path, previews_root: Path, checkpoint: Path) -> None:
        for root in (Path(units_root), Path(previews_root)):
            if not root.is_dir():
                continue
            for path in root.iterdir():
                if path.is_file():
                    path.unlink(missing_ok=True)
        Path(checkpoint).unlink(missing_ok=True)

    @staticmethod
    def _resolve_resume_start_unit(
        state: dict,
        requested_start_unit: int,
        *,
        force_restart: bool = False,
    ) -> int:
        requested = max(1, int(requested_start_unit))
        if force_restart or not state:
            return requested
        try:
            existing = int(state.get("start_page", requested))
        except (TypeError, ValueError):
            return requested
        return max(1, existing)

    def _active_backends(self, target_language: str) -> tuple[str, ...]:
        try:
            active = self.engine.active_backends(target_language, "smart2")
        except AttributeError:
            active = [object()]
        if not active:
            raise RuntimeError(
                "医学精译质量模型未就绪；PPTX/DOCX正式译文不允许降级到Smart1。"
            )
        formal_names = getattr(self.engine, "formal_backend_names", None)
        if callable(formal_names):
            names = tuple(formal_names(target_language))
            if names:
                return names
        try:
            legacy = {"marian_en_zh", "nllb_600m_en_zh"}
            names = tuple(
                str(name) for name in self.engine.available_backends()
                if str(name) not in legacy
            )
            return names or ("medical_translation",)
        except AttributeError:
            return ("medical_translation",)

    def _translate_sources(
        self,
        sources: list[str],
        target_language: str,
    ) -> list[object]:
        batch = getattr(self.engine, "translate_segments", None)
        if callable(batch):
            return list(
                batch(
                    sources,
                    target_language,
                    smart_level="smart2",
                )
            )
        return [
            self.engine.translate(
                source,
                target_language,
                smart_level="smart2",
            )
            for source in sources
        ]

    @staticmethod
    def _decision_audit(segment: OfficeSegment, decision) -> dict:
        quality = getattr(decision, "quality", None)
        return {
            "id": segment.segment_id,
            "source": segment.source,
            "translated": str(getattr(decision, "text", segment.source) or segment.source),
            "backend": str(getattr(decision, "backend", "unknown")),
            "quality_ok": bool(getattr(quality, "ok", False)),
            "quality_score": round(float(getattr(quality, "score", 0.0) or 0.0), 4),
            "reasons": list(getattr(quality, "reasons", ()) or ()),
            "needs_review": bool(getattr(decision, "needs_review", True)),
        }

    @staticmethod
    def _load_completed_unit(
        path: Path,
        unit: OfficeUnit,
        *,
        source_sha256: str,
        target_language: str,
    ) -> tuple[dict[str, str], int, list[dict]] | None:
        payload = _read_json(path)
        if not payload or payload.get("source_sha256") != source_sha256:
            return None
        if payload.get("target_language") != target_language:
            return None
        rows = payload.get("translations")
        if not isinstance(rows, list):
            return None
        expected = {segment.segment_id: segment.source for segment in unit.segments}
        translated: dict[str, str] = {}
        audits: list[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            segment_id = str(row.get("id", ""))
            if segment_id not in expected or row.get("source") != expected[segment_id]:
                return None
            translated[segment_id] = str(row.get("translated", ""))
            audits.append(dict(row))
        if set(translated) != set(expected):
            return None
        warnings = int(payload.get("warning_count", 0) or 0)
        return translated, warnings, audits

    @staticmethod
    def _write_preview(path: Path, unit: OfficeUnit, translations: dict[str, str]) -> str:
        lines = [f"===== {unit.label} =====", ""]
        if unit.segments:
            lines.extend(
                translations.get(segment.segment_id, segment.source).strip()
                for segment in unit.segments
            )
        else:
            lines.append("[本单元没有需要翻译的文字；原版式和媒体已保留。]")
        text = "\n".join(lines).rstrip() + "\n"
        path.write_text(text, encoding="utf-8")
        return text

    @staticmethod
    def _emit_preview(
        callback: PreviewCallback | None,
        unit: OfficeUnit,
        text: str,
        path: Path,
    ) -> None:
        if callback is None:
            return
        try:
            callback(unit.number, text, path)
        except Exception:
            pass

    def _build_output(
        self,
        source: Path,
        final_output: Path,
        replacements: dict[tuple[str, int], str],
    ) -> dict:
        final_output.parent.mkdir(parents=True, exist_ok=True)
        temp = final_output.with_name(
            f".{final_output.stem}.new{final_output.suffix}"
        )
        backup = final_output.with_name(
            f".{final_output.stem}.old{final_output.suffix}"
        )
        publish_marker = final_output.with_name(
            f".{final_output.stem}.publish.json"
        )
        temp.unlink(missing_ok=True)
        if publish_marker.is_file():
            interrupted = _read_json(publish_marker)
            had_previous = bool(interrupted.get("had_previous", False))
            if had_previous and backup.is_file():
                final_output.unlink(missing_ok=True)
                os.replace(backup, final_output)
            elif not had_previous:
                final_output.unlink(missing_ok=True)
            publish_marker.unlink(missing_ok=True)
        elif backup.is_file() and not final_output.is_file():
            os.replace(backup, final_output)
        else:
            backup.unlink(missing_ok=True)
        by_part: dict[str, dict[int, str]] = {}
        for (part, node_index), translated in replacements.items():
            by_part.setdefault(part, {})[int(node_index)] = translated

        try:
            with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(
                temp,
                "w",
                allowZip64=True,
            ) as output_zip:
                for info in source_zip.infolist():
                    raw = source_zip.read(info)
                    if info.filename in by_part:
                        raw = _replace_xml_nodes(raw, by_part[info.filename])
                    output_zip.writestr(info, raw)

            report = validate_office_package(
                source,
                temp,
                expected_replacements=replacements,
            )
            had_previous = final_output.is_file()
            _write_json(
                publish_marker,
                {"had_previous": had_previous, "output": str(final_output)},
            )
            if had_previous:
                os.replace(final_output, backup)
            try:
                os.replace(temp, final_output)
                report = validate_office_package(
                    source,
                    final_output,
                    expected_replacements=replacements,
                )
            except Exception:
                final_output.unlink(missing_ok=True)
                if had_previous and backup.is_file():
                    os.replace(backup, final_output)
                publish_marker.unlink(missing_ok=True)
                raise
            publish_marker.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)
            return report
        finally:
            temp.unlink(missing_ok=True)
            if backup.is_file() and not final_output.is_file():
                os.replace(backup, final_output)

    def translate_document(
        self,
        source_path: Path,
        *,
        start_page: int = 1,
        target_language: str = "中文",
        progress: ProgressCallback | None = None,
        page_preview: PreviewCallback | None = None,
        force_restart: bool = False,
        retry_warning_pages: bool = False,
        smart_level: str = "smart2",
        medical_quality_required: bool = True,
        output_layout: str | None = None,
        export_format: str | None = None,
        part_pages: int = 0,
        should_pause: PauseCallback | None = None,
    ) -> TranslationResult:
        del output_layout, export_format, part_pages
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_OFFICE_TRANSLATION_EXTENSIONS:
            raise ValueError(f"仅支持PPTX/DOCX同格式翻译：{source}")
        # PPTX/DOCX are formal same-format deliverables. Legacy callers may
        # still pass these arguments, but neither can downgrade the route.
        del medical_quality_required, smart_level
        smart_level = "smart2"
        available_backends = self._active_backends(target_language)
        digest = _sha256_file(source)
        root, units_root, previews_root, checkpoint, final_output = self._task_paths(
            source,
            digest,
            target_language,
        )
        if force_restart:
            self._clear_checkpoints(units_root, previews_root, checkpoint)

        previous = _read_json(checkpoint)
        if previous:
            if previous.get("source_sha256") != digest:
                raise OfficeTranslationError("源文档内容已变化，请使用重新翻译模式。")
            if previous.get("target_language") != target_language:
                raise OfficeTranslationError("目标语言已变化，请使用重新翻译模式。")

        with zipfile.ZipFile(source) as archive:
            broken = archive.testzip()
            if broken:
                raise OfficeTranslationError(f"源Office文件损坏：{broken}")
            parts = _candidate_parts(archive.namelist(), suffix)
            by_part = _extract_segments(archive, parts, target_language)
            units = (
                _ppt_units(archive, parts, by_part)
                if suffix == ".pptx"
                else _docx_units(parts, by_part)
            )
            image_count = len(_media_names(archive.namelist(), suffix))

        total_units = len(units)
        start_page = self._resolve_resume_start_unit(
            previous,
            start_page,
            force_restart=force_restart,
        )
        if start_page > total_units:
            raise ValueError(
                f"开始单元 {start_page} 超出总单元数 {total_units}。"
            )
        selected_total = total_units - start_page + 1
        state = {
            "source_path": str(source),
            "source_sha256": digest,
            "target_language": target_language,
            "format": suffix.lstrip("."),
            "start_page": start_page,
            "total_units": total_units,
            "status": "running",
            "smart_level": "smart2",
            "last_completed_unit": int(previous.get("last_completed_unit", start_page - 1) or start_page - 1),
            "warning_units": int(previous.get("warning_units", 0) or 0),
        }
        _write_json(checkpoint, state)

        replacements: dict[tuple[str, int], str] = {}
        translation_cache: dict[str, str] = {}
        pages_done = 0
        resumed_pages = 0
        warning_pages = 0

        try:
            for unit in units:
                if unit.number < start_page:
                    continue
                if should_pause and should_pause():
                    state["status"] = "paused"
                    state["warning_units"] = warning_pages
                    _write_json(checkpoint, state)
                    return TranslationResult(
                        output_path=final_output,
                        source_path=source,
                        start_page=start_page,
                        total_pages=total_units,
                        target_language=target_language,
                        pages_done=pages_done,
                        resumed_pages=resumed_pages,
                        warning_pages=warning_pages,
                        available_backends=available_backends,
                        output_paths=(),
                        image_count=image_count,
                        paused=True,
                        smart_level="smart2",
                        output_layout=OFFICE_OUTPUT_LAYOUT,
                        export_format=suffix.lstrip("."),
                        part_pages=0,
                    )

                unit_file = units_root / f"{unit.number:06d}.json"
                preview_file = previews_root / f"{unit.number:06d}.txt"
                completed = self._load_completed_unit(
                    unit_file,
                    unit,
                    source_sha256=digest,
                    target_language=target_language,
                )
                # A formal Office delivery may never reuse a warning unit.
                # It is automatically retried on the next run instead of
                # publishing source text or asking the reader to review it.
                if completed is not None and completed[1] <= 0:
                    translated, unit_warnings, _audits = completed
                    for segment in unit.segments:
                        value = translated[segment.segment_id]
                        replacements[(segment.part, segment.node_index)] = segment.rendered(value)
                        if value and not unit_warnings:
                            translation_cache.setdefault(segment.source, value)
                    pages_done += 1
                    resumed_pages += 1
                    if unit_warnings:
                        warning_pages += 1
                    preview_text = self._write_preview(preview_file, unit, translated)
                    self._emit_preview(page_preview, unit, preview_text, preview_file)
                    if progress:
                        progress(
                            pages_done,
                            selected_total,
                            f"{unit.label} 已翻译，直接续用",
                        )
                    continue

                if progress:
                    progress(
                        pages_done,
                        selected_total,
                        f"正在医学精译 {unit.label}；完成后立即显示……",
                    )
                translated_by_id: dict[str, str] = {}
                audits: list[dict] = []
                warning_count = 0

                for batch in _segment_batches(unit.segments):
                    pending: list[OfficeSegment] = []
                    representative: dict[str, OfficeSegment] = {}
                    for segment in batch:
                        cached = translation_cache.get(segment.source)
                        if cached is not None:
                            translated_by_id[segment.segment_id] = cached
                            audits.append({
                                "id": segment.segment_id,
                                "source": segment.source,
                                "translated": cached,
                                "backend": "document_cache",
                                "quality_ok": True,
                                "quality_score": 1.0,
                                "reasons": [],
                                "needs_review": False,
                            })
                        elif segment.source in representative:
                            continue
                        else:
                            representative[segment.source] = segment
                            pending.append(segment)

                    if pending:
                        try:
                            decisions = self._translate_sources(
                                [segment.source for segment in pending],
                                target_language,
                            )
                            if len(decisions) != len(pending):
                                raise OfficeTranslationError(
                                    "批量翻译返回数量与输入片段数不一致。"
                                )
                            decision_by_source = dict(
                                zip((segment.source for segment in pending), decisions)
                            )
                        except Exception as batch_exc:
                            # A malformed/failed batch should not throw away an
                            # entire slide or paragraph group. Retry only those
                            # pending rows individually through the same Smart2
                            # no-reasoning translation profile.
                            decision_by_source = {}
                            for segment in pending:
                                try:
                                    decision = self.engine.translate(
                                        segment.source,
                                        target_language,
                                        smart_level="smart2",
                                    )
                                    row = self._decision_audit(segment, decision)
                                    value = str(
                                        row["translated"] or segment.source
                                    ).strip()
                                    translated_by_id[segment.segment_id] = value
                                    audits.append(row)
                                    if row["needs_review"]:
                                        warning_count += 1
                                    else:
                                        translation_cache[segment.source] = value
                                except Exception as item_exc:
                                    warning_count += 1
                                    translated_by_id[segment.segment_id] = segment.source
                                    audits.append({
                                        "id": segment.segment_id,
                                        "source": segment.source,
                                        "translated": segment.source,
                                        "backend": "failed_preserve_source",
                                        "quality_ok": False,
                                        "quality_score": 0.0,
                                        "reasons": [
                                            f"batch={type(batch_exc).__name__}: {batch_exc}",
                                            f"item={type(item_exc).__name__}: {item_exc}",
                                        ],
                                        "needs_review": True,
                                    })
                        else:
                            for segment in pending:
                                decision = decision_by_source[segment.source]
                                row = self._decision_audit(segment, decision)
                                value = str(row["translated"] or segment.source).strip()
                                translated_by_id[segment.segment_id] = value
                                audits.append(row)
                                if row["needs_review"]:
                                    warning_count += 1
                                else:
                                    translation_cache[segment.source] = value

                    for segment in batch:
                        if segment.segment_id in translated_by_id:
                            continue
                        primary = representative.get(segment.source)
                        if primary is not None and primary.segment_id in translated_by_id:
                            value = translated_by_id[primary.segment_id]
                            translated_by_id[segment.segment_id] = value
                            primary_row = next(
                                (
                                    row for row in reversed(audits)
                                    if row.get("id") == primary.segment_id
                                ),
                                {},
                            )
                            accepted = bool(primary_row.get("quality_ok", False)) and not bool(
                                primary_row.get("needs_review", True)
                            )
                            if accepted:
                                translation_cache.setdefault(segment.source, value)
                            audits.append({
                                "id": segment.segment_id,
                                "source": segment.source,
                                "translated": value,
                                "backend": (
                                    "unit_deduplicated"
                                    if accepted
                                    else "unit_deduplicated_needs_review"
                                ),
                                "quality_ok": accepted,
                                "quality_score": float(
                                    primary_row.get("quality_score", 0.0) or 0.0
                                ),
                                "reasons": list(primary_row.get("reasons") or ()),
                                "needs_review": not accepted,
                            })

                for segment in unit.segments:
                    value = translated_by_id.get(segment.segment_id, segment.source)
                    replacements[(segment.part, segment.node_index)] = segment.rendered(value)

                _write_json(
                    unit_file,
                    {
                        "source_sha256": digest,
                        "target_language": target_language,
                        "unit": unit.number,
                        "label": unit.label,
                        "warning_count": warning_count,
                        "translations": audits,
                    },
                )
                preview_text = self._write_preview(
                    preview_file,
                    unit,
                    translated_by_id,
                )
                self._emit_preview(page_preview, unit, preview_text, preview_file)
                pages_done += 1
                if warning_count:
                    warning_pages += 1
                state["last_completed_unit"] = unit.number
                state["warning_units"] = warning_pages
                _write_json(checkpoint, state)
                if progress:
                    progress(
                        pages_done,
                        selected_total,
                        f"已完成 {unit.label} | 待复核单元={warning_pages}",
                    )

                if should_pause and should_pause():
                    state["status"] = "paused"
                    _write_json(checkpoint, state)
                    return TranslationResult(
                        output_path=final_output,
                        source_path=source,
                        start_page=start_page,
                        total_pages=total_units,
                        target_language=target_language,
                        pages_done=pages_done,
                        resumed_pages=resumed_pages,
                        warning_pages=warning_pages,
                        available_backends=available_backends,
                        output_paths=(),
                        image_count=image_count,
                        paused=True,
                        smart_level="smart2",
                        output_layout=OFFICE_OUTPUT_LAYOUT,
                        export_format=suffix.lstrip("."),
                        part_pages=0,
                    )

            audited_segments = 0
            accepted_segments = 0
            for unit in units:
                if unit.number < start_page:
                    continue
                payload = _read_json(units_root / f"{unit.number:06d}.json")
                for row in payload.get("translations") or ():
                    if not isinstance(row, dict):
                        continue
                    audited_segments += 1
                    if (
                        bool(row.get("quality_ok", False))
                        and not bool(row.get("needs_review", True))
                        and str(row.get("backend", "")) != "failed_preserve_source"
                    ):
                        accepted_segments += 1
            if audited_segments and accepted_segments != audited_segments:
                raise OfficeTranslationError(
                    "存在未通过医学质量校验的文字；Phoenix已保留逐单元"
                    "checkpoint并拒绝发布不合格Office成品。检查模型/API连接后"
                    "再次开始即可自动重译失败单元，无需人工修改。"
                )

            report = self._build_output(source, final_output, replacements)
            report.update({
                "unit_count": total_units,
                "translated_units": selected_total,
                "warning_units": warning_pages,
                "source_sha256": digest,
                "output_sha256": _sha256_file(final_output),
            })
            _write_json(root / "Office翻译完整性报告.json", report)
            state.update({
                "status": "completed",
                "last_completed_unit": total_units,
                "warning_units": warning_pages,
                "output_path": str(final_output),
                "size_ratio": report["size_ratio"],
            })
            _write_json(checkpoint, state)
            if progress:
                progress(
                    selected_total,
                    selected_total,
                    f"{suffix.lstrip('.').upper()} 同格式医学翻译完成。",
                )
            return TranslationResult(
                output_path=final_output,
                source_path=source,
                start_page=start_page,
                total_pages=total_units,
                target_language=target_language,
                pages_done=pages_done,
                resumed_pages=resumed_pages,
                warning_pages=warning_pages,
                available_backends=available_backends,
                output_paths=(final_output,),
                image_count=image_count,
                paused=False,
                smart_level="smart2",
                output_layout=OFFICE_OUTPUT_LAYOUT,
                export_format=suffix.lstrip("."),
                part_pages=0,
            )
        except Exception as exc:
            state["status"] = "failed"
            state["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(checkpoint, state)
            raise
        finally:
            try:
                self.engine.unload()
            except Exception:
                pass
