from __future__ import annotations

import hashlib
import json
import posixpath
import re
import shutil
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from xml.etree import ElementTree as ET

from .chunker import chunk_text
from .config import WorkbenchPaths
from .db import KnowledgeDB
from .ingest import IngestResult, LibraryIngestor
from .pdf_assets import PDFAssetStore


ProgressCallback = Callable[[int, int, str], None]
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".txt", ".md"}

_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

_NS = {
    "rel": _REL_NS,
    "a": _A_NS,
    "w": _W_NS,
    "r": _R_NS,
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            data = fh.read(chunk_size)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def _safe_member(name: str) -> str:
    return str(PurePosixPath(name.lstrip("/")))


def _rels_path(part_name: str) -> str:
    part = PurePosixPath(part_name)
    return str(part.parent / "_rels" / f"{part.name}.rels")


def _resolve_target(part_name: str, target: str) -> str:
    if target.startswith("/"):
        return _safe_member(target)
    return posixpath.normpath(posixpath.join(posixpath.dirname(part_name), target))


def _xml_root(payload: bytes):
    return ET.fromstring(payload)


def _texts_from_xml(payload: bytes) -> list[str]:
    root = _xml_root(payload)
    result: list[str] = []
    for node in root.iter():
        tag = str(node.tag)
        if tag.endswith("}t") or tag == "t":
            text = (node.text or "").strip()
            if text:
                result.append(text)
    return result


def _read_relationships(zf: zipfile.ZipFile, part_name: str) -> dict[str, dict[str, str]]:
    rel_name = _rels_path(part_name)
    if rel_name not in zf.namelist():
        return {}
    try:
        root = _xml_root(zf.read(rel_name))
    except Exception:
        return {}
    result: dict[str, dict[str, str]] = {}
    for rel in root:
        rid = rel.attrib.get("Id", "")
        if not rid:
            continue
        target = rel.attrib.get("Target", "")
        rel_type = rel.attrib.get("Type", "")
        if not target:
            continue
        result[rid] = {
            "type": rel_type,
            "target": _resolve_target(part_name, target),
        }
    return result


def _copy_zip_member(zf: zipfile.ZipFile, member: str, target: Path) -> bool:
    member = _safe_member(member)
    if member not in zf.namelist():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(zf.read(member))
    return True


@dataclass(frozen=True)
class ParsedUnit:
    number: int
    label: str
    text: str
    image_members: tuple[str, ...] = ()


class MultiDocumentIngestor:
    """Unified offline ingest for PDF/PPTX/DOCX/TXT/Markdown.

    The SQLite schema still calls the ordinal field ``page`` for backward
    compatibility. PPTX uses the real slide number. DOCX/TXT/MD use stable
    document units and include a human-readable unit marker in every chunk.
    """

    def __init__(self, db: KnowledgeDB, paths: WorkbenchPaths):
        self.db = db
        self.paths = paths
        self.pdf = LibraryIngestor(db, paths)
        self.assets = PDFAssetStore(paths.runtime_root)

    def supported(self, source: Path) -> bool:
        return Path(source).suffix.lower() in SUPPORTED_EXTENSIONS

    def _library_copy(self, source: Path) -> Path:
        source = Path(source).resolve()
        target = self.paths.source_root / source.name
        if target.resolve() == source:
            return source
        if target.exists():
            try:
                if sha256_file(target) == sha256_file(source):
                    return target
            except OSError:
                pass
            stem, suffix = source.stem, source.suffix
            counter = 2
            while target.exists():
                target = self.paths.source_root / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(source, target)
        return target

    @staticmethod
    def _pptx_units(source: Path) -> list[ParsedUnit]:
        units: list[ParsedUnit] = []
        with zipfile.ZipFile(source) as zf:
            slide_names = [
                name
                for name in zf.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
            ]
            slide_names.sort(key=lambda value: int(re.search(r"(\d+)", Path(value).stem).group(1)))
            for index, slide_name in enumerate(slide_names, start=1):
                text_parts = _texts_from_xml(zf.read(slide_name))
                rels = _read_relationships(zf, slide_name)
                notes_parts: list[str] = []
                image_members: list[str] = []
                for rel in rels.values():
                    rel_type = rel["type"].lower()
                    target = rel["target"]
                    if rel_type.endswith("/notesslide") and target in zf.namelist():
                        try:
                            notes_parts.extend(_texts_from_xml(zf.read(target)))
                        except Exception:
                            pass
                    elif rel_type.endswith("/image") and target in zf.namelist():
                        if target not in image_members:
                            image_members.append(target)

                lines = [f"[PPTX 幻灯片 {index}]"]
                if text_parts:
                    lines.extend(text_parts)
                else:
                    lines.append("[本张幻灯片未提取到文字]")
                if notes_parts:
                    lines.extend(["", "[演讲者备注]", *notes_parts])
                if image_members:
                    lines.extend(["", f"[关联图片 {len(image_members)} 张]"])
                units.append(
                    ParsedUnit(
                        number=index,
                        label=f"幻灯片{index}",
                        text="\n".join(lines).strip(),
                        image_members=tuple(image_members),
                    )
                )
        return units

    @staticmethod
    def _docx_blocks(source: Path):
        with zipfile.ZipFile(source) as zf:
            if "word/document.xml" not in zf.namelist():
                raise RuntimeError("DOCX缺少 word/document.xml")
            root = _xml_root(zf.read("word/document.xml"))
            rels = _read_relationships(zf, "word/document.xml")
            body = root.find(f".//{{{_W_NS}}}body")
            if body is None:
                return [], zf.namelist()

            blocks: list[tuple[str, list[str], bool]] = []
            for child in list(body):
                texts = []
                image_members: list[str] = []
                page_break = False
                for node in child.iter():
                    tag = str(node.tag)
                    if tag.endswith("}t"):
                        value = (node.text or "").strip()
                        if value:
                            texts.append(value)
                    elif tag.endswith("}br") and node.attrib.get(f"{{{_W_NS}}}type") == "page":
                        page_break = True
                    elif tag.endswith("}lastRenderedPageBreak"):
                        page_break = True
                    elif tag.endswith("}blip"):
                        rid = node.attrib.get(f"{{{_R_NS}}}embed", "")
                        rel = rels.get(rid)
                        if rel and rel["type"].lower().endswith("/image"):
                            target = rel["target"]
                            if target in zf.namelist() and target not in image_members:
                                image_members.append(target)
                text = " ".join(texts).strip()
                if text or image_members or page_break:
                    blocks.append((text, image_members, page_break))
            return blocks, zf.namelist()

    @classmethod
    def _docx_units(cls, source: Path) -> list[ParsedUnit]:
        blocks, _names = cls._docx_blocks(source)
        if not blocks:
            return [ParsedUnit(1, "段落组1", "[DOCX未提取到正文文字]")]

        units: list[ParsedUnit] = []
        current_text: list[str] = []
        current_images: list[str] = []
        char_count = 0
        block_count = 0

        def flush() -> None:
            nonlocal current_text, current_images, char_count, block_count
            if not current_text and not current_images:
                return
            number = len(units) + 1
            lines = [f"[DOCX 文档单元 {number}]"]
            lines.extend(current_text or ["[本单元无文字]"])
            if current_images:
                lines.extend(["", f"[关联图片 {len(current_images)} 张]"])
            units.append(
                ParsedUnit(
                    number=number,
                    label=f"文档单元{number}",
                    text="\n".join(lines).strip(),
                    image_members=tuple(dict.fromkeys(current_images)),
                )
            )
            current_text = []
            current_images = []
            char_count = 0
            block_count = 0

        for text, images, page_break in blocks:
            if text:
                current_text.append(text)
                char_count += len(text)
                block_count += 1
            current_images.extend(images)
            if page_break or char_count >= 6000 or block_count >= 24:
                flush()
        flush()
        return units or [ParsedUnit(1, "段落组1", "[DOCX未提取到正文文字]")]

    @staticmethod
    def _text_units(source: Path) -> list[ParsedUnit]:
        raw = Path(source).read_text(encoding="utf-8", errors="replace")
        parts = chunk_text(raw, max_chars=6000, overlap_chars=0) or [raw]
        suffix = source.suffix.lower()
        kind = "Markdown" if suffix == ".md" else "TXT"
        return [
            ParsedUnit(
                number=index,
                label=f"文本单元{index}",
                text=f"[{kind} 文本单元 {index}]\n{part.strip()}".strip(),
            )
            for index, part in enumerate(parts, start=1)
        ]

    def _write_asset_manifest(self, source: Path, units: list[ParsedUnit]) -> int:
        source = Path(source)
        if source.suffix.lower() not in {".pptx", ".docx"}:
            return 0

        doc_root = self.assets.document_root(source)
        if doc_root.exists():
            shutil.rmtree(doc_root, ignore_errors=True)
        doc_root.mkdir(parents=True, exist_ok=True)

        pages: dict[str, list[dict]] = {}
        image_count = 0
        with zipfile.ZipFile(source) as zf:
            for unit in units:
                items: list[dict] = []
                page_dir = doc_root / f"page_{unit.number:06d}"
                for image_index, member in enumerate(unit.image_members, start=1):
                    suffix = Path(member).suffix.lower() or ".bin"
                    target = page_dir / f"image_{image_index:03d}{suffix}"
                    if not _copy_zip_member(zf, member, target):
                        continue
                    items.append(
                        {
                            "path": target.relative_to(doc_root).as_posix(),
                            "width": 0,
                            "height": 0,
                            "source_member": member,
                        }
                    )
                    image_count += 1
                if items:
                    pages[str(unit.number)] = items

        manifest = {
            "source_path": str(source.resolve()),
            "source_type": source.suffix.lower().lstrip("."),
            "page_count": len(units),
            "image_count": image_count,
            "pages": pages,
        }
        manifest_path = self.assets.manifest_path(source)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
        return image_count

    def _ingest_units(
        self,
        source: Path,
        units: list[ParsedUnit],
        *,
        copy_into_library: bool,
        progress: ProgressCallback | None,
        stop_event: threading.Event | None,
    ) -> IngestResult:
        source = Path(source).resolve()
        stored = self._library_copy(source) if copy_into_library else source
        if stored != source:
            if stored.suffix.lower() == ".pptx":
                units = self._pptx_units(stored)
            elif stored.suffix.lower() == ".docx":
                units = self._docx_units(stored)
            else:
                units = self._text_units(stored)

        digest = sha256_file(stored)
        total = max(1, len(units))
        document_id = self.db.upsert_document(stored, digest, stored.stem, total)
        empty = 0
        processed = 0

        for unit in units:
            if stop_event is not None and stop_event.is_set():
                self.db.mark_document(document_id, "paused", "任务被暂停，可继续导入")
                break
            if self.db.page_is_indexed(document_id, unit.number):
                processed += 1
                if progress:
                    progress(processed, total, f"跳过已完成 {unit.label}")
                continue

            chunks = chunk_text(unit.text)
            if not chunks:
                empty += 1
                chunks = [f"[{unit.label} 未提取到可检索文字]"]
            self.db.replace_page_chunks(document_id, unit.number, chunks)
            processed += 1
            if progress:
                progress(processed, total, f"已索引 {unit.label} ({processed}/{total})")

        warning = ""
        if empty:
            warning = f"{empty}/{total} 个文档单元未提取到文字"
        if stop_event is None or not stop_event.is_set():
            self.db.mark_document(document_id, "indexed", warning)

        image_count = 0
        if stop_event is None or not stop_event.is_set():
            try:
                image_count = self._write_asset_manifest(stored, units)
                if progress and image_count:
                    progress(total, total, f"已保存 {image_count} 张关联图片")
            except Exception as exc:
                warning = (warning + "；" if warning else "") + f"图片提取失败: {exc}"
                self.db.mark_document(document_id, "indexed", warning)

        row = self.db.get_document(document_id)
        indexed = int(row["indexed_pages"]) if row else processed
        return IngestResult(
            document_id=document_id,
            pages_total=total,
            pages_indexed=indexed,
            chunks_total=self.db.count_chunks(),
            empty_pages=empty,
            copied_to_library=stored,
            warning=warning,
            image_count=image_count,
        )

    def ingest(
        self,
        source: Path,
        *,
        copy_into_library: bool = True,
        progress: ProgressCallback | None = None,
        stop_event: threading.Event | None = None,
        extract_images: bool = True,
        refresh_images: bool = False,
    ) -> IngestResult:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的资料格式: {source.suffix}；支持 PDF/PPTX/DOCX/TXT/MD"
            )

        if suffix == ".pdf":
            return self.pdf.ingest_pdf(
                source,
                copy_into_library=copy_into_library,
                progress=progress,
                stop_event=stop_event,
                extract_images=extract_images,
                refresh_images=refresh_images,
            )
        if suffix == ".pptx":
            units = self._pptx_units(source)
        elif suffix == ".docx":
            units = self._docx_units(source)
        else:
            units = self._text_units(source)

        result = self._ingest_units(
            source,
            units,
            copy_into_library=copy_into_library,
            progress=progress,
            stop_event=stop_event,
        )
        if not extract_images and result.image_count:
            # Images are useful to downstream evidence layout, but the caller
            # explicitly requested a text-only ingest. Remove the generated
            # manifest/assets after indexing.
            shutil.rmtree(self.assets.document_root(result.copied_to_library), ignore_errors=True)
            result.image_count = 0
        return result
