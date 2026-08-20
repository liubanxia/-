from __future__ import annotations

import json
import shutil
import threading
import zipfile
from pathlib import Path

from .chunker import chunk_text
from .config import WorkbenchPaths
from .db import KnowledgeDB
from .document_ingest import (
    MultiDocumentIngestor,
    ParsedUnit,
    SUPPORTED_EXTENSIONS as BASE_SUPPORTED_EXTENSIONS,
    sha256_file,
)
from .ingest import IngestResult
from .legacy_ppt import LegacyPPTConverter


SUPPORTED_EXTENSIONS = set(BASE_SUPPORTED_EXTENSIONS) | {".ppt"}


class ProductDocumentIngestor(MultiDocumentIngestor):
    """Product-grade document ingest with deduplicated storage and legacy PPT."""

    def __init__(self, db: KnowledgeDB, paths: WorkbenchPaths):
        super().__init__(db, paths)
        self.legacy_ppt = LegacyPPTConverter(paths)
        self.pdf._library_copy = self._library_copy

    def supported(self, source: Path) -> bool:
        return Path(source).suffix.lower() in SUPPORTED_EXTENSIONS

    def _library_copy(self, source: Path) -> Path:
        source = Path(source).resolve()
        root = Path(self.paths.source_root)
        root.mkdir(parents=True, exist_ok=True)
        direct = root / source.name
        if direct.resolve() == source:
            return source

        source_digest = sha256_file(source)
        stem, suffix = source.stem, source.suffix
        candidates: list[Path] = []
        if direct.exists():
            candidates.append(direct)
        candidates.extend(
            path for path in sorted(root.glob(f"{stem}_*{suffix}"))
            if path.is_file()
        )

        for candidate in candidates:
            try:
                if sha256_file(candidate) == source_digest:
                    return candidate
            except OSError:
                continue

        if not direct.exists():
            target = direct
        else:
            counter = 2
            while True:
                target = root / f"{stem}_{counter}{suffix}"
                if not target.exists():
                    break
                counter += 1

        shutil.copy2(source, target)
        return target

    def legacy_ppt_status(self) -> dict:
        return self.legacy_ppt.status().as_dict()

    @classmethod
    def _legacy_units(cls, converted_pptx: Path) -> list[ParsedUnit]:
        result: list[ParsedUnit] = []
        for unit in cls._pptx_units(converted_pptx):
            text = unit.text
            if text.startswith("[PPTX 幻灯片"):
                text = text.replace("[PPTX 幻灯片", "[PPT 幻灯片", 1)
            result.append(
                ParsedUnit(
                    number=unit.number,
                    label=unit.label,
                    text=text,
                    image_members=unit.image_members,
                )
            )
        return result

    def _write_legacy_asset_manifest(
        self,
        original_ppt: Path,
        converted_pptx: Path,
        units: list[ParsedUnit],
    ) -> int:
        doc_root = self.assets.document_root(original_ppt)
        if doc_root.exists():
            shutil.rmtree(doc_root, ignore_errors=True)
        doc_root.mkdir(parents=True, exist_ok=True)

        pages: dict[str, list[dict]] = {}
        image_count = 0
        with zipfile.ZipFile(converted_pptx) as zf:
            members = set(zf.namelist())
            for unit in units:
                items: list[dict] = []
                page_dir = doc_root / f"page_{unit.number:06d}"
                for image_index, member in enumerate(unit.image_members, start=1):
                    if member not in members:
                        continue
                    suffix = Path(member).suffix.lower() or ".bin"
                    target = page_dir / f"image_{image_index:03d}{suffix}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
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
            "source_path": str(original_ppt.resolve()),
            "source_sha256": sha256_file(original_ppt),
            "source_type": "ppt",
            "converted_cache": str(converted_pptx.resolve()),
            "page_count": len(units),
            "image_count": image_count,
            "pages": pages,
        }
        manifest_path = self.assets.manifest_path(original_ppt)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
        return image_count

    def _ingest_legacy_ppt(
        self,
        source: Path,
        *,
        copy_into_library: bool,
        progress,
        stop_event: threading.Event | None,
        extract_images: bool,
    ) -> IngestResult:
        source = Path(source).resolve()
        stored = self._library_copy(source) if copy_into_library else source

        if progress:
            progress(0, 1, "正在自动兼容老式PPT，不需要手工转换……")
        converted = self.legacy_ppt.convert(stored)
        units = self._legacy_units(converted)
        total = max(1, len(units))
        digest = sha256_file(stored)
        document_id = self.db.upsert_document(stored, digest, stored.stem, total)

        processed = 0
        empty = 0
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
            warning = f"{empty}/{total} 张幻灯片未提取到文字"
        if stop_event is None or not stop_event.is_set():
            self.db.mark_document(document_id, "indexed", warning)

        image_count = 0
        if extract_images and (stop_event is None or not stop_event.is_set()):
            try:
                image_count = self._write_legacy_asset_manifest(stored, converted, units)
                if progress and image_count:
                    progress(total, total, f"已保存 {image_count} 张PPT关联图片")
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
        progress=None,
        stop_event: threading.Event | None = None,
        extract_images: bool = True,
        refresh_images: bool = False,
    ) -> IngestResult:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() == ".ppt":
            return self._ingest_legacy_ppt(
                source,
                copy_into_library=copy_into_library,
                progress=progress,
                stop_event=stop_event,
                extract_images=extract_images,
            )
        return super().ingest(
            source,
            copy_into_library=copy_into_library,
            progress=progress,
            stop_event=stop_event,
            extract_images=extract_images,
            refresh_images=refresh_images,
        )
