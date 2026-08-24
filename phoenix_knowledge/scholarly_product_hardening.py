from __future__ import annotations

import shutil
from pathlib import Path

from .product_document_ingest import ProductDocumentIngestor


_INSTALLED = False


def install() -> None:
    """Preserve sidecar figures when HTML/JATS files are copied into library."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def _ingest_scholarly(
        self,
        source: Path,
        *,
        copy_into_library: bool,
        progress,
        stop_event,
        extract_images: bool,
    ):
        source = Path(source).resolve()

        # Parse before copying the article file so relative JATS/HTML figure
        # references still resolve against the user's original download folder.
        if progress:
            progress(0, 1, f"正在解析学术文献格式 {source.suffix.upper()}……")
        parsed = self.scholarly.parse(source)
        stored = self._library_copy(source) if copy_into_library else source
        title = parsed.primary_title or stored.stem

        result = self._index_units_for_source(
            stored,
            parsed.units,
            title=title,
            progress=progress,
            stop_event=stop_event,
            warning=parsed.warning,
        )

        image_count = 0
        if extract_images and (
            stop_event is None or not stop_event.is_set()
        ):
            image_count = self.scholarly.write_asset_manifest(
                stored,
                parsed,
                self.assets,
            )
            if progress and image_count:
                progress(
                    result.pages_total,
                    result.pages_total,
                    f"已保存 {image_count} 张论文关联图片",
                )
        elif not extract_images:
            shutil.rmtree(
                self.assets.document_root(stored),
                ignore_errors=True,
            )

        self.scholarly.catalog.register(stored, parsed.records)
        result.image_count = image_count
        return result

    ProductDocumentIngestor._ingest_scholarly = _ingest_scholarly
