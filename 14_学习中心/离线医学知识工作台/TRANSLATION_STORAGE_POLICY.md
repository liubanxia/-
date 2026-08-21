# Phoenix translated-PDF storage policy

Release defaults:

- Default translated PDF is **原版图文中文译本**: keep the original PDF page size, images, vector drawings and tables, remove only native source text blocks, then write Chinese into the same text regions.
- The source page is never rasterized and is never duplicated below/above another page in the default mode.
- Numeric-only page labels, URLs and DOI/PMID/PMCID identifiers are preserved rather than translated.
- If one source text block cannot hold its complete Chinese allocation, overflow is moved to a compact footer instead of being truncated.
- Scanned/image-only or rotated specialty pages preserve the source image and use a compact translation panel/footer; they do not duplicate the page image.
- Normal export produces one complete PDF only. Split volumes are opt-in (`part_pages > 0`) because they duplicate nearly one full book of storage.
- PDF writes are atomic and lossless. Normal writes use `garbage=2`, Flate compression, font/image stream compression and object streams. A `garbage=3` duplicate-object pass is attempted only when the first output exceeds the storage budget; `garbage=4` is forbidden for whole-book translation because large stream scanning caused the old 383-page final-save stall.
- Storage budget for the complete translated PDF is `max(source × 1.18, source + 2.5 MiB)`. The fixed allowance covers the CJK text/font layer on small PDFs. Larger medical books should normally remain close to 1.0×.
- The GUI displays source size, complete translated-PDF size, ratio and whether the storage target passed.
- A `PDF体积报告.json` is written beside the output with the measured ratio and any scan/overflow fallback pages.

Legacy layouts remain available when a user explicitly wants an English/Chinese bilingual page, but they are no longer the release default.
