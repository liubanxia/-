# Phoenix translated-PDF storage policy

Release defaults:

- Normal bilingual export produces one complete PDF only.
- Split volumes are opt-in (`part_pages > 0`) because they duplicate nearly one full book of storage.
- Original bilingual pages reuse copied source PDF page objects and append only the translated text layer; source pages are not rasterized.
- PDF writes are atomic and use lightweight compression (`deflate`, image/font deflate, object streams where supported) without the old heavy `garbage=3` final pass.
- Existing stale `PDF分册` outputs are removed when a normal no-split export is generated.
- The GUI displays source size, complete translated-PDF size, and ratio after export.

The release regression suite includes an image-heavy source PDF and rejects book-wide source-payload duplication.
