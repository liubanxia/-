from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


_SAFE_RE = re.compile(r'[\\/:*?"<>|\r\n]+')


def _safe_name(text: str) -> str:
    return (_SAFE_RE.sub('_', text).strip(' ._') or 'document')[:96]


def _sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PageAsset:
    page: int
    path: Path
    width: int = 0
    height: int = 0


class PDFAssetStore:
    """Persist source-linked images without cross-document cache collisions."""

    def __init__(self, runtime_root: Path):
        self.root = Path(runtime_root) / 'pdf_assets'
        self.root.mkdir(parents=True, exist_ok=True)
        self._fingerprints: dict[str, tuple[int, int, str]] = {}

    @staticmethod
    def _source_identity(source_path: Path) -> str:
        source = Path(source_path).expanduser()
        try:
            resolved = source.resolve()
        except OSError:
            resolved = source.absolute()
        return str(resolved).casefold()

    def _source_sha(self, source_path: Path) -> str:
        source = Path(source_path).resolve()
        stat = source.stat()
        identity = self._source_identity(source)
        cached = self._fingerprints.get(identity)
        signature = (int(stat.st_size), int(stat.st_mtime_ns))
        if cached is not None and cached[:2] == signature:
            return cached[2]
        digest = _sha256_file(source)
        self._fingerprints[identity] = (signature[0], signature[1], digest)
        return digest

    def document_root(self, pdf_path: Path) -> Path:
        source = Path(pdf_path)
        identity = hashlib.sha256(
            self._source_identity(source).encode('utf-8', errors='surrogatepass')
        ).hexdigest()[:16]
        suffix = _safe_name(source.suffix.lower().lstrip('.') or 'file')
        return self.root / f'{_safe_name(source.stem)}_{suffix}_{identity}'

    def manifest_path(self, pdf_path: Path) -> Path:
        return self.document_root(pdf_path) / 'manifest.json'

    def _read_manifest(self, pdf_path: Path) -> dict:
        path = self.manifest_path(pdf_path)
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def extract(self, pdf_path: Path, *, force: bool = False) -> dict:
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.is_file() or pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f'不是可读取PDF: {pdf_path}')

        current_sha = self._source_sha(pdf_path)
        existing = self._read_manifest(pdf_path)
        if (
            existing
            and not force
            and str(existing.get('source_sha256') or '') == current_sha
            and str(existing.get('source_path') or '') == str(pdf_path)
        ):
            return existing

        import fitz

        doc_root = self.document_root(pdf_path)
        if doc_root.exists():
            shutil.rmtree(doc_root, ignore_errors=True)
        doc_root.mkdir(parents=True, exist_ok=True)
        pages: dict[str, list[dict]] = {}
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        try:
            for page_index in range(total_pages):
                page_number = page_index + 1
                page = doc[page_index]
                page_dir = doc_root / f'page_{page_number:06d}'
                seen_xrefs: set[int] = set()
                items: list[dict] = []
                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    try:
                        xref = int(image_info[0])
                    except Exception:
                        continue
                    if xref <= 0 or xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)
                    try:
                        payload = doc.extract_image(xref)
                    except Exception:
                        continue
                    raw = payload.get('image')
                    if not raw:
                        continue
                    ext = str(payload.get('ext') or 'bin').lower()
                    if ext == 'jpeg':
                        ext = 'jpg'
                    page_dir.mkdir(parents=True, exist_ok=True)
                    image_path = page_dir / f'image_{image_index:03d}.{ext}'
                    image_path.write_bytes(raw)
                    items.append({
                        'path': str(image_path.relative_to(doc_root)).replace('\\', '/'),
                        'width': int(payload.get('width') or 0),
                        'height': int(payload.get('height') or 0),
                        'xref': xref,
                    })
                if items:
                    pages[str(page_number)] = items
        finally:
            doc.close()

        manifest = {
            'source_path': str(pdf_path),
            'source_sha256': current_sha,
            'pdf_pages': total_pages,
            'image_count': sum(len(v) for v in pages.values()),
            'pages': pages,
        }
        temp = self.manifest_path(pdf_path).with_suffix('.json.tmp')
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(self.manifest_path(pdf_path))
        return manifest

    def page_assets(self, pdf_path: Path, page: int, *, ensure: bool = False) -> list[PageAsset]:
        pdf_path = Path(pdf_path)
        manifest = self._read_manifest(pdf_path)
        if pdf_path.suffix.lower() == '.pdf' and ensure:
            try:
                current_sha = self._source_sha(pdf_path)
            except OSError:
                current_sha = ''
            if (
                not manifest
                or str(manifest.get('source_sha256') or '') != current_sha
                or str(manifest.get('source_path') or '') != str(pdf_path.resolve())
            ):
                manifest = self.extract(pdf_path, force=True)

        doc_root = self.document_root(pdf_path)
        result: list[PageAsset] = []
        for item in (manifest.get('pages') or {}).get(str(int(page)), []):
            path = doc_root / str(item.get('path') or '')
            if path.is_file():
                result.append(PageAsset(
                    page=int(page),
                    path=path,
                    width=int(item.get('width') or 0),
                    height=int(item.get('height') or 0),
                ))
        return result

    def copy_page_assets(
        self,
        pdf_path: Path,
        page: int,
        destination: Path,
        *,
        prefix: str = '',
        ensure: bool = True,
    ) -> list[Path]:
        destination = Path(destination)
        assets = self.page_assets(pdf_path, page, ensure=ensure)
        if not assets:
            return []
        destination.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for index, asset in enumerate(assets, start=1):
            suffix = asset.path.suffix.lower() or '.bin'
            target = destination / f'{prefix}p{int(page):06d}_{index:03d}{suffix}'
            if not target.exists():
                shutil.copy2(asset.path, target)
            copied.append(target)
        return copied


def markdown_images(paths: list[Path], *, relative_to: Path, label: str) -> str:
    lines: list[str] = []
    safe_label = (
        str(label or "图像")
        .replace("[", "（")
        .replace("]", "）")
        .replace("\r", " ")
        .replace("\n", " ")
    )
    for index, path in enumerate(paths, start=1):
        rel = Path(path).relative_to(relative_to).as_posix()
        # Angle-bracket destinations are CommonMark-safe when an asset folder
        # or filename contains spaces or parentheses. Qt's Markdown renderer
        # and the PDF/DOCX exporters all resolve this form consistently.
        lines.extend(['', f'![{safe_label} 图{index}](<{rel}>)', ''])
    return '\n'.join(lines)


def html_images(paths: list[Path], *, relative_to: Path, label: str) -> str:
    blocks: list[str] = []
    for index, path in enumerate(paths, start=1):
        rel = Path(path).relative_to(relative_to).as_posix()
        blocks.append(
            '<figure>'
            f'<img src="{html.escape(rel)}" alt="{html.escape(label)} 图{index}" '
            'style="max-width:100%;height:auto">'
            f'<figcaption>{html.escape(label)} 图{index}</figcaption>'
            '</figure>'
        )
    return '\n'.join(blocks)
