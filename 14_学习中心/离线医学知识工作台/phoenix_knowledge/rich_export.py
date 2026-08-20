from __future__ import annotations

import html
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape


_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MARKUP_RE = re.compile(r"[*_`~]+")
_SAFE_RE = re.compile(r'[\\/:*?"<>|\r\n]+')


@dataclass(frozen=True)
class ExportBundle:
    output_dir: Path
    markdown: Path
    text: Path
    docx: Path
    pdf: Path

    @property
    def output_paths(self) -> tuple[Path, ...]:
        return (self.pdf, self.docx, self.markdown, self.text)


def _safe_name(value: str) -> str:
    return (_SAFE_RE.sub("_", value).strip(" ._") or "Phoenix医学专题")[:96]


def markdown_to_plain(markdown: str) -> str:
    text = (markdown or "").replace("\r\n", "\n")
    text = _IMAGE_RE.sub(lambda m: f"[图像：{m.group(1) or Path(m.group(2)).name}]", text)
    text = _LINK_RE.sub(lambda m: m.group(1), text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            continue
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "• ", line)
        line = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", line)
        line = _MARKUP_RE.sub("", line)
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _docx_document_xml(text: str) -> str:
    paragraphs: list[str] = []
    for line in (text or "").splitlines():
        if not line:
            paragraphs.append("<w:p/>")
            continue
        safe = xml_escape(line)
        paragraphs.append(
            "<w:p><w:r><w:t xml:space=\"preserve\">"
            + safe
            + "</w:t></w:r></w:p>"
        )
    body = "".join(paragraphs)
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f"{body}"
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
        "</w:body></w:document>"
    )


def _write_docx(path: Path, text: str) -> None:
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", _docx_document_xml(text))
    temp.replace(path)


def _html_document(markdown: str) -> str:
    blocks: list[str] = []
    in_list = False
    for raw in (markdown or "").splitlines():
        line = raw.rstrip()
        if not line:
            if in_list:
                blocks.append("</ul>")
                in_list = False
            blocks.append("<div style='height:6pt'></div>")
            continue

        image = _IMAGE_RE.fullmatch(line.strip())
        if image:
            if in_list:
                blocks.append("</ul>")
                in_list = False
            blocks.append(
                f"<p><b>图像：</b>{html.escape(image.group(1) or Path(image.group(2)).name)}</p>"
            )
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            if in_list:
                blocks.append("</ul>")
                in_list = False
            level = min(len(heading.group(1)), 4)
            blocks.append(f"<h{level}>{html.escape(heading.group(2))}</h{level}>")
            continue

        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet:
            if not in_list:
                blocks.append("<ul>")
                in_list = True
            blocks.append(f"<li>{html.escape(bullet.group(1))}</li>")
            continue

        if in_list:
            blocks.append("</ul>")
            in_list = False

        if line.lstrip().startswith(">"):
            value = line.lstrip()[1:].strip()
            blocks.append(
                "<div style='border-left:2px solid #888;padding-left:8px;color:#444'>"
                + html.escape(value)
                + "</div>"
            )
        else:
            value = _MARKUP_RE.sub("", line)
            blocks.append(f"<p>{html.escape(value)}</p>")
    if in_list:
        blocks.append("</ul>")
    return (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:sans-serif;line-height:1.5;font-size:10.5pt;}"
        "h1{font-size:18pt}h2{font-size:15pt}h3{font-size:12.5pt}"
        "p{margin:0 0 5pt 0}li{margin:0 0 3pt 0}"
        "</style></head><body>"
        + "".join(blocks)
        + "</body></html>"
    )


def _write_pdf(path: Path, markdown: str) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 595.0, 842.0
    margin = 32.0
    lines = (markdown or "").splitlines()
    chunks: list[str] = []
    current: list[str] = []
    chars = 0
    for line in lines:
        current.append(line)
        chars += max(len(line), 1)
        if chars >= 3600:
            chunks.append("\n".join(current))
            current = []
            chars = 0
    if current or not chunks:
        chunks.append("\n".join(current))

    doc = fitz.open()
    try:
        for chunk in chunks:
            page = doc.new_page(width=page_width, height=page_height)
            rect = fitz.Rect(margin, margin, page_width - margin, page_height - margin)
            markup = _html_document(chunk)
            try:
                page.insert_htmlbox(rect, markup, scale_low=0.55)
            except TypeError:
                page.insert_htmlbox(rect, markup)
        if path.exists():
            path.unlink()
        doc.save(path, garbage=3, deflate=True)
    finally:
        doc.close()


class MultiFormatExporter:
    """Export one organized medical topic to PDF, DOCX, Markdown and TXT."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def export_text(
        self,
        markdown: str,
        *,
        title: str,
        source_path: Path | None = None,
    ) -> ExportBundle:
        title = _safe_name(title)
        bundle_root = self.output_root / title
        bundle_root.mkdir(parents=True, exist_ok=True)

        md_path = bundle_root / f"{title}.md"
        txt_path = bundle_root / f"{title}.txt"
        docx_path = bundle_root / f"{title}.docx"
        pdf_path = bundle_root / f"{title}.pdf"

        md_path.write_text((markdown or "").rstrip() + "\n", encoding="utf-8")
        plain = markdown_to_plain(markdown)
        txt_path.write_text(plain, encoding="utf-8")
        _write_docx(docx_path, plain)
        _write_pdf(pdf_path, markdown)

        if source_path is not None:
            source_path = Path(source_path)
            source_assets = source_path.with_name(source_path.stem + "_assets")
            if source_assets.is_dir():
                target_assets = bundle_root / source_assets.name
                if target_assets.exists():
                    shutil.rmtree(target_assets, ignore_errors=True)
                shutil.copytree(source_assets, target_assets)

        return ExportBundle(
            output_dir=bundle_root,
            markdown=md_path,
            text=txt_path,
            docx=docx_path,
            pdf=pdf_path,
        )

    def export_path(self, source: Path, *, title: str | None = None) -> ExportBundle:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        markdown = source.read_text(encoding="utf-8", errors="replace")
        return self.export_text(
            markdown,
            title=title or source.stem,
            source_path=source,
        )
