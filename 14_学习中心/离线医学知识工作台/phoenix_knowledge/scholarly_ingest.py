from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET

from .document_ingest import ParsedUnit, sha256_file
from .pdf_assets import PDFAssetStore


DIRECT_SCHOLARLY_EXTENSIONS = {
    ".html", ".htm", ".xml", ".nxml", ".jats",
    ".nbib", ".ris", ".bib", ".bibtex", ".json", ".csljson",
}

_XLINK = "http://www.w3.org/1999/xlink"
_WS_RE = re.compile(r"\s+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
_PMID_RE = re.compile(r"\bPMID\s*[:#]?\s*(\d{4,12})\b", re.I)
_PMCID_RE = re.compile(r"\bPMC\d{4,12}\b", re.I)


def _clean(text: str | None) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def _first(items) -> str:
    for item in items:
        value = _clean(item)
        if value:
            return value
    return ""


def _year(text: str | None) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", str(text or ""))
    return match.group(0) if match else ""


def _norm_doi(value: str | None) -> str:
    text = _clean(value)
    if not text:
        return ""
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.I)
    match = _DOI_RE.search(text)
    return match.group(0).rstrip(".,;)").lower() if match else text.lower()


def _canonical_key(record: "ScholarlyRecord") -> str:
    if record.doi:
        return f"doi:{_norm_doi(record.doi)}"
    if record.pmid:
        return f"pmid:{record.pmid}"
    if record.pmcid:
        return f"pmcid:{record.pmcid.upper()}"
    if record.title:
        normalized = re.sub(r"\W+", "", record.title, flags=re.UNICODE).casefold()
        if normalized:
            return f"title:{normalized[:220]}"
    return ""


@dataclass(frozen=True)
class ScholarlyRecord:
    title: str = ""
    authors: tuple[str, ...] = ()
    journal: str = ""
    year: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    abstract: str = ""
    keywords: tuple[str, ...] = ()
    publication_type: str = ""
    source_format: str = ""
    extras: dict[str, str] = field(default_factory=dict)

    @property
    def canonical_key(self) -> str:
        return _canonical_key(self)

    def evidence_header(self) -> str:
        lines = ["[论文元数据]"]
        if self.title:
            lines.append(f"题名：{self.title}")
        if self.authors:
            lines.append("作者：" + "；".join(self.authors))
        if self.journal:
            lines.append(f"期刊/来源：{self.journal}")
        if self.year:
            lines.append(f"年份：{self.year}")
        if self.doi:
            lines.append(f"DOI：{_norm_doi(self.doi)}")
        if self.pmid:
            lines.append(f"PMID：{self.pmid}")
        if self.pmcid:
            lines.append(f"PMCID：{self.pmcid}")
        if self.publication_type:
            lines.append(f"文献类型：{self.publication_type}")
        if self.keywords:
            lines.append("关键词：" + "；".join(self.keywords))
        if self.abstract:
            lines.extend(["", "[摘要]", self.abstract])
        return "\n".join(lines).strip()


@dataclass
class ScholarlyDocument:
    units: list[ParsedUnit]
    records: list[ScholarlyRecord]
    images: dict[int, tuple[Path, ...]] = field(default_factory=dict)
    warning: str = ""

    @property
    def primary_title(self) -> str:
        for record in self.records:
            if record.title:
                return record.title
        return ""


class ScholarlyCatalog:
    """Small offline identifier index used to bind citations and full text."""

    def __init__(self, runtime_root: Path):
        self.path = Path(runtime_root) / "scholarly_catalog.json"

    def _read(self) -> dict:
        if not self.path.is_file():
            return {"records": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"records": {}}
        if not isinstance(data, dict):
            return {"records": {}}
        data.setdefault("records", {})
        return data

    def register(self, source: Path, records: list[ScholarlyRecord]) -> None:
        source = Path(source).resolve()
        data = self._read()
        store = data.setdefault("records", {})
        changed = False
        for record in records:
            key = record.canonical_key
            if not key:
                continue
            item = {
                "path": str(source),
                "title": record.title,
                "source_format": record.source_format,
                "doi": _norm_doi(record.doi),
                "pmid": record.pmid,
                "pmcid": record.pmcid,
            }
            bucket = store.setdefault(key, [])
            if item not in bucket:
                bucket.append(item)
                changed = True
        if not changed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)


class _ArticleHTMLParser(HTMLParser):
    BLOCKS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "caption", "th", "td", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.blocks: list[str] = []
        self.images: list[str] = []
        self._stack: list[str] = []
        self._buf: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()
        attrs_dict = {str(k).lower(): str(v or "") for k, v in attrs}
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
            return
        if tag == "meta":
            key = (
                attrs_dict.get("name")
                or attrs_dict.get("property")
                or attrs_dict.get("http-equiv")
                or ""
            ).strip().lower()
            content = _clean(attrs_dict.get("content"))
            if key and content:
                self.meta.setdefault(key, []).append(content)
        if tag == "img":
            src = _clean(attrs_dict.get("src"))
            if src:
                self.images.append(src)
        if tag in self.BLOCKS:
            if self._buf:
                self._flush()
            self._stack.append(tag)

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1
            return
        if tag in self.BLOCKS:
            self._flush()
            if self._stack and self._stack[-1] == tag:
                self._stack.pop()

    def handle_data(self, data: str):
        if self._skip:
            return
        value = _clean(data)
        if value:
            self._buf.append(value)

    def _flush(self):
        text = _clean(" ".join(self._buf))
        if text:
            self.blocks.append(text)
        self._buf = []


class ScholarlyParser:
    def __init__(self, runtime_root: Path):
        self.catalog = ScholarlyCatalog(runtime_root)

    def parse(self, source: Path) -> ScholarlyDocument:
        source = Path(source).resolve()
        suffix = source.suffix.lower()
        if suffix in {".html", ".htm"}:
            return self._parse_html(source)
        if suffix in {".xml", ".nxml", ".jats"}:
            return self._parse_xml(source)
        if suffix == ".nbib":
            return self._parse_nbib(source)
        if suffix == ".ris":
            return self._parse_ris(source)
        if suffix in {".bib", ".bibtex"}:
            return self._parse_bibtex(source)
        if suffix in {".json", ".csljson"}:
            return self._parse_csl_json(source)
        raise ValueError(f"不支持的学术文献格式: {source.suffix}")

    @staticmethod
    def _resolve_local_image(source: Path, raw: str) -> Path | None:
        raw = html_lib.unescape(str(raw or "")).strip()
        if not raw or raw.startswith(("http://", "https://", "data:", "//")):
            return None
        raw = raw.split("#", 1)[0].split("?", 1)[0]
        candidate = (source.parent / raw).resolve()
        try:
            candidate.relative_to(source.parent.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    @staticmethod
    def _units_from_text(kind: str, text: str, *, first_header: str = "") -> list[ParsedUnit]:
        from .chunker import chunk_text
        pieces = chunk_text(text, max_chars=6000, overlap_chars=0) or [text]
        result: list[ParsedUnit] = []
        for index, piece in enumerate(pieces, start=1):
            lines = [f"[{kind} 论文单元 {index}]"]
            if index == 1 and first_header:
                lines.extend([first_header, ""])
            lines.append(piece.strip())
            result.append(
                ParsedUnit(
                    number=index,
                    label=f"论文单元{index}",
                    text="\n".join(lines).strip(),
                )
            )
        return result

    @staticmethod
    def _record_from_meta(meta: dict[str, list[str]], source_format: str) -> ScholarlyRecord:
        title = _first(meta.get("citation_title", []) + meta.get("dc.title", []) + meta.get("og:title", []))
        authors = tuple(
            dict.fromkeys(
                _clean(x)
                for x in meta.get("citation_author", []) + meta.get("dc.creator", [])
                if _clean(x)
            )
        )
        journal = _first(meta.get("citation_journal_title", []) + meta.get("prism.publicationname", []))
        date = _first(
            meta.get("citation_publication_date", [])
            + meta.get("citation_date", [])
            + meta.get("dc.date", [])
        )
        doi = _first(meta.get("citation_doi", []) + meta.get("dc.identifier", []))
        pmid = _first(meta.get("citation_pmid", []))
        pmcid = _first(meta.get("citation_pmcid", []))
        abstract = _first(
            meta.get("citation_abstract", [])
            + meta.get("description", [])
            + meta.get("dc.description", [])
        )
        keywords: list[str] = []
        for raw in meta.get("citation_keywords", []) + meta.get("keywords", []):
            keywords.extend(_clean(x) for x in re.split(r"[;,|]", raw) if _clean(x))
        return ScholarlyRecord(
            title=title,
            authors=authors,
            journal=journal,
            year=_year(date),
            doi=_norm_doi(doi),
            pmid=pmid,
            pmcid=pmcid,
            abstract=abstract,
            keywords=tuple(dict.fromkeys(keywords)),
            source_format=source_format,
        )

    def _parse_html(self, source: Path) -> ScholarlyDocument:
        raw = source.read_text(encoding="utf-8", errors="replace")
        parser = _ArticleHTMLParser()
        parser.feed(raw)
        parser._flush()
        record = self._record_from_meta(parser.meta, source.suffix.lower().lstrip("."))
        body = "\n\n".join(parser.blocks).strip()
        if not record.title:
            for block in parser.blocks[:5]:
                if 8 <= len(block) <= 400:
                    record = ScholarlyRecord(**{**asdict(record), "title": block})
                    break
        header = record.evidence_header()
        units = self._units_from_text("HTML", body or header, first_header=header)
        images: dict[int, tuple[Path, ...]] = {}
        local = [self._resolve_local_image(source, x) for x in parser.images]
        local = [x for x in local if x is not None]
        if local and units:
            images[units[0].number] = tuple(dict.fromkeys(local))
        return ScholarlyDocument(units=units, records=[record], images=images)

    @staticmethod
    def _tag_name(node) -> str:
        return str(node.tag).split("}")[-1].lower()

    @classmethod
    def _find_texts(cls, root, names: set[str]) -> list[str]:
        result = []
        for node in root.iter():
            if cls._tag_name(node) in names:
                text = _clean(" ".join(node.itertext()))
                if text:
                    result.append(text)
        return result

    @classmethod
    def _xml_record(cls, root, source_format: str) -> ScholarlyRecord:
        title = _first(cls._find_texts(root, {"article-title", "title"}))
        journal = _first(cls._find_texts(root, {"journal-title", "journal-id"}))
        abstract = _first(cls._find_texts(root, {"abstract"}))
        authors: list[str] = []
        for contrib in root.iter():
            if cls._tag_name(contrib) not in {"contrib", "author"}:
                continue
            surname = _first(cls._find_texts(contrib, {"surname"}))
            given = _first(cls._find_texts(contrib, {"given-names", "given-name"}))
            full = _clean(" ".join(x for x in [given, surname] if x))
            if not full:
                full = _first(cls._find_texts(contrib, {"name", "string-name"}))
            if full:
                authors.append(full)
        doi = ""
        pmid = ""
        pmcid = ""
        for node in root.iter():
            tag = cls._tag_name(node)
            value = _clean(" ".join(node.itertext()))
            if tag in {"article-id", "pub-id"}:
                kind = (
                    node.attrib.get("pub-id-type")
                    or node.attrib.get("id-type")
                    or node.attrib.get("specific-use")
                    or ""
                ).lower()
                if kind == "doi":
                    doi = _norm_doi(value)
                elif kind in {"pmid", "pubmed"}:
                    pmid = value
                elif kind in {"pmcid", "pmc"}:
                    pmcid = value
            if not doi and tag == "doi":
                doi = _norm_doi(value)
        years = cls._find_texts(root, {"year"})
        keywords = tuple(dict.fromkeys(cls._find_texts(root, {"kwd", "keyword"})))
        pub_type = _clean(root.attrib.get("article-type") or root.attrib.get("publication-type"))
        return ScholarlyRecord(
            title=title,
            authors=tuple(dict.fromkeys(authors)),
            journal=journal,
            year=_year(_first(years)),
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            abstract=abstract,
            keywords=keywords,
            publication_type=pub_type,
            source_format=source_format,
        )

    @classmethod
    def _graphics(cls, source: Path, node) -> tuple[Path, ...]:
        result: list[Path] = []
        for item in node.iter():
            if cls._tag_name(item) not in {"graphic", "inline-graphic", "media"}:
                continue
            href = (
                item.attrib.get(f"{{{_XLINK}}}href")
                or item.attrib.get("href")
                or item.attrib.get("src")
                or ""
            )
            image = cls._resolve_local_image(source, href)
            if image is not None:
                result.append(image)
        return tuple(dict.fromkeys(result))

    def _parse_xml(self, source: Path) -> ScholarlyDocument:
        try:
            root = ET.parse(source).getroot()
        except ET.ParseError as exc:
            raise RuntimeError(f"XML/JATS解析失败: {exc}") from exc
        source_format = source.suffix.lower().lstrip(".")
        record = self._xml_record(root, source_format)
        units: list[ParsedUnit] = []
        images: dict[int, tuple[Path, ...]] = {}

        header = record.evidence_header()
        units.append(ParsedUnit(1, "论文元数据", header or "[论文元数据未提取]"))
        root_images = self._graphics(source, root)
        if root_images:
            images[1] = root_images

        abstract = record.abstract
        if abstract:
            units.append(
                ParsedUnit(
                    len(units) + 1,
                    "摘要",
                    f"[论文摘要]\n{abstract}",
                )
            )

        body = next((x for x in root.iter() if self._tag_name(x) == "body"), None)
        sections = []
        if body is not None:
            sections = [x for x in list(body) if self._tag_name(x) == "sec"]
        if sections:
            for sec in sections:
                title = _first(self._find_texts(sec, {"title"})) or f"正文部分{len(units)}"
                text = _clean(" ".join(sec.itertext()))
                if not text:
                    continue
                number = len(units) + 1
                units.append(
                    ParsedUnit(
                        number,
                        f"论文单元{number}",
                        f"[论文正文：{title}]\n{text}",
                    )
                )
                graphics = self._graphics(source, sec)
                if graphics:
                    images[number] = graphics
        else:
            body_text = _clean(" ".join(body.itertext())) if body is not None else _clean(" ".join(root.itertext()))
            if body_text:
                for piece in self._units_from_text("XML/JATS", body_text):
                    number = len(units) + 1
                    units.append(
                        ParsedUnit(number, f"论文单元{number}", piece.text)
                    )

        refs = next((x for x in root.iter() if self._tag_name(x) in {"ref-list", "references"}), None)
        if refs is not None:
            ref_text = _clean(" ".join(refs.itertext()))
            if ref_text:
                number = len(units) + 1
                units.append(
                    ParsedUnit(number, f"参考文献单元{number}", f"[参考文献]\n{ref_text}")
                )

        return ScholarlyDocument(units=units, records=[record], images=images)

    @staticmethod
    def _medline_records(text: str) -> list[dict[str, list[str]]]:
        records: list[dict[str, list[str]]] = []
        current: dict[str, list[str]] = {}
        last_tag = ""
        for raw in text.splitlines():
            if not raw.strip():
                continue
            match = re.match(r"^([A-Z0-9]{2,4})\s*-\s?(.*)$", raw)
            if match:
                tag, value = match.group(1), match.group(2).strip()
                if tag == "PMID" and current.get("PMID"):
                    records.append(current)
                    current = {}
                current.setdefault(tag, []).append(value)
                last_tag = tag
            elif raw.startswith("      ") and last_tag:
                current[last_tag][-1] = _clean(current[last_tag][-1] + " " + raw.strip())
        if current:
            records.append(current)
        return records

    @staticmethod
    def _record_from_medline(data: dict[str, list[str]]) -> ScholarlyRecord:
        aid = data.get("AID", [])
        doi = _first(x.split(" [doi]")[0] for x in aid if "[doi]" in x.lower())
        return ScholarlyRecord(
            title=_first(data.get("TI", [])),
            authors=tuple(data.get("FAU", []) or data.get("AU", [])),
            journal=_first(data.get("JT", []) or data.get("TA", [])),
            year=_year(_first(data.get("DP", []))),
            doi=_norm_doi(doi),
            pmid=_first(data.get("PMID", [])),
            abstract=_first(data.get("AB", [])),
            keywords=tuple(dict.fromkeys(data.get("OT", []) + data.get("MH", []))),
            publication_type="；".join(data.get("PT", [])),
            source_format="nbib",
        )

    def _parse_nbib(self, source: Path) -> ScholarlyDocument:
        text = source.read_text(encoding="utf-8", errors="replace")
        records = [self._record_from_medline(x) for x in self._medline_records(text)]
        if not records:
            raise RuntimeError("NBIB未解析到PubMed记录")
        units = [
            ParsedUnit(
                i,
                f"文献记录{i}",
                f"[NBIB 文献记录 {i}]\n{record.evidence_header()}",
            )
            for i, record in enumerate(records, 1)
        ]
        return ScholarlyDocument(units=units, records=records)

    @staticmethod
    def _ris_records(text: str) -> list[dict[str, list[str]]]:
        result: list[dict[str, list[str]]] = []
        current: dict[str, list[str]] = {}
        for raw in text.splitlines():
            match = re.match(r"^([A-Z0-9]{2})  -\s?(.*)$", raw)
            if not match:
                continue
            tag, value = match.group(1), match.group(2).strip()
            if tag == "TY" and current:
                result.append(current)
                current = {}
            current.setdefault(tag, []).append(value)
            if tag == "ER":
                result.append(current)
                current = {}
        if current:
            result.append(current)
        return result

    @staticmethod
    def _record_from_ris(data: dict[str, list[str]]) -> ScholarlyRecord:
        doi = _first(data.get("DO", []))
        if not doi:
            for value in data.get("UR", []):
                if "doi.org/" in value.lower():
                    doi = value
                    break
        return ScholarlyRecord(
            title=_first(data.get("TI", []) or data.get("T1", [])),
            authors=tuple(data.get("AU", []) or data.get("A1", [])),
            journal=_first(data.get("JO", []) or data.get("JF", []) or data.get("T2", [])),
            year=_year(_first(data.get("PY", []) or data.get("Y1", []))),
            doi=_norm_doi(doi),
            abstract=_first(data.get("AB", [])),
            keywords=tuple(dict.fromkeys(data.get("KW", []))),
            publication_type=_first(data.get("TY", [])),
            source_format="ris",
        )

    def _parse_ris(self, source: Path) -> ScholarlyDocument:
        text = source.read_text(encoding="utf-8", errors="replace")
        records = [self._record_from_ris(x) for x in self._ris_records(text)]
        if not records:
            raise RuntimeError("RIS未解析到文献记录")
        units = [
            ParsedUnit(i, f"文献记录{i}", f"[RIS 文献记录 {i}]\n{record.evidence_header()}")
            for i, record in enumerate(records, 1)
        ]
        return ScholarlyDocument(units=units, records=records)

    @staticmethod
    def _split_top_level(text: str) -> list[str]:
        result: list[str] = []
        buf: list[str] = []
        depth = 0
        quote = False
        escape = False
        for ch in text:
            if escape:
                buf.append(ch)
                escape = False
                continue
            if ch == "\\":
                buf.append(ch)
                escape = True
                continue
            if ch == '"':
                quote = not quote
                buf.append(ch)
                continue
            if not quote:
                if ch in "{(":
                    depth += 1
                elif ch in "})" and depth > 0:
                    depth -= 1
                elif ch == "," and depth == 0:
                    result.append("".join(buf).strip())
                    buf = []
                    continue
            buf.append(ch)
        if buf:
            result.append("".join(buf).strip())
        return result

    @classmethod
    def _bibtex_entries(cls, text: str) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        pos = 0
        while True:
            match = re.search(r"@([A-Za-z]+)\s*([\{\(])", text[pos:])
            if not match:
                break
            kind = match.group(1)
            opener = match.group(2)
            closer = "}" if opener == "{" else ")"
            start = pos + match.end()
            depth = 1
            quote = False
            escape = False
            end = start
            while end < len(text):
                ch = text[end]
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    quote = not quote
                elif not quote:
                    if ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            break
                end += 1
            body = text[start:end]
            pos = end + 1
            parts = cls._split_top_level(body)
            if not parts:
                continue
            fields: dict[str, str] = {"entry_type": kind, "citation_key": parts[0].strip()}
            for part in parts[1:]:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                value = value.strip().strip(",").strip()
                while len(value) >= 2 and (
                    (value[0] == "{" and value[-1] == "}")
                    or (value[0] == '"' and value[-1] == '"')
                ):
                    value = value[1:-1].strip()
                fields[key.strip().lower()] = _clean(value.replace("\n", " "))
            entries.append(fields)
        return entries

    @staticmethod
    def _record_from_bib(data: dict[str, str]) -> ScholarlyRecord:
        authors = tuple(_clean(x) for x in re.split(r"\s+and\s+", data.get("author", ""), flags=re.I) if _clean(x))
        keywords = tuple(_clean(x) for x in re.split(r"[;,]", data.get("keywords", "")) if _clean(x))
        return ScholarlyRecord(
            title=_clean(data.get("title")),
            authors=authors,
            journal=_clean(data.get("journal") or data.get("booktitle")),
            year=_year(data.get("year")),
            doi=_norm_doi(data.get("doi")),
            pmid=_clean(data.get("pmid")),
            abstract=_clean(data.get("abstract")),
            keywords=keywords,
            publication_type=_clean(data.get("entry_type")),
            source_format="bibtex",
            extras={"citation_key": _clean(data.get("citation_key"))},
        )

    def _parse_bibtex(self, source: Path) -> ScholarlyDocument:
        text = source.read_text(encoding="utf-8", errors="replace")
        records = [self._record_from_bib(x) for x in self._bibtex_entries(text)]
        if not records:
            raise RuntimeError("BibTeX未解析到文献记录")
        units = [
            ParsedUnit(i, f"文献记录{i}", f"[BibTeX 文献记录 {i}]\n{record.evidence_header()}")
            for i, record in enumerate(records, 1)
        ]
        return ScholarlyDocument(units=units, records=records)

    @staticmethod
    def _record_from_csl(data: dict) -> ScholarlyRecord:
        authors: list[str] = []
        for item in data.get("author") or []:
            if isinstance(item, dict):
                literal = _clean(item.get("literal"))
                if literal:
                    authors.append(literal)
                    continue
                given = _clean(item.get("given"))
                family = _clean(item.get("family"))
                full = _clean(" ".join(x for x in [given, family] if x))
                if full:
                    authors.append(full)
        issued = data.get("issued") or {}
        parts = issued.get("date-parts") if isinstance(issued, dict) else None
        year = ""
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            year = str(parts[0][0])
        container = data.get("container-title") or ""
        if isinstance(container, list):
            container = _first(container)
        keyword = data.get("keyword") or ""
        if isinstance(keyword, list):
            keywords = tuple(_clean(x) for x in keyword if _clean(x))
        else:
            keywords = tuple(_clean(x) for x in re.split(r"[;,]", str(keyword)) if _clean(x))
        return ScholarlyRecord(
            title=_clean(data.get("title")),
            authors=tuple(authors),
            journal=_clean(container),
            year=_year(year),
            doi=_norm_doi(data.get("DOI") or data.get("doi")),
            pmid=_clean(data.get("PMID") or data.get("pmid")),
            pmcid=_clean(data.get("PMCID") or data.get("pmcid")),
            abstract=_clean(data.get("abstract")),
            keywords=keywords,
            publication_type=_clean(data.get("type")),
            source_format="csl-json",
        )

    def _parse_csl_json(self, source: Path) -> ScholarlyDocument:
        try:
            data = json.loads(source.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise RuntimeError(f"CSL-JSON解析失败: {exc}") from exc
        items = data if isinstance(data, list) else [data]
        if not items or not all(isinstance(x, dict) for x in items):
            raise RuntimeError("JSON不是可识别的CSL-JSON文献记录")
        records = [self._record_from_csl(x) for x in items]
        if not any(record.title or record.doi or record.pmid for record in records):
            raise RuntimeError("JSON不是可识别的CSL-JSON文献记录")
        units = [
            ParsedUnit(i, f"文献记录{i}", f"[CSL-JSON 文献记录 {i}]\n{record.evidence_header()}")
            for i, record in enumerate(records, 1)
        ]
        return ScholarlyDocument(units=units, records=records)

    def write_asset_manifest(
        self,
        source: Path,
        document: ScholarlyDocument,
        assets: PDFAssetStore,
    ) -> int:
        source = Path(source).resolve()
        if not document.images:
            return 0
        doc_root = assets.document_root(source)
        if doc_root.exists():
            import shutil
            shutil.rmtree(doc_root, ignore_errors=True)
        doc_root.mkdir(parents=True, exist_ok=True)
        pages: dict[str, list[dict]] = {}
        count = 0
        import shutil
        for number, image_paths in document.images.items():
            items: list[dict] = []
            page_dir = doc_root / f"page_{int(number):06d}"
            for index, image in enumerate(image_paths, start=1):
                image = Path(image)
                if not image.is_file():
                    continue
                suffix = image.suffix.lower() or ".bin"
                target = page_dir / f"image_{index:03d}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target)
                items.append({
                    "path": target.relative_to(doc_root).as_posix(),
                    "width": 0,
                    "height": 0,
                    "source_file": str(image),
                })
                count += 1
            if items:
                pages[str(int(number))] = items
        manifest = {
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "source_type": source.suffix.lower().lstrip("."),
            "page_count": len(document.units),
            "image_count": count,
            "pages": pages,
        }
        manifest_path = assets.manifest_path(source)
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
        return count
