from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .scholarly_ingest import ScholarlyDocument, ScholarlyParser, ScholarlyRecord, _clean, _first, _norm_doi, _year


_INSTALLED = False


def _tag(node) -> str:
    return str(node.tag).split("}")[-1].lower()


def _nodes(root, name: str):
    wanted = name.lower()
    return [node for node in root.iter() if _tag(node) == wanted]


def _text(node) -> str:
    return _clean(" ".join(node.itertext())) if node is not None else ""


def _first_node_text(root, names: tuple[str, ...]) -> str:
    for name in names:
        for node in _nodes(root, name):
            value = _text(node)
            if value:
                return value
    return ""


def _is_pubmed_xml(root) -> bool:
    return any(_tag(node) == "pubmedarticle" for node in root.iter())


def _pubmed_record(article) -> ScholarlyRecord:
    title = _first_node_text(article, ("articletitle",))
    pmid = _first_node_text(article, ("pmid",))

    journal = ""
    for journal_node in _nodes(article, "journal"):
        journal = _first_node_text(journal_node, ("title", "isoabbreviation"))
        if journal:
            break

    authors: list[str] = []
    for author in _nodes(article, "author"):
        collective = _first_node_text(author, ("collectivename",))
        if collective:
            authors.append(collective)
            continue
        given = _first_node_text(author, ("forename", "initials"))
        family = _first_node_text(author, ("lastname",))
        full = _clean(" ".join(value for value in (given, family) if value))
        if full:
            authors.append(full)

    abstract_parts: list[str] = []
    for node in _nodes(article, "abstracttext"):
        value = _text(node)
        if not value:
            continue
        label = _clean(node.attrib.get("Label") or node.attrib.get("NlmCategory"))
        abstract_parts.append(f"{label}：{value}" if label else value)
    abstract = "\n".join(abstract_parts)

    doi = ""
    pmcid = ""
    for node in _nodes(article, "articleid"):
        kind = _clean(node.attrib.get("IdType")).lower()
        value = _text(node)
        if kind == "doi" and value:
            doi = _norm_doi(value)
        elif kind in {"pmc", "pmcid"} and value:
            pmcid = value
    if not doi:
        for node in _nodes(article, "elocationid"):
            kind = _clean(node.attrib.get("EIdType")).lower()
            if kind == "doi":
                doi = _norm_doi(_text(node))
                if doi:
                    break

    year = ""
    for pub_date in _nodes(article, "pubdate"):
        year = _year(_first_node_text(pub_date, ("year", "medlinedate")))
        if year:
            break
    if not year:
        year = _year(_first_node_text(article, ("datecompleted", "daterevised")))

    keywords = tuple(
        dict.fromkeys(
            value
            for value in (_text(node) for node in _nodes(article, "keyword"))
            if value
        )
    )
    publication_types = tuple(
        dict.fromkeys(
            value
            for value in (
                _text(node) for node in _nodes(article, "publicationtype")
            )
            if value
        )
    )

    return ScholarlyRecord(
        title=title,
        authors=tuple(dict.fromkeys(authors)),
        journal=journal,
        year=year,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        abstract=abstract,
        keywords=keywords,
        publication_type="；".join(publication_types),
        source_format="pubmed-xml",
    )


def _parse_pubmed_xml(source: Path, root) -> ScholarlyDocument:
    from .document_ingest import ParsedUnit

    records: list[ScholarlyRecord] = []
    units: list[ParsedUnit] = []
    for article in (node for node in root.iter() if _tag(node) == "pubmedarticle"):
        record = _pubmed_record(article)
        if not (record.title or record.pmid or record.doi):
            continue
        records.append(record)
        number = len(records)
        lines = [f"[PubMed XML 文献记录 {number}]", record.evidence_header()]
        units.append(
            ParsedUnit(
                number=number,
                label=f"文献记录{number}",
                text="\n".join(lines).strip(),
            )
        )
    if not records:
        raise RuntimeError("PubMed XML未解析到文献记录")
    return ScholarlyDocument(units=units, records=records)


def install() -> None:
    """Extend the XML parser with multi-record PubMed XML support."""

    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    original_parse_xml = ScholarlyParser._parse_xml

    def _parse_xml(self, source: Path):
        source = Path(source).resolve()
        try:
            root = ET.parse(source).getroot()
        except ET.ParseError as exc:
            raise RuntimeError(f"XML解析失败: {exc}") from exc
        if _is_pubmed_xml(root):
            return _parse_pubmed_xml(source, root)
        return original_parse_xml(self, source)

    ScholarlyParser._parse_xml = _parse_xml
