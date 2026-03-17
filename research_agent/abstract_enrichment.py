from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import pymupdf

from .advanced_scraper import fetch_page_html_with_scrapling
from .config import AppConfig
from .crossref_client import CrossrefClient
from .openai_client import DEFAULT_ANSWER_MODEL, OpenAIAnswerClient
from .db import fetch_pdf_catalog_paths, fetch_paper, get_connection, initialize_database, update_paper_abstract

USER_AGENT = "Mozilla/5.0 (ResearchAgent/1.0; +https://localhost)"
ABSTRACT_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|summary)\s*[:\n]\s*(.+?)(?=\n\s*(?:keywords?|index terms?|highlights|introduction|background|1\.?\s+introduction)\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)
WHITESPACE_RE = re.compile(r"\s+")
JSON_ABSTRACT_PATTERNS = (
    re.compile(r'"abstract"\s*:\s*"(.+?)"', re.IGNORECASE | re.DOTALL),
    re.compile(r'"description"\s*:\s*"(.+?)"', re.IGNORECASE | re.DOTALL),
)
ELSEVIER_PII_RE = re.compile(r"/pii/([A-Z0-9]+)", re.IGNORECASE)
ABSTRACT_HEADING_RE = re.compile(r"^(?:abstract|a\s*b\s*s\s*t\s*r\s*a\s*c\s*t|summary)$", re.IGNORECASE)
STOP_HEADING_RE = re.compile(
    r"^(?:keywords?|index terms?|introduction|background|1\.?\s+introduction|references|acknowledg(?:e)?ments?)$",
    re.IGNORECASE,
)
MIN_ABSTRACT_LENGTH = 100
LLM_ABSTRACT_PAGE_LIMIT = 3


INVALID_ABSTRACT_PATTERNS = (
    re.compile(r"^information systems research, ahead of print\.?$", re.IGNORECASE),
    re.compile(r"^ahead of print\.?$", re.IGNORECASE),
    re.compile(r"^[A-Za-z &]+,\s*ahead of print\.?$", re.IGNORECASE),
    re.compile(r"^information systems research, volume .*issue .*page .*\.?$", re.IGNORECASE),
    re.compile(r"^[A-Za-z ]+, volume .*issue .*page .*\.?$", re.IGNORECASE),
)


def is_valid_abstract(value: str) -> bool:
    cleaned = _clean_text(value)
    if len(cleaned) < MIN_ABSTRACT_LENGTH:
        return False
    lowered = cleaned.lower()
    if lowered in {"abstract", "summary", "no abstract available."}:
        return False
    if any(pattern.fullmatch(cleaned) for pattern in INVALID_ABSTRACT_PATTERNS):
        return False
    if ('volume ' in lowered and 'issue ' in lowered and 'page ' in lowered and len(cleaned) < 220):
        return False
    return True


class _MetaTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        attr_map = {key.lower(): (value or "") for key, value in attrs}
        key = attr_map.get("name") or attr_map.get("property") or attr_map.get("http-equiv")
        content = attr_map.get("content", "").strip()
        if key and content and key not in self.meta:
            self.meta[key.lower()] = content


class _TextCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)



def _clean_text(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()



def _normalize_html_text(value: str) -> str:
    collector = _TextCollector()
    collector.feed(value)
    return _clean_text(unescape(" ".join(collector.parts)))



def _extract_pii(url: str) -> str:
    match = ELSEVIER_PII_RE.search(url)
    return match.group(1).strip() if match else ""



def _sciencedirect_urls_from_pii(pii: str) -> list[str]:
    if not pii:
        return []
    return [
        f"https://www.sciencedirect.com/science/article/pii/{pii}",
        f"https://www.sciencedirect.com/science/article/pii/{pii}/pdf",
        f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true",
    ]



def _sciencedirect_urls_from_doi(doi: str) -> list[str]:
    normalized = (doi or "").strip()
    if not normalized:
        return []
    encoded = quote(normalized, safe="")
    return [
        f"https://doi.org/{normalized}",
        f"https://www.sciencedirect.com/science/article/pii/{encoded}",
        f"https://www.sciencedirect.com/science/article/abs/pii/{encoded}",
    ]



def _candidate_pdf_urls(paper: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("pdf_url", "link"):
        value = str(paper.get(key) or "").strip()
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            urls.append(value)
    pii = _extract_pii(str(paper.get("link") or "").strip())
    if pii:
        urls.extend(_sciencedirect_urls_from_pii(pii))
    doi = str(paper.get("doi") or "").strip()
    if doi and "elsevier" in str(paper.get("source") or "").lower():
        urls.extend(_sciencedirect_urls_from_doi(doi))
    return list(dict.fromkeys(urls))



def _candidate_page_urls(paper: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for key in ("link", "pdf_url"):
        value = str(paper.get(key) or "").strip()
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"}:
            urls.append(value)
    pii = _extract_pii(str(paper.get("link") or "").strip())
    if pii:
        urls.append(f"https://www.sciencedirect.com/science/article/pii/{pii}")
    doi = str(paper.get("doi") or "").strip()
    if doi:
        urls.append(f"https://doi.org/{doi}")
        if "elsevier" in str(paper.get("source") or "").lower():
            urls.extend(_sciencedirect_urls_from_doi(doi)[:1])
    return list(dict.fromkeys(urls))



def _fetch_url_bytes(url: str, *, accept: str = "*/*") -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept, "Referer": "https://www.google.com/"})
    with urlopen(request, timeout=20) as response:
        return response.read()



def _extract_abstract_from_text(text: str) -> str:
    normalized = text.replace("\r", "\n")
    match = ABSTRACT_LABEL_RE.search(normalized)
    if match:
        candidate = _clean_text(match.group(1))
        if is_valid_abstract(candidate):
            return candidate

    lines = [line.strip() for line in normalized.splitlines()]
    for index, line in enumerate(lines[:100]):
        if not line:
            continue
        if ABSTRACT_HEADING_RE.fullmatch(line):
            collected: list[str] = []
            for candidate in lines[index + 1:index + 30]:
                if not candidate:
                    if collected:
                        break
                    continue
                if STOP_HEADING_RE.fullmatch(candidate):
                    break
                collected.append(candidate)
            cleaned = _clean_text(" ".join(collected))
            if len(cleaned) >= 200 and is_valid_abstract(cleaned):
                return cleaned

    paragraphs = [_clean_text(chunk) for chunk in re.split(r"\n\s*\n", normalized) if _clean_text(chunk)]
    for paragraph in paragraphs[:10]:
        lowered = paragraph.lower()
        if lowered.startswith("abstract"):
            cleaned = _clean_text(paragraph[len("abstract"):].lstrip(" :-"))
            if len(cleaned) >= 200 and is_valid_abstract(cleaned):
                return cleaned
        if 500 <= len(paragraph) <= 3500 and "copyright" not in lowered and "all rights reserved" not in lowered and is_valid_abstract(paragraph):
            return paragraph
    return ""



def _pdf_page_texts(document: pymupdf.Document, *, max_pages: int = LLM_ABSTRACT_PAGE_LIMIT) -> list[str]:
    chunks: list[str] = []
    for index in range(min(max_pages, document.page_count)):
        page = document.load_page(index)
        page_parts: list[str] = []
        text = page.get_text("text")
        if text:
            page_parts.append(text)
        blocks = page.get_text("blocks")
        if blocks:
            ordered_blocks = "\n".join(
                _clean_text(str(block[4]))
                for block in sorted(blocks, key=lambda item: (round(item[1], 1), round(item[0], 1)))
                if len(block) >= 5 and _clean_text(str(block[4]))
            )
            if ordered_blocks:
                page_parts.append(ordered_blocks)
        merged = _clean_text("\n\n".join(page_parts))
        if merged:
            chunks.append(merged)
    return chunks


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _build_llm_abstract_prompt(page_texts: list[str]) -> str:
    pages = "\n\n".join(
        f"[Page {index}]\n{text}"
        for index, text in enumerate(page_texts, start=1)
    )
    return f"""You are extracting an abstract from the opening pages of an academic paper.
Use only the provided text.
If the paper does not contain an abstract in these pages, return an empty abstract.
Do not summarize the paper from the introduction.
Do not infer or rewrite findings.
Return strict JSON only in this exact shape:
{{"found": true, "abstract": "..."}}
or
{{"found": false, "abstract": ""}}

Rules:
- Extract the paper's abstract as it appears, or as close to the original wording as possible.
- If the visible text is not an abstract section, return found=false.
- If the candidate text is only journal metadata, ahead-of-print text, copyright text, or article navigation text, return found=false.
- Never fabricate a new abstract.

Paper opening pages:
{pages}
"""


def _extract_abstract_with_llm(page_texts: list[str], *, model: str = DEFAULT_ANSWER_MODEL) -> str:
    if not page_texts:
        return ""
    try:
        response = OpenAIAnswerClient().create_answer(_build_llm_abstract_prompt(page_texts), model=model)
    except Exception:
        return ""
    payload = _extract_json_object(response.text)
    if not payload or not payload.get("found"):
        return ""
    candidate = _clean_text(str(payload.get("abstract") or ""))
    return candidate if is_valid_abstract(candidate) else ""


def _extract_abstract_from_pdf_document(document: pymupdf.Document, *, max_pages: int = LLM_ABSTRACT_PAGE_LIMIT) -> str:
    return _extract_abstract_from_text("\n\n".join(_pdf_page_texts(document, max_pages=max_pages)))



def _extract_abstract_from_local_pdf(path: Path) -> tuple[str, list[str]]:
    if not path.exists() or path.suffix.lower() != ".pdf":
        return ""
    try:
        document = pymupdf.open(path)
    except Exception:
        return "", []
    try:
        page_texts = _pdf_page_texts(document)
        return _extract_abstract_from_text("\n\n".join(page_texts)), page_texts
    finally:
        document.close()



def _extract_abstract_from_remote_pdf(url: str) -> tuple[str, list[str]]:
    try:
        payload = _fetch_url_bytes(url, accept="application/pdf,*/*")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return "", []
    try:
        document = pymupdf.Document(stream=payload)
    except Exception:
        return "", []
    try:
        page_texts = _pdf_page_texts(document)
        return _extract_abstract_from_text("\n\n".join(page_texts)), page_texts
    finally:
        document.close()



def _extract_json_string(value: str) -> str:
    return _clean_text(unescape(value.encode("utf-8").decode("unicode_escape", errors="ignore")))



def _extract_abstract_from_html(html: str, url: str = "") -> str:
    parser = _MetaTagParser()
    parser.feed(html)
    for key in (
        "citation_abstract",
        "dc.description.abstract",
        "dc.description",
        "description",
        "og:description",
        "twitter:description",
    ):
        cleaned = _normalize_html_text(parser.meta.get(key, ""))
        if cleaned and is_valid_abstract(cleaned):
            return cleaned

    if "sciencedirect.com" in url or "linkinghub.elsevier.com" in url:
        for pattern in (
            r'"abstracts"\s*:\s*\[\s*\{\s*"content"\s*:\s*"(.*?)"',
            r'"abstract"\s*:\s*"(.*?)"',
            r'<div[^>]+class="[^"]*abstract[^"]*"[^>]*>(.*?)</div>',
            r'<section[^>]+id="abstracts?"[^>]*>(.*?)</section>',
        ):
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = _normalize_html_text(match.group(1))
                cleaned = re.sub(r'^Policy and Practice Abstract\s*', '', cleaned, flags=re.IGNORECASE)
                if cleaned and is_valid_abstract(cleaned):
                    return cleaned

    if "nature.com" in url:
        for pattern in (
            r'<div[^>]+id="Abs1-content"[^>]*>(.*?)</div>',
            r'<section[^>]+aria-labelledby="Abs1-heading"[^>]*>(.*?)</section>',
            r'"description"\s*:\s*"(.*?)"',
        ):
            match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = _normalize_html_text(match.group(1))
                cleaned = re.sub(r'^Policy and Practice Abstract\s*', '', cleaned, flags=re.IGNORECASE)
                if cleaned and is_valid_abstract(cleaned):
                    return cleaned

    for pattern in JSON_ABSTRACT_PATTERNS:
        match = pattern.search(html)
        if match:
            cleaned = _extract_json_string(match.group(1))
            cleaned = re.sub(r'^Policy and Practice Abstract\s*', '', cleaned, flags=re.IGNORECASE)
            if cleaned and is_valid_abstract(cleaned):
                return cleaned

    for match in re.finditer(r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', html, re.IGNORECASE | re.DOTALL):
        snippet = match.group(1)
        json_description = re.search(r'"description"\s*:\s*"(.*?)"', snippet, re.IGNORECASE | re.DOTALL)
        if json_description:
            cleaned = _extract_json_string(json_description.group(1))
            cleaned = re.sub(r'^Policy and Practice Abstract\s*', '', cleaned, flags=re.IGNORECASE)
            if cleaned and is_valid_abstract(cleaned):
                return cleaned
    return ""



def _extract_abstract_from_web_page(url: str) -> str:
    try:
        html = _fetch_url_bytes(url, accept="text/html,application/xhtml+xml,*/*").decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return ""
    return _extract_abstract_from_html(html, url)



def _extract_abstract_with_scrapling(url: str) -> str:
    html = fetch_page_html_with_scrapling(url)
    if not html:
        return ""
    return _extract_abstract_from_html(html, url)



def enrich_paper_abstract(config: AppConfig, paper: dict[str, Any]) -> dict[str, str]:
    abstract = _clean_text(str(paper.get("abstract") or ""))
    if is_valid_abstract(abstract):
        return {"abstract": abstract, "source": "existing"}

    entry_id = str(paper.get("entry_id") or "")
    if not entry_id:
        return {"abstract": "", "source": "missing-entry-id"}

    with get_connection(config) as connection:
        initialize_database(connection)
        local_paths = fetch_pdf_catalog_paths(connection, entry_id)

    for raw_path in local_paths:
        extracted, page_texts = _extract_abstract_from_local_pdf(Path(raw_path))
        if extracted:
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, extracted)
            return {"abstract": extracted, "source": f"local-pdf:{raw_path}"}
        llm_extracted = _extract_abstract_with_llm(page_texts)
        if llm_extracted:
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, llm_extracted)
            return {"abstract": llm_extracted, "source": f"llm-pdf:{raw_path}"}

    for url in _candidate_pdf_urls(paper):
        extracted, page_texts = _extract_abstract_from_remote_pdf(url)
        if extracted:
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, extracted)
            return {"abstract": extracted, "source": f"remote-pdf:{url}"}
        llm_extracted = _extract_abstract_with_llm(page_texts)
        if llm_extracted:
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, llm_extracted)
            return {"abstract": llm_extracted, "source": f"llm-pdf:{url}"}

    for url in _candidate_page_urls(paper):
        extracted = _extract_abstract_from_web_page(url)
        if extracted:
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, extracted)
            return {"abstract": extracted, "source": f"web:{url}"}
        extracted = _extract_abstract_with_scrapling(url)
        if extracted:
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, extracted)
            return {"abstract": extracted, "source": f"scrapling:{url}"}

    doi = str(paper.get("doi") or "").strip()
    if doi:
        try:
            record = CrossrefClient().fetch_work(doi)
        except Exception:
            record = None
        if record and is_valid_abstract(record.abstract):
            with get_connection(config) as connection:
                initialize_database(connection)
                update_paper_abstract(connection, entry_id, record.abstract)
            return {"abstract": record.abstract, "source": f"crossref:{doi}"}

    return {"abstract": "", "source": "unresolved"}



def enrich_missing_abstracts(config: AppConfig, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for paper in papers:
        updated = dict(paper)
        result = enrich_paper_abstract(config, updated)
        if result["abstract"]:
            updated["abstract"] = result["abstract"]
        updated["abstract_source"] = result["source"]
        enriched.append(updated)
    return enriched



def _load_missing_abstract_candidates(config: AppConfig, *, limit: int, days: int | None = None) -> list[dict[str, Any]]:
    with get_connection(config) as connection:
        initialize_database(connection)
        sql = [
            "SELECT p.entry_id, p.title, p.published, p.abstract, p.link, p.source, p.note_path, p.added_at, p.doi, p.pdf_url, p.ai_summary, p.tags,",
            "(SELECT GROUP_CONCAT(name, '||') FROM (SELECT a.name AS name FROM paper_authors pa JOIN authors a ON a.id = pa.author_id WHERE pa.paper_entry_id = p.entry_id ORDER BY pa.author_order)) AS author_names",
            "FROM papers p",
            "WHERE trim(coalesce(p.abstract, '')) = ''",
        ]
        params: list[Any] = []
        if days is not None:
            sql.append("AND p.added_at >= datetime('now', ?)")
            params.append(f"-{days} days")
        sql.append("ORDER BY p.published DESC, p.added_at DESC LIMIT ?")
        params.append(limit)
        rows = connection.execute("\n".join(sql), params).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidates.append({
            "entry_id": row["entry_id"],
            "title": row["title"],
            "published": row["published"],
            "abstract": row["abstract"],
            "link": row["link"],
            "source": row["source"],
            "note_path": row["note_path"],
            "added_at": row["added_at"],
            "doi": row["doi"],
            "pdf_url": row["pdf_url"],
            "ai_summary": row["ai_summary"],
            "tags": json.loads(row["tags"] or "[]") if row["tags"] else [],
            "authors": row["author_names"].split("||") if row["author_names"] else [],
        })
    return candidates



def enrich_abstract_targets(config: AppConfig, *, entry_id: str | None = None, limit: int = 20, days: int | None = None) -> dict[str, Any]:
    if entry_id:
        with get_connection(config) as connection:
            initialize_database(connection)
            paper = fetch_paper(connection, entry_id)
        if paper is None:
            raise ValueError(f"Paper not found: {entry_id}")
        targets = [paper]
    else:
        targets = _load_missing_abstract_candidates(config, limit=limit, days=days)

    results: list[dict[str, Any]] = []
    enriched_count = 0
    for paper in targets:
        before = _clean_text(str(paper.get("abstract") or ""))
        result = enrich_paper_abstract(config, paper)
        enriched = bool(result["abstract"]) and not before
        if enriched:
            enriched_count += 1
        results.append({
            "entry_id": paper.get("entry_id"),
            "title": paper.get("title"),
            "published": paper.get("published"),
            "source": paper.get("source"),
            "enriched": enriched,
            "abstract_source": result["source"],
            "abstract_length": len(result["abstract"] or before),
        })

    return {
        "requested": len(targets),
        "enriched": enriched_count,
        "results": results,
    }
