from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .openai_client import load_dotenv


CROSSREF_API_BASE = "https://api.crossref.org/works"
JATS_TAG_PATTERN = re.compile(r"<[^>]+>")
DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)


@dataclass(frozen=True)
class CrossrefRecord:
    doi: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    link: str
    source: str
    pdf_url: str
    tags: list[str]


@dataclass(frozen=True)
class CrossrefMatch:
    record: CrossrefRecord
    score: float


class CrossrefClient:
    def __init__(self, mailto: str | None = None) -> None:
        load_dotenv()
        import os

        self._mailto = mailto or os.getenv("CROSSREF_MAILTO") or os.getenv("OPENAI_EMAIL") or ""

    def fetch_work(self, doi: str) -> CrossrefRecord:
        normalized_doi = normalize_doi(doi)
        encoded_doi = urllib.parse.quote(normalized_doi, safe="")
        url = f"{CROSSREF_API_BASE}/{encoded_doi}"
        payload = self._get_json(url)
        return self._parse_message(payload["message"])

    def search_best_match(self, title: str, *, top_rows: int = 5) -> CrossrefMatch | None:
        normalized_title = normalize_title(title)
        if not normalized_title:
            return None

        params = {"query.title": normalized_title, "rows": str(top_rows)}
        query_string = urllib.parse.urlencode(params)
        payload = self._get_json(f"{CROSSREF_API_BASE}?{query_string}")
        items = payload.get("message", {}).get("items", [])

        best_match: CrossrefMatch | None = None
        for item in items:
            record = self._parse_message(item)
            score = title_similarity(normalized_title, normalize_title(record.title))
            match = CrossrefMatch(record=record, score=score)
            if best_match is None or match.score > best_match.score:
                best_match = match
        return best_match

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    )
    def _get_json(self, url: str) -> dict[str, Any]:
        full_url = url
        if self._mailto:
            separator = "&" if "?" in url else "?"
            full_url = f"{url}{separator}mailto={urllib.parse.quote(self._mailto, safe='@')}"

        with httpx.Client(timeout=30) as client:
            response = client.get(
                full_url,
                headers={
                    "User-Agent": self._build_user_agent(),
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()

    def _build_user_agent(self) -> str:
        if self._mailto:
            return f"ResearchAgent/1.0 (mailto:{self._mailto})"
        return "ResearchAgent/1.0"

    def _parse_message(self, message: dict[str, Any]) -> CrossrefRecord:
        title = _first_text(message.get("title")) or _first_text(message.get("short-title")) or "Untitled"
        source = _first_text(message.get("container-title")) or _first_text(message.get("publisher")) or "Crossref"
        link = _extract_landing_page(message)
        pdf_url = _extract_pdf_url(message)
        abstract = _clean_abstract(message.get("abstract", ""))
        published = _extract_published_date(message)
        authors = _extract_authors(message.get("author", []))
        doi = normalize_doi(message.get("DOI", "")) if message.get("DOI") else ""
        tags = ["crossref"]
        return CrossrefRecord(doi=doi, title=title, authors=authors, published=published, abstract=abstract, link=link, source=source, pdf_url=pdf_url, tags=tags)


def normalize_doi(raw_doi: str) -> str:
    candidate = raw_doi.strip().lstrip("\ufeff")
    candidate = re.sub(r"^https?://(dx\.)?doi\.org/", "", candidate, flags=re.IGNORECASE)
    match = DOI_PATTERN.search(candidate)
    if not match:
        raise ValueError(f"Invalid DOI: {raw_doi}")
    return match.group(0).rstrip(".,;) ")


def load_dois(path: str) -> list[str]:
    from pathlib import Path

    doi_path = Path(path)
    if not doi_path.exists():
        return []

    dois: list[str] = []
    for raw_line in doi_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.lstrip("\ufeff")
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            dois.append(normalize_doi(stripped))
    return dois


def normalize_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def title_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_authors(author_items: list[dict[str, Any]]) -> list[str]:
    authors: list[str] = []
    for item in author_items:
        given = str(item.get("given", "")).strip()
        family = str(item.get("family", "")).strip()
        full_name = " ".join(part for part in [given, family] if part).strip()
        if not full_name:
            full_name = str(item.get("name", "")).strip()
        if full_name:
            authors.append(full_name)
    return authors


def _extract_published_date(message: dict[str, Any]) -> str:
    date_fields = [message.get("published-print"), message.get("published-online"), message.get("published"), message.get("created")]
    for field in date_fields:
        date_parts = (field or {}).get("date-parts", [])
        if not date_parts:
            continue
        parts = date_parts[0]
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        day = int(parts[2]) if len(parts) > 2 else 1
        return datetime(year, month, day, tzinfo=UTC).date().isoformat()
    return datetime.now(UTC).date().isoformat()


def _clean_abstract(raw_abstract: str) -> str:
    if not raw_abstract:
        return ""
    cleaned = JATS_TAG_PATTERN.sub(" ", raw_abstract)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_landing_page(message: dict[str, Any]) -> str:
    resource = message.get("resource", {})
    primary = resource.get("primary", {})
    if isinstance(primary, dict) and primary.get("URL"):
        return str(primary["URL"]).strip()
    if message.get("URL"):
        return str(message["URL"]).strip()
    return ""


def _extract_pdf_url(message: dict[str, Any]) -> str:
    for link in message.get("link", []) or []:
        content_type = str(link.get("content-type", "")).strip().lower()
        url = str(link.get("URL", "")).strip()
        if url and (content_type == "application/pdf" or url.lower().endswith(".pdf")):
            return url
    return ""
