from __future__ import annotations

import hashlib
import re
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import feedparser

from .abstract_enrichment import is_valid_abstract
from .config import AppConfig
from .crossref_client import CrossrefClient, load_dois, normalize_doi
from .db import (
    create_ingestion_run,
    fetch_recent_papers,
    finish_ingestion_run,
    get_connection,
    get_setting,
    initialize_database,
    insert_paper,
    paper_exists,
    paper_exists_by_doi,
    paper_exists_by_source_identity,
    set_setting,
    upsert_feed,
)
from .huggingface_client import HUGGINGFACE_DAILY_PAPERS_URL, HuggingFaceDailyPapersClient
from .models import PaperEntry


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
CROSSREF_SOURCE_URL = "https://api.crossref.org"
HUGGINGFACE_SOURCE_TITLE = "Hugging Face Daily Papers"
HUGGINGFACE_LAST_DATE_KEY = "huggingface_daily_papers_last_date"
HUGGINGFACE_MAX_BACKFILL_DAYS = 7
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_SOURCE_URL = "https://arxiv.org"
ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b", re.IGNORECASE)


def load_feeds(path: Path) -> list[str]:
    if not path.exists():
        return []

    feeds: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            feeds.append(stripped)
    return feeds


def sanitize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def slugify(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_length].strip("-") or "untitled"


def parse_published(entry: Any) -> datetime:
    candidates = [
        getattr(entry, "published", ""),
        getattr(entry, "updated", ""),
        getattr(entry, "created", ""),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = parsedate_to_datetime(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except (TypeError, ValueError, IndexError):
            continue
    return datetime.now(UTC)


def extract_authors(entry: Any) -> list[str]:
    authors = []
    for author in getattr(entry, "authors", []) or []:
        name = sanitize_text(getattr(author, "name", ""))
        if name:
            authors.append(name)
    return authors


def extract_abstract(entry: Any) -> str:
    summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
    clean = re.sub(r"<[^>]+>", " ", summary)
    sanitized = sanitize_text(clean)
    return sanitized if is_valid_abstract(sanitized) else ""


def extract_doi(entry: Any, abstract: str) -> str:
    candidates = [
        getattr(entry, "id", ""),
        getattr(entry, "link", ""),
        abstract,
    ]
    for candidate in candidates:
        match = DOI_PATTERN.search(candidate or "")
        if match:
            return match.group(0).rstrip(".,;) ")
    return ""


def extract_pdf_url(entry: Any) -> str:
    for link in getattr(entry, "links", []) or []:
        href = sanitize_text(getattr(link, "href", ""))
        content_type = sanitize_text(getattr(link, "type", ""))
        if href and (href.lower().endswith(".pdf") or content_type == "application/pdf"):
            return href

    fallback = sanitize_text(getattr(entry, "link", ""))
    return fallback if fallback.lower().endswith(".pdf") else ""


def make_entry_id(title: str, link: str, published: datetime, doi: str = "") -> str:
    raw = f"{title}|{link}|{published.isoformat()}|{doi.lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def frontmatter_list(values: list[str]) -> str:
    if not values:
        return "[]"
    escaped = [value.replace('"', '\\"') for value in values]
    return "[" + ", ".join(f'"{value}"' for value in escaped) + "]"


def render_note(entry: PaperEntry) -> str:
    safe_title = entry.title.replace('"', '\\"')
    safe_source = entry.source.replace('"', '\\"')
    authors = ", ".join(entry.authors) if entry.authors else ""
    abstract = entry.abstract or "No abstract available."
    summary = entry.ai_summary or "Pending AI or manual summary."
    tags = frontmatter_list(entry.tags)

    return f"""---
title: \"{safe_title}\"
authors: {frontmatter_list(entry.authors)}
source: \"{safe_source}\"
published: \"{entry.published}\"
url: \"{entry.link}\"
doi: \"{entry.doi}\"
pdf_url: \"{entry.pdf_url}\"
tags: {tags}
entry_id: \"{entry.entry_id}\"
added_at: \"{entry.added_at}\"
---

# {entry.title}

## Abstract

{abstract}

## AI Summary

{summary}

## Notes

- Authors: {authors or "Unknown"}
- Source: {entry.source}
- DOI: {entry.doi or "N/A"}
- PDF: {entry.pdf_url or "N/A"}

## Relevance to Research

- Add your interpretation here.
"""


def write_note(config: AppConfig, entry: PaperEntry) -> str:
    published_token = entry.published.replace("-", "") if entry.published else datetime.now(UTC).strftime("%Y%m%d")
    filename = f"{published_token}-{slugify(entry.title)}.md"
    note_path = config.literature_dir / filename

    note_entry = PaperEntry(
        entry_id=entry.entry_id,
        title=entry.title,
        authors=entry.authors,
        published=entry.published,
        abstract=entry.abstract,
        link=entry.link,
        source=entry.source,
        note_path=str(note_path),
        added_at=entry.added_at,
        doi=entry.doi,
        pdf_url=entry.pdf_url,
        ai_summary=entry.ai_summary,
        tags=entry.tags,
    )
    note_path.write_text(render_note(note_entry), encoding="utf-8")
    return str(note_path)


def _build_paper_entry(
    *,
    title: str,
    authors: list[str],
    published: str,
    abstract: str,
    link: str,
    source: str,
    added_at: str,
    doi: str = "",
    pdf_url: str = "",
    ai_summary: str = "",
    tags: list[str] | None = None,
) -> PaperEntry:
    published_dt = datetime.fromisoformat(f"{published}T00:00:00+00:00") if published else datetime.now(UTC)
    entry_id = make_entry_id(title, link, published_dt, doi=doi)
    return PaperEntry(
        entry_id=entry_id,
        title=title,
        authors=authors,
        published=published,
        abstract=abstract,
        link=link,
        source=source,
        note_path="",
        added_at=added_at,
        doi=doi,
        pdf_url=pdf_url,
        ai_summary=ai_summary,
        tags=tags or [],
    )


def ingest_feeds(config: AppConfig) -> list[PaperEntry]:
    config.ensure_directories()
    feeds = load_feeds(config.feeds_path)
    captured_at = datetime.now(UTC)
    new_entries: list[PaperEntry] = []

    with get_connection(config) as connection:
        initialize_database(connection)
        run_id = create_ingestion_run(connection, captured_at.isoformat(), len(feeds) + 1)

        for feed_url in feeds:
            parsed = feedparser.parse(feed_url)
            source_title = sanitize_text(getattr(parsed.feed, "title", "")) or feed_url
            upsert_feed(connection, feed_url, source_title, captured_at.isoformat())

            for raw_entry in parsed.entries:
                title = sanitize_text(getattr(raw_entry, "title", "Untitled"))
                published_dt = parse_published(raw_entry)
                link = sanitize_text(getattr(raw_entry, "link", ""))
                abstract = extract_abstract(raw_entry)
                authors = extract_authors(raw_entry)
                doi = extract_doi(raw_entry, abstract)
                pdf_url = extract_pdf_url(raw_entry)
                entry_id = make_entry_id(title, link, published_dt, doi=doi)

                if entry_id in {item.entry_id for item in new_entries}:
                    continue
                if paper_exists(connection, entry_id) or (doi and paper_exists_by_doi(connection, doi)):
                    continue
                if paper_exists_by_source_identity(
                    connection,
                    source=source_title,
                    link=link,
                    title=title,
                    published=published_dt.date().isoformat(),
                ):
                    continue

                paper = PaperEntry(
                    entry_id=entry_id,
                    title=title,
                    authors=authors,
                    published=published_dt.date().isoformat(),
                    abstract=abstract,
                    link=link,
                    source=source_title,
                    note_path="",
                    added_at=captured_at.isoformat(),
                    doi=doi,
                    pdf_url=pdf_url,
                    ai_summary="",
                    tags=[],
                )
                paper.note_path = write_note(config, paper)
                insert_paper(connection, paper, feed_url)
                new_entries.append(paper)

        new_entries.extend(_ingest_huggingface_daily_papers(connection, config, captured_at, existing_ids={item.entry_id for item in new_entries}))
        finish_ingestion_run(connection, run_id, datetime.now(UTC).isoformat(), len(new_entries))

    return new_entries



def ingest_huggingface_daily_papers_for_date(config: AppConfig, target_date: str) -> list[PaperEntry]:
    config.ensure_directories()
    captured_at = datetime.now(UTC)
    try:
        requested_date = date.fromisoformat(target_date).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid target date: {target_date}") from exc

    with get_connection(config) as connection:
        initialize_database(connection)
        upsert_feed(connection, HUGGINGFACE_DAILY_PAPERS_URL, HUGGINGFACE_SOURCE_TITLE, captured_at.isoformat())
        return _ingest_huggingface_records_for_dates(
            connection,
            config,
            captured_at,
            [requested_date],
            existing_ids=set(),
            update_last_date=False,
        )


def _ingest_huggingface_daily_papers(connection: Any, config: AppConfig, captured_at: datetime, existing_ids: set[str]) -> list[PaperEntry]:
    upsert_feed(connection, HUGGINGFACE_DAILY_PAPERS_URL, HUGGINGFACE_SOURCE_TITLE, captured_at.isoformat())
    current_date = captured_at.date()
    return _ingest_huggingface_records_for_dates(
        connection,
        config,
        captured_at,
        _huggingface_dates_to_fetch(connection, current_date),
        existing_ids=existing_ids,
        update_last_date=True,
    )



def _ingest_huggingface_records_for_dates(
    connection: Any,
    config: AppConfig,
    captured_at: datetime,
    target_dates: list[str],
    *,
    existing_ids: set[str],
    update_last_date: bool,
) -> list[PaperEntry]:
    client = HuggingFaceDailyPapersClient()
    new_entries: list[PaperEntry] = []
    for target_date in target_dates:
        try:
            records = client.fetch_daily_papers(date=target_date)
        except Exception:
            continue
        for record in records:
            paper = _build_paper_entry(
                title=record.title,
                authors=record.authors,
                published=record.published,
                abstract=record.abstract,
                link=record.link,
                source=record.source,
                added_at=captured_at.isoformat(),
                doi=record.doi,
                pdf_url=record.pdf_url,
                tags=record.tags or [],
            )
            if paper.entry_id in existing_ids:
                continue
            if paper_exists(connection, paper.entry_id) or (paper.doi and paper_exists_by_doi(connection, paper.doi)):
                continue
            paper.note_path = write_note(config, paper)
            insert_paper(connection, paper, HUGGINGFACE_DAILY_PAPERS_URL)
            new_entries.append(paper)
            existing_ids.add(paper.entry_id)
    if update_last_date:
        set_setting(connection, HUGGINGFACE_LAST_DATE_KEY, captured_at.date().isoformat())
    return new_entries


def _huggingface_dates_to_fetch(connection: Any, current_date: date) -> list[str]:
    stored_value = get_setting(connection, HUGGINGFACE_LAST_DATE_KEY)
    if not stored_value:
        return [current_date.isoformat()]
    try:
        last_date = date.fromisoformat(stored_value)
    except ValueError:
        return [current_date.isoformat()]
    start_date = min(last_date, current_date)
    if (current_date - start_date).days >= HUGGINGFACE_MAX_BACKFILL_DAYS:
        start_date = current_date - timedelta(days=HUGGINGFACE_MAX_BACKFILL_DAYS - 1)
    return [(start_date + timedelta(days=offset)).isoformat() for offset in range((current_date - start_date).days + 1)]


def import_dois(config: AppConfig, doi_path: str | None = None) -> dict[str, Any]:
    config.ensure_directories()
    path = doi_path or str(config.dois_path)
    dois = load_dois(path)
    client = CrossrefClient()
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    captured_at = datetime.now(UTC).isoformat()

    with get_connection(config) as connection:
        initialize_database(connection)
        upsert_feed(connection, CROSSREF_SOURCE_URL, "Crossref DOI Import", captured_at)

        for doi in dois:
            try:
                normalized_doi = normalize_doi(doi)
                if paper_exists_by_doi(connection, normalized_doi):
                    skipped.append({"doi": normalized_doi, "reason": "existing doi"})
                    continue

                record = client.fetch_work(normalized_doi)
                paper = _build_paper_entry(
                    title=record.title,
                    authors=record.authors,
                    published=record.published,
                    abstract=record.abstract,
                    link=record.link,
                    source=record.source,
                    added_at=captured_at,
                    doi=record.doi,
                    pdf_url=record.pdf_url,
                    tags=record.tags,
                )
                if paper_exists(connection, paper.entry_id):
                    skipped.append({"doi": normalized_doi, "reason": "existing entry_id"})
                    continue

                paper.note_path = write_note(config, paper)
                insert_paper(connection, paper, CROSSREF_SOURCE_URL)
                imported.append({"doi": record.doi, "entry_id": paper.entry_id, "title": paper.title})
            except Exception as exc:
                failed.append({"doi": doi, "error": str(exc)})

    return {
        "doi_file": str(Path(path).resolve()),
        "requested": len(dois),
        "imported": len(imported),
        "skipped": len(skipped),
        "failed": len(failed),
        "imported_items": imported,
        "skipped_items": skipped,
        "failed_items": failed,
    }


def import_url(config: AppConfig, raw_value: str) -> dict[str, Any]:
    config.ensure_directories()
    requested = sanitize_text(raw_value)
    if not requested:
        raise ValueError("URL or identifier is required.")

    captured_at = datetime.now(UTC).isoformat()
    imported: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    with get_connection(config) as connection:
        initialize_database(connection)

        try:
            paper, feed_url, feed_title, identifier = _build_paper_from_url_input(requested, captured_at)
            upsert_feed(connection, feed_url, feed_title, captured_at)

            if paper.doi and paper_exists_by_doi(connection, paper.doi):
                skipped.append({"url": requested, "identifier": paper.doi, "reason": "existing doi"})
            elif paper_exists(connection, paper.entry_id):
                skipped.append({"url": requested, "identifier": identifier, "reason": "existing entry_id"})
            elif paper_exists_by_source_identity(
                connection,
                source=paper.source,
                link=paper.link,
                title=paper.title,
                published=paper.published,
            ):
                skipped.append({"url": requested, "identifier": identifier, "reason": "existing source identity"})
            else:
                paper.note_path = write_note(config, paper)
                insert_paper(connection, paper, feed_url)
                imported.append({"url": requested, "identifier": identifier, "entry_id": paper.entry_id, "title": paper.title})
        except Exception as exc:
            failed.append({"url": requested, "error": str(exc)})

    return {
        "requested_url": requested,
        "requested": 1,
        "imported": len(imported),
        "skipped": len(skipped),
        "failed": len(failed),
        "imported_items": imported,
        "skipped_items": skipped,
        "failed_items": failed,
    }



def _build_paper_from_url_input(raw_value: str, added_at: str) -> tuple[PaperEntry, str, str, str]:
    doi = _try_extract_doi_from_input(raw_value)
    if doi:
        client = CrossrefClient()
        record = client.fetch_work(doi)
        paper = _build_paper_entry(
            title=record.title,
            authors=record.authors,
            published=record.published,
            abstract=record.abstract,
            link=record.link,
            source=record.source,
            added_at=added_at,
            doi=record.doi,
            pdf_url=record.pdf_url,
            tags=list(dict.fromkeys([*record.tags, "url-import"])),
        )
        return paper, CROSSREF_SOURCE_URL, "Crossref URL Import", record.doi or doi

    arxiv_id = _extract_arxiv_id_from_input(raw_value)
    if arxiv_id:
        return _build_arxiv_paper_entry(arxiv_id, added_at)

    raise ValueError("Unsupported URL. Paste a DOI URL, arXiv URL, DOI, or arXiv identifier.")



def _try_extract_doi_from_input(raw_value: str) -> str:
    candidate = sanitize_text(unquote(raw_value))
    if not candidate:
        return ""
    try:
        return normalize_doi(candidate)
    except ValueError:
        pass

    match = DOI_PATTERN.search(candidate)
    if not match:
        return ""
    try:
        return normalize_doi(match.group(0))
    except ValueError:
        return ""



def _extract_arxiv_id_from_input(raw_value: str) -> str:
    candidate = sanitize_text(unquote(raw_value))
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    query_values: list[str] = []
    if parsed.query:
        query = parse_qs(parsed.query)
        query_values.extend(query.get("id_list", []))
        query_values.extend(query.get("id", []))

    path_candidates: list[str] = []
    if parsed.scheme and parsed.netloc:
        host = parsed.netloc.lower()
        path = parsed.path.strip("/")
        if "arxiv.org" in host and path:
            segments = [segment for segment in path.split("/") if segment]
            if segments:
                if segments[0] in {"abs", "pdf", "html"} and len(segments) > 1:
                    path_candidates.append(segments[1])
                else:
                    path_candidates.append(segments[-1])
    else:
        path_candidates.append(candidate)

    for value in [*query_values, *path_candidates, candidate]:
        cleaned = sanitize_text(value)
        if cleaned.lower().endswith(".pdf"):
            cleaned = cleaned[:-4]
        cleaned = cleaned.strip("/")
        match = ARXIV_ID_PATTERN.search(cleaned)
        if match:
            return match.group(1)
    return ""



def _build_arxiv_paper_entry(arxiv_id: str, added_at: str) -> tuple[PaperEntry, str, str, str]:
    parsed = feedparser.parse(f"{ARXIV_API_URL}?id_list={arxiv_id}")
    entries = getattr(parsed, "entries", []) or []
    if not entries:
        raise ValueError(f"arXiv item not found: {arxiv_id}")

    raw_entry = entries[0]
    title = sanitize_text(getattr(raw_entry, "title", "Untitled"))
    published_dt = parse_published(raw_entry)
    published = published_dt.date().isoformat()
    link = sanitize_text(getattr(raw_entry, "link", "")) or f"https://arxiv.org/abs/{arxiv_id}"
    abstract = extract_abstract(raw_entry)
    authors = extract_authors(raw_entry)
    pdf_url = extract_pdf_url(raw_entry) or f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    paper = _build_paper_entry(
        title=title,
        authors=authors,
        published=published,
        abstract=abstract,
        link=link,
        source="arXiv",
        added_at=added_at,
        doi="",
        pdf_url=pdf_url,
        tags=["arxiv", "url-import"],
    )
    return paper, ARXIV_SOURCE_URL, "arXiv URL Import", arxiv_id


def generate_digest(config: AppConfig, days: int = 7) -> Path:
    config.ensure_directories()
    with get_connection(config) as connection:
        initialize_database(connection)
        recent = fetch_recent_papers(connection, days)

    now = datetime.now(UTC)
    digest_path = config.weekly_notes_dir / f"{now.strftime('%Y-W%W')}.md"
    digest_path.write_text(render_digest(recent, now, days), encoding="utf-8")
    return digest_path


def render_digest(entries: list[dict[str, Any]], now: datetime, days: int) -> str:
    lines = [
        "---",
        f'title: "Weekly Research Digest {now.strftime("%Y-W%W")}"',
        f'date: "{now.date().isoformat()}"',
        f'window_days: "{days}"',
        'tags: ["weekly-digest"]',
        "---",
        "",
        f"# Weekly Research Digest {now.strftime('%Y-W%W')}",
        "",
        f"Captured papers from the last {days} days: {len(entries)}",
        "",
        "## New Papers",
        "",
    ]

    if not entries:
        lines.append("- No new papers captured in this period.")
    else:
        for item in entries:
            authors = ", ".join(item.get("authors", [])) or "Unknown authors"
            doi = f" | DOI: {item['doi']}" if item.get("doi") else ""
            lines.append(
                f"- [{item['title']}]({item['link']}) | {authors} | {item['source']}{doi} | note: {item['note_path']}"
            )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "- Add synthesized findings here.",
            "",
            "## Potential Citations",
            "",
            "- Add the most relevant papers here.",
            "",
            "## Research Ideas",
            "",
            "- Add emerging themes, gaps, and next-step ideas here.",
            "",
        ]
    )
    return "\n".join(lines)
