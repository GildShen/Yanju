from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz

from .config import AppConfig
from .crossref_client import CrossrefClient, DOI_PATTERN
from .db import (
    fetch_paper_by_doi,
    get_connection,
    initialize_database,
    insert_paper,
    paper_exists,
    paper_exists_by_doi,
    upsert_feed,
    upsert_pdf_catalog_entry,
)
from .models import PaperEntry
from .pipeline import make_entry_id, write_note


CROSSREF_SOURCE_URL = "https://api.crossref.org"
MIN_CONFIDENCE_AUTO_IMPORT = 0.72
TITLE_NOISE_PREFIXES = (
    "abstract",
    "introduction",
    "keywords",
    "proceedings",
    "copyright",
)
ARXIV_PATTERN = re.compile(r"\b(?:arxiv:\s*)?(\d{4}\.\d{4,5}(?:v\d+)?)\b", re.IGNORECASE)
ARXIV_OLD_PATTERN = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b", re.IGNORECASE)
PII_PATTERN = re.compile(r"\b(S\d{15,17})\b", re.IGNORECASE)
PMID_PATTERN = re.compile(r"\bPMID[:\s]+(\d{7,9})\b", re.IGNORECASE)
FILENAME_SAFE_PATTERN = re.compile(r'[<>:"/\\|?*]+')


def catalog_pdfs(config: AppConfig, pdf_dir: str | None = None) -> dict[str, Any]:
    target_dir = Path(pdf_dir) if pdf_dir else config.default_pdf_dir
    pdf_files = sorted(target_dir.rglob("*.pdf")) if target_dir.exists() else []
    client = CrossrefClient()
    scanned = 0
    imported = 0
    linked_existing = 0
    needs_review = 0
    failed = 0
    moved_to_done = 0
    items: list[dict[str, Any]] = []

    with get_connection(config) as connection:
        initialize_database(connection)
        upsert_feed(connection, CROSSREF_SOURCE_URL, "Crossref PDF Catalog", datetime.now(UTC).isoformat())

        for pdf_path in pdf_files:
            scanned += 1
            timestamp = datetime.now(UTC).isoformat()
            final_path = pdf_path
            try:
                file_hash = _hash_file(pdf_path)
                extracted_title = extract_title_from_pdf(pdf_path)
                identifier = extract_research_identifier(pdf_path)
                if not extracted_title:
                    final_path = _apply_identifier_filename(final_path, identifier)
                    upsert_pdf_catalog_entry(connection, file_path=str(final_path.resolve()), file_name=final_path.name, file_hash=file_hash, title_extracted="", title_matched="", match_confidence=0.0, catalog_status="failed", paper_entry_id=None, notes="Could not extract a title from the PDF.", updated_at=timestamp)
                    failed += 1
                    items.append({"file": str(pdf_path), "final_file": str(final_path), "status": "failed", "reason": "title extraction failed", "identifier": identifier})
                    continue

                match = client.search_best_match(extracted_title)
                if match is None:
                    final_path = _apply_identifier_filename(final_path, identifier)
                    upsert_pdf_catalog_entry(connection, file_path=str(final_path.resolve()), file_name=final_path.name, file_hash=file_hash, title_extracted=extracted_title, title_matched="", match_confidence=0.0, catalog_status="needs_review", paper_entry_id=None, notes="Crossref returned no candidate match.", updated_at=timestamp)
                    needs_review += 1
                    items.append({"file": str(pdf_path), "final_file": str(final_path), "status": "needs_review", "title_extracted": extracted_title, "reason": "no crossref match", "identifier": identifier})
                    continue

                record = match.record
                confidence = match.score
                identifier = record.doi or identifier
                existing_entry_id: str | None = None
                status = "needs_review"
                notes = f"Crossref title match confidence: {confidence:.3f}"

                if record.doi and paper_exists_by_doi(connection, record.doi):
                    existing_paper = fetch_paper_by_doi(connection, record.doi)
                    existing_entry_id = existing_paper["entry_id"] if existing_paper else None
                    status = "linked_existing"
                    linked_existing += 1
                elif confidence >= MIN_CONFIDENCE_AUTO_IMPORT:
                    paper = _record_to_paper_entry(record, added_at=timestamp)
                    if paper_exists(connection, paper.entry_id):
                        status = "linked_existing"
                        existing_entry_id = paper.entry_id
                        linked_existing += 1
                    else:
                        paper.note_path = write_note(config, paper)
                        insert_paper(connection, paper, CROSSREF_SOURCE_URL)
                        existing_entry_id = paper.entry_id
                        status = "imported"
                        imported += 1
                else:
                    needs_review += 1

                if status in {"imported", "linked_existing"} and not _is_in_done_dir(config, final_path):
                    final_path = _move_to_done(config, final_path, published=record.published)
                    moved_to_done += 1
                    notes = f"{notes} | moved to {final_path}"

                renamed_path = _apply_identifier_filename(final_path, identifier)
                if renamed_path != final_path:
                    notes = f"{notes} | renamed to {renamed_path.name}"
                    final_path = renamed_path

                upsert_pdf_catalog_entry(connection, file_path=str(final_path.resolve()), file_name=final_path.name, file_hash=file_hash, title_extracted=extracted_title, title_matched=record.title, match_confidence=confidence, catalog_status=status, paper_entry_id=existing_entry_id, notes=notes, updated_at=timestamp)
                items.append({"file": str(pdf_path), "final_file": str(final_path), "status": status, "title_extracted": extracted_title, "title_matched": record.title, "doi": record.doi, "identifier": identifier, "confidence": round(confidence, 3), "paper_entry_id": existing_entry_id})
            except Exception as exc:
                upsert_pdf_catalog_entry(connection, file_path=str(final_path.resolve()), file_name=final_path.name, file_hash=_safe_hash(final_path), title_extracted="", title_matched="", match_confidence=0.0, catalog_status="failed", paper_entry_id=None, notes=str(exc), updated_at=timestamp)
                failed += 1
                items.append({"file": str(pdf_path), "final_file": str(final_path), "status": "failed", "reason": str(exc)})

    report = {"pdf_dir": str(target_dir.resolve()) if target_dir.exists() else str(target_dir), "done_dir": str(config.done_pdf_dir.resolve()), "scanned": scanned, "imported": imported, "linked_existing": linked_existing, "needs_review": needs_review, "failed": failed, "moved_to_done": moved_to_done, "items": items}
    report_path = config.data_dir / "pdf_catalog_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def extract_title_from_pdf(pdf_path: Path) -> str:
    with fitz.open(pdf_path) as document:
        metadata_title = str((document.metadata or {}).get("title", "") or "").strip()
        if _looks_like_title(metadata_title):
            return _normalize_extracted_title(metadata_title)
        candidate_lines: list[str] = []
        for page_index in range(min(2, document.page_count)):
            candidate_lines.extend(_candidate_lines(document.load_page(page_index).get_text("text") or ""))
    for line in candidate_lines:
        if _looks_like_title(line):
            return _normalize_extracted_title(line)
    return ""


def extract_research_identifier(pdf_path: Path) -> str:
    try:
        with fitz.open(pdf_path) as document:
            metadata = document.metadata or {}
            for candidate in (
                str(metadata.get("doi", "") or "").strip(),
                str(metadata.get("subject", "") or "").strip(),
                str(metadata.get("keywords", "") or "").strip(),
            ):
                identifier = _extract_identifier_from_text(candidate)
                if identifier:
                    return identifier

            page_text_chunks: list[str] = []
            for page_index in range(min(2, document.page_count)):
                page_text_chunks.append(document.load_page(page_index).get_text("text") or "")
    except Exception:
        return _extract_identifier_from_text(pdf_path.name)

    return _extract_identifier_from_text("\n".join(page_text_chunks)) or _extract_identifier_from_text(pdf_path.name)


def _candidate_lines(text: str) -> list[str]:
    return [line for line in (re.sub(r"\s+", " ", raw).strip() for raw in text.splitlines()) if line][:40]


def _looks_like_title(value: str) -> bool:
    line = value.strip()
    lower = line.lower()
    if not line:
        return False
    if any(lower.startswith(prefix) for prefix in TITLE_NOISE_PREFIXES):
        return False
    if len(line) < 25 or len(line) > 280:
        return False
    if sum(char.isalpha() for char in line) < 12:
        return False
    if re.fullmatch(r"[A-Z ,.&-]+", line) and len(line.split()) <= 3:
        return False
    return True


def _normalize_extracted_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().strip("-:| ")


def _extract_identifier_from_text(value: str) -> str:
    if not value:
        return ""
    doi_match = DOI_PATTERN.search(value)
    if doi_match:
        return doi_match.group(0).rstrip(".,;) ")
    for pattern in (ARXIV_PATTERN, ARXIV_OLD_PATTERN, PII_PATTERN, PMID_PATTERN):
        match = pattern.search(value)
        if match:
            return match.group(1).rstrip(".,;) ")
    return ""


def _apply_identifier_filename(pdf_path: Path, identifier: str) -> Path:
    if not identifier:
        return pdf_path
    safe_stem = _sanitize_identifier_for_filename(identifier)
    if not safe_stem:
        return pdf_path
    destination = _unique_destination_path(pdf_path.with_name(f"{safe_stem}{pdf_path.suffix.lower()}"), pdf_path)
    if destination.resolve() == pdf_path.resolve():
        return pdf_path
    return pdf_path.rename(destination)


def _sanitize_identifier_for_filename(identifier: str) -> str:
    normalized = identifier.strip()
    normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^doi:\s*", "", normalized, flags=re.IGNORECASE)
    normalized = normalized.replace("/", "_")
    normalized = FILENAME_SAFE_PATTERN.sub("_", normalized)
    normalized = re.sub(r"\s+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip(" ._")


def _year_folder_name(published: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", published or ""):
        return f"{published[:4]}"
    if re.fullmatch(r"\d{4}", published or ""):
        return f"{published}"
    return "unknown"


def _hash_file(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_hash(pdf_path: Path) -> str:
    try:
        return _hash_file(pdf_path)
    except Exception:
        return ""


def _is_in_done_dir(config: AppConfig, pdf_path: Path) -> bool:
    done_dir = config.done_pdf_dir.resolve()
    resolved_path = pdf_path.resolve()
    return resolved_path.parent == done_dir or done_dir in resolved_path.parents


def _move_to_done(config: AppConfig, pdf_path: Path, *, published: str = "") -> Path:
    year_dir = config.done_pdf_dir / _year_folder_name(published)
    year_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination_path(year_dir / pdf_path.name, pdf_path)
    shutil.move(str(pdf_path), str(destination))
    return destination


def _unique_destination_path(destination: Path, source_path: Path) -> Path:
    if destination.exists() and destination.resolve() == source_path.resolve():
        return source_path
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    candidate = destination
    while candidate.exists():
        candidate = destination.with_name(f"{stem}-{counter}{suffix}")
        counter += 1
    return candidate


def _record_to_paper_entry(record: Any, *, added_at: str) -> PaperEntry:
    published_dt = datetime.fromisoformat(f"{record.published}T00:00:00+00:00") if record.published else datetime.now(UTC)
    entry_id = make_entry_id(record.title, record.link, published_dt, doi=record.doi)
    return PaperEntry(entry_id=entry_id, title=record.title, authors=record.authors, published=record.published, abstract=record.abstract, link=record.link, source=record.source, note_path="", added_at=added_at, doi=record.doi, pdf_url=record.pdf_url, ai_summary="", tags=list(dict.fromkeys(record.tags + ["pdf-catalog"])))
