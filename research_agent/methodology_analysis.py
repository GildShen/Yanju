from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pymupdf

from .config import AppConfig
from .db import (
    fetch_latest_methodology_run,
    fetch_methodology_run,
    fetch_paper,
    fetch_pdf_catalog_paths,
    get_connection,
    initialize_database,
    save_methodology_run,
    update_methodology_run_note_path,
    update_paper_metadata,
    update_paper_abstract,
)
from .openai_client import DEFAULT_ANSWER_MODEL, OpenAIAnswerClient

PROMPT_VERSION = "v1-methodology-main-body"
USER_AGENT = "Mozilla/5.0 (ResearchAgent/1.0; +https://localhost)"
SECTION_STOP_RE = re.compile(r"^\s*(?:[ivx]+\.\s*)?(references|bibliography|appendix|appendices|online appendix|supplementary materials?)\s*(?:[:.-]|$)", re.IGNORECASE)
MIN_MAIN_BODY_CHARS = 1200


def _fetch_url_bytes(url: str, *, accept: str = "*/*") -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept, "Referer": "https://www.google.com/"})
    with urlopen(request, timeout=30) as response:
        return response.read()



def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()



def _is_stop_heading(line: str) -> bool:
    cleaned = _clean_text(line)
    if not cleaned:
        return False
    if len(cleaned) > 60:
        return False
    return bool(SECTION_STOP_RE.match(cleaned))



def _extract_pdf_pages(document: pymupdf.Document, *, max_pages: int = 25) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    stop = False
    for index in range(min(max_pages, document.page_count)):
        page = document.load_page(index)
        text = page.get_text("text") or ""
        blocks = page.get_text("blocks") or []
        block_text = "\n".join(
            _clean_text(str(block[4]))
            for block in sorted(blocks, key=lambda item: (round(item[1], 1), round(item[0], 1)))
            if len(block) >= 5 and _clean_text(str(block[4]))
        )
        merged = "\n\n".join(part for part in [text, block_text] if part).strip()
        if not merged:
            continue
        kept_lines: list[str] = []
        for raw_line in merged.splitlines():
            line = raw_line.strip()
            if _is_stop_heading(line):
                stop = True
                break
            kept_lines.append(raw_line)
        page_text = "\n".join(kept_lines).strip()
        if page_text:
            pages.append({"page": index + 1, "text": page_text})
        if stop:
            break
    return pages



def _load_pdf_pages_from_local(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.suffix.lower() != ".pdf":
        return []
    try:
        document = pymupdf.open(path)
    except Exception:
        return []
    try:
        return _extract_pdf_pages(document)
    finally:
        document.close()



def _load_pdf_pages_from_remote(url: str) -> list[dict[str, Any]]:
    try:
        payload = _fetch_url_bytes(url, accept="application/pdf,*/*")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return []
    try:
        document = pymupdf.Document(stream=payload)
    except Exception:
        return []
    try:
        return _extract_pdf_pages(document)
    finally:
        document.close()



def _resolve_source_pages(config: AppConfig, paper: dict[str, Any]) -> tuple[list[dict[str, Any]], str, str]:
    entry_id = str(paper.get("entry_id") or "")
    with get_connection(config) as connection:
        initialize_database(connection)
        local_paths = fetch_pdf_catalog_paths(connection, entry_id)
    for raw_path in local_paths:
        pages = _load_pdf_pages_from_local(Path(raw_path))
        if sum(len(page["text"]) for page in pages) >= MIN_MAIN_BODY_CHARS:
            return pages, "local_pdf", raw_path
    pdf_url = str(paper.get("pdf_url") or "").strip()
    if pdf_url:
        pages = _load_pdf_pages_from_remote(pdf_url)
        if sum(len(page["text"]) for page in pages) >= MIN_MAIN_BODY_CHARS:
            return pages, "remote_pdf", pdf_url
    return [], "metadata_only", str(paper.get("link") or "")



def _build_analysis_prompt(paper: dict[str, Any], pages: list[dict[str, Any]]) -> str:
    page_text = "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in pages)
    abstract = str(paper.get("abstract") or "").strip()
    metadata_block = "\n".join([
        f"Title: {paper.get('title', '')}",
        f"Authors: {', '.join(paper.get('authors', []))}",
        f"Source: {paper.get('source', '')}",
        f"Published: {paper.get('published', '')}",
        f"DOI: {paper.get('doi', '')}",
        f"Abstract: {abstract}",
    ])
    return f"""You are analyzing a single academic paper for research methodology.
Use only the provided metadata and paper main body.
Ignore references, bibliography, appendix, appendices, online appendix, supplementary material, and author biographies.
Do not use any evidence from those ignored sections even if fragments appear in the text.
If a field is not clear from the paper main body, explicitly say \"Not clear from the main body.\".
Write in Traditional Chinese.
Return clean markdown with exactly these sections and no others:

## Abstract
## Research Problem
## Theory And Context
## Data And Sample
## Research Design
## Measures And Constructs
## Analysis Method
## Main Findings
## Limitations
## Notes For Future Research

Requirements:
- Focus on the research body, not citation lists.
- Prefer concrete details over generic summary language.
- Mention specific methods, datasets, sample details, variables, models, or empirical strategy when available.
- Do not invent hypotheses, data, or findings.
- Keep the analysis concise but useful for later literature review and study design.

Paper metadata:
{metadata_block}

Paper main body pages:
{page_text}
"""





def _slugify(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (value or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_length].strip("-") or "untitled"


def _obsidian_wikilink(path_value: str) -> str:
    if not path_value:
        return ""
    return f"[[{Path(path_value).stem}]]"


def _yaml_escape(value: Any) -> str:
    return str(value or "").replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))


def _build_obsidian_uri(config: AppConfig, note_path: Path) -> str:
    relative = note_path.relative_to(config.vault_dir).as_posix()
    return f"obsidian://open?vault={quote(config.vault_dir.name)}&file={quote(relative, safe='/')}"


def _render_methodology_note(paper: dict[str, Any], run: dict[str, Any]) -> str:
    tags = '["methodology-analysis", "research-analysis"]'
    literature_note = _obsidian_wikilink(str(paper.get("note_path") or ""))
    title = str(paper.get("title") or "Untitled")
    authors = ", ".join(paper.get("authors", [])) or "Unknown authors"
    link = str(paper.get("link") or "")
    pdf_url = str(paper.get("pdf_url") or "")
    doi = str(paper.get("doi") or "")
    analysis_text = str(run.get("analysis_text") or "").strip()
    lines = [
        "---",
        f'title: "{_yaml_escape(title)} - Methodology Analysis"',
        f'paper_title: "{_yaml_escape(title)}"',
        f'paper_entry_id: "{_yaml_escape(paper.get("entry_id", ""))}"',
        f'methodology_run_id: "{_yaml_escape(run.get("id", ""))}"',
        f'published: "{_yaml_escape(paper.get("published", ""))}"',
        f'source: "{_yaml_escape(paper.get("source", ""))}"',
        f'model: "{_yaml_escape(run.get("model", ""))}"',
        f'prompt_version: "{_yaml_escape(run.get("prompt_version", ""))}"',
        f'source_type: "{_yaml_escape(run.get("source_type", ""))}"',
        f'page_count: "{_yaml_escape(run.get("page_count", 0))}"',
        f'doi: "{_yaml_escape(doi)}"',
        f'url: "{_yaml_escape(link)}"',
        f'pdf_url: "{_yaml_escape(pdf_url)}"',
        f'tags: {tags}',
        "---",
        "",
        f"# {title} - Methodology Analysis",
        "",
        "## Linked Paper",
        "",
        f"- Literature note: {literature_note or 'N/A'}",
        f"- Authors: {authors}",
        f"- Source: {paper.get('source', '')}",
        f"- Published: {paper.get('published', '')}",
        f"- DOI: {doi or 'N/A'}",
        f"- Source URL: {link or 'N/A'}",
        f"- PDF URL: {pdf_url or 'N/A'}",
        "",
        "## Analysis Metadata",
        "",
        f"- Run ID: {run.get('id', '')}",
        f"- Status: {run.get('status', '')}",
        f"- Model: {run.get('model', '')}",
        f"- Prompt Version: {run.get('prompt_version', '')}",
        f"- Source Type: {run.get('source_type', '')}",
        f"- Source Ref: {run.get('source_ref', '') or 'N/A'}",
        f"- Page Count: {run.get('page_count', 0)}",
        f"- Updated At: {run.get('updated_at', '')}",
        "",
        "## Full Analysis",
        "",
        analysis_text or "No analysis available.",
        "",
    ]
    return "\n".join(lines)


def export_methodology_note(config: AppConfig, run_id: int) -> dict[str, Any]:
    with get_connection(config) as connection:
        initialize_database(connection)
        run = fetch_methodology_run(connection, run_id)
        if run is None:
            raise ValueError(f"Methodology run not found: {run_id}")
        paper = fetch_paper(connection, str(run.get("paper_entry_id") or ""))
        if paper is None:
            raise ValueError(f"Paper not found for methodology run: {run_id}")

        existing_note = str(run.get("note_path") or "").strip()
        if existing_note:
            note_path = Path(existing_note)
        else:
            published_token = str(paper.get("published") or "").replace("-", "") or datetime.now(UTC).strftime("%Y%m%d")
            filename = f"{published_token}-{_slugify(str(paper.get('title') or 'untitled'))}-methodology.md"
            note_path = config.analysis_notes_dir / filename

        note_path.write_text(_render_methodology_note(paper, run), encoding="utf-8")
        updated = update_methodology_run_note_path(connection, run_id, str(note_path), datetime.now(UTC).isoformat())

    result = dict(updated or run)
    result["title"] = paper.get("title")
    result["paper_title"] = paper.get("title")
    result["note_path"] = str(note_path)
    result["obsidian_uri"] = _build_obsidian_uri(config, note_path)
    return result


def analyze_paper_methodology(config: AppConfig, entry_id: str, *, model: str = DEFAULT_ANSWER_MODEL, force: bool = False, pdf_url_override: str | None = None) -> dict[str, Any]:
    override_url = str(pdf_url_override or "").strip()
    if override_url:
        force = True

    with get_connection(config) as connection:
        initialize_database(connection)
        paper = fetch_paper(connection, entry_id)
        if paper is None:
            raise ValueError(f"Paper not found: {entry_id}")
        latest = fetch_latest_methodology_run(connection, entry_id, prompt_version=PROMPT_VERSION, model=model)
        if latest and not force:
            result = dict(latest)
            result["paper_pdf_url"] = str(paper.get("pdf_url") or "")
            return result

    if override_url:
        paper["pdf_url"] = override_url

    extracted_abstract = ""
    pages, source_type, source_ref = _resolve_source_pages(config, paper)
    if not pages:
        analysis_text = "## Abstract\nNot clear from the main body.\n\n## Research Problem\nNot clear from the main body.\n\n## Theory And Context\nNot clear from the main body.\n\n## Data And Sample\nNot clear from the main body.\n\n## Research Design\nNot clear from the main body.\n\n## Measures And Constructs\nNot clear from the main body.\n\n## Analysis Method\nNot clear from the main body.\n\n## Main Findings\nNot clear from the main body.\n\n## Limitations\nNot clear from the main body.\n\n## Notes For Future Research\nPDF full text was not available, so this analysis could not inspect the main body."
        status = "no_source"
    else:
        prompt = _build_analysis_prompt(paper, pages)
        response = OpenAIAnswerClient().create_answer(prompt=prompt, model=model)
        analysis_text = response.text.strip()
        status = "completed"

        abstract_match = re.search(r"##\s+Abstract\s*\n(.*?)(?=\n## |\Z)", analysis_text, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            extracted_abstract = abstract_match.group(1).strip()

    now = datetime.now(UTC).isoformat()
    payload = {
        "paper_entry_id": entry_id,
        "title": paper.get("title"),
        "status": status,
        "source_type": source_type,
        "source_ref": source_ref,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "analysis_text": analysis_text,
        "note_path": "",
        "page_count": len(pages),
        "created_at": now,
        "updated_at": now,
    }

    with get_connection(config) as connection:
        initialize_database(connection)
        if override_url and status == "completed" and source_type == "remote_pdf" and source_ref == override_url:
            update_paper_metadata(connection, entry_id, pdf_url=override_url)
        if status == "completed" and extracted_abstract:
            update_paper_abstract(connection, entry_id, extracted_abstract)
        run_id = save_methodology_run(connection, payload)
        latest = fetch_latest_methodology_run(connection, entry_id, prompt_version=PROMPT_VERSION, model=model)
    result = dict(latest) if latest else payload
    result["run_id"] = run_id
    result["paper_pdf_url"] = str(paper.get("pdf_url") or "")
    return result
