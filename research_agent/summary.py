"""Shared summary logic extracted from web.py.

All functions here are framework-agnostic and can be called from FastAPI,
Streamlit, CLI, or any other Python entry point.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from .abstract_enrichment import enrich_missing_abstracts
from .cli import _embed_paper_records
from .config import AppConfig
from .db import (
    fetch_embeddings_for_entries,
    get_connection,
    get_setting,
    initialize_database,
    list_papers,
    set_setting,
)
from .openai_client import DEFAULT_EMBEDDING_MODEL, OpenAIAnswerClient, OpenAIEmbeddingClient
from .pipeline import ingest_feeds, ingest_huggingface_daily_papers_for_date

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAIPEI_TZ = timezone(timedelta(hours=8))

LANGUAGE_LABELS = {
    "zh-TW": "Traditional Chinese",
    "en": "English",
}

SUMMARY_FILTER_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
SUMMARY_PROMPT_VERSION = "v13-summary-abstract-filter-and-i18n"

RESEARCH_TOPIC_KEY = "research_topic"

EMPTY_MESSAGES = {
    "zh-TW": "今天還沒有新匯入的 paper。請先執行 ingest，再重新整理這個摘要欄位。",
    "en": "No new papers were added today. Run ingest first, then refresh this summary.",
}

# ---------------------------------------------------------------------------
# Data class placeholder ??mirrors TodaySummaryRequest from web.py
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Data class placeholder – mirrors TodaySummaryRequest from web.py
# ---------------------------------------------------------------------------

class SummaryRequest:
    """Framework-agnostic summary request. Can be constructed manually or
    converted from a Pydantic model in web.py."""

    def __init__(
        self,
        language: str = "zh-TW",
        model: str = "gpt-5-mini",
        limit: int = 15,
        target_date: str | None = None,
        force_refresh: bool = False,
        starred_only: bool = False,
    ):
        self.language = language
        self.model = model
        self.limit = limit
        self.target_date = target_date
        self.force_refresh = force_refresh
        self.starred_only = starred_only


# ---------------------------------------------------------------------------
# Research topic helpers
# ---------------------------------------------------------------------------

def get_research_topic(config: AppConfig) -> str:
    with get_connection(config) as connection:
        initialize_database(connection)
        return (get_setting(connection, RESEARCH_TOPIC_KEY) or "").strip()


def set_research_topic(config: AppConfig, topic: str) -> str:
    cleaned = " ".join(topic.split()).strip()
    with get_connection(config) as connection:
        initialize_database(connection)
        set_setting(connection, RESEARCH_TOPIC_KEY, cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def ensure_summary_embeddings(config: AppConfig, papers: list[dict[str, Any]], model: str = SUMMARY_FILTER_EMBEDDING_MODEL) -> None:
    entry_ids = [str(item.get("entry_id")) for item in papers if item.get("entry_id")]
    if not entry_ids:
        return
    with get_connection(config) as connection:
        initialize_database(connection)
        existing = fetch_embeddings_for_entries(connection, entry_ids, model)
        missing_ids = [entry_id for entry_id in entry_ids if entry_id not in existing]
        missing_papers = [paper for paper in papers if str(paper.get("entry_id")) in missing_ids]
    if not missing_papers:
        return
    _embed_paper_records(config, missing_papers, model=model, dimensions=None, init_vec=False)


@lru_cache(maxsize=64)
def _cached_topic_embedding(topic: str, model: str) -> tuple[float, ...]:
    return tuple(OpenAIEmbeddingClient().create_embeddings([topic], model=model).embeddings[0])


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def resolve_target_date(target_date: str | None) -> str:
    if not target_date:
        return datetime.now(TAIPEI_TZ).date().isoformat()
    try:
        return date.fromisoformat(target_date).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid target date: {target_date}") from exc


# ---------------------------------------------------------------------------
# Paper selection
# ---------------------------------------------------------------------------

def papers_for_date(config: AppConfig, target_date: str, limit: int) -> list[dict[str, Any]]:
    resolved_date = resolve_target_date(target_date)
    with get_connection(config) as connection:
        initialize_database(connection)
        return list_papers(connection, limit=limit, published=resolved_date)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return sum(l * r for l, r in zip(left, right)) / (left_norm * right_norm)


def select_summary_papers(config: AppConfig, target_date: str, limit: int, starred_only: bool = False) -> tuple[list[dict[str, Any]], str, str]:
    daily_items = papers_for_date(config, target_date, 200)
    if starred_only:
        daily_items = [item for item in daily_items if "starred" in (item.get("tags") or [])]
    topic = get_research_topic(config)
    if not topic:
        return daily_items[:limit], "latest", topic

    ensure_summary_embeddings(config, daily_items, SUMMARY_FILTER_EMBEDDING_MODEL)
    topic_embedding = list(_cached_topic_embedding(topic, SUMMARY_FILTER_EMBEDDING_MODEL))
    entry_ids = [str(item.get("entry_id")) for item in daily_items if item.get("entry_id")]
    with get_connection(config) as connection:
        initialize_database(connection)
        embeddings = fetch_embeddings_for_entries(connection, entry_ids, SUMMARY_FILTER_EMBEDDING_MODEL)

    scored: list[dict[str, Any]] = []
    for item in daily_items:
        entry_id = str(item.get("entry_id") or "")
        embedding = embeddings.get(entry_id)
        if not embedding:
            continue
        score = cosine_similarity(topic_embedding, embedding)
        ranked = dict(item)
        ranked["relevance_score"] = score
        scored.append(ranked)

    scored.sort(key=lambda item: float(item.get("relevance_score", -999.0)), reverse=True)
    if scored:
        return scored[:limit], "topic", topic
    return daily_items[:limit], "latest", topic


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def encode_summary_cache(summary: str, topic: str) -> str:
    topic_line = topic.replace("\n", " ").strip()
    header = f"<!-- topic: {topic_line} | prompt_version: {SUMMARY_PROMPT_VERSION} -->"
    return f"{header}\n\n{summary}"


def read_summary_cache(cache_path: Path) -> tuple[str, str, str]:
    raw = cache_path.read_text(encoding="utf-8")
    first_line, _, remainder = raw.partition("\n")
    if first_line.startswith("<!-- ") and first_line.endswith(" -->"):
        metadata = first_line[len("<!-- ") : -len(" -->")]
        parts = [part.strip() for part in metadata.split("|")]
        values: dict[str, str] = {}
        for part in parts:
            key, sep, value = part.partition(":")
            if sep:
                values[key.strip()] = value.strip()
        summary = remainder.lstrip("\n")
        return values.get("topic", ""), summary, values.get("prompt_version", "")
    return "", raw, ""


def summary_cache_path(config: AppConfig, language: str, day_label: str, starred_only: bool = False) -> Path:
    safe_language = language.replace("/", "-").replace("\\", "-")
    target_dir = config.data_dir / "daily-summaries" / safe_language
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-B" if starred_only else ""
    return target_dir / f"{day_label}{suffix}.md"


def summary_eligible_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if str(item.get("abstract") or "").strip()]


def summary_empty_message(language: str, *, had_items_without_abstract: bool) -> str:
    if had_items_without_abstract:
        if language == "zh-TW":
            return "今天選到的 paper 缺少可用摘要，因此本次摘要已略過這些文獻。若需要納入，請先補 abstract 或改用全文抽取流程。"
        return "The selected papers for this date do not have usable abstracts, so they were skipped from the summary. Add abstracts or use the full-text extraction flow first."
    return EMPTY_MESSAGES.get(language, EMPTY_MESSAGES["en"])


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_daily_summary_prompt(items: list[dict[str, Any]], *, language: str, day_label: str, topic: str) -> str:
    language_label = LANGUAGE_LABELS.get(language, language)
    papers_block: list[str] = []
    for index, item in enumerate(items, start=1):
        authors = ", ".join(item.get("authors", [])) if isinstance(item.get("authors"), list) else ""
        papers_block.append(
            "\n".join(
                [
                    f"Title: {item.get('title', '')}",
                    f"Authors: {authors}",
                    f"Source: {item.get('source', '')}",
                    f"Published: {item.get('published', '')}",
                    f"Paper URL: {item.get('link', '')}",
                    f"PDF URL: {item.get('pdf_url', '')}",
                    f"Abstract: {item.get('abstract', '')}",
                ]
            )
        )
    joined_papers = "\n\n".join(papers_block)
    topic_instruction = (
        f"Research topic: {topic}. Prioritize relevance to this topic and mention the topic explicitly in the notes."
        if topic
        else "No research topic is set. Summarize the latest papers of the day as research notes."
    )
    return f"""You are writing a research notes entry for a scholar.
Write the answer in {language_label}.
Date: {day_label}.
Use only the provided papers.
Do not invent findings not supported by titles or abstracts.
If a paper has no abstract, skip it entirely. Do not infer, reconstruct, or fabricate its problem, method, findings, or contribution.
Only discuss papers that include a non-empty abstract in the provided input.
{topic_instruction}

Target style: research notes.
- Write like a researcher organizing today's reading notes for later thinking and possible study design.
- Start from a self-reflection (or murmuration) that makes this set of papers worth paying attention to.
- Mention that you recently read some related papers and specify the source of papers. Specify the number of papers.
- When discussing each paper, include the research problem, method, and main finding, but keep the prose flowing naturally.
- Keep the tone rational, clear, and analytical, but not overly formal.
- You may include a small amount of natural commentary, for example: this is interesting, to some extent, simply put.
- After introducing several papers, synthesize them into a cross-paper observation.
- Include concept-level contrasts when useful, such as strong coordination but weak impact, or clear structure but limited evolution.
- End from a researcher perspective with an open reflection: what kind of study could be designed next, what mechanism is still unclear, or what this phenomenon may imply.

Formatting rules:
- Do not output internal record markers such as [Paper 1], [Paper 2], Record 1, or similar references.
- Do not append bracketed paper ids or index labels anywhere in the notes.
- The main body must be written in connected paragraphs, like a notebook entry.
- If you mention an individual paper, identify it by title in natural prose, not by index.
- Avoid generic AI writing habits: do not use empty framing such as "it is worth noting", "overall", "in conclusion", "delves into", "highlights the importance of", or similar stock phrases unless they add concrete meaning.
- Avoid repetitive sentence templates, abstract filler, and over-smoothed transitions.
- Prefer specific observations, concrete contrasts, and researcher-like phrasing over polished assistant-style exposition.
- Write like someone actually keeping research notes, not like a chatbot summarizing content for a general audience.
- Do not start with heading-like labels or framing phrases such as "Current phenomenon", "Key concern", "Background", "Overview", or similar label-style openers. Start directly with natural prose.
- Avoid first-person narration. Do not use "I", "we", or the Chinese first-person form "\u6211" unless absolutely unavoidable.
- Avoid semi-academic stock phrasing such as "this study", "the discussion above", "in summary", or similar formulaic transitions unless they are truly necessary.
- After the main body, add a separate section titled `References`.
- In `References`, you must list the papers you used with explicit numbering.
- Each reference line must include: number, title, authors, source, and URL.
- Prefer the Paper URL; if missing, use the PDF URL; if both are missing, write `URL: N/A`.
- The tone of the summary should be natural and conversational, not academic or formal.

Suggested note flow:
- Paragraph 1: why this cluster of papers matters now, and what thread connects them. Begin with a short random murmuration of the papers.
- Paragraphs 2-4: discuss representative papers as notes, each with problem, method, and finding.
- Paragraph 5: synthesize patterns, contrast concepts, and end with a research-facing question or design idea.
- Final section: `References` only limited to the papers used in this summary.
- Format each reference exactly like this:
  1. Title: <title> | Authors: <authors> | Source: <source> | URL: <url>
Papers for the selected date:

{joined_papers}
"""


def build_summary_rewrite_prompt(*, language: str, day_label: str, topic: str, full_summary: str, selected_text: str, instruction: str, prefix: str, suffix: str) -> str:
    language_label = LANGUAGE_LABELS.get(language, language)
    topic_line = topic if topic else "No research topic is set."
    return f"""You are revising a selected passage inside an existing research-notes style daily summary.
Write the answer in {language_label}.
Date: {day_label}.
Research topic: {topic_line}

Task:
- Rewrite only the selected passage.
- Follow the user's instruction exactly.
- Keep the revised passage consistent with the surrounding context.
- Preserve the same language as the selected passage unless the instruction explicitly asks to change it.
- Do not rewrite anything outside the selected passage.
- Do not add headings, labels, markdown fences, commentary, or explanations.
- Return only the replacement text for the selected passage.

User instruction:
{instruction}

Full summary:
{full_summary}

Context before selection:
{prefix}

Selected passage to rewrite:
{selected_text}

Context after selection:
{suffix}
"""


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Cached summary response check
# ---------------------------------------------------------------------------

def cached_summary_response(*, request: SummaryRequest, target_date: str, items: list[dict[str, Any]], selection_mode: str, topic: str, cache_path: Path) -> dict[str, Any] | None:
    if cache_path.exists() and not request.force_refresh:
        cached_topic, cached_summary, cached_prompt_version = read_summary_cache(cache_path)
        if cached_topic == topic and cached_prompt_version == SUMMARY_PROMPT_VERSION:
            return {
                "date": target_date,
                "language": request.language,
                "paper_count": len(items),
                "summary": cached_summary,
                "papers": [
                    {
                        "entry_id": item.get("entry_id"),
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "published": item.get("published"),
                        "doi": item.get("doi"),
                    }
                    for item in items
                ],
                "cached": True,
                "path": str(cache_path),
                "selection_mode": selection_mode,
                "topic": topic,
            }
    return None


# ---------------------------------------------------------------------------
# Selection resolution (for rewrite)
# ---------------------------------------------------------------------------

def resolve_summary_selection(summary_text: str, start_offset: int, end_offset: int, selected_text: str) -> tuple[int, int, str]:
    if start_offset < 0 or end_offset < start_offset:
        raise ValueError("Invalid selection range.")
    if end_offset > len(summary_text):
        raise ValueError("Selection range exceeds summary length.")
    selected_from_offsets = summary_text[start_offset:end_offset]
    if selected_from_offsets == selected_text:
        return start_offset, end_offset, selected_from_offsets
    snippet_index = summary_text.find(selected_text)
    if snippet_index < 0:
        raise ValueError("The selected text no longer matches the current summary.")
    return snippet_index, snippet_index + len(selected_text), selected_text


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------

def generate_summary_from_selection(config: AppConfig, request: SummaryRequest, *, target_date: str, items: list[dict[str, Any]], selection_mode: str, topic: str) -> dict[str, Any]:
    cache_path = summary_cache_path(config, request.language, target_date, request.starred_only)
    cached = cached_summary_response(
        request=request,
        target_date=target_date,
        items=items,
        selection_mode=selection_mode,
        topic=topic,
        cache_path=cache_path,
    )
    if cached is not None:
        return cached

    enriched_items = enrich_missing_abstracts(config, items)
    eligible = summary_eligible_items(enriched_items)

    if not eligible:
        summary = summary_empty_message(request.language, had_items_without_abstract=bool(enriched_items))
        cache_path.write_text(encode_summary_cache(summary, topic), encoding="utf-8")
        return {
            "date": target_date,
            "language": request.language,
            "paper_count": 0,
            "summary": summary,
            "papers": [],
            "cached": False,
            "path": str(cache_path),
            "selection_mode": selection_mode,
            "topic": topic,
        }

    prompt = build_daily_summary_prompt(eligible, language=request.language, day_label=target_date, topic=topic)
    summary = OpenAIAnswerClient().create_answer(prompt=prompt, model=request.model).text
    cache_path.write_text(encode_summary_cache(summary, topic), encoding="utf-8")
    return {
        "date": target_date,
        "language": request.language,
        "paper_count": len(eligible),
        "summary": summary,
        "papers": [
            {
                "entry_id": item.get("entry_id"),
                "title": item.get("title"),
                "source": item.get("source"),
                "published": item.get("published"),
                "doi": item.get("doi"),
            }
            for item in eligible
        ],
        "cached": False,
        "path": str(cache_path),
        "selection_mode": selection_mode,
        "topic": topic,
    }


def stream_summary(config: AppConfig, request: SummaryRequest):
    target_date = resolve_target_date(request.target_date)
    items, selection_mode, topic = select_summary_papers(config, target_date, request.limit, request.starred_only)
    cache_path = summary_cache_path(config, request.language, target_date, request.starred_only)

    cached = cached_summary_response(
        request=request,
        target_date=target_date,
        items=items,
        selection_mode=selection_mode,
        topic=topic,
        cache_path=cache_path,
    )
    if cached is not None:
        meta = {k: cached[k] for k in ["date", "language", "paper_count", "selection_mode", "topic", "path"]}
        yield sse_event("meta", {**meta, "cached": True})
        yield sse_event("delta", {"text": str(cached["summary"])})
        yield sse_event("done", {**meta, "cached": True})
        return

    enriched_items = enrich_missing_abstracts(config, items)
    eligible = summary_eligible_items(enriched_items)
    meta = {
        "date": target_date,
        "language": request.language,
        "paper_count": len(eligible),
        "selection_mode": selection_mode,
        "topic": topic,
        "path": str(cache_path),
    }

    if not eligible:
        summary = summary_empty_message(request.language, had_items_without_abstract=bool(enriched_items))
        cache_path.write_text(encode_summary_cache(summary, topic), encoding="utf-8")
        yield sse_event("meta", {**meta, "cached": False})
        yield sse_event("delta", {"text": summary})
        yield sse_event("done", {**meta, "cached": False})
        return

    prompt = build_daily_summary_prompt(eligible, language=request.language, day_label=target_date, topic=topic)
    answer_client = OpenAIAnswerClient()
    collected: list[str] = []
    yield sse_event("meta", {**meta, "cached": False})
    try:
        for chunk in answer_client.stream_answer(prompt=prompt, model=request.model):
            if not chunk:
                continue
            collected.append(chunk)
            yield sse_event("delta", {"text": chunk})
    except Exception as exc:
        yield sse_event("error", {"detail": str(exc)})
        return

    summary = "".join(collected).strip()
    cache_path.write_text(encode_summary_cache(summary, topic), encoding="utf-8")
    yield sse_event("done", {**meta, "cached": False})


def generate_summary(config: AppConfig, request: SummaryRequest) -> dict[str, Any]:
    target_date = resolve_target_date(request.target_date)
    items, selection_mode, topic = select_summary_papers(config, target_date, request.limit, request.starred_only)
    return generate_summary_from_selection(
        config,
        request,
        target_date=target_date,
        items=items,
        selection_mode=selection_mode,
        topic=topic,
    )


def ensure_daily_ready(config: AppConfig, request: SummaryRequest) -> dict[str, Any]:
    target_date = resolve_target_date(request.target_date)
    cache_path = summary_cache_path(config, request.language, target_date, request.starred_only)
    items, selection_mode, topic = select_summary_papers(config, target_date, request.limit, request.starred_only)
    actions: list[str] = []
    imported_count = 0

    if not items:
        if target_date == datetime.now(TAIPEI_TZ).date().isoformat():
            new_entries = ingest_feeds(config)
        else:
            new_entries = ingest_huggingface_daily_papers_for_date(config, target_date)
        imported_count = len(new_entries)
        actions.append("ingest")
        items, selection_mode, topic = select_summary_papers(config, target_date, request.limit, request.starred_only)

    needs_summary = request.force_refresh or not cache_path.exists() or bool(actions)
    if needs_summary:
        summary_result = generate_summary_from_selection(
            config,
            SummaryRequest(
                language=request.language,
                model=request.model,
                limit=request.limit,
                target_date=target_date,
                force_refresh=True,
                starred_only=request.starred_only,
            ),
            target_date=target_date,
            items=items,
            selection_mode=selection_mode,
            topic=topic,
        )
        actions.append("summarize")
    else:
        summary_result = generate_summary_from_selection(
            config,
            request,
            target_date=target_date,
            items=items,
            selection_mode=selection_mode,
            topic=topic,
        )
        actions.append("load-cache")

    summary_result["actions"] = actions
    summary_result["imported_count"] = imported_count
    return summary_result


def preview_summary_selection_rewrite(config: AppConfig, *, language: str, model: str, target_date: str | None, summary_text: str, start_offset: int, end_offset: int, selected_text: str, instruction: str) -> dict[str, Any]:
    resolved_date = resolve_target_date(target_date)
    text = summary_text or ""
    start, end, resolved_text = resolve_summary_selection(text, start_offset, end_offset, selected_text)

    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Rewrite instruction is required.")

    prefix = text[max(0, start - 1200):start]
    suffix_text = text[end:min(len(text), end + 1200)]
    topic = get_research_topic(config)
    prompt = build_summary_rewrite_prompt(
        language=language,
        day_label=resolved_date,
        topic=topic,
        full_summary=text,
        selected_text=resolved_text,
        instruction=instruction,
        prefix=prefix,
        suffix=suffix_text,
    )
    rewritten = OpenAIAnswerClient().create_answer(prompt=prompt, model=model).text.strip()
    if not rewritten:
        raise ValueError("The model returned an empty rewrite.")

    return {
        "date": resolved_date,
        "language": language,
        "topic": topic,
        "selected_text": resolved_text,
        "rewritten_text": rewritten,
        "start_offset": start,
        "end_offset": end,
    }


def apply_summary_selection_rewrite(config: AppConfig, *, language: str, target_date: str | None, starred_only: bool = False, summary_text: str, start_offset: int, end_offset: int, selected_text: str, rewritten_text: str) -> dict[str, Any]:
    resolved_date = resolve_target_date(target_date)
    text = summary_text or ""
    start, end, _ = resolve_summary_selection(text, start_offset, end_offset, selected_text)
    rewritten = rewritten_text.strip()
    if not rewritten:
        raise ValueError("Rewritten text is required.")

    topic = get_research_topic(config)
    updated = text[:start] + rewritten + text[end:]
    cache = summary_cache_path(config, language, resolved_date, starred_only)
    cache.write_text(encode_summary_cache(updated, topic), encoding="utf-8")
    return {
        "date": resolved_date,
        "language": language,
        "topic": topic,
        "summary": updated,
        "rewritten_text": rewritten,
        "path": str(cache),
    }


def save_summary_text(config: AppConfig, *, language: str, target_date: str | None, starred_only: bool = False, summary_text: str) -> dict[str, Any]:
    resolved_date = resolve_target_date(target_date)
    text = (summary_text or "").strip()
    if not text:
        raise ValueError("Summary text is required.")
    items, _, topic = select_summary_papers(config, resolved_date, 15, starred_only)
    cache = summary_cache_path(config, language, resolved_date, starred_only)
    cache.write_text(encode_summary_cache(text, topic), encoding="utf-8")
    return {
        "date": resolved_date,
        "language": language,
        "topic": topic,
        "summary": text,
        "path": str(cache),
        "paper_count": len(items),
        "cached": False,
        "selection_mode": "manual-edit",
    }
