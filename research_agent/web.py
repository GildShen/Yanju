from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import re
from urllib.parse import quote
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from .abstract_enrichment import enrich_missing_abstracts
from .catalog import catalog_pdfs
from .cli import _build_answer_prompt, _embed_paper_records
from .config import AppConfig
from .db import (
    fetch_latest_methodology_run,
    fetch_paper,
    fetch_recent_papers,
    fetch_stats,
    get_connection,
    get_setting,
    initialize_database,
    list_embedding_candidates,
    list_paper_dates,
    list_papers,
    search_similar_embeddings,
    set_setting,
    toggle_paper_starred,
    update_methodology_run_text,
    fetch_embeddings_for_entries,
)
from .openai_client import DEFAULT_ANSWER_MODEL, DEFAULT_EMBEDDING_MODEL, OpenAIAnswerClient, OpenAIEmbeddingClient
from .methodology_analysis import analyze_paper_methodology, export_methodology_note
from .pipeline import generate_digest, import_dois, import_url, ingest_feeds, ingest_huggingface_daily_papers_for_date


class CatalogRequest(BaseModel):
    pdf_dir: str = "papers/tmp"
    embed: bool = False
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    init_vec: bool = False


class SearchRequest(BaseModel):
    query: str
    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    limit: int = 5


class AskRequest(BaseModel):
    query: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL
    dimensions: int | None = None
    top_k: int = 5
    temperature: float | None = None


class ImportDoisRequest(BaseModel):
    doi_file: str = "dois.txt"
    embed: bool = False
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    init_vec: bool = False


class ImportUrlRequest(BaseModel):
    url: str
    embed: bool = False
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int | None = None
    init_vec: bool = False


class DigestRequest(BaseModel):
    days: int = 7


class TodaySummaryRequest(BaseModel):
    language: str = "zh-TW"
    model: str = DEFAULT_ANSWER_MODEL
    limit: int = 15
    target_date: str | None = None
    force_refresh: bool = False


class ResearchTopicRequest(BaseModel):
    topic: str = ""


class ToggleStarRequest(BaseModel):
    entry_id: str


class PaperAnalysisRequest(BaseModel):
    entry_id: str
    model: str = DEFAULT_ANSWER_MODEL
    force: bool = False
    pdf_url_override: str | None = None


class PaperAnalysisSaveRequest(BaseModel):
    run_id: int
    analysis_text: str


class PaperAnalysisExportRequest(BaseModel):
    run_id: int


class SummaryRewriteRequest(BaseModel):
    language: str = "zh-TW"
    model: str = DEFAULT_ANSWER_MODEL
    target_date: str | None = None
    summary_text: str
    start_offset: int
    end_offset: int
    selected_text: str
    instruction: str


class SummaryRewriteApplyRequest(SummaryRewriteRequest):
    rewritten_text: str


class SummarySaveRequest(BaseModel):
    language: str = "zh-TW"
    target_date: str | None = None
    summary_text: str

TAIPEI_TZ = timezone(timedelta(hours=8))
LANGUAGE_LABELS = {
    "zh-TW": "Traditional Chinese",
    "en": "English",
}
SUMMARY_FILTER_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
SUMMARY_PROMPT_VERSION = "v13-summary-abstract-filter-and-i18n"

EMPTY_MESSAGES = {
    "zh-TW": "\u4eca\u5929\u9084\u6c92\u6709\u65b0\u532f\u5165\u7684 paper\u3002\u8acb\u5148\u57f7\u884c ingest\uff0c\u518d\u91cd\u65b0\u6574\u7406\u9019\u500b\u6458\u8981\u6b04\u4f4d\u3002",
    "en": "No new papers were added today. Run ingest first, then refresh this summary.",
}



@lru_cache(maxsize=1)
def _load_ui_translations() -> dict[str, dict[str, str]]:
    path = Path(__file__).with_name("i18n") / "ui_translations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _render_app_html() -> str:
    return APP_HTML.replace("__UI_TRANSLATIONS_JSON__", json.dumps(_load_ui_translations(), ensure_ascii=False))


APP_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Agent</title>
  <style>
    :root {
      --bg: #f3eee5;
      --paper: rgba(255, 252, 246, 0.94);
      --panel: rgba(255, 255, 255, 0.72);
      --ink: #1e293b;
      --muted: #6b7280;
      --accent: #0f766e;
      --accent-soft: rgba(15, 118, 110, 0.12);
      --accent-2: #b45309;
      --line: rgba(120, 113, 108, 0.18);
      --shadow: 0 20px 44px rgba(30, 41, 59, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Noto Serif TC", "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 22%),
        linear-gradient(180deg, #f7f2e9 0%, #f1ebdf 100%);
    }
    .wrap {
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 22px 52px;
    }
    .hero {
      display: grid;
      gap: 18px;
      margin-bottom: 22px;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,249,240,0.78));
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }
    .hero-top {
      display: flex;
      gap: 18px;
      justify-content: space-between;
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .hero-copy { display: grid; gap: 10px; max-width: 760px; }
    .hero-kicker {
      display: inline-flex;
      width: fit-content;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.10);
      color: var(--accent);
      font-size: 0.85rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(2.4rem, 5vw, 4.8rem);
      line-height: 0.92;
      letter-spacing: -0.03em;
    }
    .hero p { margin: 0; max-width: 820px; color: var(--muted); font-size: 1.02rem; line-height: 1.65; }
    .hero-status {
      min-width: 240px;
      padding: 16px 18px;
      border-radius: 20px;
      background: rgba(255,255,255,0.70);
      border: 1px solid rgba(15,118,110,0.16);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
    }
    .hero-status span {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .hero-status b {
      display: block;
      font-size: 1.1rem;
      line-height: 1.4;
    }
    .hero-nav {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .hero-nav a {
      padding: 10px 14px;
      border-radius: 999px;
      text-decoration: none;
      color: var(--ink);
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      font-size: 0.92rem;
    }
    .hero-nav a:hover { color: var(--accent); border-color: rgba(15,118,110,0.25); }
    .stats {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    }
    .stat {
      padding: 16px;
      border-radius: 20px;
      background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(250,245,237,0.96));
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
    }
    .stat span { display: block; margin-bottom: 8px; color: var(--muted); font-size: 0.88rem; }
    .stat b { display: block; font-size: 2rem; }
    .page-grid {
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.95fr);
      align-items: start;
    }
    .stack { display: grid; gap: 18px; }
    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 26px;
      padding: 22px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .card-head {
      display: flex;
      gap: 14px;
      align-items: flex-start;
      justify-content: space-between;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .card-head h2 { margin: 0; font-size: 1.28rem; }
    .card-head p { margin: 6px 0 0; color: var(--muted); line-height: 1.55; }
    .summary-controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .summary-controls input,
    .summary-controls select {
      min-width: 150px;
      width: auto;
    }
    .meta-chip-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }
    .meta-chip {
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.88rem;
    }
    .topic-row {
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
    }
    .topic-status { margin-top: 10px; color: var(--muted); font-size: 0.92rem; }
    .summary-box {
      margin-top: 14px;
      border-radius: 20px;
      border: 1px solid rgba(180, 83, 9, 0.16);
      background: rgba(255, 255, 255, 0.78);
      padding: 18px 18px;
      min-height: 180px;
      white-space: pre-wrap;
      line-height: 1.8;
      font-size: 0.99rem;
    }
    .summary-box a { color: var(--accent); text-decoration: underline; }
    .summary-box.selectable { cursor: text; }
    .summary-box.editing { display: none; }
    .summary-editor { display: none; margin-top: 14px; }
    .summary-editor.open { display: grid; gap: 10px; }
    .summary-editor textarea { min-height: 320px; line-height: 1.8; }
    .summary-actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
    .summary-selection-status { color: var(--muted); font-size: 0.9rem; }
    .summary-meta { margin-top: 10px; color: var(--muted); font-size: 0.9rem; }
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.44);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      z-index: 50;
    }
    .modal-backdrop.open { display: flex; }
    .modal-card {
      width: min(720px, 100%);
      max-height: 88vh;
      overflow: auto;
      padding: 22px;
      border-radius: 24px;
      background: rgba(255, 252, 246, 0.98);
      border: 1px solid var(--line);
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.22);
    }
    .modal-card h3 { font-size: 1.16rem; }
    .modal-card p { color: var(--muted); line-height: 1.6; }
    .modal-preview {
      margin-top: 10px;
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.85);
      border: 1px solid rgba(120,113,108,0.18);
      white-space: pre-wrap;
      line-height: 1.75;
      max-height: 220px;
      overflow: auto;
    }
    .workbench-shell {
      display: grid;
      gap: 14px;
    }
    .workbench-toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 6px;
    }
    .workbench-tab {
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.62);
      color: var(--ink);
    }
    .workbench-tab.active {
      background: linear-gradient(180deg, #12786f, #0f766e);
      color: white;
      border-color: transparent;
    }
    .workbench-panel { display: none; }
    .workbench-panel.active { display: grid; gap: 12px; }
    .tool-accordion {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255,255,255,0.58);
      overflow: hidden;
    }
    .tool-accordion summary {
      list-style: none;
      cursor: pointer;
      padding: 16px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-weight: 600;
    }
    .tool-accordion summary::-webkit-details-marker { display: none; }
    .tool-accordion summary span { color: var(--muted); font-weight: 400; font-size: 0.92rem; }
    .tool-body {
      padding: 0 18px 18px;
      border-top: 1px solid rgba(120,113,108,0.12);
      background: rgba(255,255,255,0.34);
    }
    .output {
      margin-top: 14px;
      padding: 16px;
      border-radius: 18px;
      background: #17202b;
      color: #e7f1ff;
      min-height: 120px;
      overflow: auto;
      font-family: Consolas, monospace;
      font-size: 0.85rem;
      white-space: pre-wrap;
    }
    .paper-toolbar {
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(0, 1fr) 120px;
      align-items: end;
      margin-bottom: 14px;
    }
    .paper-search-row {
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr;
      align-items: end;
      margin-bottom: 14px;
    }
    .paper-actions-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .mini-actions,
    .actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
    h2, h3 { margin: 0; }
    h3 { font-size: 1rem; }
    label { display: block; margin: 10px 0 6px; font-size: 0.9rem; color: var(--muted); }
    input, textarea, select {
      width: 100%;
      padding: 11px 12px;
      border-radius: 12px;
      border: 1px solid rgba(120,113,108,0.22);
      background: rgba(255,255,255,0.94);
      color: var(--ink);
      font: inherit;
    }
    textarea { min-height: 100px; resize: vertical; }
    .row { display: grid; gap: 10px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button.alt { background: var(--accent-2); }
    button.ghost { background: rgba(255,255,255,0.76); color: var(--ink); border: 1px solid var(--line); }
    .list { display: grid; gap: 14px; }
    .paper-item {
      border: 1px solid rgba(120,113,108,0.15);
      border-radius: 22px;
      padding: 16px 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(249,246,239,0.96));
      box-shadow: 0 10px 24px rgba(31, 41, 51, 0.04);
    }
    .paper-head { display: grid; grid-template-columns: 52px minmax(0, 1fr); gap: 14px; align-items: start; }
    .paper-title { margin: 0; font-size: 1.02rem; line-height: 1.42; }
    .paper-title-link { color: var(--ink); text-decoration: none; }
    .paper-title-link:hover { color: var(--accent); text-decoration: underline; }
    .paper-meta-row { margin-top: 6px; color: var(--muted); font-size: 0.92rem; line-height: 1.5; }
    .paper-authors { margin-top: 4px; color: #475569; font-size: 0.94rem; line-height: 1.5; }
    .paper-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .paper-chip {
      border: 1px solid rgba(120,113,108,0.18);
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.82rem;
      line-height: 1;
      text-decoration: none;
      cursor: pointer;
      transition: border-color 120ms ease, transform 120ms ease, background 120ms ease;
    }
    .paper-chip:hover { transform: translateY(-1px); border-color: rgba(15,118,110,0.34); background: rgba(239,250,249,0.92); }
    .paper-chip.active { background: rgba(15,118,110,0.12); border-color: rgba(15,118,110,0.28); color: #0f766e; }
    .paper-abstract {
      margin-top: 12px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(120,113,108,0.12);
      background: rgba(255,255,255,0.62);
      color: #334155;
      font-size: 0.93rem;
      line-height: 1.72;
      white-space: normal;
    }
    .paper-abstract.empty { color: var(--muted); font-style: italic; }
    .paper-analysis {
      margin-top: 12px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(120,113,108,0.12);
      background: rgba(255,255,255,0.72);
      color: #334155;
      font-size: 0.92rem;
      line-height: 1.7;
    }
    .paper-analysis h3,
    .paper-analysis h4,
    .paper-analysis strong { color: var(--ink); }
    .paper-analysis .meta { margin-top: 8px; font-size: 0.82rem; }
    .paper-chip.warn { background: rgba(180,83,9,0.08); border-color: rgba(180,83,9,0.2); color: #9a3412; }
    .analysis-dialog { width: min(980px, 100%); max-height: 90vh; }
    .analysis-meta { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
    .analysis-badge { padding: 6px 10px; border-radius: 999px; background: rgba(15,118,110,0.1); color: #0f766e; border: 1px solid rgba(15,118,110,0.14); font-size: 0.8rem; }
    .analysis-grid { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 14px; }
    .analysis-tile { padding: 14px 16px; border-radius: 16px; border: 1px solid rgba(120,113,108,0.16); background: rgba(255,255,255,0.84); }
    .analysis-tile h4 { margin: 0 0 8px; font-size: 0.95rem; }
    .analysis-body { margin-top: 16px; padding: 16px; border-radius: 18px; background: rgba(255,255,255,0.9); border: 1px solid rgba(120,113,108,0.16); white-space: pre-wrap; line-height: 1.75; }
    .analysis-editor { min-height: 320px; margin-top: 16px; }
    .analysis-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
    .star-btn {
      width: 52px;
      height: 52px;
      border-radius: 16px;
      border: 1px solid rgba(15,118,110,0.18);
      background: linear-gradient(180deg, #1b8a81, #0f766e);
      color: #fff4c2;
      padding: 0;
      font-size: 1.4rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 10px 18px rgba(15, 118, 110, 0.18);
    }
    .star-btn.active {
      background: linear-gradient(180deg, #f3c14a, #cd8f12);
      color: #fffaf0;
      border-color: rgba(181,125,9,0.22);
    }
    .footer { margin-top: 16px; color: var(--muted); font-size: 0.88rem; }
    .mono { font-family: Consolas, monospace; }
    .empty-note {
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px dashed rgba(120,113,108,0.25);
      color: var(--muted);
      background: rgba(255,255,255,0.54);
    }
    @media (max-width: 1100px) {
      .page-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 820px) {
      .wrap { padding: 18px 14px 38px; }
      .hero { padding: 22px; }
      .hero-top,
      .card-head,
      .summary-head { align-items: flex-start; }
      .paper-toolbar,
      .row,
      .topic-row { grid-template-columns: 1fr; }
      .summary-controls { justify-content: flex-start; }
      .summary-controls input,
      .summary-controls select { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="hero-top">
        <div class="hero-copy">
          <span class="hero-kicker" data-i18n="hero_kicker">Local Research Workflow</span>
          <h1>Research Agent</h1>
          <p data-i18n="hero_desc">Start from the daily summary, then move into paper exploration and focused intake workflows. PDF cataloging, DOI import, and retrieval are grouped into a cleaner workbench so the page feels like one research desk instead of many forms.</p>
        </div>
        <div class="hero-status">
          <span data-i18n="current_focus">Current Focus</span>
          <b id="topicStatus">No research topic set. Daily Summary will use the latest 15 papers for the selected date.</b>
          <label for="uiLanguage" data-i18n="ui_language" style="margin-top: 12px;">UI Language</label>
          <select id="uiLanguage" onchange="setUiLanguage(this.value)">
            <option value="zh-TW">Traditional Chinese</option>
            <option value="en">English</option>
          </select>
        </div>
      </div>
      <div class="hero-nav">
        <a href="#summary-card" data-i18n="nav_summary">Daily Summary</a>
        <a href="#papers-card" data-i18n="nav_papers">Paper Explorer</a>
        <a href="#workbench-card" data-i18n="nav_workbench">Workbench</a>
      </div>
      <div class="stats" id="stats"></div>
    </section>

    <section class="page-grid">
      <div class="stack">
        <section class="card" id="summary-card">
          <div class="card-head summary-head">
            <div>
              <h2 data-i18n="daily_summary_title">Daily Summary</h2>
              <p data-i18n="daily_summary_desc">Select a date, language, and research topic to generate a compact research-note style summary.</p>
            </div>
            <div class="summary-controls">
              <input id="summaryDate" type="date" onchange="syncPaperDateFromSummary()">
              <select id="summaryLanguage">
                <option value="zh-TW" selected>Traditional Chinese</option>
                <option value="en">English</option>
              </select>
              <button onclick="loadTodaySummary(false)" data-i18n="load_summary">Load Summary</button>
              <button class="ghost" onclick="loadTodaySummary(true)" data-i18n="regenerate">Regenerate</button>
            </div>
          </div>
          <div class="meta-chip-row">
            <span class="meta-chip">Narrative summary</span>
            <span class="meta-chip">Semantic topic filter</span>
            <span class="meta-chip">Date-based cache</span>
          </div>
          <div class="topic-row">
            <input id="researchTopic" data-i18n-placeholder="research_topic_placeholder" placeholder="Set your research topic, e.g. human-AI collaboration in knowledge management">
            <button class="ghost" onclick="saveResearchTopic()" data-i18n="save_topic">Save Topic</button>
          </div>
          <div class="summary-box selectable" id="todaySummaryOutput" ondblclick="enterSummaryEditMode()">Loading summary...</div>
          <div class="summary-editor" id="summaryEditor">
            <textarea id="summaryEditorInput"></textarea>
            <div class="actions">
              <button onclick="saveSummaryEdit()" data-i18n="save_summary">Save Summary</button>
              <button class="ghost" onclick="cancelSummaryEdit()" data-i18n="cancel">Cancel</button>
            </div>
          </div>
          <div class="summary-actions">
            <button class="ghost" id="rewriteSelectionButton" onclick="openSelectionRewriteModal()" disabled data-i18n="rewrite_selection">Rewrite Selection</button>
            <span class="summary-selection-status" id="summarySelectionStatus">Select a passage in the summary to regenerate only that part, or double-click the summary to edit it directly.</span>
          </div>
          <div class="summary-meta" id="todaySummaryMeta"></div>
        </section>

        <section class="card" id="workbench-card">
          <div class="card-head">
            <div>
              <h2 data-i18n="workbench_title">Workbench</h2>
              <p data-i18n="workbench_desc">Group ingestion, cataloging, import, search, and Q&A into one collapsible workbench.</p>
            </div>
          </div>
          <div class="workbench-shell">
            <div class="workbench-toolbar">
              <button class="workbench-tab active" id="tab-data" onclick="switchWorkbench('data')" data-i18n="tab_data">Data Intake</button>
              <button class="workbench-tab" id="tab-search" onclick="switchWorkbench('search')" data-i18n="tab_search">Search And Ask</button>
            </div>

            <div class="workbench-panel active" id="panel-data">
              <details class="tool-accordion" open>
                <summary>
                  <strong data-i18n="ingest_digest_title">Ingest And Digest</strong>
                  <span data-i18n="ingest_digest_desc">Fetch new items and produce a weekly note</span>
                </summary>
                <div class="tool-body">
                  <div class="actions">
                    <button onclick="runIngest()" data-i18n="run_ingest">Run Ingest</button>
                    <button class="alt" onclick="runDigest()" data-i18n="run_digest">Run Digest</button>
                  </div>
                </div>
              </details>

              <details class="tool-accordion">
                <summary>
                  <strong data-i18n="catalog_pdfs_title">Catalog PDFs</strong>
                  <span data-i18n="catalog_pdfs_desc">Catalog files from papers/tmp and optionally embed them</span>
                </summary>
                <div class="tool-body">
                  <label data-i18n="pdf_directory">PDF directory</label>
                  <input id="catalogPdfDir" value="papers/tmp">
                  <div class="row">
                    <div>
                      <label data-i18n="embedding_model">Embedding model</label>
                      <input id="catalogEmbeddingModel" value="text-embedding-3-small">
                    </div>
                    <div>
                      <label data-i18n="dimensions_optional">Dimensions (optional)</label>
                      <input id="catalogDimensions" value="">
                    </div>
                  </div>
                  <div class="actions">
                    <button onclick="catalogPdfs(false)" data-i18n="catalog_pdfs">Catalog PDFs</button>
                    <button class="alt" onclick="catalogPdfs(true)" data-i18n="catalog_embed">Catalog + Embed</button>
                  </div>
                </div>
              </details>

              <details class="tool-accordion">
                <summary>
                  <strong data-i18n="import_dois_title">Import DOIs</strong>
                  <span data-i18n="import_dois_desc">Backfill metadata and notes from a DOI list</span>
                </summary>
                <div class="tool-body">
                  <label data-i18n="doi_file">DOI file</label>
                  <input id="doiFile" value="dois.txt">
                  <div class="actions">
                    <button onclick="importDois(false)" data-i18n="import_dois">Import DOIs</button>
                    <button class="alt" onclick="importDois(true)" data-i18n="import_embed">Import + Embed</button>
                  </div>
                </div>
              </details>

              <details class="tool-accordion">
                <summary>
                  <strong data-i18n="import_url_title">Fetch New Item</strong>
                  <span data-i18n="import_url_desc">Paste a DOI or arXiv URL and import it directly</span>
                </summary>
                <div class="tool-body">
                  <label data-i18n="paper_url">Paper URL</label>
                  <input id="importUrlInput" data-i18n-placeholder="paper_url_placeholder" placeholder="https://arxiv.org/abs/2602.17221">
                  <div class="actions">
                    <button onclick="importUrl(false)" data-i18n="import_url">Fetch New Item</button>
                    <button class="alt" onclick="importUrl(true)" data-i18n="import_url_embed">Fetch + Embed</button>
                  </div>
                </div>
              </details>
            </div>

            <div class="workbench-panel" id="panel-search">
              <details class="tool-accordion" open>
                <summary>
                  <strong data-i18n="semantic_search_title">Semantic Search</strong>
                  <span data-i18n="semantic_search_desc">Find related papers before asking a question</span>
                </summary>
                <div class="tool-body">
                  <label data-i18n="semantic_search_query">Semantic search query</label>
                  <textarea id="semanticQuery">human AI collaboration in knowledge management</textarea>
                  <div class="row">
                    <div>
                      <label data-i18n="embedding_model">Embedding model</label>
                      <input id="semanticModel" value="text-embedding-3-small">
                    </div>
                    <div>
                      <label data-i18n="limit">Limit</label>
                      <input id="semanticLimit" value="5">
                    </div>
                  </div>
                  <div class="actions">
                    <button onclick="semanticSearch()" data-i18n="semantic_search">Semantic Search</button>
                  </div>
                </div>
              </details>

              <details class="tool-accordion">
                <summary>
                  <strong data-i18n="ask_title">Ask</strong>
                  <span data-i18n="ask_desc">Answer with top-k retrieved papers as context</span>
                </summary>
                <div class="tool-body">
                  <label data-i18n="ask_query">Ask query</label>
                  <textarea id="askQuery">What themes appear in my current embedded papers?</textarea>
                  <div class="row">
                    <div>
                      <label data-i18n="answer_model">Answer model</label>
                      <input id="answerModel" value="gpt-5-mini">
                    </div>
                    <div>
                      <label data-i18n="top_k">Top K</label>
                      <input id="askTopK" value="5">
                    </div>
                  </div>
                  <div class="actions">
                    <button class="alt" onclick="askQuestion()" data-i18n="ask">Ask</button>
                  </div>
                </div>
              </details>
            </div>

            <div class="output" id="operationsOutput">Ready.</div>
            <div class="output" id="searchOutput">Ready.</div>
          </div>
        </section>
      </div>

      <aside class="stack">
        <section class="card" id="papers-card">
          <div class="card-head">
            <div>
              <h2 data-i18n="paper_explorer_title">Paper Explorer</h2>
              <p data-i18n="paper_explorer_desc">Browse one day at a time, open the source, and star items worth keeping.</p>
            </div>
          </div>
          <div class="paper-toolbar">
            <div>
              <label data-i18n="date">Date</label>
              <select id="paperDate" onchange="syncSummaryDateFromPapers(); loadPapers();"></select>
            </div>
            <div>
              <label data-i18n="limit">Limit</label>
              <input id="paperLimit" value="20">
            </div>
          </div>
          <div class="paper-search-row">
            <div>
              <label data-i18n="paper_search">Keyword Search</label>
              <input id="paperSearch" data-i18n-placeholder="paper_search_placeholder" placeholder="title, DOI, arXiv id" onkeydown="handlePaperSearchKeydown(event)">
            </div>
          </div>
          <div class="paper-actions-row">
            <button class="ghost" onclick="loadPapers()" data-i18n="refresh_papers">Refresh Papers</button>
            <button class="ghost" onclick="clearPaperSearch()" data-i18n="clear_search">Clear Search</button>
            <button class="ghost" onclick="loadStats(); loadTodaySummary(false);" data-i18n="refresh_summary">Refresh Summary</button>
          </div>
          <div class="list" id="paperList"></div>
          <div class="footer"><span data-i18n="server_command">Server command:</span> <span class="mono">python -m research_agent.web</span></div>
        </section>
      </aside>
    </section>
  </div>
  <div class="modal-backdrop" id="paperAnalysisModal" onclick="closeAnalysisDialog(event)">
    <div class="modal-card analysis-dialog" onclick="event.stopPropagation()">
      <div class="card-head">
        <div>
          <h3 id="analysisTitle"></h3>
          <p id="analysisEditHint" data-i18n="analysis_edit_hint">Double-click the full analysis below to edit and save your revised version.</p>
        </div>
      </div>
      <div class="analysis-meta" id="analysisMeta"></div>
      <div class="footer" id="analysisStatusLine"></div>
      <label for="analysisPdfUrlInput" data-i18n="analysis_pdf_url">PDF URL</label>
      <div class="row">
        <input id="analysisPdfUrlInput" data-i18n-placeholder="analysis_pdf_url_placeholder" placeholder="https://arxiv.org/pdf/2603.12345.pdf">
        <button class="ghost" id="analysisPdfUrlButton" onclick="analyzeWithPdfUrl()" data-i18n="analyze_with_pdf_url">Analyze with PDF URL</button>
      </div>
      <div id="analysisSnapshot"></div>
      <div id="analysisDisplay" ondblclick="beginAnalysisEdit()"></div>
      <div id="analysisEditor" hidden>
        <textarea id="analysisEditorInput" class="analysis-editor"></textarea>
      </div>
      <div class="footer" id="analysisNoteMeta"></div>
      <div class="analysis-actions">
        <button class="alt" id="analysisRegenerateButton" onclick="regenerateAnalysis()" data-i18n="regenerate">Regenerate</button>
        <button class="ghost" id="analysisExportButton" onclick="exportAnalysisNote()" data-i18n="export_analysis_note">Export to Note</button>
        <button class="ghost" id="analysisOpenObsidianButton" onclick="openAnalysisInObsidian()" hidden data-i18n="open_in_obsidian">Open in Obsidian</button>
        <button id="analysisSaveButton" onclick="saveAnalysisEdit()" hidden data-i18n="save">Save</button>
        <button class="ghost" id="analysisCancelButton" onclick="cancelAnalysisEdit()" hidden data-i18n="cancel">Cancel</button>
        <button class="ghost" id="analysisCloseButton" onclick="closeAnalysisDialog()" data-i18n="close">Close</button>
      </div>
    </div>
  </div>
  <div class="modal-backdrop" id="selectionRewriteModal" onclick="closeSelectionRewriteModal(event)">
    <div class="modal-card" onclick="event.stopPropagation()">
      <div class="card-head">
        <div>
          <h3 data-i18n="rewrite_modal_title">Rewrite Selected Summary Passage</h3>
          <p data-i18n="rewrite_modal_desc">Adjust only the selected part. The rest of the daily summary stays unchanged.</p>
        </div>
      </div>
      <label data-i18n="original_passage">Original passage</label>
      <div class="modal-preview" id="selectionRewritePreview"></div>
      <label for="selectionRewriteInstruction" data-i18n="instruction">Instruction</label>
      <textarea id="selectionRewriteInstruction" data-i18n-placeholder="instruction_placeholder" placeholder="Example: Make this paragraph more concrete and add the paper sources."></textarea>
      <div class="actions">
        <button onclick="previewSelectionRewrite()" data-i18n="generate_preview">Generate Preview</button>
        <button class="ghost" id="applySelectionRewriteButton" onclick="applySelectionRewrite()" disabled data-i18n="apply_rewrite">Apply Rewrite</button>
        <button class="ghost" onclick="closeSelectionRewriteModal()" data-i18n="cancel">Cancel</button>
      </div>
      <label data-i18n="rewritten_preview">Rewritten preview</label>
      <div class="modal-preview" id="selectionRewriteResult">Generate a preview to compare the rewritten passage here.</div>
    </div>
  </div>
  <script>
    async function api(url, options = {}) {
      const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
      return data;
    }

    async function streamApi(url, payload, handlers = {}) {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || JSON.stringify(data));
      }
      if (!response.body) {
        throw new Error('Streaming response body is unavailable.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const messages = buffer.split('\n\n');
        buffer = messages.pop() || '';

        for (const message of messages) {
          if (!message.trim()) continue;
          let eventName = 'message';
          let dataText = '';
          for (const line of message.split('\n')) {
            if (line.startsWith('event:')) eventName = line.slice(6).trim();
            if (line.startsWith('data:')) dataText += line.slice(5).trim();
          }
          if (!dataText) continue;
          const parsed = JSON.parse(dataText);
          if (eventName === 'meta' && handlers.onMeta) handlers.onMeta(parsed);
          if (eventName === 'delta' && handlers.onDelta) handlers.onDelta(parsed);
          if (eventName === 'done' && handlers.onDone) handlers.onDone(parsed);
          if (eventName === 'error') throw new Error(parsed.detail || JSON.stringify(parsed));
        }

        if (done) break;
      }
    }

    const UI_TRANSLATIONS = __UI_TRANSLATIONS_JSON__;

    let currentUiLanguage = 'zh-TW';
    let currentStatsData = null;

    function t(key, vars = {}) {
      const table = UI_TRANSLATIONS[currentUiLanguage] || UI_TRANSLATIONS['en'];
      let value = table[key] || UI_TRANSLATIONS['en'][key] || key;
      for (const [name, replacement] of Object.entries(vars)) {
        value = value.replace(`{${name}}`, String(replacement));
      }
      return value;
    }

    function applyUiLanguage() {
      document.documentElement.lang = currentUiLanguage === 'zh-TW' ? 'zh-Hant' : 'en';
      document.querySelectorAll('[data-i18n]').forEach((element) => {
        element.textContent = t(element.dataset.i18n);
      });
      document.querySelectorAll('[data-i18n-placeholder]').forEach((element) => {
        element.placeholder = t(element.dataset.i18nPlaceholder);
      });
      document.getElementById('uiLanguage').value = currentUiLanguage;
      if (currentStatsData) renderStats(currentStatsData);
      if (!document.getElementById('topicStatus').dataset.topicValue) {
        document.getElementById('topicStatus').textContent = t('no_topic');
      }
      if (!currentSummaryText) {
        document.getElementById('todaySummaryOutput').textContent = t('loading_summary');
      }
      if (!currentSummarySelection) {
        setSummarySelectionStatus(t('summary_selection_default'));
      }
      if (document.getElementById('operationsOutput').textContent.trim() === 'Ready.' || document.getElementById('operationsOutput').textContent.trim() === '???') {
        document.getElementById('operationsOutput').textContent = t('ready');
      }
      if (document.getElementById('searchOutput').textContent.trim() === 'Ready.' || document.getElementById('searchOutput').textContent.trim() === '???') {
        document.getElementById('searchOutput').textContent = t('ready');
      }
    }

    function setUiLanguage(language) {
      currentUiLanguage = UI_TRANSLATIONS[language] ? language : 'zh-TW';
      localStorage.setItem('research-agent-ui-language', currentUiLanguage);
      applyUiLanguage();
      refreshTopicStatus(document.getElementById('topicStatus').dataset.topicValue || '');
    }

    function refreshTopicStatus(topic) {
      const status = document.getElementById('topicStatus');
      status.dataset.topicValue = topic || '';
      status.textContent = topic ? t('topic_active', { topic }) : t('no_topic');
    }

    function summarySelectionModeLabel(mode) {
      if (mode === 'topic') return t('selection_mode_topic');
      if (mode === 'latest') return t('selection_mode_latest');
      if (mode === 'manual-edit') return t('selection_mode_manual');
      return mode || '';
    }

    function summaryStateLabel(state) {
      if (state === 'cached') return t('summary_state_cached');
      if (state === 'generated') return t('summary_state_generated');
      if (state === 'streaming') return t('summary_state_streaming');
      return state || '';
    }

    function renderSummaryMetaLine({ date, language, paperCount, selectionMode, state, actions = [], topic = '', path = '' }) {
      const parts = [date, language, t('paper_count_label', { count: paperCount }), summarySelectionModeLabel(selectionMode), summaryStateLabel(state)];
      if (actions.length) parts.push(t('actions_label', { actions: actions.join(', ') }));
      if (topic) parts.push(t('topic_label', { topic }));
      if (path) parts.push(path);
      return parts.filter(Boolean).join(' | ');
    }

    let currentSummaryText = '';
    let currentSummarySelection = null;
    let currentSummaryRewritePreview = null;
    let isSummaryEditing = false;

    function show(id, value) {
      document.getElementById(id).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    }

    function setSummarySelectionStatus(message) {
      document.getElementById('summarySelectionStatus').textContent = message;
    }

    function setSummaryEditing(editing) {
      isSummaryEditing = editing;
      document.getElementById('todaySummaryOutput').classList.toggle('editing', editing);
      document.getElementById('summaryEditor').classList.toggle('open', editing);
      document.getElementById('rewriteSelectionButton').disabled = editing || !currentSummarySelection;
    }

    function enterSummaryEditMode() {
      if (!currentSummaryText) return;
      setSummaryEditing(true);
      document.getElementById('summaryEditorInput').value = currentSummaryText;
      setSummarySelectionStatus(t('editing_summary_status'));
      document.getElementById('summaryEditorInput').focus();
    }

    function cancelSummaryEdit() {
      setSummaryEditing(false);
      document.getElementById('summaryEditorInput').value = currentSummaryText;
      clearSummarySelection();
    }

    function clearSummarySelection(message = t('summary_selection_short')) {
      currentSummarySelection = null;
      currentSummaryRewritePreview = null;
      document.getElementById('rewriteSelectionButton').disabled = true;
      setSummarySelectionStatus(message);
    }

    function getSelectionOffsets(container) {
      const selection = window.getSelection();
      if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
      const range = selection.getRangeAt(0);
      if (!container.contains(range.commonAncestorContainer)) return null;
      const prefixRange = range.cloneRange();
      prefixRange.selectNodeContents(container);
      prefixRange.setEnd(range.startContainer, range.startOffset);
      const start = prefixRange.toString().length;
      const selectedText = range.toString();
      return { start, end: start + selectedText.length, selectedText };
    }

    function updateSummarySelection() {
      const container = document.getElementById('todaySummaryOutput');
      if (isSummaryEditing || !container || !currentSummaryText) {
        clearSummarySelection();
        return;
      }
      const selectionInfo = getSelectionOffsets(container);
      if (!selectionInfo || !selectionInfo.selectedText.trim()) {
        clearSummarySelection();
        return;
      }
      currentSummarySelection = {
        startOffset: selectionInfo.start,
        endOffset: selectionInfo.end,
        selectedText: selectionInfo.selectedText,
      };
      document.getElementById('rewriteSelectionButton').disabled = false;
      setSummarySelectionStatus(t('selected_characters_status', { count: selectionInfo.selectedText.length }));
    }

    function openSelectionRewriteModal() {
      if (!currentSummarySelection || !currentSummarySelection.selectedText.trim()) return;
      currentSummaryRewritePreview = null;
      document.getElementById('selectionRewritePreview').textContent = currentSummarySelection.selectedText;
      document.getElementById('selectionRewriteInstruction').value = '';
      document.getElementById('selectionRewriteResult').textContent = t('rewrite_preview_placeholder');
      document.getElementById('applySelectionRewriteButton').disabled = true;
      document.getElementById('selectionRewriteModal').classList.add('open');
      document.getElementById('selectionRewriteInstruction').focus();
    }

    function closeSelectionRewriteModal(event = null) {
      if (event && event.target && event.target.id !== 'selectionRewriteModal') return;
      document.getElementById('selectionRewriteModal').classList.remove('open');
    }

    function switchWorkbench(name) {
      const panels = ['data', 'search'];
      for (const panelName of panels) {
        document.getElementById(`tab-${panelName}`).classList.toggle('active', panelName === name);
        document.getElementById(`panel-${panelName}`).classList.toggle('active', panelName === name);
      }
    }

    function renderStats(data) {
      currentStatsData = data;
      const stats = [
        [t('stat_papers'), data.papers],
        [t('stat_feeds'), data.feeds],
        [t('stat_authors'), data.authors],
      ];
      document.getElementById('stats').innerHTML = stats.map(([label, value]) => `<div class="stat"><span>${label}</span><b>${value ?? 0}</b></div>`).join('');
    }

    function paperAbstractId(entryId) {
      return `paper-abstract-${entryId}`;
    }

    function paperAbstractButtonId(entryId) {
      return `paper-abstract-button-${entryId}`;
    }

    let currentAnalysisRecord = null;
    let currentAnalysisEditing = false;
    let currentAnalysisLoading = false;

    function paperAnalysisId(entryId) {
      return `paper-analysis-${entryId}`;
    }

    function paperAnalysisButtonId(entryId) {
      return `paper-analysis-button-${entryId}`;
    }

    function setAnalysisStatus(message = '') {
      const statusLine = document.getElementById('analysisStatusLine');
      if (statusLine) statusLine.textContent = message || '';
    }

    function setAnalysisBusyState(isBusy, button = null) {
      currentAnalysisLoading = isBusy;
      if (button) {
        button.disabled = isBusy;
        button.classList.toggle('active', isBusy);
      }
      const regenerateButton = document.getElementById('analysisRegenerateButton');
      const exportButton = document.getElementById('analysisExportButton');
      const pdfUrlInput = document.getElementById('analysisPdfUrlInput');
      const pdfUrlButton = document.getElementById('analysisPdfUrlButton');
      const openObsidianButton = document.getElementById('analysisOpenObsidianButton');
      const saveButton = document.getElementById('analysisSaveButton');
      const cancelButton = document.getElementById('analysisCancelButton');
      const closeButton = document.getElementById('analysisCloseButton');
      if (regenerateButton) regenerateButton.disabled = isBusy;
      if (exportButton) exportButton.disabled = isBusy;
      if (pdfUrlInput) pdfUrlInput.disabled = isBusy;
      if (pdfUrlButton) pdfUrlButton.disabled = isBusy;
      if (openObsidianButton) openObsidianButton.disabled = isBusy;
      if (saveButton) saveButton.disabled = isBusy;
      if (cancelButton) cancelButton.disabled = isBusy;
      if (closeButton) closeButton.disabled = isBusy;
    }

    function analysisSectionMap(markdown) {
      const sections = {};
      const matches = String(markdown || '').split(/\n## /);
      for (const chunk of matches) {
        const cleaned = chunk.trim();
        if (!cleaned) continue;
        const normalized = cleaned.startsWith('## ') ? cleaned.slice(3) : cleaned;
        const firstNewline = normalized.indexOf('\n');
        const heading = (firstNewline === -1 ? normalized : normalized.slice(0, firstNewline)).trim();
        const body = (firstNewline === -1 ? '' : normalized.slice(firstNewline + 1)).trim();
        if (heading) sections[heading] = body;
      }
      return sections;
    }

    function renderAnalysisSnapshot(markdown) {
      const sections = analysisSectionMap(markdown);
      const pairs = [
        ['Research Problem', t('snapshot_problem')],
        ['Research Design', t('snapshot_design')],
        ['Data And Sample', t('snapshot_data')],
        ['Analysis Method', t('snapshot_analysis')],
        ['Main Findings', t('snapshot_findings')],
        ['Limitations', t('snapshot_limitations')],
      ];
      return `<div class="analysis-grid">${pairs.map(([key, label]) => {
        const value = sections[key] || t('analysis_not_available');
        return `<section class="analysis-tile"><h4>${escapeHtml(label)}</h4><div>${linkifyText(value).split('\n').join('<br>')}</div></section>`;
      }).join('')}</div>`;
    }

    function setAnalysisEditing(editing) {
      currentAnalysisEditing = editing;
      document.getElementById('analysisDisplay').hidden = editing;
      document.getElementById('analysisEditor').hidden = !editing;
      document.getElementById('analysisSaveButton').hidden = !editing;
      document.getElementById('analysisCancelButton').hidden = !editing;
      document.getElementById('analysisRegenerateButton').hidden = editing;
      document.getElementById('analysisEditHint').hidden = editing;
      if (editing && currentAnalysisRecord) {
        document.getElementById('analysisEditorInput').value = currentAnalysisRecord.analysis_text || '';
      }
    }

    function closeAnalysisDialog(event) {
      if (event && event.target && event.target.id !== 'paperAnalysisModal') return;
      document.getElementById('paperAnalysisModal').classList.remove('open');
      currentAnalysisRecord = null;
      setAnalysisEditing(false);
      document.getElementById('analysisSnapshot').innerHTML = '';
      document.getElementById('analysisDisplay').innerHTML = '';
      document.getElementById('analysisMeta').innerHTML = '';
      setAnalysisStatus('');
      document.getElementById('analysisNoteMeta').textContent = '';
      document.getElementById('analysisPdfUrlInput').value = '';
      document.getElementById('analysisOpenObsidianButton').hidden = true;
      document.getElementById('analysisTitle').textContent = '';
    }

    function openAnalysisDialog(data) {
      currentAnalysisRecord = data;
      document.getElementById('analysisTitle').textContent = data.title || '';
      const badges = [data.source_type, data.page_count ? `${data.page_count} pages` : '', data.model, data.status].filter(Boolean);
      document.getElementById('analysisMeta').innerHTML = badges.map(value => `<span class="analysis-badge">${escapeHtml(String(value))}</span>`).join('');
      document.getElementById('analysisPdfUrlInput').value = data.paper_pdf_url || '';
      setAnalysisStatus('');
      document.getElementById('analysisSnapshot').innerHTML = renderAnalysisSnapshot(data.analysis_text || '');
      document.getElementById('analysisDisplay').innerHTML = `<div class="analysis-body">${linkifyText(data.analysis_text || '').split('\n').join('<br>')}</div>`;
      const noteMeta = data.note_path ? `${t('analysis_note_saved')}: ${data.note_path}` : t('analysis_note_not_exported');
      document.getElementById('analysisNoteMeta').textContent = noteMeta;
      document.getElementById('analysisOpenObsidianButton').hidden = !data.obsidian_uri;
      document.getElementById('paperAnalysisModal').classList.add('open');
      setAnalysisEditing(false);
    }

    async function openPaperAnalysis(entryId, force = false, pdfUrlOverride = "") {
      if (currentAnalysisLoading) return;
      const button = document.getElementById(paperAnalysisButtonId(entryId));
      const originalLabel = button ? button.textContent : '';
      if (button) button.textContent = t('analyzing_paper');
      setAnalysisStatus(t('analysis_running'));
      setAnalysisBusyState(true, button);
      try {
        const data = await api('/api/actions/analyze-paper', {
          method: 'POST',
          body: JSON.stringify({ entry_id: entryId, force, pdf_url_override: pdfUrlOverride || null }),
        });
        openAnalysisDialog(data);
      } catch (error) {
        alert(String(error.message || error));
      } finally {
        setAnalysisBusyState(false, button);
        if (button) button.textContent = originalLabel || t('analyze_paper');
      }
    }

    function beginAnalysisEdit() {
      if (!currentAnalysisRecord) return;
      setAnalysisEditing(true);
    }

    function cancelAnalysisEdit() {
      setAnalysisEditing(false);
    }

    async function saveAnalysisEdit() {
      if (!currentAnalysisRecord || currentAnalysisLoading) return;
      const analysisText = document.getElementById('analysisEditorInput').value.trim();
      if (!analysisText) return;
      setAnalysisStatus(t('analysis_saving'));
      setAnalysisBusyState(true);
      try {
        const data = await api('/api/actions/analyze-paper/save', {
        method: 'POST',
        body: JSON.stringify({ run_id: currentAnalysisRecord.id || currentAnalysisRecord.run_id, analysis_text: analysisText }),
      });
        currentAnalysisRecord = data;
        openAnalysisDialog(data);
      } finally {
        setAnalysisBusyState(false);
      }
    }

    async function regenerateAnalysis() {
      if (!currentAnalysisRecord || currentAnalysisLoading) return;
      await openPaperAnalysis(currentAnalysisRecord.paper_entry_id, true);
    }

    async function analyzeWithPdfUrl() {
      if (!currentAnalysisRecord || currentAnalysisLoading) return;
      const pdfUrl = document.getElementById('analysisPdfUrlInput').value.trim();
      if (!pdfUrl) {
        alert(t('analysis_pdf_url_required'));
        return;
      }
      setAnalysisStatus(t('analysis_running_pdf_url'));
      await openPaperAnalysis(currentAnalysisRecord.paper_entry_id, true, pdfUrl);
    }

    async function exportAnalysisNote() {
      if (!currentAnalysisRecord || currentAnalysisLoading) return;
      setAnalysisStatus(t('analysis_exporting_note'));
      setAnalysisBusyState(true);
      try {
        const data = await api('/api/actions/analyze-paper/export-note', {
          method: 'POST',
          body: JSON.stringify({ run_id: currentAnalysisRecord.id || currentAnalysisRecord.run_id }),
        });
        currentAnalysisRecord = { ...currentAnalysisRecord, ...data };
        openAnalysisDialog(currentAnalysisRecord);
      } catch (error) {
        alert(String(error.message || error));
      } finally {
        setAnalysisBusyState(false);
      }
    }

    function openAnalysisInObsidian() {
      if (!currentAnalysisRecord || !currentAnalysisRecord.obsidian_uri) return;
      window.location.href = currentAnalysisRecord.obsidian_uri;
    }

    function renderPaperAbstract(abstract) {
      const cleaned = String(abstract || '').trim();
      if (!cleaned) {
        return `<div class="paper-abstract empty">${escapeHtml(t('no_abstract_available'))}</div>`;
      }
      return `<div class="paper-abstract">${linkifyText(cleaned).split('\n').join('<br>')}</div>`;
    }

    function togglePaperAbstract(entryId) {
      const container = document.getElementById(paperAbstractId(entryId));
      const button = document.getElementById(paperAbstractButtonId(entryId));
      if (!container || !button) {
        return;
      }
      const expanded = container.hidden;
      container.hidden = !expanded;
      button.classList.toggle('active', expanded);
      button.textContent = expanded ? t('hide_abstract') : t('show_abstract');
    }

    function renderPapers(items) {
      const root = document.getElementById('paperList');
      if (!items.length) {
        root.innerHTML = `<div class="meta">${escapeHtml(t('no_papers_for_date'))}</div>`;
        return;
      }

      root.innerHTML = items.map(item => {
        const titleHref = item.link || item.pdf_url || '#';
        const titleHtml = titleHref === '#'
          ? `<span class="paper-title"><strong>${escapeHtml(item.title)}</strong></span>`
          : `<a class="paper-title-link paper-title" href="${titleHref}" target="_blank" rel="noopener noreferrer"><strong>${escapeHtml(item.title)}</strong></a>`;
        const starred = Array.isArray(item.tags) && item.tags.includes('starred');
        const authors = (item.authors || []).join(', ') || t('unknown_authors');
        const starIcon = starred ? '&#9733;' : '&#9734;';
        const hasSource = Boolean(item.link);
        const hasPdf = Boolean(item.pdf_url);
        return `
          <div class="paper-item">
            <div class="paper-head">
              <button class="star-btn ${starred ? 'active' : ''}" onclick="toggleStar('${item.entry_id}')" title="${t('collect_paper')}" aria-label="${t('collect_paper')}">${starIcon}</button>
              <div>
                <div>${titleHtml}</div>
                <div class="paper-meta-row">${escapeHtml(item.published)} | ${escapeHtml(item.source)}${item.doi ? ` | DOI: ${escapeHtml(item.doi)}` : ''}</div>
                <div class="paper-authors">${escapeHtml(authors)}</div>
                <div class="paper-actions">
                  <button type="button" class="paper-chip" id="${paperAbstractButtonId(item.entry_id)}" onclick="togglePaperAbstract('${item.entry_id}')">${t('show_abstract')}</button>
                  ${hasSource ? `<a class="paper-chip" href="${item.link}" target="_blank" rel="noopener noreferrer">${t('open_source')}</a>` : ''}
                  ${hasPdf ? `<a class="paper-chip" href="${item.pdf_url}" target="_blank" rel="noopener noreferrer">${t('open_pdf')}</a>` : ''}
                  <button type="button" class="paper-chip warn" id="${paperAnalysisButtonId(item.entry_id)}" onclick="openPaperAnalysis('${item.entry_id}')">${t('analyze_paper')}</button>
                </div>
                <div id="${paperAbstractId(item.entry_id)}" hidden>${renderPaperAbstract(item.abstract)}</div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    async function toggleStar(entryId) {
      await api('/api/actions/toggle-star', {
        method: 'POST',
        body: JSON.stringify({ entry_id: entryId }),
      });
      await loadPapers();
    }


    async function loadResearchTopic() {
      const data = await api('/api/research-topic');
      const topic = data.topic || '';
      document.getElementById('researchTopic').value = topic;
      refreshTopicStatus(topic);
    }

    async function saveResearchTopic() {
      const topic = document.getElementById('researchTopic').value || '';
      const data = await api('/api/research-topic', {
        method: 'POST',
        body: JSON.stringify({ topic }),
      });
      document.getElementById('researchTopic').value = data.topic || '';
      refreshTopicStatus(data.topic || '');
      await loadTodaySummary(true);
    }

    async function loadStats() {
      renderStats(await api('/api/stats'));
    }

    async function loadPaperDates() {
      const dates = await api('/api/paper-dates');
      const select = document.getElementById('paperDate');
      select.innerHTML = dates.map(date => `<option value="${date}">${date}</option>`).join('');
      if (!select.value && dates.length) {
        select.value = dates[0];
      }
    }

    function syncSummaryDateFromPapers() {
      const paperDate = document.getElementById('paperDate').value;
      if (paperDate) {
        document.getElementById('summaryDate').value = paperDate;
      }
    }

    function syncPaperDateFromSummary() {
      const summaryDate = document.getElementById('summaryDate').value;
      const paperDate = document.getElementById('paperDate');
      if (!summaryDate || !paperDate) {
        return;
      }
      const hasOption = Array.from(paperDate.options).some(option => option.value === summaryDate);
      if (hasOption) {
        paperDate.value = summaryDate;
        loadPapers();
      }
    }

    function handlePaperSearchKeydown(event) {
      if (event.key === 'Enter') {
        event.preventDefault();
        loadPapers();
      }
    }

    function clearPaperSearch() {
      const input = document.getElementById('paperSearch');
      if (!input) return;
      input.value = '';
      loadPapers();
    }

    async function loadPapers() {
      const published = encodeURIComponent(document.getElementById('paperDate').value || '');
      const limit = encodeURIComponent(document.getElementById('paperLimit').value || '20');
      const text = encodeURIComponent(document.getElementById('paperSearch').value || '');
      renderPapers(await api(`/api/papers?limit=${limit}&published=${published}&text=${text}`));
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function linkifyText(value) {
      const escaped = escapeHtml(value);
      return escaped.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    }

    function renderSummaryText(summary) {
      currentSummaryText = summary || '';
      setSummaryEditing(false);
      document.getElementById('todaySummaryOutput').innerHTML = linkifyText(currentSummaryText).split('\n').join('<br>');
      document.getElementById('summaryEditorInput').value = currentSummaryText;
      clearSummarySelection();
    }

    async function saveSummaryEdit() {
      const summaryText = document.getElementById('summaryEditorInput').value.trim();
      if (!summaryText) {
        setSummarySelectionStatus(t('summary_empty'));
        return;
      }
      const language = document.getElementById('summaryLanguage').value;
      const targetDate = document.getElementById('summaryDate').value || new Date().toISOString().slice(0, 10);
      setSummarySelectionStatus(t('saving_summary'));
      try {
        const data = await api('/api/actions/today-summary/save', {
          method: 'POST',
          body: JSON.stringify({
            language,
            target_date: targetDate,
            summary_text: summaryText,
          }),
        });
        renderSummaryText(data.summary || '');
        document.getElementById('todaySummaryMeta').textContent = renderSummaryMetaLine({ date: data.date, language: data.language, paperCount: '', selectionMode: 'manual-edit', state: 'generated', topic: data.topic || '', path: data.path || '' });
        setSummarySelectionStatus(t('summary_saved'));
      } catch (error) {
        setSummarySelectionStatus(String(error.message || error));
      }
    }

    function renderTodaySummary(data) {
      renderSummaryText(data.summary || '');
      const cacheText = data.cached ? 'cached' : 'generated';
      const actions = Array.isArray(data.actions) ? data.actions : [];
      document.getElementById('todaySummaryMeta').textContent = renderSummaryMetaLine({ date: data.date, language: data.language, paperCount: data.paper_count, selectionMode: data.selection_mode, state: cacheText, actions, topic: data.topic || '', path: data.path || '' });
    }

    async function previewSelectionRewrite() {
      if (!currentSummarySelection || !currentSummarySelection.selectedText.trim()) {
        return;
      }
      const instruction = document.getElementById('selectionRewriteInstruction').value.trim();
      if (!instruction) {
        setSummarySelectionStatus(t('add_instruction'));
        return;
      }
      const language = document.getElementById('summaryLanguage').value;
      const targetDate = document.getElementById('summaryDate').value || new Date().toISOString().slice(0, 10);
      document.getElementById('selectionRewriteResult').textContent = t('generating_preview');
      document.getElementById('applySelectionRewriteButton').disabled = true;
      try {
        const data = await api('/api/actions/today-summary/rewrite-selection/preview', {
          method: 'POST',
          body: JSON.stringify({
            language,
            model: document.getElementById('answerModel').value || 'gpt-5-mini',
            target_date: targetDate,
            summary_text: currentSummaryText,
            start_offset: currentSummarySelection.startOffset,
            end_offset: currentSummarySelection.endOffset,
            selected_text: currentSummarySelection.selectedText,
            instruction,
          }),
        });
        currentSummaryRewritePreview = data;
        document.getElementById('selectionRewriteResult').textContent = data.rewritten_text || '';
        document.getElementById('applySelectionRewriteButton').disabled = !data.rewritten_text;
        setSummarySelectionStatus(t('preview_ready'));
      } catch (error) {
        currentSummaryRewritePreview = null;
        document.getElementById('selectionRewriteResult').textContent = String(error.message || error);
        setSummarySelectionStatus(String(error.message || error));
      }
    }

    async function applySelectionRewrite() {
      if (!currentSummarySelection || !currentSummarySelection.selectedText.trim() || !currentSummaryRewritePreview?.rewritten_text) {
        return;
      }
      const instruction = document.getElementById('selectionRewriteInstruction').value.trim();
      const language = document.getElementById('summaryLanguage').value;
      const targetDate = document.getElementById('summaryDate').value || new Date().toISOString().slice(0, 10);
      closeSelectionRewriteModal();
      setSummarySelectionStatus(t('applying_rewrite'));
      try {
        const data = await api('/api/actions/today-summary/rewrite-selection/apply', {
          method: 'POST',
          body: JSON.stringify({
            language,
            model: document.getElementById('answerModel').value || 'gpt-5-mini',
            target_date: targetDate,
            summary_text: currentSummaryText,
            start_offset: currentSummaryRewritePreview.start_offset ?? currentSummarySelection.startOffset,
            end_offset: currentSummaryRewritePreview.end_offset ?? currentSummarySelection.endOffset,
            selected_text: currentSummaryRewritePreview.selected_text || currentSummarySelection.selectedText,
            instruction,
            rewritten_text: currentSummaryRewritePreview.rewritten_text,
          }),
        });
        renderSummaryText(data.summary || '');
        document.getElementById('todaySummaryMeta').textContent = renderSummaryMetaLine({ date: data.date, language: data.language, paperCount: '', selectionMode: 'manual-edit', state: 'generated', topic: data.topic || '', path: data.path || '' });
        setSummarySelectionStatus(t('rewrite_saved'));
      } catch (error) {
        setSummarySelectionStatus(String(error.message || error));
      }
    }

    async function loadTodaySummary(forceRefresh) {
      const language = document.getElementById('summaryLanguage').value;
      const targetDate = document.getElementById('summaryDate').value || new Date().toISOString().slice(0, 10);
      const payload = { language, model: document.getElementById('answerModel').value || 'gpt-5-mini', limit: 15, target_date: targetDate, force_refresh: !!forceRefresh };
      currentSummaryText = '';
      clearSummarySelection(forceRefresh ? t('regenerating_summary') : t('loading_summary'));
      document.getElementById('todaySummaryOutput').textContent = forceRefresh ? t('regenerating_summary') : t('loading_summary');
      document.getElementById('todaySummaryMeta').textContent = '';

      let streamedSummary = '';
      let streamedMeta = null;
      try {
        await streamApi('/api/actions/today-summary/stream', payload, {
          onMeta(data) {
            streamedMeta = data;
            streamedSummary = '';
            document.getElementById('todaySummaryOutput').textContent = data.cached ? t('loading_cached_summary') : t('streaming_summary');
            clearSummarySelection(data.cached ? t('loading_cached_summary') : t('streaming_summary'));
            document.getElementById('todaySummaryMeta').textContent = renderSummaryMetaLine({ date: data.date, language: data.language, paperCount: data.paper_count, selectionMode: data.selection_mode, state: data.cached ? 'cached' : 'streaming', topic: data.topic || '', path: data.path || '' });
          },
          onDelta(data) {
            streamedSummary += data.text || '';
            currentSummaryText = streamedSummary;
            document.getElementById('todaySummaryOutput').innerHTML = linkifyText(streamedSummary).split('\n').join('<br>');
          },
          onDone(data) {
            renderTodaySummary({ ...(streamedMeta || {}), ...(data || {}), summary: streamedSummary });
          },
        });
      } catch (error) {
        try {
          const data = await api('/api/actions/today-summary', {
            method: 'POST',
            body: JSON.stringify(payload),
          });
          renderTodaySummary(data);
        } catch (fallbackError) {
          document.getElementById('todaySummaryOutput').textContent = String(fallbackError.message || fallbackError || error.message || error);
        }
      }
    }

    async function ensureDailyReady() {
      const language = document.getElementById('summaryLanguage').value;
      const targetDate = document.getElementById('summaryDate').value || new Date().toISOString().slice(0, 10);
      currentSummaryText = '';
      clearSummarySelection(t('checking_summary'));
      document.getElementById('todaySummaryOutput').textContent = t('checking_summary');
      document.getElementById('todaySummaryMeta').textContent = '';
      try {
        const data = await api('/api/actions/ensure-daily-ready', {
          method: 'POST',
          body: JSON.stringify({ language, model: document.getElementById('answerModel').value || 'gpt-5-mini', limit: 15, target_date: targetDate, force_refresh: false }),
        });
        renderTodaySummary(data);
        if ((data.imported_count || 0) > 0 || (Array.isArray(data.actions) && data.actions.includes('ingest'))) {
          await loadStats();
          await loadPaperDates();
          syncPaperDateFromSummary();
        }
      } catch (error) {
        document.getElementById('todaySummaryOutput').textContent = String(error.message || error);
      }
    }

    async function runIngest() {
      show('operationsOutput', t('running_ingest'));
      const data = await api('/api/actions/ingest', { method: 'POST', body: '{}' });
      show('operationsOutput', data);
      await loadStats();
      await loadPaperDates();
      syncPaperDateFromSummary();
      await loadTodaySummary(false);
    }

    async function runDigest() {
      show('operationsOutput', t('generating_digest'));
      const data = await api('/api/actions/digest', { method: 'POST', body: JSON.stringify({ days: 7 }) });
      show('operationsOutput', data);
    }

    async function catalogPdfs(embed) {
      show('operationsOutput', t('cataloging_pdfs'));
      const data = await api('/api/actions/catalog-pdfs', {
        method: 'POST',
        body: JSON.stringify({
          pdf_dir: document.getElementById('catalogPdfDir').value,
          embed,
          embedding_model: document.getElementById('catalogEmbeddingModel').value,
          dimensions: parseInt(document.getElementById('catalogDimensions').value || '0') || null,
          init_vec: false,
        }),
      });
      show('operationsOutput', data);
      await loadStats();
      await loadPaperDates();
      syncPaperDateFromSummary();
      await loadTodaySummary(false);
    }

    async function importDois(embed) {
      show('operationsOutput', t('importing_dois'));
      const data = await api('/api/actions/import-dois', {
        method: 'POST',
        body: JSON.stringify({
          doi_file: document.getElementById('doiFile').value,
          embed,
          embedding_model: document.getElementById('catalogEmbeddingModel').value,
          dimensions: parseInt(document.getElementById('catalogDimensions').value || '0') || null,
          init_vec: false,
        }),
      });
      show('operationsOutput', data);
      await loadStats();
      await loadPaperDates();
      syncPaperDateFromSummary();
      await loadTodaySummary(false);
    }

    async function importUrl(embed) {
      show('operationsOutput', t('importing_url'));
      const data = await api('/api/actions/import-url', {
        method: 'POST',
        body: JSON.stringify({
          url: document.getElementById('importUrlInput').value,
          embed,
          embedding_model: document.getElementById('catalogEmbeddingModel').value,
          dimensions: parseInt(document.getElementById('catalogDimensions').value || '0') || null,
          init_vec: false,
        }),
      });
      show('operationsOutput', data);
      await loadStats();
      await loadPaperDates();
      syncPaperDateFromSummary();
      await loadTodaySummary(false);
    }

    async function semanticSearch() {
      show('searchOutput', t('running_semantic_search'));
      const data = await api('/api/actions/semantic-search', {
        method: 'POST',
        body: JSON.stringify({
          query: document.getElementById('semanticQuery').value,
          model: document.getElementById('semanticModel').value,
          limit: parseInt(document.getElementById('semanticLimit').value || '5'),
        }),
      });
      show('searchOutput', data);
    }

    async function askQuestion() {
      show('searchOutput', t('running_ask'));
      const data = await api('/api/actions/ask', {
        method: 'POST',
        body: JSON.stringify({
          query: document.getElementById('askQuery').value,
          embedding_model: document.getElementById('semanticModel').value,
          answer_model: document.getElementById('answerModel').value,
          top_k: parseInt(document.getElementById('askTopK').value || '5'),
        }),
      });
      show('searchOutput', data);
    }

    async function initializeDashboard() {
      currentUiLanguage = localStorage.getItem('research-agent-ui-language') || 'zh-TW';
      applyUiLanguage();
      document.getElementById('summaryDate').value = new Date().toISOString().slice(0, 10);
      const summaryBox = document.getElementById('todaySummaryOutput');
      summaryBox.addEventListener('mouseup', () => setTimeout(updateSummarySelection, 0));
      summaryBox.addEventListener('keyup', () => setTimeout(updateSummarySelection, 0));
      document.addEventListener('selectionchange', () => {
        const modalOpen = document.getElementById('selectionRewriteModal').classList.contains('open');
        if (!modalOpen) {
          setTimeout(updateSummarySelection, 0);
        }
      });
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') closeSelectionRewriteModal();
      });

      try {
        await loadResearchTopic();
      } catch (error) {
        document.getElementById('topicStatus').textContent = String(error.message || error);
        document.getElementById('topicStatus').dataset.topicValue = '';
      }

      try {
        await ensureDailyReady();
      } catch (error) {
        document.getElementById('todaySummaryOutput').textContent = String(error.message || error);
      }

      await Promise.allSettled([
        loadPaperDates(),
        loadStats(),
      ]);

      syncPaperDateFromSummary();

      try {
        await loadPapers();
      } catch (error) {
        document.getElementById('paperList').innerHTML = `<div class="meta">${escapeHtml(String(error.message || error))}</div>`;
      }
    }

    initializeDashboard();
  </script>
</body>
</html>
"""


def make_config() -> AppConfig:
    root = Path.cwd()
    return AppConfig(feeds_path=root / "feeds.txt", data_dir=root / "data", vault_dir=root / "vault")


RESEARCH_TOPIC_KEY = "research_topic"


def _attach_analysis_note_metadata(config: AppConfig, data: dict[str, object]) -> dict[str, object]:
    record = dict(data)
    paper_entry_id = str(record.get("paper_entry_id") or "").strip()
    if paper_entry_id and (not record.get("title") or record.get("paper_pdf_url") is None):
        with get_connection(config) as connection:
            initialize_database(connection)
            paper = fetch_paper(connection, paper_entry_id)
        if paper is not None:
            record.setdefault("title", paper.get("title"))
            record.setdefault("paper_title", paper.get("title"))
            record["paper_pdf_url"] = str(paper.get("pdf_url") or "")
    note_path = str(record.get("note_path") or "").strip()
    if note_path:
        try:
            note_path_obj = Path(note_path)
            relative = note_path_obj.relative_to(config.vault_dir).as_posix()
            record["obsidian_uri"] = f"obsidian://open?vault={quote(config.vault_dir.name)}&file={quote(relative, safe='/')}"
        except Exception:
            pass
    return record


def _get_research_topic(config: AppConfig) -> str:
    with get_connection(config) as connection:
        initialize_database(connection)
        return (get_setting(connection, RESEARCH_TOPIC_KEY) or "").strip()


def _set_research_topic(config: AppConfig, topic: str) -> str:
    cleaned = " ".join(topic.split()).strip()
    with get_connection(config) as connection:
        initialize_database(connection)
        set_setting(connection, RESEARCH_TOPIC_KEY, cleaned)
    return cleaned


def _ensure_summary_embeddings(config: AppConfig, papers: list[dict[str, object]], model: str = SUMMARY_FILTER_EMBEDDING_MODEL) -> None:
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


def _resolve_target_date(target_date: str | None) -> str:
    if not target_date:
        return datetime.now(TAIPEI_TZ).date().isoformat()
    try:
        return date.fromisoformat(target_date).isoformat()
    except ValueError as exc:
        raise ValueError(f"Invalid target date: {target_date}") from exc



def _papers_for_date(config: AppConfig, target_date: str, limit: int) -> list[dict[str, object]]:
    resolved_date = _resolve_target_date(target_date)
    with get_connection(config) as connection:
        initialize_database(connection)
        return list_papers(connection, limit=limit, published=resolved_date)



def _select_summary_papers(config: AppConfig, target_date: str, limit: int) -> tuple[list[dict[str, object]], str]:
    daily_items = _papers_for_date(config, target_date, 200)
    topic = _get_research_topic(config)
    if not topic:
        return daily_items[:limit], "latest"

    _ensure_summary_embeddings(config, daily_items, SUMMARY_FILTER_EMBEDDING_MODEL)
    topic_embedding = OpenAIEmbeddingClient().create_embeddings([topic], model=SUMMARY_FILTER_EMBEDDING_MODEL).embeddings[0]
    entry_ids = [str(item.get("entry_id")) for item in daily_items if item.get("entry_id")]
    with get_connection(config) as connection:
        initialize_database(connection)
        embeddings = fetch_embeddings_for_entries(connection, entry_ids, SUMMARY_FILTER_EMBEDDING_MODEL)

    scored: list[dict[str, object]] = []
    for item in daily_items:
        entry_id = str(item.get("entry_id") or "")
        embedding = embeddings.get(entry_id)
        if not embedding:
            continue
        score = _cosine_similarity(topic_embedding, embedding)
        ranked = dict(item)
        ranked["relevance_score"] = score
        scored.append(ranked)

    scored.sort(key=lambda item: float(item.get("relevance_score", -999.0)), reverse=True)
    if scored:
        return scored[:limit], "topic"
    return daily_items[:limit], "latest"


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return sum(l * r for l, r in zip(left, right)) / (left_norm * right_norm)


def _encode_summary_cache(summary: str, topic: str) -> str:
    topic_line = topic.replace("\n", " ").strip()
    header = f"<!-- topic: {topic_line} | prompt_version: {SUMMARY_PROMPT_VERSION} -->"
    return f"{header}\n\n{summary}"


def _read_summary_cache(cache_path: Path) -> tuple[str, str, str]:
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


def _summary_cache_path(config: AppConfig, language: str, day_label: str) -> Path:
    safe_language = language.replace("/", "-").replace("\\", "-")
    target_dir = config.data_dir / "daily-summaries" / safe_language
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{day_label}.md"


def _summary_eligible_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    return [item for item in items if str(item.get("abstract") or "").strip()]


def _summary_empty_message(language: str, *, had_items_without_abstract: bool) -> str:
    if had_items_without_abstract:
        if language == "zh-TW":
            return "今天選到的 paper 缺少可用摘要，因此本次摘要已略過這些文獻。若需要納入，請先補 abstract 或改用全文抽取流程。"
        return "The selected papers for this date do not have usable abstracts, so they were skipped from the summary. Add abstracts or use the full-text extraction flow first."
    return EMPTY_MESSAGES.get(language, EMPTY_MESSAGES["en"])


def _build_daily_summary_prompt(items: list[dict[str, object]], *, language: str, day_label: str, topic: str) -> str:
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
- Mention that you recently read several related papers and specify the source of papers.
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
- Avoid first-person narration. Do not use "I", "we", or the Chinese first-person form "?" unless absolutely unavoidable.
- Avoid semi-academic stock phrasing such as "this study", "the discussion above", "in summary", or similar formulaic transitions unless they are truly necessary.
- After the main body, add a separate section titled `References`.
- In `References`, you must list the papers you used with explicit numbering.
- Each reference line must include: number, title, authors, source, and URL.
- Prefer the Paper URL; if missing, use the PDF URL; if both are missing, write `URL: N/A`.
- The tone of the summary should be natural and conversational, not academic or formal.

Suggested note flow:
- Paragraph 1: why this cluster of papers matters now, and what thread connects them.
- Paragraphs 2-4: discuss representative papers as notes, each with problem, method, and finding.
- Paragraph 5: synthesize patterns, contrast concepts, and end with a research-facing question or design idea.
- Final section: `References` only limited to the papers used in this summary.
- Format each reference exactly like this:
  1. Title: <title> | Authors: <authors> | Source: <source> | URL: <url>
Papers for the selected date:

{joined_papers}
"""


def _sse_event(event: str, payload: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"



def _build_summary_rewrite_prompt(*, language: str, day_label: str, topic: str, full_summary: str, selected_text: str, instruction: str, prefix: str, suffix: str) -> str:
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


def _resolve_summary_selection(summary_text: str, start_offset: int, end_offset: int, selected_text: str) -> tuple[int, int, str]:
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



def _preview_summary_selection_rewrite(config: AppConfig, request: SummaryRewriteRequest) -> dict[str, object]:
    target_date = _resolve_target_date(request.target_date)
    summary_text = request.summary_text or ""
    start_offset, end_offset, selected_text = _resolve_summary_selection(
        summary_text,
        request.start_offset,
        request.end_offset,
        request.selected_text,
    )

    instruction = request.instruction.strip()
    if not instruction:
        raise ValueError("Rewrite instruction is required.")

    prefix = summary_text[max(0, start_offset - 1200):start_offset]
    suffix = summary_text[end_offset:min(len(summary_text), end_offset + 1200)]
    topic = _get_research_topic(config)
    prompt = _build_summary_rewrite_prompt(
        language=request.language,
        day_label=target_date,
        topic=topic,
        full_summary=summary_text,
        selected_text=selected_text,
        instruction=instruction,
        prefix=prefix,
        suffix=suffix,
    )
    rewritten_text = OpenAIAnswerClient().create_answer(prompt=prompt, model=request.model).text.strip()
    if not rewritten_text:
        raise ValueError("The model returned an empty rewrite.")

    return {
        "date": target_date,
        "language": request.language,
        "topic": topic,
        "selected_text": selected_text,
        "rewritten_text": rewritten_text,
        "start_offset": start_offset,
        "end_offset": end_offset,
    }



def _apply_summary_selection_rewrite(config: AppConfig, request: SummaryRewriteApplyRequest) -> dict[str, object]:
    target_date = _resolve_target_date(request.target_date)
    summary_text = request.summary_text or ""
    start_offset, end_offset, _ = _resolve_summary_selection(
        summary_text,
        request.start_offset,
        request.end_offset,
        request.selected_text,
    )
    rewritten_text = request.rewritten_text.strip()
    if not rewritten_text:
        raise ValueError("Rewritten text is required.")

    topic = _get_research_topic(config)
    updated_summary = summary_text[:start_offset] + rewritten_text + summary_text[end_offset:]
    cache_path = _summary_cache_path(config, request.language, target_date)
    cache_path.write_text(_encode_summary_cache(updated_summary, topic), encoding="utf-8")
    return {
        "date": target_date,
        "language": request.language,
        "topic": topic,
        "summary": updated_summary,
        "rewritten_text": rewritten_text,
        "path": str(cache_path),
    }


def _save_summary_text(config: AppConfig, request: SummarySaveRequest) -> dict[str, object]:
    target_date = _resolve_target_date(request.target_date)
    summary_text = (request.summary_text or "").strip()
    if not summary_text:
        raise ValueError("Summary text is required.")
    topic = _get_research_topic(config)
    cache_path = _summary_cache_path(config, request.language, target_date)
    cache_path.write_text(_encode_summary_cache(summary_text, topic), encoding="utf-8")
    paper_count = len(_select_summary_papers(config, target_date, 15)[0])
    return {
        "date": target_date,
        "language": request.language,
        "topic": topic,
        "summary": summary_text,
        "path": str(cache_path),
        "paper_count": paper_count,
        "cached": False,
        "selection_mode": "manual-edit",
    }


def _stream_today_summary(config: AppConfig, request: TodaySummaryRequest):
    target_date = _resolve_target_date(request.target_date)
    items, selection_mode = _select_summary_papers(config, target_date, request.limit)
    items = enrich_missing_abstracts(config, items)
    summary_items = _summary_eligible_items(items)
    topic = _get_research_topic(config)
    cache_path = _summary_cache_path(config, request.language, target_date)
    meta = {
        "date": target_date,
        "language": request.language,
        "paper_count": len(summary_items),
        "selection_mode": selection_mode,
        "topic": topic,
        "path": str(cache_path),
    }

    if cache_path.exists() and not request.force_refresh:
        cached_topic, cached_summary, cached_prompt_version = _read_summary_cache(cache_path)
        if cached_topic == topic and cached_prompt_version == SUMMARY_PROMPT_VERSION:
            yield _sse_event("meta", {**meta, "cached": True})
            yield _sse_event("delta", {"text": cached_summary})
            yield _sse_event("done", {**meta, "cached": True})
            return

    if not summary_items:
        summary = _summary_empty_message(request.language, had_items_without_abstract=bool(items))
        cache_path.write_text(_encode_summary_cache(summary, topic), encoding="utf-8")
        yield _sse_event("meta", {**meta, "cached": False})
        yield _sse_event("delta", {"text": summary})
        yield _sse_event("done", {**meta, "cached": False})
        return

    prompt = _build_daily_summary_prompt(summary_items, language=request.language, day_label=target_date, topic=topic)
    answer_client = OpenAIAnswerClient()
    collected: list[str] = []
    yield _sse_event("meta", {**meta, "cached": False})
    try:
        for chunk in answer_client.stream_answer(prompt=prompt, model=request.model):
            if not chunk:
                continue
            collected.append(chunk)
            yield _sse_event("delta", {"text": chunk})
    except Exception as exc:
        yield _sse_event("error", {"detail": str(exc)})
        return

    summary = "".join(collected).strip()
    cache_path.write_text(_encode_summary_cache(summary, topic), encoding="utf-8")
    yield _sse_event("done", {**meta, "cached": False})


def _generate_today_summary(config: AppConfig, request: TodaySummaryRequest) -> dict[str, object]:
    target_date = _resolve_target_date(request.target_date)
    items, selection_mode = _select_summary_papers(config, target_date, request.limit)
    items = enrich_missing_abstracts(config, items)
    summary_items = _summary_eligible_items(items)
    topic = _get_research_topic(config)
    cache_path = _summary_cache_path(config, request.language, target_date)

    if cache_path.exists() and not request.force_refresh:
        cached_topic, cached_summary, cached_prompt_version = _read_summary_cache(cache_path)
        if cached_topic == topic and cached_prompt_version == SUMMARY_PROMPT_VERSION:
            return {
                "date": target_date,
                "language": request.language,
                "paper_count": len(summary_items),
                "summary": cached_summary,
                "papers": [
                    {
                        "entry_id": item.get("entry_id"),
                        "title": item.get("title"),
                        "source": item.get("source"),
                        "published": item.get("published"),
                        "doi": item.get("doi"),
                    }
                    for item in summary_items
                ],
                "cached": True,
                "path": str(cache_path),
                "selection_mode": selection_mode,
                "topic": topic,
            }

    if not summary_items:
        summary = _summary_empty_message(request.language, had_items_without_abstract=bool(items))
        cache_path.write_text(_encode_summary_cache(summary, topic), encoding="utf-8")
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

    prompt = _build_daily_summary_prompt(summary_items, language=request.language, day_label=target_date, topic=topic)
    summary = OpenAIAnswerClient().create_answer(prompt=prompt, model=request.model).text
    cache_path.write_text(_encode_summary_cache(summary, topic), encoding="utf-8")
    return {
        "date": target_date,
        "language": request.language,
        "paper_count": len(summary_items),
        "summary": summary,
        "papers": [
            {
                "entry_id": item.get("entry_id"),
                "title": item.get("title"),
                "source": item.get("source"),
                "published": item.get("published"),
                "doi": item.get("doi"),
            }
            for item in summary_items
        ],
        "cached": False,
        "path": str(cache_path),
        "selection_mode": selection_mode,
        "topic": topic,
    }


def _ensure_today_ready(config: AppConfig, request: TodaySummaryRequest) -> dict[str, object]:
    target_date = _resolve_target_date(request.target_date)
    cache_path = _summary_cache_path(config, request.language, target_date)
    papers, _ = _select_summary_papers(config, target_date, request.limit)
    actions: list[str] = []
    imported_count = 0

    if not papers:
        if target_date == datetime.now(TAIPEI_TZ).date().isoformat():
            new_entries = ingest_feeds(config)
        else:
            new_entries = ingest_huggingface_daily_papers_for_date(config, target_date)
        imported_count = len(new_entries)
        actions.append("ingest")
        papers, _ = _select_summary_papers(config, target_date, request.limit)

    needs_summary = request.force_refresh or not cache_path.exists() or bool(actions)
    if needs_summary:
        summary_result = _generate_today_summary(
            config,
            TodaySummaryRequest(
                language=request.language,
                model=request.model,
                limit=request.limit,
                target_date=target_date,
                force_refresh=True,
            ),
        )
        actions.append("summarize")
    else:
        summary_result = _generate_today_summary(config, request)
        actions.append("load-cache")

    summary_result["actions"] = actions
    summary_result["imported_count"] = imported_count
    return summary_result

def create_app() -> FastAPI:
    app = FastAPI(title="Research Agent Web")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _render_app_html()

    @app.get("/api/stats")
    async def api_stats() -> dict[str, object]:
        config = make_config()
        def _load_stats() -> dict[str, object]:
            with get_connection(config) as connection:
                initialize_database(connection)
                return fetch_stats(connection)
        return await run_in_threadpool(_load_stats)

    @app.get("/api/papers")
    async def api_papers(limit: int = 20, text: str | None = None, published: str | None = None) -> list[dict[str, object]]:
        config = make_config()
        def _load_papers() -> list[dict[str, object]]:
            with get_connection(config) as connection:
                initialize_database(connection)
                return list_papers(connection, limit=limit, text=text, published=published)
        return await run_in_threadpool(_load_papers)

    @app.get("/api/paper-dates")
    async def api_paper_dates(limit: int = 90) -> list[str]:
        config = make_config()
        def _load_dates() -> list[str]:
            with get_connection(config) as connection:
                initialize_database(connection)
                return list_paper_dates(connection, limit=limit)
        return await run_in_threadpool(_load_dates)

    @app.get("/api/research-topic")
    async def api_research_topic() -> dict[str, str]:
        config = make_config()
        topic = await run_in_threadpool(_get_research_topic, config)
        return {"topic": topic}

    @app.post("/api/research-topic")
    async def api_save_research_topic(request: ResearchTopicRequest) -> dict[str, str]:
        config = make_config()
        topic = await run_in_threadpool(_set_research_topic, config, request.topic)
        return {"topic": topic}

    @app.post("/api/actions/ingest")
    async def api_ingest() -> dict[str, object]:
        config = make_config()
        entries = await run_in_threadpool(ingest_feeds, config)
        return {"imported": len(entries), "items": [item.to_dict() for item in entries[:20]]}

    @app.post("/api/actions/digest")
    async def api_digest(request: DigestRequest) -> dict[str, object]:
        config = make_config()
        path = await run_in_threadpool(generate_digest, config, request.days)
        return {"days": request.days, "path": str(path)}

    @app.post("/api/actions/ensure-daily-ready")
    async def api_ensure_daily_ready(request: TodaySummaryRequest) -> dict[str, object]:
        config = make_config()
        try:
            return await run_in_threadpool(_ensure_today_ready, config, request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/today-summary")
    async def api_today_summary(request: TodaySummaryRequest) -> dict[str, object]:
        config = make_config()
        try:
            return await run_in_threadpool(_generate_today_summary, config, request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/today-summary/stream")
    async def api_today_summary_stream(request: TodaySummaryRequest) -> StreamingResponse:
        config = make_config()
        return StreamingResponse(_stream_today_summary(config, request), media_type="text/event-stream")

    @app.post("/api/actions/today-summary/rewrite-selection/preview")
    async def api_preview_summary_selection_rewrite(request: SummaryRewriteRequest) -> dict[str, object]:
        config = make_config()
        try:
            return await run_in_threadpool(_preview_summary_selection_rewrite, config, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/today-summary/rewrite-selection/apply")
    async def api_apply_summary_selection_rewrite(request: SummaryRewriteApplyRequest) -> dict[str, object]:
        config = make_config()
        try:
            return await run_in_threadpool(_apply_summary_selection_rewrite, config, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/today-summary/save")
    async def api_save_today_summary(request: SummarySaveRequest) -> dict[str, object]:
        config = make_config()
        try:
            return await run_in_threadpool(_save_summary_text, config, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/analyze-paper")
    async def api_analyze_paper(request: PaperAnalysisRequest) -> dict[str, object]:
        config = make_config()
        try:
            result = await run_in_threadpool(analyze_paper_methodology, config, request.entry_id, model=request.model, force=request.force, pdf_url_override=request.pdf_url_override)
            return _attach_analysis_note_metadata(config, result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/analyze-paper/save")
    async def api_save_paper_analysis(request: PaperAnalysisSaveRequest) -> dict[str, object]:
        config = make_config()
        def _save_analysis() -> dict[str, object]:
            with get_connection(config) as connection:
                initialize_database(connection)
                updated = update_methodology_run_text(connection, request.run_id, request.analysis_text, datetime.now(UTC).isoformat())
                if updated is None:
                    raise ValueError(f"Methodology run not found: {request.run_id}")
                return updated
        try:
            result = await run_in_threadpool(_save_analysis)
            return _attach_analysis_note_metadata(config, result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/analyze-paper/export-note")
    async def api_export_paper_analysis_note(request: PaperAnalysisExportRequest) -> dict[str, object]:
        config = make_config()
        try:
            result = await run_in_threadpool(export_methodology_note, config, request.run_id)
            return _attach_analysis_note_metadata(config, result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/toggle-star")
    async def api_toggle_star(request: ToggleStarRequest) -> dict[str, object]:
        config = make_config()
        try:
            def _toggle_star() -> dict[str, object]:
                with get_connection(config) as connection:
                    initialize_database(connection)
                    return toggle_paper_starred(connection, request.entry_id)
            return await run_in_threadpool(_toggle_star)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/catalog-pdfs")
    async def api_catalog_pdfs(request: CatalogRequest) -> dict[str, object]:
        config = make_config()
        result = await run_in_threadpool(catalog_pdfs, config, pdf_dir=request.pdf_dir)
        if request.embed:
            entry_ids = {item["paper_entry_id"] for item in result["items"] if item.get("status") in {"imported", "linked_existing"} and item.get("paper_entry_id")}
            papers = []
            if entry_ids:
                def _fetch_papers():
                    with get_connection(config) as connection:
                        initialize_database(connection)
                        return [p for eid in entry_ids if (p := fetch_paper(connection, eid)) is not None]
                papers = await run_in_threadpool(_fetch_papers)

            result["embedding"] = await run_in_threadpool(
                _embed_paper_records, config, papers, model=request.embedding_model, dimensions=request.dimensions, init_vec=request.init_vec
            )
        return result

    @app.post("/api/actions/import-dois")
    async def api_import_dois(request: ImportDoisRequest) -> dict[str, object]:
        config = make_config()
        result = await run_in_threadpool(import_dois, config, doi_path=request.doi_file)
        if request.embed and result.get("imported_items"):
            imported_entry_ids = {item["entry_id"] for item in result["imported_items"]}

            def _fetch_papers():
                with get_connection(config) as connection:
                    initialize_database(connection)
                    return [p for eid in imported_entry_ids if (p := fetch_paper(connection, eid)) is not None]

            papers = await run_in_threadpool(_fetch_papers)
            result["embedding"] = await run_in_threadpool(
                _embed_paper_records, config, papers, model=request.embedding_model, dimensions=request.dimensions, init_vec=request.init_vec
            )
        return result

    @app.post("/api/actions/import-url")
    async def api_import_url(request: ImportUrlRequest) -> dict[str, object]:
        config = make_config()
        try:
            result = await run_in_threadpool(import_url, config, request.url)
            if request.embed and result.get("imported_items"):
                imported_entry_ids = {item["entry_id"] for item in result["imported_items"]}

                def _fetch_papers():
                    with get_connection(config) as connection:
                        initialize_database(connection)
                        return [p for eid in imported_entry_ids if (p := fetch_paper(connection, eid)) is not None]

                papers = await run_in_threadpool(_fetch_papers)
                result["embedding"] = await run_in_threadpool(
                    _embed_paper_records, config, papers, model=request.embedding_model, dimensions=request.dimensions, init_vec=request.init_vec
                )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/semantic-search")
    async def api_semantic_search(request: SearchRequest) -> dict[str, object]:
        config = make_config()

        def _semantic_search_sync() -> dict[str, object]:
            embedding_client = OpenAIEmbeddingClient()
            response = embedding_client.create_embeddings([request.query], model=request.model, dimensions=request.dimensions)
            with get_connection(config) as connection:
                initialize_database(connection)
                results = search_similar_embeddings(connection, query_vector=response.embeddings[0], limit=request.limit, model=request.model)
            return {"query": request.query, "model": request.model, "dimension": len(response.embeddings[0]), "results": results}

        try:
            return await run_in_threadpool(_semantic_search_sync)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/actions/ask")
    async def api_ask(request: AskRequest) -> dict[str, object]:
        config = make_config()

        def _ask_sync() -> dict[str, object]:
            embedding_client = OpenAIEmbeddingClient()
            query_embedding = embedding_client.create_embeddings([request.query], model=request.embedding_model, dimensions=request.dimensions).embeddings[0]
            with get_connection(config) as connection:
                initialize_database(connection)
                results = search_similar_embeddings(connection, query_vector=query_embedding, limit=request.top_k, model=request.embedding_model)
            if not results:
                raise HTTPException(status_code=400, detail="No matching embeddings found.")
            answer = OpenAIAnswerClient().create_answer(prompt=_build_answer_prompt(request.query, results), model=request.answer_model, temperature=request.temperature)
            return {
                "query": request.query,
                "embedding_model": answer.model, # Changed from request.embedding_model to answer.model to reflect actual model used for answer
                "answer_model": answer.model,
                "top_k": request.top_k,
                "answer": answer.text,
                "sources": [
                    {
                        "source_id": index,
                        "paper_entry_id": item.get("paper_entry_id"),
                        "title": item.get("title"),
                        "doi": item.get("doi"),
                        "link": item.get("link"),
                        "note_path": item.get("note_path"),
                        "distance": item.get("distance"),
                    }
                    for index, item in enumerate(results, start=1)
                ],
            }

        try:
            return JSONResponse(await run_in_threadpool(_ask_sync))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/embedding-candidates")
    async def embedding_candidates(limit: int = 20, model: str = DEFAULT_EMBEDDING_MODEL) -> JSONResponse:
        config = make_config()

        def _load_candidates() -> list[dict[str, object]]:
            with get_connection(config) as connection:
                initialize_database(connection)
                return list_embedding_candidates(connection, model=model, limit=limit, include_existing=False)

        return JSONResponse(await run_in_threadpool(_load_candidates))

    return app


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Install FastAPI and uvicorn first: pip install -r requirements.txt") from exc
    uvicorn.run("research_agent.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
















