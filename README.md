# Research Agent MVP

This project implements a local-first research agent for:

- RSS ingestion
- Hugging Face Daily Papers ingestion via the official API
- retrospective DOI import via Crossref
- SQLite-based paper storage
- Obsidian literature note generation
- OpenAI embeddings for retrieval
- retrieval-augmented question answering
- weekly digest generation
- local web UI for day-to-day operation

The current implementation is designed around a local SQLite database plus an Obsidian-style vault.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional native vector index:

```bash
pip install sqlite-vec
```

## Collaboration

Recommended default workflow for ongoing changes:

1. create a branch from `main`
2. make one focused change
3. open a pull request into `main`
4. merge after CI passes

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch naming convention and a sample PR flow.

## .env

Place `.env` in `D:\dev\ResearchAgent`:

```dotenv
OPENAI_API_KEY=your_key_here
CROSSREF_MAILTO=you@example.com
# Optional
OPENAI_BASE_URL=
OPENAI_ORG_ID=
OPENAI_PROJECT_ID=
```

The app auto-loads `.env` from the project root. `CROSSREF_MAILTO` is recommended for polite Crossref API usage.

## Recommended Start

If you want a browser-based workflow, use the web UI:

```bash
python -m research_agent.web
```

Then open:

```text
http://127.0.0.1:8000
```

If you prefer terminal workflows, the CLI remains available:

```bash
python -m research_agent.cli --help
```

## Web UI

The local web dashboard is the simplest way to operate the project without memorizing commands.

Current web UI features:

- dashboard stats
- run ingest
- generate weekly digest
- catalog PDFs from `papers/tmp`
- import DOIs from `dois.txt`
- semantic search
- ask with retrieved papers
- browse recent papers

Current web UI routes:

- `/` main dashboard
- `/api/stats` project stats
- `/api/papers` paper list API
- `/api/actions/ingest` run RSS + Hugging Face ingest
- `/api/actions/digest` generate weekly digest
- `/api/actions/catalog-pdfs` catalog local PDFs
- `/api/actions/import-dois` import DOI file
- `/api/actions/semantic-search` run semantic retrieval
- `/api/actions/ask` run RAG question answering

Notes:

- the web UI reuses the same SQLite database, note generation, embedding pipeline, and OpenAI integration as the CLI
- the default web UI server binds to `127.0.0.1:8000`
- the web UI is local-only by design in this MVP
- blocking SQLite, file, OpenAI, RSS, and Crossref work is offloaded from FastAPI async routes via threadpool execution, so the event loop stays responsive during longer operations
- this is an async web-layer improvement, not a full `aiosqlite` rewrite; core storage and pipeline modules remain synchronous internally
- if `sqlite-vec` is unavailable, semantic retrieval falls back to Python cosine similarity over stored embeddings

## OpenAI Defaults

Current defaults:

- Answer model: `gpt-5-mini`
- Embedding model: `text-embedding-3-small`
- Answer API path: `Responses API` first, `chat.completions` fallback only

Important note for `gpt-5-mini`:

- do not pass `temperature` unless you explicitly know the model supports it
- the CLI and web UI leave `temperature` unset by default for `ask`

This flow has been verified end-to-end in the project:

1. query embedding with OpenAI
2. retrieval from SQLite embeddings
3. answer generation with `gpt-5-mini`
4. JSON output to terminal or browser API response

## Core CLI Usage

```bash
python -m research_agent.cli ingest
python -m research_agent.cli import-dois --doi-file dois.txt
python -m research_agent.cli import-dois --doi-file dois.txt --embed --embedding-model text-embedding-3-small
python -m research_agent.cli digest
python -m research_agent.cli run
python -m research_agent.cli migrate-json
```

## Ingest Sources

`python -m research_agent.cli ingest` does two things in one run:

1. reads RSS sources from `feeds.txt`
2. fetches Hugging Face Daily Papers from the official API endpoint `https://huggingface.co/api/daily_papers`

Hugging Face ingestion behavior:

- stores papers as source `Hugging Face Daily Papers`
- tracks the last ingested date in SQLite settings
- backfills up to the last 7 days when needed
- skips duplicates by `entry_id` and DOI
- tags imported Hugging Face papers with `huggingface-daily-papers`

## DOI Retrospective Import

Create `dois.txt` with one DOI per line:

```text
10.1038/s41586-023-06291-2
10.1145/3543507.3583288
```

Import into SQLite and generate literature notes:

```bash
python -m research_agent.cli import-dois --doi-file dois.txt
```

Import and immediately embed only the newly imported papers:

```bash
python -m research_agent.cli import-dois --doi-file dois.txt --embed --embedding-model text-embedding-3-small --init-vec
```

The DOI importer will:

1. fetch metadata from Crossref
2. deduplicate by DOI and entry id
3. write the paper to SQLite
4. create a note in `vault/literature`
5. optionally generate embeddings for the newly imported papers

## PDF Cataloging

The PDF catalog flow defaults to `papers/tmp` and moves successfully processed files to `papers/done/[YYYY]`.

CLI examples:

```bash
python -m research_agent.cli catalog-pdfs
python -m research_agent.cli catalog-pdfs --embed --embedding-model text-embedding-3-small
```

Behavior:

- extracts a title from PDF metadata or the first pages via `pymupdf`
- searches Crossref for the best match
- imports or links the paper in SQLite
- records the file in `pdf_catalog`
- renames PDFs to the detected research identifier when one is found, preferring DOI and otherwise using codes such as arXiv, PII, or PMID
- keeps `needs_review` and failed files in place
- moves only successful files to `papers/done/[YYYY]`, grouped by publication year when available

## Ask with Retrieval

Default answer model is `gpt-5-mini`:

```bash
python -m research_agent.cli ask "What papers in my library discuss human-AI collaboration in knowledge management?" --embedding-model text-embedding-3-small --top-k 5
```

Override the answer model explicitly if needed:

```bash
python -m research_agent.cli ask "Summarize the main themes in my library" --answer-model gpt-5-mini --embedding-model text-embedding-3-small
```

Only set temperature if you know the selected model supports it:

```bash
python -m research_agent.cli ask "Compare the main themes in my current papers" --temperature 1
```

## Query CLI

```bash
python -m research_agent.cli query papers --limit 10 --text llm
python -m research_agent.cli query paper <entry_id>
python -m research_agent.cli query stats
python -m research_agent.cli query sql "SELECT title, doi FROM papers ORDER BY added_at DESC LIMIT 5"
```

## OpenAI Embeddings

```bash
python -m research_agent.cli embed papers --model text-embedding-3-small --limit 20
python -m research_agent.cli semantic-search "knowledge management and human AI collaboration" --model text-embedding-3-small --limit 5
```

## Vector Operations

```bash
python -m research_agent.cli vector init --dimension 1536
python -m research_agent.cli vector status
python -m research_agent.cli vector upsert <entry_id> --model text-embedding-3-small --chunk-text "paper abstract" --embedding-json "[0.1, 0.2, 0.3]"
python -m research_agent.cli vector search --query-vector "[0.1, 0.2, 0.3]" --model text-embedding-3-small --limit 5
```

If `sqlite-vec` is unavailable, semantic retrieval falls back to Python cosine similarity over stored embeddings.

## Daily Summary Cache

The web UI `Daily Summary` panel can summarize any selected date, not only today.

Behavior:

- choose a date in the web UI summary panel
- if papers for that date already exist in SQLite, the app summarizes them directly
- if the selected date has no papers yet, the app first tries a historical backfill from Hugging Face Daily Papers for that exact date, then generates the summary
- summaries are cached by date and language

The web UI cache rule is:

Storage rule:

- `data/daily-summaries/<language>/YYYY-MM-DD.md`

Examples:

- `data/daily-summaries/zh-TW/2026-03-14.md`
- `data/daily-summaries/en/2026-03-14.md`

Behavior:

- if today's summary for the selected language already exists, the web UI loads it directly
- if not, it generates a new summary and saves it
- `Load Summary` prefers the cached file
- `Regenerate` forces a new summary and overwrites today's cached file for that language


## Web Auto Bootstrap

When the web UI loads, it now performs an automatic daily bootstrap for the selected summary language.

Behavior on page load:

- checks whether today's papers exist
- checks whether today's summary cache exists for the selected language
- if today's papers are missing, it automatically runs `ingest`
- if today's summary is missing, it automatically generates and saves it
- if both already exist, it loads the cached summary directly

This bootstrap uses:

- `POST /api/actions/ensure-daily-ready`

The cache file remains stored under:

- `data/daily-summaries/<language>/YYYY-MM-DD.md`

## Paper List UI

The web UI paper list now supports:

- clickable paper titles that open the paper link in a new tab
- author display instead of raw entry id strings
- a star button that toggles paper collection state and stores it in SQLite using the `starred` tag

## Research Topic Filter

The web UI now supports a saved research topic for `Today Summary`.

Behavior:

- if a research topic is set, the app first applies a local relevance filter against today's papers using title, abstract, source, and authors
- if relevant papers are found, the summary is generated from the top 15 relevant papers
- if no topic is set, the summary uses the latest 15 papers from today
- if a topic is set but nothing matches, it falls back to the latest 15 papers from today

Topic settings are stored in SQLite `settings` using the `research_topic` key.

## Topic Semantic Filter

The saved research topic now uses embedding-based semantic filtering for `Today Summary`.

Behavior:

- when a research topic is set, the app embeds the topic text with `text-embedding-3-small`
- it ensures today's candidate papers have embeddings available
- it ranks today's papers by cosine similarity against the topic embedding
- it summarizes the top 15 semantically relevant papers
- if no topic is set, it falls back to the latest 15 papers from today
- if a topic is changed, the cached summary for the same date and language is regenerated instead of reusing a stale cache

## Papers By Date

The web UI `Papers` section now works as a date-based view instead of one long mixed list.

Behavior:

- the UI loads available publication dates from `/api/paper-dates`
- you select one date from the dropdown
- the list shows only papers for that date
- star toggling still works inside the filtered date view
