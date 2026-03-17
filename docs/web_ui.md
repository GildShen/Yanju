# Web UI

The project includes a local web dashboard so common workflows can be run in a browser instead of the terminal.

## Start

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m research_agent.web
```

Open:

```text
http://127.0.0.1:8000
```

## What It Uses

The web UI reuses the same backend modules as the CLI:

- SQLite database in `data/research_agent.db`
- note generation in `vault/literature`
- weekly digest generation in `vault/weekly-notes`
- OpenAI embeddings and `ask`
- Crossref DOI import and PDF cataloging
- RSS plus Hugging Face Daily Papers ingestion

This means the browser UI and CLI operate on the same data.

## Current Pages And APIs

Current page:

- `/` main dashboard

Current API endpoints:

- `GET /api/stats`
- `GET /api/papers`
- `POST /api/actions/ingest`
- `POST /api/actions/digest`
- `POST /api/actions/catalog-pdfs`
- `POST /api/actions/import-dois`
- `POST /api/actions/semantic-search`
- `POST /api/actions/ask`
- `GET /api/embedding-candidates`

## Current Browser Actions

The dashboard currently supports:

- run ingest
- generate weekly digest
- catalog PDFs from `papers/tmp`
- optionally embed newly cataloged papers
- import DOIs from `dois.txt`
- optionally embed newly imported papers
- semantic search over stored embeddings
- ask questions with retrieved paper context
- view recent papers and project stats

## Operational Notes

- The server binds to `127.0.0.1:8000` by default.
- This MVP is local-only and does not include authentication.
- If `sqlite-vec` is not installed, semantic retrieval falls back to Python cosine similarity.
- `ask` and semantic search still require valid embeddings in the database.
- OpenAI settings are loaded from `.env` in the project root.

## Known Gaps

Current missing pieces in the web UI:

- paper detail page
- manual review page for `needs_review` PDF matches
- edit paper metadata from the browser
- upload DOI files or PDFs directly in the browser
- background job progress tracking

The CLI remains useful for advanced or bulk workflows until those pages are added.
