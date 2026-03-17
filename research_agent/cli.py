from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .abstract_enrichment import enrich_abstract_targets
from .config import AppConfig
from .dedupe import dedupe_papers
from .db import (
    ensure_vector_table,
    execute_readonly_query,
    fetch_paper,
    fetch_stats,
    get_connection,
    initialize_database,
    list_embedding_candidates,
    list_papers,
    migrate_legacy_json,
    search_similar_embeddings,
    update_paper_metadata,
    upsert_embedding,
    vector_status,
)
from .openai_client import DEFAULT_ANSWER_MODEL, DEFAULT_EMBEDDING_MODEL, OpenAIAnswerClient, OpenAIEmbeddingClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research agent MVP")
    parser.add_argument("--feeds", default="feeds.txt", help="Path to feeds file")
    parser.add_argument("--data-dir", default="data", help="Directory for state files")
    parser.add_argument("--vault", default="vault", help="Directory for generated notes")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Fetch RSS feeds and create notes")

    catalog_pdfs = subparsers.add_parser("catalog-pdfs", help="Catalog local PDFs by title and Crossref match")
    catalog_pdfs.add_argument("--pdf-dir", default="papers/tmp", help="Directory containing PDFs to catalog")
    catalog_pdfs.add_argument("--embed", action="store_true", help="Generate embeddings for cataloged papers that do not yet have embeddings")
    catalog_pdfs.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    catalog_pdfs.add_argument("--dimensions", type=int)
    catalog_pdfs.add_argument("--init-vec", action="store_true", help="Initialize sqlite-vec table when available")

    import_dois = subparsers.add_parser("import-dois", help="Import retrospective papers from a DOI list via Crossref")
    import_dois.add_argument("--doi-file", default="dois.txt", help="Path to DOI list file")
    import_dois.add_argument("--embed", action="store_true", help="Generate embeddings for newly imported papers")
    import_dois.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    import_dois.add_argument("--dimensions", type=int)
    import_dois.add_argument("--init-vec", action="store_true", help="Initialize sqlite-vec table when available")

    import_url = subparsers.add_parser("import-url", help="Import one paper from a DOI or arXiv URL")
    import_url.add_argument("url", help="DOI URL, arXiv URL, DOI, or arXiv identifier")
    import_url.add_argument("--embed", action="store_true", help="Generate embeddings for a newly imported paper")
    import_url.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    import_url.add_argument("--dimensions", type=int)
    import_url.add_argument("--init-vec", action="store_true", help="Initialize sqlite-vec table when available")

    enrich_abstracts = subparsers.add_parser("enrich-abstracts", help="Fill missing abstracts from local PDFs, remote PDFs, or paper pages")
    enrich_abstracts.add_argument("--entry-id", help="Enrich one paper by entry_id")
    enrich_abstracts.add_argument("--limit", type=int, default=20, help="How many missing-abstract papers to try in batch mode")
    enrich_abstracts.add_argument("--days", type=int, help="Only consider papers added in the last N days in batch mode")

    digest = subparsers.add_parser("digest", help="Generate weekly digest")
    digest.add_argument("--days", type=int, default=7, help="Digest window in days")

    run = subparsers.add_parser("run", help="Run ingest and digest")
    run.add_argument("--days", type=int, default=7, help="Digest window in days")

    subparsers.add_parser("migrate-json", help="Import legacy seen.json and library.json into SQLite")

    dedupe = subparsers.add_parser("dedupe-papers", help="Deduplicate papers while preserving the oldest record")
    dedupe.add_argument("--dry-run", action="store_true", help="Preview duplicate groups without modifying the database")

    update = subparsers.add_parser("update-paper", help="Update doi, pdf_url, ai_summary, or tags for a paper")
    update.add_argument("entry_id", help="Paper entry_id")
    update.add_argument("--doi", help="DOI value")
    update.add_argument("--pdf-url", help="PDF URL")
    update.add_argument("--ai-summary", help="AI summary text")
    update.add_argument("--tags", help="JSON array of tags, e.g. [\"rss\",\"theory\"]")

    query = subparsers.add_parser("query", help="Query SQLite content")
    query_subparsers = query.add_subparsers(dest="query_command", required=True)

    query_papers = query_subparsers.add_parser("papers", help="List papers")
    query_papers.add_argument("--limit", type=int, default=20)
    query_papers.add_argument("--source")
    query_papers.add_argument("--days", type=int)
    query_papers.add_argument("--text")

    query_paper = query_subparsers.add_parser("paper", help="Show one paper")
    query_paper.add_argument("entry_id")

    query_subparsers.add_parser("stats", help="Show database stats")

    query_sql = query_subparsers.add_parser("sql", help="Run a read-only SQL query")
    query_sql.add_argument("statement")

    vector = subparsers.add_parser("vector", help="Vector index operations")
    vector_subparsers = vector.add_subparsers(dest="vector_command", required=True)

    vector_init = vector_subparsers.add_parser("init", help="Initialize sqlite-vec table")
    vector_init.add_argument("--dimension", type=int, required=True)

    vector_subparsers.add_parser("status", help="Show vector index status")

    vector_upsert = vector_subparsers.add_parser("upsert", help="Upsert one embedding row")
    vector_upsert.add_argument("entry_id")
    vector_upsert.add_argument("--model", required=True)
    vector_upsert.add_argument("--embedding-json", required=True)
    vector_upsert.add_argument("--chunk-text", required=True)
    vector_upsert.add_argument("--chunk-index", type=int, default=0)

    vector_search = vector_subparsers.add_parser("search", help="Search by query embedding")
    vector_search.add_argument("--query-vector", required=True)
    vector_search.add_argument("--limit", type=int, default=5)
    vector_search.add_argument("--model")

    embed = subparsers.add_parser("embed", help="Generate embeddings with OpenAI")
    embed_subparsers = embed.add_subparsers(dest="embed_command", required=True)

    embed_papers = embed_subparsers.add_parser("papers", help="Embed papers into SQLite")
    embed_papers.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    embed_papers.add_argument("--limit", type=int, default=20)
    embed_papers.add_argument("--dimensions", type=int)
    embed_papers.add_argument("--all", action="store_true", help="Re-embed papers even if embeddings exist")
    embed_papers.add_argument("--init-vec", action="store_true", help="Initialize sqlite-vec table when available")

    semantic = subparsers.add_parser("semantic-search", help="Query papers by natural language")
    semantic.add_argument("query")
    semantic.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    semantic.add_argument("--dimensions", type=int)
    semantic.add_argument("--limit", type=int, default=5)

    ask = subparsers.add_parser("ask", help="Answer a question using retrieved papers")
    ask.add_argument("query")
    ask.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    ask.add_argument("--dimensions", type=int)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--temperature", type=float, default=None)

    return parser


def make_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(feeds_path=Path(args.feeds), data_dir=Path(args.data_dir), vault_dir=Path(args.vault))


def _print_json(data: Any) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        print(payload)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(payload.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _parse_tags(raw_tags: str | None) -> list[str] | None:
    if raw_tags is None:
        return None
    parsed = json.loads(raw_tags)
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("--tags must be a JSON array of strings")
    return parsed


def _parse_vector(raw_vector: str) -> list[float]:
    parsed = json.loads(raw_vector)
    if not isinstance(parsed, list):
        raise ValueError("Vector must be a JSON array")
    return [float(value) for value in parsed]


def _build_embedding_text(paper: dict[str, Any]) -> str:
    authors = ", ".join(paper.get("authors", [])) or "Unknown authors"
    parts = [f"Title: {paper.get('title', '')}", f"Authors: {authors}", f"Source: {paper.get('source', '')}", f"Published: {paper.get('published', '')}"]
    if paper.get("doi"):
        parts.append(f"DOI: {paper['doi']}")
    if paper.get("abstract"):
        parts.append(f"Abstract: {paper['abstract']}")
    if paper.get("ai_summary"):
        parts.append(f"AI Summary: {paper['ai_summary']}")
    return "\n".join(parts)


def _format_context(results: list[dict[str, Any]]) -> str:
    chunks = []
    for index, item in enumerate(results, start=1):
        chunks.append("\n".join([f"[Source {index}]", f"Paper Entry ID: {item.get('paper_entry_id', '')}", f"Title: {item.get('title', '')}", f"Source: {item.get('source', '')}", f"DOI: {item.get('doi', '')}", f"Link: {item.get('link', '')}", f"Note Path: {item.get('note_path', '')}", f"Retrieved Distance: {item.get('distance', '')}", f"Chunk Text:\n{item.get('chunk_text', '')}"]))
    return "\n\n".join(chunks)


def _build_answer_prompt(question: str, results: list[dict[str, Any]]) -> str:
    return f"""You are helping with academic research triage.
Use only the provided context.
If the context is insufficient, say so clearly.
Cite source ids like [Source 1], [Source 2].

Question:
{question}

Retrieved context:
{_format_context(results)}

Return:
1. Direct answer
2. Key supporting points
3. Gaps or uncertainty
4. Relevant sources cited inline
"""


def _embed_paper_records(config: AppConfig, papers: list[dict[str, Any]], *, model: str, dimensions: int | None, init_vec: bool) -> dict[str, Any]:
    if not papers:
        return {"model": model, "requested": 0, "embedded": 0, "results": []}
    client = OpenAIEmbeddingClient()
    texts = [_build_embedding_text(paper) for paper in papers]
    response = client.create_embeddings(texts, model=model, dimensions=dimensions)
    with get_connection(config) as connection:
        initialize_database(connection)
        if init_vec and response.embeddings:
            ensure_vector_table(connection, len(response.embeddings[0]))
        results = []
        for paper, embedding, chunk_text in zip(papers, response.embeddings, texts):
            embedding_id = upsert_embedding(connection, paper_entry_id=paper["entry_id"], model=response.model, chunk_text=chunk_text, embedding=embedding, chunk_index=0)
            results.append({"entry_id": paper["entry_id"], "title": paper["title"], "embedding_id": embedding_id, "dimension": len(embedding)})
    return {"model": response.model, "requested": len(papers), "embedded": len(results), "results": results}


def _run_embed_papers(config: AppConfig, args: argparse.Namespace) -> None:
    with get_connection(config) as connection:
        initialize_database(connection)
        candidates = list_embedding_candidates(connection, model=args.model, limit=args.limit, include_existing=args.all)
    if not candidates:
        print("No papers to embed.")
        return
    _print_json(_embed_paper_records(config, candidates, model=args.model, dimensions=args.dimensions, init_vec=args.init_vec))


def _run_semantic_search(config: AppConfig, args: argparse.Namespace) -> None:
    client = OpenAIEmbeddingClient()
    response = client.create_embeddings([args.query], model=args.model, dimensions=args.dimensions)
    with get_connection(config) as connection:
        initialize_database(connection)
        results = search_similar_embeddings(connection, query_vector=response.embeddings[0], limit=args.limit, model=args.model)
    _print_json({"query": args.query, "model": args.model, "dimension": len(response.embeddings[0]), "results": results})


def _run_ask(config: AppConfig, args: argparse.Namespace) -> None:
    embedding_client = OpenAIEmbeddingClient()
    query_embedding = embedding_client.create_embeddings([args.query], model=args.embedding_model, dimensions=args.dimensions).embeddings[0]
    with get_connection(config) as connection:
        initialize_database(connection)
        results = search_similar_embeddings(connection, query_vector=query_embedding, limit=args.top_k, model=args.embedding_model)
    if not results:
        raise RuntimeError("No matching embeddings found. Run `embed papers` first.")
    answer = OpenAIAnswerClient().create_answer(prompt=_build_answer_prompt(args.query, results), model=args.answer_model, temperature=args.temperature)
    _print_json({"query": args.query, "embedding_model": args.embedding_model, "answer_model": answer.model, "top_k": args.top_k, "answer": answer.text, "sources": [{"source_id": index, "paper_entry_id": item.get("paper_entry_id"), "title": item.get("title"), "doi": item.get("doi"), "link": item.get("link"), "note_path": item.get("note_path"), "distance": item.get("distance")} for index, item in enumerate(results, start=1)]})


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = make_config(args)

    if args.command in {"ingest", "digest", "run", "import-dois", "import-url"}:
        from .pipeline import generate_digest, import_dois, import_url, ingest_feeds
        if args.command == "ingest":
            print(f"Ingested {len(ingest_feeds(config))} new entries.")
            return
        if args.command == "import-dois":
            result = import_dois(config, doi_path=args.doi_file)
            if args.embed and result["imported_items"]:
                imported_entry_ids = {item["entry_id"] for item in result["imported_items"]}
                with get_connection(config) as connection:
                    initialize_database(connection)
                    papers = [paper for entry_id in imported_entry_ids if (paper := fetch_paper(connection, entry_id)) is not None]
                result["embedding"] = _embed_paper_records(config, papers, model=args.embedding_model, dimensions=args.dimensions, init_vec=args.init_vec)
            _print_json(result)
            return
        if args.command == "import-url":
            result = import_url(config, args.url)
            if args.embed and result["imported_items"]:
                imported_entry_ids = {item["entry_id"] for item in result["imported_items"]}
                with get_connection(config) as connection:
                    initialize_database(connection)
                    papers = [paper for entry_id in imported_entry_ids if (paper := fetch_paper(connection, entry_id)) is not None]
                result["embedding"] = _embed_paper_records(config, papers, model=args.embedding_model, dimensions=args.dimensions, init_vec=args.init_vec)
            _print_json(result)
            return
        if args.command == "digest":
            print(f"Wrote digest to {generate_digest(config, days=args.days)}")
            return
        if args.command == "run":
            entries = ingest_feeds(config)
            path = generate_digest(config, days=args.days)
            print(f"Ingested {len(entries)} new entries.")
            print(f"Wrote digest to {path}")
            return

    if args.command == "catalog-pdfs":
        from .catalog import catalog_pdfs
        result = catalog_pdfs(config, pdf_dir=args.pdf_dir)
        if args.embed:
            entry_ids = {item["paper_entry_id"] for item in result["items"] if item.get("status") in {"imported", "linked_existing"} and item.get("paper_entry_id")}
            with get_connection(config) as connection:
                initialize_database(connection)
                papers = [paper for entry_id in entry_ids if (paper := fetch_paper(connection, entry_id)) is not None]
            result["embedding"] = _embed_paper_records(config, papers, model=args.embedding_model, dimensions=args.dimensions, init_vec=args.init_vec)
        _print_json(result)
        return

    if args.command == "migrate-json":
        _print_json(migrate_legacy_json(config))
        return

    if args.command == "enrich-abstracts":
        _print_json(enrich_abstract_targets(config, entry_id=args.entry_id, limit=args.limit, days=args.days))
        return

    if args.command == "dedupe-papers":
        _print_json(dedupe_papers(config, dry_run=args.dry_run, keep="oldest"))
        return

    if args.command == "update-paper":
        with get_connection(config) as connection:
            initialize_database(connection)
            update_paper_metadata(connection, entry_id=args.entry_id, doi=args.doi, pdf_url=args.pdf_url, ai_summary=args.ai_summary, tags=_parse_tags(args.tags))
        print(f"Updated paper {args.entry_id}")
        return

    if args.command == "query":
        with get_connection(config) as connection:
            initialize_database(connection)
            if args.query_command == "papers":
                _print_json(list_papers(connection, limit=args.limit, source=args.source, days=args.days, text=args.text))
                return
            if args.query_command == "paper":
                _print_json(fetch_paper(connection, args.entry_id))
                return
            if args.query_command == "stats":
                _print_json(fetch_stats(connection))
                return
            if args.query_command == "sql":
                columns, rows = execute_readonly_query(connection, args.statement)
                _print_json({"columns": columns, "rows": rows})
                return

    if args.command == "vector":
        with get_connection(config) as connection:
            initialize_database(connection)
            if args.vector_command == "init":
                ok = ensure_vector_table(connection, args.dimension)
                if not ok:
                    raise RuntimeError("sqlite-vec is not available. Install sqlite-vec first.")
                print(f"Initialized vector table with dimension {args.dimension}")
                return
            if args.vector_command == "status":
                _print_json(vector_status(connection))
                return
            if args.vector_command == "upsert":
                print(f"Upserted embedding {upsert_embedding(connection, paper_entry_id=args.entry_id, model=args.model, chunk_text=args.chunk_text, embedding=_parse_vector(args.embedding_json), chunk_index=args.chunk_index)}")
                return
            if args.vector_command == "search":
                _print_json(search_similar_embeddings(connection, _parse_vector(args.query_vector), limit=args.limit, model=args.model))
                return

    if args.command == "embed" and args.embed_command == "papers":
        _run_embed_papers(config, args)
        return

    if args.command == "semantic-search":
        _run_semantic_search(config, args)
        return

    if args.command == "ask":
        _run_ask(config, args)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
