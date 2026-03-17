from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Iterator

from .config import AppConfig
from .models import PaperEntry


BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feeds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    title TEXT,
    created_at TEXT NOT NULL,
    last_checked_at TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    feed_count INTEGER NOT NULL DEFAULT 0,
    new_paper_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS papers (
    entry_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    published TEXT NOT NULL,
    abstract TEXT,
    link TEXT,
    source TEXT NOT NULL,
    feed_url TEXT NOT NULL,
    note_path TEXT NOT NULL,
    added_at TEXT NOT NULL,
    doi TEXT NOT NULL DEFAULT '',
    pdf_url TEXT NOT NULL DEFAULT '',
    ai_summary TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_entry_id TEXT NOT NULL,
    author_id INTEGER NOT NULL,
    author_order INTEGER NOT NULL,
    PRIMARY KEY (paper_entry_id, author_id),
    FOREIGN KEY (paper_entry_id) REFERENCES papers(entry_id) ON DELETE CASCADE,
    FOREIGN KEY (author_id) REFERENCES authors(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_entry_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(paper_entry_id, chunk_index, model),
    FOREIGN KEY (paper_entry_id) REFERENCES papers(entry_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS pdf_catalog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    title_extracted TEXT NOT NULL DEFAULT '',
    title_matched TEXT NOT NULL DEFAULT '',
    match_confidence REAL NOT NULL DEFAULT 0,
    catalog_status TEXT NOT NULL,
    paper_entry_id TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (paper_entry_id) REFERENCES papers(entry_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS paper_methodology_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_entry_id TEXT NOT NULL,
    status TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    analysis_text TEXT NOT NULL,
    note_path TEXT NOT NULL DEFAULT '',
    page_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (paper_entry_id) REFERENCES papers(entry_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_papers_added_at ON papers(added_at);
CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published);
CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_paper_authors_order ON paper_authors(paper_entry_id, author_order);
CREATE INDEX IF NOT EXISTS idx_embeddings_paper_model ON paper_embeddings(paper_entry_id, model);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_dimension ON paper_embeddings(model, dimension);
CREATE INDEX IF NOT EXISTS idx_pdf_catalog_status ON pdf_catalog(catalog_status);
CREATE INDEX IF NOT EXISTS idx_pdf_catalog_paper ON pdf_catalog(paper_entry_id);
CREATE INDEX IF NOT EXISTS idx_methodology_runs_paper ON paper_methodology_runs(paper_entry_id, updated_at DESC);
"""

PAPER_REQUIRED_COLUMNS: dict[str, str] = {
    "doi": "TEXT NOT NULL DEFAULT ''",
    "pdf_url": "TEXT NOT NULL DEFAULT ''",
    "ai_summary": "TEXT NOT NULL DEFAULT ''",
    "tags": "TEXT NOT NULL DEFAULT '[]'",
}

METHODOLOGY_REQUIRED_COLUMNS: dict[str, str] = {
    "note_path": "TEXT NOT NULL DEFAULT ''",
}


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


@contextmanager
def get_connection(config: AppConfig) -> Iterator[sqlite3.Connection]:
    config.ensure_directories()
    connection = sqlite3.connect(config.db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(BASE_SCHEMA)
    existing_columns = _table_columns(connection, "papers")
    for column_name, column_type in PAPER_REQUIRED_COLUMNS.items():
        if column_name not in existing_columns:
            connection.execute(f"ALTER TABLE papers ADD COLUMN {column_name} {column_type}")
    methodology_columns = _table_columns(connection, "paper_methodology_runs")
    for column_name, column_type in METHODOLOGY_REQUIRED_COLUMNS.items():
        if column_name not in methodology_columns:
            connection.execute(f"ALTER TABLE paper_methodology_runs ADD COLUMN {column_name} {column_type}")


def create_ingestion_run(connection: sqlite3.Connection, started_at: str, feed_count: int) -> int:
    return int(connection.execute("INSERT INTO ingestion_runs (started_at, feed_count) VALUES (?, ?)", (started_at, feed_count)).lastrowid)


def finish_ingestion_run(connection: sqlite3.Connection, run_id: int, completed_at: str, new_paper_count: int) -> None:
    connection.execute("UPDATE ingestion_runs SET completed_at = ?, new_paper_count = ? WHERE id = ?", (completed_at, new_paper_count, run_id))


def upsert_feed(connection: sqlite3.Connection, url: str, title: str, checked_at: str) -> None:
    connection.execute(
        """
        INSERT INTO feeds (url, title, created_at, last_checked_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET title = excluded.title, last_checked_at = excluded.last_checked_at
        """,
        (url, title, checked_at, checked_at),
    )


def paper_exists(connection: sqlite3.Connection, entry_id: str) -> bool:
    return connection.execute("SELECT 1 FROM papers WHERE entry_id = ? LIMIT 1", (entry_id,)).fetchone() is not None


def paper_exists_by_doi(connection: sqlite3.Connection, doi: str) -> bool:
    return connection.execute("SELECT 1 FROM papers WHERE lower(doi) = lower(?) LIMIT 1", (doi,)).fetchone() is not None


def paper_exists_by_source_identity(connection: sqlite3.Connection, *, source: str, link: str, title: str, published: str) -> bool:
    normalized_link = (link or "").strip()
    normalized_title = " ".join((title or "").split()).strip()
    normalized_source = (source or "").strip()
    normalized_published = (published or "").strip()

    if normalized_link:
        row = connection.execute(
            "SELECT 1 FROM papers WHERE source = ? AND link = ? LIMIT 1",
            (normalized_source, normalized_link),
        ).fetchone()
        if row is not None:
            return True

    if normalized_title and normalized_published:
        row = connection.execute(
            "SELECT 1 FROM papers WHERE source = ? AND lower(title) = lower(?) AND published = ? LIMIT 1",
            (normalized_source, normalized_title, normalized_published),
        ).fetchone()
        if row is not None:
            return True

    return False


def _ensure_author(connection: sqlite3.Connection, author_name: str) -> int:
    connection.execute("INSERT INTO authors (name) VALUES (?) ON CONFLICT(name) DO NOTHING", (author_name,))
    return int(connection.execute("SELECT id FROM authors WHERE name = ?", (author_name,)).fetchone()["id"])


def insert_paper(connection: sqlite3.Connection, paper: PaperEntry, feed_url: str) -> None:
    connection.execute(
        """
        INSERT INTO papers (entry_id, title, published, abstract, link, source, feed_url, note_path, added_at, doi, pdf_url, ai_summary, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (paper.entry_id, paper.title, paper.published, paper.abstract, paper.link, paper.source, feed_url, paper.note_path, paper.added_at, paper.doi, paper.pdf_url, paper.ai_summary, json.dumps(paper.tags, ensure_ascii=False)),
    )
    for index, author_name in enumerate(paper.authors):
        author_id = _ensure_author(connection, author_name)
        connection.execute("INSERT INTO paper_authors (paper_entry_id, author_id, author_order) VALUES (?, ?, ?)", (paper.entry_id, author_id, index))


def update_paper_metadata(connection: sqlite3.Connection, entry_id: str, doi: str | None = None, pdf_url: str | None = None, ai_summary: str | None = None, tags: list[str] | None = None) -> None:
    current = connection.execute("SELECT doi, pdf_url, ai_summary, tags FROM papers WHERE entry_id = ?", (entry_id,)).fetchone()
    if current is None:
        raise ValueError(f"Paper not found: {entry_id}")
    connection.execute(
        "UPDATE papers SET doi = ?, pdf_url = ?, ai_summary = ?, tags = ? WHERE entry_id = ?",
        (current["doi"] if doi is None else doi, current["pdf_url"] if pdf_url is None else pdf_url, current["ai_summary"] if ai_summary is None else ai_summary, current["tags"] if tags is None else json.dumps(tags, ensure_ascii=False), entry_id),
    )


def update_paper_abstract(connection: sqlite3.Connection, entry_id: str, abstract: str) -> None:
    current = connection.execute("SELECT 1 FROM papers WHERE entry_id = ?", (entry_id,)).fetchone()
    if current is None:
        raise ValueError(f"Paper not found: {entry_id}")
    connection.execute("UPDATE papers SET abstract = ? WHERE entry_id = ?", (abstract, entry_id))


def toggle_paper_starred(connection: sqlite3.Connection, entry_id: str) -> dict[str, Any]:
    row = connection.execute("SELECT title, tags FROM papers WHERE entry_id = ?", (entry_id,)).fetchone()
    if row is None:
        raise ValueError(f"Paper not found: {entry_id}")
    tags = _decode_tags(row["tags"])
    if "starred" in tags:
        tags = [tag for tag in tags if tag != "starred"]
        starred = False
    else:
        tags.append("starred")
        tags = list(dict.fromkeys(tags))
        starred = True
    connection.execute("UPDATE papers SET tags = ? WHERE entry_id = ?", (json.dumps(tags, ensure_ascii=False), entry_id))
    return {"entry_id": entry_id, "title": row["title"], "starred": starred, "tags": tags}


def upsert_pdf_catalog_entry(connection: sqlite3.Connection, *, file_path: str, file_name: str, file_hash: str, title_extracted: str, title_matched: str, match_confidence: float, catalog_status: str, paper_entry_id: str | None, notes: str, updated_at: str) -> None:
    created_row = connection.execute("SELECT created_at FROM pdf_catalog WHERE file_path = ?", (file_path,)).fetchone()
    created_at = created_row["created_at"] if created_row else updated_at
    connection.execute(
        """
        INSERT INTO pdf_catalog (file_path, file_name, file_hash, title_extracted, title_matched, match_confidence, catalog_status, paper_entry_id, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            file_name = excluded.file_name,
            file_hash = excluded.file_hash,
            title_extracted = excluded.title_extracted,
            title_matched = excluded.title_matched,
            match_confidence = excluded.match_confidence,
            catalog_status = excluded.catalog_status,
            paper_entry_id = excluded.paper_entry_id,
            notes = excluded.notes,
            updated_at = excluded.updated_at
        """,
        (file_path, file_name, file_hash, title_extracted, title_matched, match_confidence, catalog_status, paper_entry_id, notes, created_at, updated_at),
    )


def save_methodology_run(connection: sqlite3.Connection, payload: dict[str, Any]) -> int:
    row = connection.execute(
        """
        INSERT INTO paper_methodology_runs (paper_entry_id, status, source_type, source_ref, model, prompt_version, analysis_text, note_path, page_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            payload["paper_entry_id"],
            payload["status"],
            payload["source_type"],
            payload.get("source_ref", ""),
            payload["model"],
            payload["prompt_version"],
            payload["analysis_text"],
            payload.get("note_path", ""),
            int(payload.get("page_count", 0)),
            payload["created_at"],
            payload["updated_at"],
        ),
    ).fetchone()
    return int(row["id"])


def update_methodology_run_text(connection: sqlite3.Connection, run_id: int, analysis_text: str, updated_at: str) -> dict[str, Any] | None:
    connection.execute(
        "UPDATE paper_methodology_runs SET analysis_text = ?, updated_at = ? WHERE id = ?",
        (analysis_text, updated_at, run_id),
    )
    row = connection.execute(
        "SELECT id, paper_entry_id, status, source_type, source_ref, model, prompt_version, analysis_text, note_path, page_count, created_at, updated_at FROM paper_methodology_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    return None if row is None else dict(row)


def fetch_latest_methodology_run(connection: sqlite3.Connection, paper_entry_id: str, *, prompt_version: str | None = None, model: str | None = None) -> dict[str, Any] | None:
    sql = [
        "SELECT id, paper_entry_id, status, source_type, source_ref, model, prompt_version, analysis_text, note_path, page_count, created_at, updated_at",
        "FROM paper_methodology_runs",
        "WHERE paper_entry_id = ?",
    ]
    params: list[Any] = [paper_entry_id]
    if prompt_version is not None:
        sql.append("AND prompt_version = ?")
        params.append(prompt_version)
    if model is not None:
        sql.append("AND model = ?")
        params.append(model)
    sql.append("ORDER BY updated_at DESC, id DESC LIMIT 1")
    row = connection.execute("\n".join(sql), params).fetchone()
    return None if row is None else dict(row)


def fetch_methodology_run(connection: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT id, paper_entry_id, status, source_type, source_ref, model, prompt_version, analysis_text, note_path, page_count, created_at, updated_at FROM paper_methodology_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    return None if row is None else dict(row)


def update_methodology_run_note_path(connection: sqlite3.Connection, run_id: int, note_path: str, updated_at: str) -> dict[str, Any] | None:
    connection.execute(
        "UPDATE paper_methodology_runs SET note_path = ?, updated_at = ? WHERE id = ?",
        (note_path, updated_at, run_id),
    )
    return fetch_methodology_run(connection, run_id)


def fetch_pdf_catalog_paths(connection: sqlite3.Connection, paper_entry_id: str) -> list[str]:
    rows = connection.execute(
        """
        SELECT file_path
        FROM pdf_catalog
        WHERE paper_entry_id = ?
        ORDER BY updated_at DESC, created_at DESC
        """,
        (paper_entry_id,),
    ).fetchall()
    return [str(row["file_path"]) for row in rows if row["file_path"]]


def _authors_subquery() -> str:
    return """
    (
        SELECT GROUP_CONCAT(name, '||')
        FROM (
            SELECT a.name AS name
            FROM paper_authors pa
            JOIN authors a ON a.id = pa.author_id
            WHERE pa.paper_entry_id = p.entry_id
            ORDER BY pa.author_order
        )
    ) AS author_names
    """


def fetch_recent_papers(connection: sqlite3.Connection, days: int) -> list[dict[str, Any]]:
    rows = connection.execute(f"SELECT p.entry_id, p.title, p.published, p.abstract, p.link, p.source, p.note_path, p.added_at, p.doi, p.pdf_url, p.ai_summary, p.tags, {_authors_subquery()} FROM papers p WHERE p.added_at >= datetime('now', ?) ORDER BY p.published DESC, p.added_at DESC", (f"-{days} days",)).fetchall()
    return [_row_to_paper_dict(row) for row in rows]


def list_papers(connection: sqlite3.Connection, limit: int = 20, source: str | None = None, days: int | None = None, text: str | None = None, published: str | None = None) -> list[dict[str, Any]]:
    sql = ["SELECT p.entry_id, p.title, p.published, p.source, p.link, p.doi, p.pdf_url, p.abstract, p.tags,", _authors_subquery(), "FROM papers p", "WHERE 1=1"]
    params: list[Any] = []
    if source:
        sql.append("AND p.source = ?")
        params.append(source)
    if days is not None:
        sql.append("AND p.added_at >= datetime('now', ?)")
        params.append(f"-{days} days")
    if published:
        sql.append("AND p.published = ?")
        params.append(published)
    if text:
        wildcard = f"%{text}%"
        sql.append("AND (p.title LIKE ? OR p.abstract LIKE ? OR p.ai_summary LIKE ? OR p.doi LIKE ?)")
        params.extend([wildcard, wildcard, wildcard, wildcard])
    sql.append("ORDER BY p.published DESC, p.added_at DESC")
    sql.append("LIMIT ?")
    params.append(limit)
    rows = connection.execute("\n".join(sql), params).fetchall()
    return [{"entry_id": row["entry_id"], "title": row["title"], "published": row["published"], "source": row["source"], "link": row["link"], "doi": row["doi"], "pdf_url": row["pdf_url"], "abstract": row["abstract"], "tags": _decode_tags(row["tags"]), "authors": row["author_names"].split("||") if row["author_names"] else []} for row in rows]


def list_paper_dates(connection: sqlite3.Connection, limit: int = 60) -> list[str]:
    rows = connection.execute(
        "SELECT published FROM papers WHERE published <> '' GROUP BY published ORDER BY published DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [str(row["published"]) for row in rows]


def fetch_paper(connection: sqlite3.Connection, entry_id: str) -> dict[str, Any] | None:
    row = connection.execute(f"SELECT p.entry_id, p.title, p.published, p.abstract, p.link, p.source, p.note_path, p.added_at, p.doi, p.pdf_url, p.ai_summary, p.tags, {_authors_subquery()} FROM papers p WHERE p.entry_id = ?", (entry_id,)).fetchone()
    return None if row is None else _row_to_paper_dict(row)


def fetch_paper_by_doi(connection: sqlite3.Connection, doi: str) -> dict[str, Any] | None:
    row = connection.execute(f"SELECT p.entry_id, p.title, p.published, p.abstract, p.link, p.source, p.note_path, p.added_at, p.doi, p.pdf_url, p.ai_summary, p.tags, {_authors_subquery()} FROM papers p WHERE lower(p.doi) = lower(?) LIMIT 1", (doi,)).fetchone()
    return None if row is None else _row_to_paper_dict(row)


def fetch_stats(connection: sqlite3.Connection) -> dict[str, Any]:
    latest = connection.execute("SELECT entry_id, title, added_at FROM papers ORDER BY added_at DESC LIMIT 1").fetchone()
    return {
        "papers": connection.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"],
        "feeds": connection.execute("SELECT COUNT(*) AS count FROM feeds").fetchone()["count"],
        "authors": connection.execute("SELECT COUNT(*) AS count FROM authors").fetchone()["count"],
        "ingestion_runs": connection.execute("SELECT COUNT(*) AS count FROM ingestion_runs").fetchone()["count"],
        "paper_embeddings": connection.execute("SELECT COUNT(*) AS count FROM paper_embeddings").fetchone()["count"],
        "pdf_catalog": connection.execute("SELECT COUNT(*) AS count FROM pdf_catalog").fetchone()["count"],
        "latest_paper": dict(latest) if latest else None,
    }


def execute_readonly_query(connection: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    normalized = sql.strip().lstrip("(").upper()
    if not normalized.startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN")):
        raise ValueError("Only read-only queries are allowed.")
    cursor = connection.execute(sql)
    return [item[0] for item in cursor.description] if cursor.description else [], [tuple(row) for row in cursor.fetchall()]


def migrate_legacy_json(config: AppConfig) -> dict[str, int]:
    imported = 0
    skipped = 0
    if not config.legacy_library_path.exists():
        return {"imported": 0, "skipped": 0}
    library = json.loads(config.legacy_library_path.read_text(encoding="utf-8"))
    seen_data = json.loads(config.legacy_seen_path.read_text(encoding="utf-8")) if config.legacy_seen_path.exists() else {}
    with get_connection(config) as connection:
        initialize_database(connection)
        for item in library:
            entry_id = item["entry_id"]
            if paper_exists(connection, entry_id):
                skipped += 1
                continue
            paper = PaperEntry(entry_id=entry_id, title=item.get("title", "Untitled"), authors=item.get("authors", []), published=item.get("published", ""), abstract=item.get("abstract", ""), link=item.get("link", ""), source=item.get("source", "unknown"), note_path=item.get("note_path", ""), added_at=item.get("added_at") or seen_data.get(entry_id, datetime.now(UTC).isoformat()), doi=item.get("doi", ""), pdf_url=item.get("pdf_url", ""), ai_summary=item.get("ai_summary", ""), tags=item.get("tags", []))
            feed_url = item.get("feed_url", item.get("source", "legacy-import"))
            upsert_feed(connection, feed_url, item.get("source", "legacy-import"), paper.added_at)
            insert_paper(connection, paper, feed_url)
            imported += 1
    return {"imported": imported, "skipped": skipped}


def set_setting(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def get_setting(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def try_load_sqlite_vec(connection: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        return False
    try:
        connection.enable_load_extension(True)
        if hasattr(sqlite_vec, "load"):
            sqlite_vec.load(connection)
        elif hasattr(sqlite_vec, "loadable_path"):
            connection.load_extension(sqlite_vec.loadable_path())
        else:
            return False
        connection.enable_load_extension(False)
        return True
    except Exception:
        return False


def ensure_vector_table(connection: sqlite3.Connection, dimension: int) -> bool:
    if not try_load_sqlite_vec(connection):
        return False
    existing_dimension = get_setting(connection, "vector_dimension")
    if existing_dimension and int(existing_dimension) != dimension:
        raise ValueError(f"Existing vector dimension is {existing_dimension}; requested {dimension}.")
    connection.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_paper_embeddings USING vec0(paper_embedding_id INTEGER PRIMARY KEY, paper_entry_id TEXT, chunk_index INTEGER, model TEXT, embedding FLOAT[{dimension}])")
    set_setting(connection, "vector_dimension", str(dimension))
    return True


def upsert_embedding(connection: sqlite3.Connection, paper_entry_id: str, model: str, chunk_text: str, embedding: list[float], chunk_index: int = 0) -> int:
    dimension = len(embedding)
    sqlite_vec_enabled = ensure_vector_table(connection, dimension)
    now = datetime.now(UTC).isoformat()
    embedding_id = int(connection.execute(
        """
        INSERT INTO paper_embeddings (paper_entry_id, chunk_index, model, dimension, chunk_text, embedding_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_entry_id, chunk_index, model) DO UPDATE SET dimension = excluded.dimension, chunk_text = excluded.chunk_text, embedding_json = excluded.embedding_json, updated_at = excluded.updated_at
        RETURNING id
        """,
        (paper_entry_id, chunk_index, model, dimension, chunk_text, json.dumps(embedding), now, now),
    ).fetchone()["id"])
    if sqlite_vec_enabled:
        connection.execute("DELETE FROM vec_paper_embeddings WHERE paper_embedding_id = ?", (embedding_id,))
        connection.execute("INSERT INTO vec_paper_embeddings (paper_embedding_id, paper_entry_id, chunk_index, model, embedding) VALUES (?, ?, ?, ?, ?)", (embedding_id, paper_entry_id, chunk_index, model, json.dumps(embedding)))
    return embedding_id


def list_embedding_candidates(connection: sqlite3.Connection, model: str, limit: int = 20, include_existing: bool = False) -> list[dict[str, Any]]:
    sql = ["SELECT p.entry_id, p.title, p.abstract, p.ai_summary, p.source, p.published, p.doi, p.link,", _authors_subquery(), "FROM papers p"]
    params: list[Any] = []
    if not include_existing:
        sql.append("LEFT JOIN paper_embeddings e ON e.paper_entry_id = p.entry_id AND e.chunk_index = 0 AND e.model = ?")
        params.append(model)
        sql.append("WHERE e.id IS NULL")
    else:
        sql.append("WHERE 1=1")
    sql.append("ORDER BY p.added_at DESC")
    sql.append("LIMIT ?")
    params.append(limit)
    rows = connection.execute("\n".join(sql), params).fetchall()
    return [{"entry_id": row["entry_id"], "title": row["title"], "abstract": row["abstract"], "ai_summary": row["ai_summary"], "source": row["source"], "published": row["published"], "doi": row["doi"], "link": row["link"], "authors": row["author_names"].split("||") if row["author_names"] else []} for row in rows]


def search_similar_embeddings(connection: sqlite3.Connection, query_vector: list[float], limit: int = 5, model: str | None = None) -> list[dict[str, Any]]:
    dimension = len(query_vector)
    if try_load_sqlite_vec(connection) and _vector_table_exists(connection):
        return _search_sqlite_vec(connection, query_vector, limit=limit, model=model)
    return _search_python_cosine(connection, query_vector, limit=limit, model=model, dimension=dimension)


def fetch_embeddings_for_entries(connection: sqlite3.Connection, entry_ids: list[str], model: str) -> dict[str, list[float]]:
    if not entry_ids:
        return {}
    placeholders = ", ".join("?" for _ in entry_ids)
    rows = connection.execute(
        f"SELECT paper_entry_id, embedding_json FROM paper_embeddings WHERE chunk_index = 0 AND model = ? AND paper_entry_id IN ({placeholders})",
        [model, *entry_ids],
    ).fetchall()
    return {str(row["paper_entry_id"]): json.loads(row["embedding_json"]) for row in rows}


def vector_status(connection: sqlite3.Connection) -> dict[str, Any]:
    models = [dict(row) for row in connection.execute("SELECT model, dimension, COUNT(*) AS count FROM paper_embeddings GROUP BY model, dimension ORDER BY count DESC").fetchall()]
    return {"sqlite_vec_loaded": try_load_sqlite_vec(connection), "vector_dimension": int(get_setting(connection, "vector_dimension")) if get_setting(connection, "vector_dimension") else None, "vector_table_exists": _vector_table_exists(connection), "embedding_count": connection.execute("SELECT COUNT(*) AS count FROM paper_embeddings").fetchone()["count"], "models": models}


def _vector_table_exists(connection: sqlite3.Connection) -> bool:
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = 'vec_paper_embeddings'").fetchone() is not None


def _search_sqlite_vec(connection: sqlite3.Connection, query_vector: list[float], limit: int, model: str | None) -> list[dict[str, Any]]:
    conditions = ["v.embedding MATCH ?", "k = ?"]
    params: list[Any] = [json.dumps(query_vector), limit]
    if model:
        conditions.append("e.model = ?")
        params.append(model)
    rows = connection.execute(f"SELECT v.distance, e.id AS embedding_id, e.paper_entry_id, e.chunk_index, e.model, e.chunk_text, p.title, p.link, p.source, p.doi, p.note_path FROM vec_paper_embeddings v JOIN paper_embeddings e ON e.id = v.paper_embedding_id JOIN papers p ON p.entry_id = e.paper_entry_id WHERE {' AND '.join(conditions)} ORDER BY v.distance ASC", params).fetchall()
    return [dict(row) for row in rows]


def _search_python_cosine(connection: sqlite3.Connection, query_vector: list[float], limit: int, model: str | None, dimension: int) -> list[dict[str, Any]]:
    sql = "SELECT e.id AS embedding_id, e.paper_entry_id, e.chunk_index, e.model, e.chunk_text, e.embedding_json, p.title, p.link, p.source, p.doi, p.note_path FROM paper_embeddings e JOIN papers p ON p.entry_id = e.paper_entry_id WHERE e.dimension = ?"
    params: list[Any] = [dimension]
    if model:
        sql += " AND e.model = ?"
        params.append(model)
    rows = connection.execute(sql, params).fetchall()
    scored = []
    for row in rows:
        score = _cosine_similarity(query_vector, json.loads(row["embedding_json"]))
        scored.append({"distance": 1 - score, "embedding_id": row["embedding_id"], "paper_entry_id": row["paper_entry_id"], "chunk_index": row["chunk_index"], "model": row["model"], "chunk_text": row["chunk_text"], "title": row["title"], "link": row["link"], "source": row["source"], "doi": row["doi"], "note_path": row["note_path"]})
    scored.sort(key=lambda item: item["distance"])
    return scored[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimension mismatch.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(l * r for l, r in zip(left, right)) / (left_norm * right_norm)


def _decode_tags(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    try:
        data = json.loads(raw_value)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _row_to_paper_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {"entry_id": row["entry_id"], "title": row["title"], "published": row["published"], "abstract": row["abstract"], "link": row["link"], "source": row["source"], "note_path": row["note_path"], "added_at": row["added_at"], "doi": row["doi"], "pdf_url": row["pdf_url"], "ai_summary": row["ai_summary"], "tags": _decode_tags(row["tags"]), "authors": row["author_names"].split("||") if row["author_names"] else []}

