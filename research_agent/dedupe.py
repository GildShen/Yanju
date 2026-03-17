from __future__ import annotations

import json
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AppConfig
from .db import get_connection, initialize_database



def _normalize_text(value: str) -> str:
    return " ".join((value or "").split()).strip()



def _normalize_link(value: str) -> str:
    return (value or "").strip()



def _group_key(row: sqlite3.Row) -> tuple[Any, ...]:
    doi = _normalize_text(row["doi"]).lower()
    link = _normalize_link(row["link"])
    source = _normalize_text(row["source"]).lower()
    title = _normalize_text(row["title"]).lower()
    published = _normalize_text(row["published"])
    if doi:
        return ("doi", doi)
    if link:
        return ("source_link", source, link)
    return ("source_title_pub", source, title, published)



def _sort_key(row: sqlite3.Row) -> tuple[str, str, str]:
    return (row["added_at"] or "", row["published"] or "", row["entry_id"])



def _load_papers(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return connection.execute(
        "SELECT entry_id, title, published, abstract, link, source, note_path, added_at, doi, pdf_url, ai_summary, tags FROM papers"
    ).fetchall()



def _backup_database(config: AppConfig) -> str:
    backup_dir = config.data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"research_agent-before-dedupe-{stamp}.db"
    shutil.copy2(config.db_path, backup_path)
    return str(backup_path)



def dedupe_papers(config: AppConfig, *, dry_run: bool = False, keep: str = "oldest") -> dict[str, Any]:
    if keep != "oldest":
        raise ValueError("Only keep='oldest' is currently supported.")

    config.ensure_directories()
    backup_path = ""
    if not dry_run:
        backup_path = _backup_database(config)

    with get_connection(config) as connection:
        initialize_database(connection)
        rows = _load_papers(connection)
        groups: dict[tuple[Any, ...], list[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            groups[_group_key(row)].append(row)

        report: dict[str, Any] = {
            "before_count": connection.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"],
            "groups_cleaned": 0,
            "papers_removed": 0,
            "kept": [],
            "dry_run": dry_run,
            "keep": keep,
            "backup_path": backup_path,
        }

        for key, items in groups.items():
            if len(items) < 2:
                continue
            items = sorted(items, key=_sort_key)
            keep_row = items[0]
            remove_rows = items[1:]
            report["groups_cleaned"] += 1
            report["papers_removed"] += len(remove_rows)

            keep_tags = json.loads(keep_row["tags"] or "[]") if keep_row["tags"] else []
            merged = {
                "abstract": keep_row["abstract"] or "",
                "doi": keep_row["doi"] or "",
                "pdf_url": keep_row["pdf_url"] or "",
                "link": keep_row["link"] or "",
                "note_path": keep_row["note_path"] or "",
                "ai_summary": keep_row["ai_summary"] or "",
                "tags": list(dict.fromkeys(keep_tags)),
            }

            for row in remove_rows:
                for field in ("abstract", "doi", "pdf_url", "link", "note_path", "ai_summary"):
                    if not (merged[field] or "").strip() and (row[field] or "").strip():
                        merged[field] = row[field]
                row_tags = json.loads(row["tags"] or "[]") if row["tags"] else []
                merged["tags"] = list(dict.fromkeys(merged["tags"] + row_tags))

            report["kept"].append(
                {
                    "key": list(key),
                    "kept_entry_id": keep_row["entry_id"],
                    "removed_entry_ids": [row["entry_id"] for row in remove_rows],
                    "title": keep_row["title"],
                }
            )

            if dry_run:
                continue

            connection.execute(
                "UPDATE papers SET abstract = ?, doi = ?, pdf_url = ?, link = ?, note_path = ?, ai_summary = ?, tags = ? WHERE entry_id = ?",
                (
                    merged["abstract"],
                    merged["doi"],
                    merged["pdf_url"],
                    merged["link"],
                    merged["note_path"],
                    merged["ai_summary"],
                    json.dumps(merged["tags"], ensure_ascii=False),
                    keep_row["entry_id"],
                ),
            )

            for row in remove_rows:
                old_id = row["entry_id"]
                connection.execute("UPDATE pdf_catalog SET paper_entry_id = ? WHERE paper_entry_id = ?", (keep_row["entry_id"], old_id))
                author_rows = connection.execute(
                    "SELECT author_id, author_order FROM paper_authors WHERE paper_entry_id = ?",
                    (old_id,),
                ).fetchall()
                for author_row in author_rows:
                    connection.execute(
                        "INSERT OR IGNORE INTO paper_authors (paper_entry_id, author_id, author_order) VALUES (?, ?, ?)",
                        (keep_row["entry_id"], author_row["author_id"], author_row["author_order"]),
                    )
                embedding_rows = connection.execute(
                    "SELECT chunk_index, model, dimension, chunk_text, embedding_json, created_at, updated_at FROM paper_embeddings WHERE paper_entry_id = ?",
                    (old_id,),
                ).fetchall()
                for embedding_row in embedding_rows:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO paper_embeddings (paper_entry_id, chunk_index, model, dimension, chunk_text, embedding_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            keep_row["entry_id"],
                            embedding_row["chunk_index"],
                            embedding_row["model"],
                            embedding_row["dimension"],
                            embedding_row["chunk_text"],
                            embedding_row["embedding_json"],
                            embedding_row["created_at"],
                            embedding_row["updated_at"],
                        ),
                    )
                connection.execute("DELETE FROM papers WHERE entry_id = ?", (old_id,))

        report["after_count"] = report["before_count"] if dry_run else connection.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"]
        report["remaining_link_dupes"] = connection.execute(
            "SELECT COUNT(*) AS c FROM (SELECT 1 FROM papers WHERE trim(coalesce(link,'')) <> '' GROUP BY source, link HAVING COUNT(*) > 1)"
        ).fetchone()["c"]
        report["remaining_doi_dupes"] = connection.execute(
            "SELECT COUNT(*) AS c FROM (SELECT 1 FROM papers WHERE trim(coalesce(doi,'')) <> '' GROUP BY lower(doi) HAVING COUNT(*) > 1)"
        ).fetchone()["c"]
        report["remaining_title_published_dupes"] = connection.execute(
            "SELECT COUNT(*) AS c FROM (SELECT 1 FROM papers GROUP BY source, lower(title), published HAVING COUNT(*) > 1)"
        ).fetchone()["c"]

    report_path = config.data_dir / "dedupe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
