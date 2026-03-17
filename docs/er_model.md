# ER Model

## Current implemented schema

```mermaid
erDiagram
    FEEDS ||--o{ PAPERS : "source feed"
    PAPERS ||--o{ PAPER_AUTHORS : "has"
    AUTHORS ||--o{ PAPER_AUTHORS : "maps"
    PAPERS ||--o{ PAPER_EMBEDDINGS : "chunk embeddings"

    FEEDS {
        int id PK
        text url UK
        text title
        text created_at
        text last_checked_at
    }

    PAPERS {
        text entry_id PK
        text title
        text published
        text abstract
        text link
        text source
        text feed_url
        text note_path
        text added_at
        text doi
        text pdf_url
        text ai_summary
        text tags
    }

    AUTHORS {
        int id PK
        text name UK
    }

    PAPER_AUTHORS {
        text paper_entry_id PK
        int author_id PK
        int author_order
    }

    PAPER_EMBEDDINGS {
        int id PK
        text paper_entry_id
        int chunk_index
        text model
        int dimension
        text chunk_text
        text embedding_json
        text created_at
        text updated_at
    }
```

## OpenAI + vector retrieval design

- OpenAI creates embeddings outside the DB layer.
- `paper_embeddings` is the canonical store for embedding payloads and chunk text.
- `vec_paper_embeddings` is an optional acceleration index when `sqlite-vec` is installed.
- Semantic retrieval first tries `sqlite-vec`; otherwise it falls back to Python cosine similarity on `paper_embeddings`.
- The application currently assumes one active embedding dimension per vector index.

## Recommended next-step research schema

```mermaid
erDiagram
    PAPERS ||--o{ PAPER_TAGS : "tagged by"
    TAGS ||--o{ PAPER_TAGS : "used by"
    PAPERS ||--o{ AI_SUMMARIES : "summarized as"
    PAPERS ||--o{ PAPER_FILES : "stores"
    PAPERS ||--o{ CITATIONS : "cites / cited by"
    PROJECTS ||--o{ PROJECT_PAPERS : "uses"
    PAPERS ||--o{ PROJECT_PAPERS : "supports"
    PAPERS ||--o{ PAPER_CONCEPTS : "mentions"
    CONCEPTS ||--o{ PAPER_CONCEPTS : "appears in"
```

## Practical recommendation

For the next iteration, keep this split:

1. Bibliographic truth in `papers`.
2. LLM-generated text in a future `ai_summaries` table.
3. Retrieval text chunks in `paper_embeddings`.
4. Fast ANN index in `sqlite-vec` only as a derived structure.

That separation will make future model migrations and re-embedding runs much less painful.
