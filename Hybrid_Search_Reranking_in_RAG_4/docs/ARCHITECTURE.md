# Architecture & phase guide (read this instead of the code when extending the project)

## Phases already done
| Phase | Scope | Files |
| --- | --- | --- |
| 1 Foundation (Day 9, unchanged) | settings, loaders+OCR, chunking, Pinecone Documents service | `config.py` (+5 Day 10 fields), `loaders.py`, `chunking.py`, `pinecone_service.py`, `embeddings.py` |
| 2 Async ingestion | SQLAlchemy models, Celery app, `process_document` | `models.py`, `database.py`, `celery_app.py`, `tasks.py`, `migrations/` |
| 3 Retrieval, streaming, API | numbered retrieval, cited prompt, Gemini stream, SSE | `rag_chain.py`, `gemini_client.py`, `schemas.py`, `main.py` |
| 4 Tests, Docker, docs | fakes, tests, compose, reports | `tests/`, `docker-compose.yml`, `README.md`, `ANSWERS.md`, `reports/` |

## Integration contracts (stable; build on these without reopening the files)
- **Index**: schema-based Pinecone index, `chunk_text` has integrated embedding (llama-text-embed-v2, 1024, cosine), plus `title`.
- `PineconeService.from_settings(settings)` validates the schema; `.query(query_text, top_k, metadata_filter, namespace)` ->
  `[{"id", "score", "metadata": {..., "text": chunk_text}}]`; `.upsert_documents(documents, namespace, batch_size=50)`;
  `.delete_ids(ids)`; `.delete_by_document_id(document_id)`; `.ping()`.
- `upsert_chunks_to_pinecone(chunks, pinecone_service, namespace) -> list[str]` (no embeddings argument).
- `load_document(path_or_url, source_type, document_id) -> [{"text", "metadata"}]`;
  `chunk_documents(docs, size, overlap, extra_metadata=...) -> [{"text", "metadata": {chunk_id, chunk_index, ...}}]`.
- Metadata every chunk needs: document_id, document_name, source_type, page_number, chunk_index, chunk_id, section,
  uploaded_by, tenant_id, created_at.
- Filters: `{"tenant_id": {"$eq": t}, "document_id": {"$in": [...]}}`; tenant always from the identity headers.
- `retrieve_chunks(query, pinecone_service, top_k, metadata_filter)` -> chunks with `source_id`, `chunk_id`, `score`, `text`,
  `document_name`, `page_number`, `section`. No embedding model is passed anywhere.
- `stream_rag_answer(query, pinecone_service, gemini_client, top_k, model, metadata_filter)` yields
  `{"event": "sources"|"token"|"done"|"error", "data": {...}}`.
- Celery: `process_document(document_id, file_path, source_url)`; DB statuses PENDING|PROCESSING|COMPLETED|FAILED;
  `run_ingestion(...)` is the testable pipeline; `requeue_stuck_documents` is the beat recovery job.

## Where to add the next feature
- New endpoint: `main.py` (+ model in `schemas.py`); keep blocking work in plain `def` endpoints or Celery.
- New document type: `loaders.py` + `EXT_TO_TYPE` in `main.py`.
- New background job: `tasks.py` (use `session_scope()` for DB, `_deps()` for settings + Pinecone).
