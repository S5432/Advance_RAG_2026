# Day 10 — Async Document Q&A (Celery + Pinecone + Gemini streaming over SSE)

Upgrade of Day 9: uploads return immediately, ingestion runs in a **Celery worker**, status lives in
**PostgreSQL**, answers and sources stream to the client over **Server-Sent Events**.

```
Client ─POST /documents/upload─▶ FastAPI ─▶ PostgreSQL (documents, ingestion_jobs = PENDING)
                                    └────▶ Redis broker ─▶ Celery worker
                                                      load/OCR → chunk → Pinecone upsert (Pinecone embeds `chunk_text`) → chunks to PostgreSQL → COMPLETED
Client ─POST /ask/stream────────▶ FastAPI ─▶ query text → Pinecone search (embeds the query, tenant filter) → `sources` event
                                         └──▶ Gemini Flash stream → `token` events → `done` (citations)
```

## Embeddings and Pinecone (same as your working Day 9)
- Index: the **schema-based Pinecone index you created for Day 9** (`chunk_text` with integrated embedding
  `llama-text-embed-v2`, 1024 dims, cosine, plus `title`). The app validates it at startup and never creates one.
- No embedding model is installed or loaded (no `sentence-transformers`/torch): documents go in through
  `index.documents.upsert` and queries through `index.documents.search(EmbedQuery(...))` — the code in
  `src/pinecone_service.py` is your Day 9 file, unchanged.
- Tip: set `PINECONE_NAMESPACE=documents_day10` to keep Day 10 data apart from Day 9 documents.
- Keep the `pinecone` SDK at the version that worked for you on Day 9 (`pip show pinecone`), since the
  Documents API only exists in recent releases.

## Run with Docker Compose
```bash
cp .env.example .env            # add GEMINI_API_KEY and PINECONE_API_KEY
docker compose up -d --build  # # api, worker, beat, redis, postgres 
docker ps 
http://127.0.0.1:8000/docs#/  # swagger UI 
docker compose down # stop docker container   
curl localhost:8000/health
```
Compose overrides `REDIS_URL`, `DATABASE_URL` and `UPLOAD_DIR` for the internal network; the `uploads` volume is
shared so the worker can read what the API saved. Schema is created from `migrations/001_init.sql` (first
Postgres start) and by `init_db()`.

## Run locally (without Docker)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # also: apt install tesseract-ocr poppler-utils
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db postgres:16-alpine
cp .env.example .env
uvicorn src.main:app --reload                                              # terminal 1
celery -A src.celery_app.celery_app worker --loglevel=info                 # terminal 2
celery -A src.celery_app.celery_app beat --loglevel=info                   # terminal 3 (optional: requeues stuck docs)
```

## API (send `X-Tenant-ID` and `X-User-ID` on every call except /health)
```bash
H='-H X-Tenant-ID:company_001 -H X-User-ID:user_001'

curl $H -F "file=@data/sample_policy.pdf" localhost:8000/documents/upload
# {"document_id":"doc_ab12…","status":"PENDING","task_id":"…","duplicate":false,…}

curl $H localhost:8000/documents/doc_ab12…/status
# {"status":"PROCESSING","progress":60,…}  →  {"status":"COMPLETED","progress":100,"total_chunks":4,…}

curl $H -X POST localhost:8000/documents/url -H 'Content-Type: application/json' \
     -d '{"url":"https://example.com/policy.html"}'

curl $H -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"How many annual leave days are allowed?"}'

curl $H -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"question":"remote work","top_k":5}'

# streaming: -N disables curl's buffering
curl -N $H -X POST localhost:8000/ask/stream -H 'Content-Type: application/json' \
     -d '{"question":"How many annual leave days are allowed?"}'
```
Sample SSE output (`reports/sample_sse_output.txt`):
```
event: sources
data: {"sources":[{"source_id":1,"document_name":"sample_policy.pdf","page_number":2,"section":"Leave Policy","chunk_id":"doc_ab12_chunk_1","score":0.71}]}

event: token
data: {"content":"Employees are entitled to "}

event: token
data: {"content":"18 days of paid annual leave. [Source 1]"}

event: done
data: {"status":"completed","citations":[{"source_id":1,…}],"model":"gemini-3.5-flash-lite","retrieved_chunks":5}
```
Browser client: `fetch('/ask/stream', {method:'POST', headers, body})` and read `response.body` with a reader
(`EventSource` only supports GET).

## Key design points
| Topic | Implementation |
| --- | --- |
| Non-blocking upload | API saves the file, inserts `PENDING` rows, calls `apply_async`, returns 202 |
| Status & progress | `documents.status`, `ingestion_jobs.progress` (5→15→30→60→90→100), `retries`, `error_message` |
| Retries | `max_retries=3`, backoff 10/20/40 s; every failure is written as FAILED before retrying; permanent errors (empty doc) don't retry |
| Worker crash | `acks_late` + `reject_on_worker_lost` redeliver the message; deterministic vector IDs + chunk rows replaced ⇒ replay is idempotent; beat job `requeue_stuck_documents` rescues PENDING/PROCESSING docs older than 15 min |
| Duplicates | `document_id = sha256(tenant + bytes)`; chunk IDs `{document_id}_chunk_{n}`; Pinecone upsert replaces same ID |
| Tenant isolation | `tenant_id` in every vector; server-side filter on every query, applied last; status lookups return 404 across tenants |
| Streaming | `sources` first, then `token`s, then `done`; failures send `error` + `done(status=failed)`; client disconnect stops generation |
| Large PDFs | upserts in 200-chunk slices (Pinecone embeds server-side) with progress updates; 30-min task time limit |

Notes: SSE uses `StreamingResponse(media_type="text/event-stream")` so it works on every FastAPI version
(newer releases also offer `EventSourceResponse`). Identity headers must come from real auth in production.

## Tests
```bash
pytest -q        # loaders, chunking, RAG chain/streaming, Celery ingestion (SQLite), API + SSE — all use fakes
```
