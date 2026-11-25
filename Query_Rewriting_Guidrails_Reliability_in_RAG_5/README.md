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

## Query rewriting + multi-query retrieval (vague questions)
Short or vague questions ("what about time off?") share few words with the passage that answers them
("18 days of paid annual leave"), so one dense + BM25 search can miss it. With rewriting on, Gemini turns the
question into a few clearer search queries and every one of them is searched:

```
question ─▶ Gemini rewrite ─▶ [original, q1, q2, q3]
                               └─ for each query: dense top-20 (Pinecone) + BM25 top-20 (PostgreSQL)
                                  └─ all ranked lists ─▶ weighted RRF ─▶ top-20 ─▶ reranker ─▶ top-5 ─▶ Gemini answer
```
- **Safe by design**: the original question is always searched (weight `ORIGINAL_QUERY_WEIGHT`), reranking and the
  final prompt use the *original* wording, and any rewrite failure (timeout, bad JSON, API error) silently falls back to the
  normal pipeline (`query_rewrite.fallback = true`).
- **Cost**: +1 small Gemini call (thinking disabled on Flash) and +1 Pinecone query per extra variant (run in parallel).
- **Config** (`.env`): `QUERY_REWRITE_ENABLED`, `MULTI_QUERY_COUNT`, `QUERY_REWRITE_MODEL`, `QUERY_REWRITE_TIMEOUT_S`,
  `ORIGINAL_QUERY_WEIGHT`. Per request: `"rewrite_query": true|false` on `/ask`, `/ask/stream` and `/search`.
- **Visibility**: `retrieval.query_rewrite` (`original`, `rewrites`, `queries`, `fallback`, `latency_ms`) and
  `retrieval.queries` (dense / keyword hits per query) in `/ask` + `/search`; `/ask/stream` sends a new first event `query_rewrite`.

```bash
curl $H -X POST localhost:8000/search -H 'Content-Type: application/json' \
     -d '{"question":"what about time off?","rewrite_query":true}'      # compare with "rewrite_query": false
```

## Retries, timeouts, fallback model and token tracking (Gemini)
Every Gemini call (query rewrite and answer, blocking and streaming) now goes through one policy in
`src/gemini_client.py`:

```
primary model ──transient error──▶ retry (1s, 2s, 4s… + jitter, max LLM_MAX_RETRIES)
              ──still failing─────▶ GEMINI_FALLBACK_MODEL (same retry policy)
              ──all failed────────▶ /ask: 502 · /ask/stream: `error` + `done(status=failed)`
```
| Situation | Behaviour |
| --- | --- |
| 408 / 425 / 429 / 5xx, network error, timeout | retried with backoff, then fallback model |
| 404 (model retired / not found) | no retries, straight to the fallback model |
| 400 / 401 / 403 (bad request, bad key) | fails at once: another model would fail the same way |
| Streaming, failure **before** the first token | retry / fallback as above (client never sees a duplicate) |
| Streaming, failure **after** the first token | not retried (it would duplicate text): `error{partial:true}` + `done(failed)` |
| Query rewrite fails | unchanged: original query is used (no fallback model for rewriting, 1 retry) |

- **Timeouts**: `LLM_TIMEOUT_S` / `LLM_STREAM_TIMEOUT_S` are sent to the SDK per attempt (`http_options.timeout`);
  streams are also cut if one attempt exceeds its limit. `LLM_TOTAL_TIMEOUT_S` is a single budget for all attempts,
  backoff sleeps and fallbacks, so a bad outage cannot hold a request for attempts × models × timeout.
- **Token tracking**: `prompt / completion / thinking / cached / total` tokens from Gemini's `usage_metadata`.
  If the API reports nothing (or a stream is cut) the numbers are estimated (~4 chars/token) and marked
  `"source": "estimated"`. Thinking tokens are billed as output, so they are reported separately.
- **Where you see it**: `/ask` returns `usage`; `/ask/stream` puts `usage` in the `done` event; `model` is the model
  that really answered. `usage = {request_id, totals{…tokens, llm_calls, attempts, fallback_used}, calls[{purpose, model_requested, model_used, attempts, usage, attempt_log}]}`.
- **Logging**: one `llm_usage {json}` line per call on logger `docqa.usage` (WARNING when it failed or used the
  fallback), and one row per call in the `llm_usage` table (`LLM_USAGE_PERSIST=false` turns the table off).
  Questions and answers are never logged or stored there.
- **Config** (`.env`): `GEMINI_FALLBACK_MODEL` (comma-separated chain, empty = none), `LLM_MAX_RETRIES`,
  `LLM_RETRY_BACKOFF_S`, `LLM_RETRY_BACKOFF_MAX_S`, `LLM_TIMEOUT_S`, `LLM_STREAM_TIMEOUT_S`, `LLM_TOTAL_TIMEOUT_S`,
  `LLM_USAGE_PERSIST`. Check that `GEMINI_FALLBACK_MODEL` is a model your key can use.
- **Existing databases**: `init_db()` creates the `llm_usage` table on startup; `migrations/004_llm_usage.sql` does the
  same for fresh Postgres volumes.

```sql
-- tokens per tenant, model and day (fallback_used = how often the primary failed)
SELECT tenant_id, model_used, date_trunc('day', created_at) AS day, count(*) AS calls,
       sum(total_tokens) AS tokens, sum(fallback_used::int) AS fallbacks, sum((status <> 'ok')::int) AS failures
FROM llm_usage GROUP BY 1, 2, 3 ORDER BY day DESC;
```
The reranker already had its own retry + RRF fallback (`src/reranker.py`) and is unchanged.

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


## Guardrails (PII redaction, prompt-injection checks, output validation)

Implemented in `src/guardrails.py`, wired into `/ask`, `/ask/stream` and `/search`. Every layer is switched by
`.env` (see `.env.example`); with the settings absent / disabled the API behaves exactly as before.

| Layer | Where it runs | What happens |
|---|---|---|
| Prompt-injection check on the **question** | before retrieval | `INJECTION_ACTION=block` -> **HTTP 400** `{"error": "guardrail_blocked", "reason": "prompt_injection", "matches": [rule names]}`; `flag` -> answered, reported in `guardrails.input`; `off` |
| PII redaction on the **question** | before retrieval, rewrite, rerank and Gemini | email, phone, Aadhaar, PAN, credit card (Luhn), SSN, IP, API keys -> `[EMAIL]`, `[PHONE]`, ... The response `question` field is the redacted text |
| Prompt-injection check on **retrieved chunks** | after rerank, before the prompt | `CONTEXT_INJECTION_ACTION=drop` removes poisoned chunks (sources renumbered); `neutralize` defangs them |
| PII redaction on chunks | same place | only when `REDACT_CONTEXT=true` |
| **Output validation** | after Gemini | system-prompt leak -> replaced; PII -> redacted; `[Source N]` that does not exist -> removed; over-long -> truncated; missing citations -> reported only |

`/ask` returns a `guardrails` object (null when everything is off):
`{"input": {"pii_redacted": {...}, "injection": {"flagged", "score", "matches"}}, "context": {"dropped", "neutralized", "pii_redacted", "dropped_chunk_ids"}, "output": {"ok", "issues", "pii_findings", "removed_citations"}}`.
It contains counts and rule names only, never the redacted values.

`/ask/stream` rejects an injection question with HTTP 400 before the stream starts. Tokens are streamed as they are
generated, so the answer is validated when it is complete: if validation changed it, a `replace` event
`{"content": <clean answer>, "issues": [...]}` is sent just before `done`, and clients should show that text instead of
the streamed tokens. `done` carries the same `guardrails` report. Event order:
`[query_rewrite] -> retrieval -> rerank -> sources -> token* -> [replace] -> done`.

`/search` has no LLM, so it only redacts PII in the query; injection wording is not rejected there.

These are heuristics (regex + scoring), one layer next to the system prompt, tenant filters and rate limits; they will
not stop a determined attacker. Tune `INJECTION_THRESHOLD` (default 5) if you see false positives or misses.
