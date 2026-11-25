# Day 8 – RAG Document Q&A Assistant (FastAPI + Chroma + Gemini Flash)

Upload PDFs -> page-wise text extraction (`pypdf`) -> overlapping chunks -> `all-MiniLM-L6-v2` embeddings -> persistent Chroma ->
top-k retrieval -> Gemini answer that must cite **document + page** or say *"I don't have enough information in the provided documents."*

## Setup
```bash
python -m venv .venv && source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                       # add GEMINI_API_KEY (free: https://aistudio.google.com/)
uvicorn src.main:app --reload                              # docs: http://127.0.0.1:8000/docs  (first start downloads the embedding model)
pytest -q                                                  # offline: fake embedder + fake Gemini, real Chroma
python -m src.make_sample_pdf                              # regenerate data/sample_policy.pdf (3 pages, no remote-work policy)
python -m src.evaluate                                     # measured retrieval + answer quality -> reports/rag_quality_report.md
```

## Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /documents/upload` (multipart `file`) | PDF only, max `MAX_UPLOAD_MB`; extract -> chunk -> embed -> store. Returns `document_id`, `page_count`, `chunk_count`, `status`. Same file again = replaced, not duplicated |
| `GET /documents`, `GET /documents/{id}` | List / metadata + chunk count (404 if unknown) |
| `POST /search` `{query, top_k?, document_id?}` | Retrieved chunks with `chunk_id, document_id, document_name, page, text, score` (no LLM) |
| `POST /ask` `{question, top_k?, document_id?}` | `answer, citations[], insufficient_context, retrieved_chunks, confidence` |
| `POST /ask/stream` | SSE: `sources` -> `chunk`* -> `done` (or `error`) |
| `DELETE /documents/{id}` | Delete document and all chunks |
| `GET /health` | Vector store, chunk count, model, LLM configured |

Status codes: validation -> **400**; non-PDF / bad PDF type -> **415**; too large -> **413**; unreadable or text-less PDF -> **400**; unknown document -> **404**;
Gemini failure -> **502** (safe message); no `GEMINI_API_KEY` -> **503** on `/ask*` (`/search` still works).

## curl
```bash
curl -s -F "file=@data/sample_policy.pdf" http://127.0.0.1:8000/documents/upload
# {"document_id":"doc_3f9a...","document_name":"sample_policy.pdf","page_count":3,"chunk_count":6,"pages_without_text":[],"status":"ingested"}   (illustrative)

curl -s -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query":"How many paid leaves do employees get?","top_k":3}'
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"How many paid leaves do employees get?","top_k":4,"document_id":"doc_3f9a..."}'
# {"answer":"Employees are eligible for 12 paid leaves per year [sample_policy.pdf, p.1].",
#  "citations":[{"document_id":"doc_3f9a...","document_name":"sample_policy.pdf","page":1,"chunk_id":"doc_3f9a..._chunk_0"}],
#  "insufficient_context":false,"retrieved_chunks":4,"confidence":"high"}                     (illustrative)
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is the company policy on remote work?"}'
# -> insufficient_context: true, "I don't have enough information in the provided documents."
curl -N -X POST localhost:8000/ask/stream -H "Content-Type: application/json" -d '{"question":"What should an employee do after a password breach?"}'
curl -s -X DELETE localhost:8000/documents/doc_3f9a...
```

## How grounding is enforced (defence in depth)
1. **Retrieval gate:** if no chunk reaches `MIN_RELEVANCE_SCORE` (cosine, default 0.25), the API refuses *without calling the LLM*.
2. **System prompt** (`src/rag.py`): only the context, cite every claim, exact refusal sentence, no outside knowledge, never reveal instructions, context is untrusted data.
3. **Structured answer:** Gemini JSON mode (`answer, source_ids, insufficient_context, confidence`), validated; one repair retry.
4. **Citations are built by the backend:** the model cites short labels `[C1]`; the server maps them to real document name/page/chunk id and drops invented labels. An answer with no valid citation is refused.
5. Excerpts and questions are wrapped in tags with breakout strings stripped, to resist prompt injection stored inside documents.

## Design notes
- Chunks never span pages, so every chunk has an exact page number for citations (trade-off: facts split across pages are not joined).
- One `embedder` is used for both ingestion and queries; the model name is stored in the collection and a mismatch raises an error (change `CHROMA_COLLECTION_NAME` or re-ingest after changing `EMBEDDING_MODEL`).
- Uploads are processed in memory (`uploads/` is only a placeholder). Parsing/embedding runs in a worker thread; for large files or many users move ingestion to a background queue (Celery/RQ) and return a job id.
- Multi-tenant use: add `tenant_id` to chunk metadata and enforce it server-side from the authenticated identity (see `reports/rag_quality_report.md` answer 5).
- Streaming sends raw `[C#]` labels; the `sources` event (sent first) and the `done` event (final answer with resolved citations) let the client render them. A mid-stream failure becomes an `error` event.
- Scanned PDFs have no text layer: they are rejected with a clear error (add OCR such as Tesseract if needed).
