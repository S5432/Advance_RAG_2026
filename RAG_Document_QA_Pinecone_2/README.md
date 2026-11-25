# Day 9 — Document Q&A Assistant (RAG + Pinecone + Gemini Flash)

FastAPI service that ingests PDF (incl. scanned), DOCX, Markdown, CSV, HTML files and web URLs,
stores 384-dim embeddings in Pinecone with rich metadata, and answers questions with citations.

```
Upload → load (OCR if scanned) → chunk (800/150) → embed (MiniLM) → Pinecone upsert
Question → embed → Pinecone top-5 (tenant filter) → cited prompt → Gemini Flash → answer + citations
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# OCR needs system packages:  Ubuntu: sudo apt install tesseract-ocr poppler-utils
#                             macOS:  brew install tesseract poppler
cp .env.example .env     # fill GEMINI_API_KEY and PINECONE_API_KEY
uvicorn src.main:app --reload
```

### Pinecone index
The service creates the index on first start if it does not exist. To create it manually:
```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="...")
pc.create_index(name="rag-document-qa", dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
```
Namespace: `documents` · Top-k: 5 · Embedding: `sentence-transformers/all-MiniLM-L6-v2`.

## API (all endpoints except /health need `X-Tenant-ID` and `X-User-ID` headers)
```bash
H='-H X-Tenant-ID:company_001 -H X-User-ID:user_001'

curl http://localhost:8000/health

curl $H -F "file=@data/sample_policy.pdf" http://localhost:8000/documents/upload
# -> {"document_id":"doc_ab12...","document_name":"sample_policy.pdf","status":"processing",...}

curl $H -X POST http://localhost:8000/documents/url -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/policy.html"}'

curl $H http://localhost:8000/documents/doc_ab12.../status

curl $H -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"How many annual leave days are allowed?"}'

curl $H -X POST http://localhost:8000/search -H "Content-Type: application/json" \
     -d '{"question":"remote work policy","top_k":5}'
```
Example `/ask` response:
```json
{"question":"How many annual leave days are allowed?",
 "answer":"Employees are entitled to 18 days of paid annual leave per year. [Source 1]",
 "citations":[{"source_id":1,"document_name":"sample_policy.pdf","page_number":2,
   "chunk_id":"doc_ab12_chunk_1","score":0.71,"text":"Leave Policy Employees are entitled to 18 days..."}],
 "model":"gemini-2.5-flash","retrieved_chunks":5}
```

## Design decisions
| Topic | Decision |
| --- | --- |
| Vector IDs | `{document_id}_chunk_{n}` — deterministic, so re-ingesting overwrites instead of duplicating |
| Duplicates | `document_id = sha256(tenant + file bytes)`; same tenant + same file → `status: duplicate` |
| Tenant isolation | every query gets `filter={"tenant_id": <header>}`, applied last so callers cannot override it |
| Scanned PDFs | pages without selectable text are OCR'd (pdf2image + pytesseract); mean confidence stored, pages < 60 flagged |
| Chunking | sentence-aware packing, 800 chars, 150 overlap; `section` detected from headings |
| Ingestion | background task; status `processing → completed / failed`; partial upserts rolled back on failure |
| Safety | file-type + magic-byte + size (20 MB) checks; SSRF guard on URLs; context treated as untrusted in prompt |
| Hallucination | zero retrieved chunks → fixed "not found" reply without calling the LLM; temperature 0.1 |

Known limits: the status registry is in-memory (use Redis/Postgres for multi-worker deployments);
identity headers must come from a real auth layer in production.

## Tests
```bash
pytest -q          # loaders, chunking and API tests; API tests use fakes (no keys / network needed)
python scripts/run_eval.py   # precision@5 / recall@5 against a running service
```

## Docker
```bash
docker compose up --build
```
