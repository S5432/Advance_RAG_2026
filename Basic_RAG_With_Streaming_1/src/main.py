"""uvicorn src.main:app --reload"""
import asyncio, contextvars, json, logging, os, re, time, uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from src import rag
from src.config import get_settings
from src.gemini_client import GeminiClient, GeminiError
from src.ingestion import IngestionError, ingest_pdf
from src.pdf_loader import PDFExtractionError
from src.schemas import (AskRequest, DeleteResponse, DocumentInfo, RAGAnswerResponse, SearchRequest, SearchResponse, UploadResponse)
from src.vector_store import ChromaVectorStore

logging.basicConfig(level=get_settings().log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag_service")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")



SAFE_MESSAGES = {
    "auth_error": "The LLM provider rejected the service credentials.", "quota_exceeded": "The LLM provider quota or rate limit was exceeded. Try again later.",
    "timeout": "The LLM provider timed out.", "bad_request": "The LLM provider could not process this request.",
    "empty_response": "The model returned no content (it may have been blocked by safety filters).",
    "invalid_output": "The model did not return a valid answer. Please retry.", "upstream_error": "The LLM provider is unavailable or returned an unexpected response.",
}
PDF_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}



@lru_cache
def get_store() -> ChromaVectorStore:
    s = get_settings()
    return ChromaVectorStore(s.chroma_db_path, s.chroma_collection_name, s.embedding_model)


@lru_cache
def _build_client(key, model, base_url, timeout, retries, thinking) -> GeminiClient:
    return GeminiClient(key, model, base_url, timeout=timeout, max_retries=retries, thinking_budget=thinking)



def get_gemini_client() -> GeminiClient:
    s = get_settings(); key = s.gemini_api_key.get_secret_value()
    if not key:
        raise HTTPException(503, "LLM is not configured: set GEMINI_API_KEY.")
    return _build_client(key, s.gemini_model, s.gemini_base_url, s.gemini_timeout_seconds, s.gemini_max_retries, s.gemini_thinking_budget)



@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(get_store)                 # load embedding model + open Chroma once at startup
        logger.info("startup_ok chunks_indexed=%d", get_store().count())
    except Exception:
        logger.exception("startup_warmup_failed (will retry on first request)")
    yield

app = FastAPI(title="RAG Document Q&A Assistant", version="1.0", lifespan=lifespan)




# ---------------- cross-cutting ----------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request_id_var.set(rid); start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    logger.info("request_id=%s %s %s status=%d latency_ms=%.0f", rid, request.method, request.url.path, response.status_code, (time.perf_counter() - start) * 1000)
    return response



@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "message": str(e["msg"])} for e in exc.errors()]
    return JSONResponse({"detail": "Invalid request.", "errors": errors}, status_code=400)



def llm_http_error(exc: GeminiError) -> HTTPException:
    logger.error("request_id=%s llm_error type=%s detail=%s", request_id_var.get(), exc.error_type, exc)
    return HTTPException(502, detail={"message": SAFE_MESSAGES.get(exc.error_type, SAFE_MESSAGES["upstream_error"]), "error_type": exc.error_type})



def internal_error(what: str) -> HTTPException:
    logger.exception("request_id=%s %s failed", request_id_var.get(), what)
    return HTTPException(500, f"Internal error while {what}.")



def safe_filename(name: str | None) -> str:
    base = os.path.basename((name or "").replace("\\", "/"))
    return re.sub(r"[^A-Za-z0-9._ \-]", "_", base)[:100] or "document.pdf"



def retrieve(question: str, top_k: int | None, document_id: str | None) -> list[dict]:
    store = get_store()
    if document_id and store.get_document(document_id) is None:
        raise HTTPException(404, f"Document '{document_id}' not found.")
    try:
        return store.search(question, top_k or get_settings().top_k, {"document_id": document_id} if document_id else None)
    except HTTPException:
        raise
    except Exception:
        raise internal_error("searching documents")



# ---------------- endpoints ----------------
@app.get("/")
async def root():
    return {
        "message": "RAG API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    s = get_settings()
    try:
        chunks = get_store().count(); store_ok = True
    except Exception:
        chunks, store_ok = None, False
    return {"status": "ok" if store_ok else "degraded", "vector_store": store_ok, "chunks_indexed": chunks,
            "embedding_model": s.embedding_model, "llm_configured": bool(s.gemini_api_key.get_secret_value())}



@app.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)):
    s = get_settings(); name = safe_filename(file.filename); limit = s.max_upload_mb * 1024 * 1024
    if not name.lower().endswith(".pdf") or (file.content_type and file.content_type not in PDF_CONTENT_TYPES):
        raise HTTPException(415, "Only PDF files are accepted.")
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File too large (max {s.max_upload_mb} MB).")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "The file is not a valid PDF.")
    try:
        result = await run_in_threadpool(ingest_pdf, data, name, get_store(), s.chunk_size, s.chunk_overlap)
    except (PDFExtractionError, IngestionError) as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise internal_error("ingesting the document")
    logger.info("request_id=%s ingested document_id=%s pages=%d chunks=%d", request_id_var.get(), result["document_id"], result["page_count"], result["chunk_count"])
    return result



@app.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return get_store().list_documents()



@app.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str):
    doc = get_store().get_document(document_id)
    if doc is None:
        raise HTTPException(404, f"Document '{document_id}' not found.")
    return doc



@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    deleted = get_store().delete_document(document_id)
    if deleted == 0:
        raise HTTPException(404, f"Document '{document_id}' not found.")
    return {"document_id": document_id, "deleted_chunks": deleted, "status": "deleted"}




@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest):
    return {"query": payload.query, "results": retrieve(payload.query, payload.top_k, payload.document_id)}



@app.post("/ask", response_model=RAGAnswerResponse)
def ask(payload: AskRequest):
    contexts = retrieve(payload.question, payload.top_k, payload.document_id)
    s = get_settings()
    if not contexts or max(c["score"] for c in contexts) < s.min_relevance_score:
        # nothing relevant: refuse WITHOUT calling (or needing) the LLM
        return rag.answer_question(payload.question, contexts, None, s.min_relevance_score)
    client = get_gemini_client()
    try:
        return rag.answer_question(payload.question, contexts, client, s.min_relevance_score)
    except GeminiError as exc:
        raise llm_http_error(exc)



def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"




@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request):
    s = get_settings()
    contexts = await run_in_threadpool(retrieve, payload.question, payload.top_k, payload.document_id)
    relevant = bool(contexts) and max(c["score"] for c in contexts) >= s.min_relevance_score
    client = get_gemini_client() if relevant else None          # 503 before streaming starts if the LLM is not configured
    rid = request_id_var.get()

    async def event_stream():
        gen = rag.stream_answer(payload.question, contexts, client, s.min_relevance_score)
        try:
            async for ev in iterate_in_threadpool(gen):
                if await request.is_disconnected():
                    logger.info("request_id=%s client_disconnected", rid); return
                yield sse(ev["event"], ev["data"])
        except GeminiError as exc:
            logger.error("request_id=%s stream_error type=%s detail=%s", rid, exc.error_type, exc)
            yield sse("error", {"error_type": exc.error_type, "message": SAFE_MESSAGES.get(exc.error_type, SAFE_MESSAGES["upstream_error"])})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("request_id=%s stream_unexpected_error", rid)
            yield sse("error", {"error_type": "internal_error", "message": "Unexpected error while streaming."})
        finally:
            try: gen.close()
            except Exception: pass

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
