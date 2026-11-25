cp .env.example .env            # add GEMINI_API_KEY and PINECONE_API_KEY

docker compose up -d --build  # # api, worker, beat, redis, postgres 

docker ps 

http://127.0.0.1:8000/docs#/  # swagger UI 

docker compose down # stop docker container 

Your Docker output looks good. **All five containers are running**, and PostgreSQL and Redis have passed their Docker health checks.

However, there is an important distinction:

- `Up` means the container process is running.
- `healthy` means its configured health check is passing.
- Neither status alone proves that your application can successfully communicate with every dependency or process a RAG ingestion job.

Let's verify each service individually and then test the complete upload-to-question-answer workflow.

# **1. Current Docker status**

5 containers running

| Service       | Container            | Current status | Initial assessment  |
| ------------- | -------------------- | -------------- | ------------------- |
| API           | `backend-api-1`      | Up 8 minutes   | Running             |
| Celery Worker | `backend-worker-1`   | Up 8 minutes   | Running             |
| Celery Beat   | `backend-beat-1`     | Up 8 minutes   | Running             |
| PostgreSQL    | `backend-postgres-1` | Up (healthy)   | Health check passed |
| Redis         | `backend-redis-1`    | Up (healthy)   | Health check passed |

Your PostgreSQL host port is `5433`, while the internal Docker port is `5432`. That is correct for the configuration we discussed.

Your API is accessible through port `8000`.

# **2. Verify the API service**

### **Step 1: Check API logs**

Run:

```
docker compose logs --tail=100 api
```

Look for:

```
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

Also check for errors such as:

- Database connection failures.
- Redis connection failures.
- Pinecone authentication or index errors.
- Gemini API configuration errors.

### **Step 2: Check the API health endpoint**

Open:

[API health endpoint](http://localhost:8000/health?utm_source=chatgpt.com)

Or run in PowerShell:

```
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json -Depth 5
```

Expected response:

```
{
  "status": "ok",
  "database": true,
  "redis": true,
  "pinecone": true,
  "model": "your-configured-model"
}
```



Expected result

&#x20;All three dependency checks should be `true`, and overall status should be `ok`.



Note: Your current health endpoint checks PostgreSQL, Redis, and Pinecone. It does not actually send a test request to Gemini, nor does it verify that Celery is processing tasks.

# **3. Verify PostgreSQL**

Your Docker health check already reports `healthy`. Now let's verify that PostgreSQL accepts SQL queries and that your application tables exist.

### **Step 1: Check database readiness**

```
docker compose exec postgres pg_isready -U postgres -d rag_db
```

Expected:

```
/var/run/postgresql:5432 - accepting connections
```

### **Step 2: Execute a database query**

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT version();"
```

You should see the PostgreSQL version.

### **Step 3: Verify your RAG application tables**

```
docker compose exec postgres psql -U postgres -d rag_db -c "\dt"
```

You should see tables similar to:

| Table             | Purpose                       |
| ----------------- | ----------------------------- |
| `documents`       | Uploaded document records     |
| `document_chunks` | Extracted document chunks     |
| `ingestion_jobs`  | Celery ingestion job tracking |

If these tables exist, execute:

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT document_id, filename, status, total_chunks FROM documents ORDER BY created_at DESC LIMIT 10;"
```

This checks whether your uploaded documents are being recorded.

# **4. Verify Redis**

Redis is your Celery message broker, so it is important that both the API and worker can communicate with it.

### **Step 1: Check Redis health**

```
docker compose exec redis redis-cli ping
```

Expected:

```
PONG
```

### **Step 2: Check Redis server information**

```
docker compose exec redis redis-cli INFO server
```

You should see Redis server information, including its version and uptime.

### **Step 3: Check Redis memory**

```
docker compose exec redis redis-cli INFO memory
```

This helps you monitor memory usage as the application grows.

# **5. Verify Celery Worker**

This is one of the most important checks in your RAG application.

Your API does not perform document ingestion directly. It sends an ingestion task to Redis, and the Celery worker processes it.

A running worker container does not automatically mean it is connected to Redis or ready to execute your tasks.

### **Step 1: Check worker logs**

```
docker compose logs --tail=150 worker
```

Look for:

```
Connected to redis://redis:6379/0
celery@... ready.
```

Also check that your ingestion task is registered.

For example, you should see a task resembling:

```
src.tasks.process_document
```

### **Step 2: Check whether the worker responds to Celery**

```
docker compose exec worker celery -A src.celery_app.celery_app inspect ping
```

Expected response:

```
-> celery@...: OK
    pong
```

If you receive `pong`, your worker is responding to Celery control commands.

### **Step 3: Check registered tasks**

```
docker compose exec worker celery -A src.celery_app.celery_app inspect registered
```

Verify that your document processing task appears in the output.

### **Step 4: Check active tasks**

```
docker compose exec worker celery -A src.celery_app.celery_app inspect active
```

This shows tasks currently being executed.

An empty active-task list is normal when no documents are being processed.

# **6. Verify Celery Beat**

Celery Beat is responsible for scheduling periodic tasks. It is different from the Celery worker:

- Worker executes tasks.
- Beat schedules tasks.

### **Step 1: Check Beat logs**

```
docker compose logs --tail=150 beat
```

Look for:

```
beat: Starting...
Scheduler: Sending due task
```

The exact output depends on your scheduler configuration and whether a scheduled task is due.

### **Step 2: Verify the container process**

```
docker compose exec beat celery -A src.celery_app.celery_app report
```

This displays Celery configuration and runtime information.

**Important:** A running Beat container does not prove that your periodic tasks are configured correctly. You should also verify the scheduled task definitions in your Celery configuration.

If you have not configured periodic tasks yet, Beat can be running without scheduling any useful work.

# **7. Verify the complete RAG workflow**

Now we need to test whether all the services work together, not just individually.

Your application workflow is:

**1. FastAPI**

Accepts document upload

**2. PostgreSQL + Redis**

Saves document and queues ingestion task

**3. Celery Worker**

Loads, extracts and chunks the document

**4. Pinecone**

Stores document vectors and metadata

**5. Gemini**

Generates answers using retrieved context

### **Step 1: Upload a test document**

Open:

[FastAPI Swagger UI](http://localhost:8000/docs?utm_source=chatgpt.com)

Test `POST /documents/upload`.

Use your existing tenant and user headers. Upload a small PDF or DOCX file.

Expected:

- HTTP `202 Accepted`.
- A `document_id`.
- A `task_id`.
- Initial status of `PENDING`.

### **Step 2: Monitor the worker**

Immediately after uploading, run:

```
docker compose logs -f worker
```

You should see the task being received and executed.

Press `Ctrl+C` to stop following the logs. This does not stop the worker container.

### **Step 3: Check document ingestion status**

Use the returned document ID:

```
GET /documents/{document_id}/status
```

Wait until the status becomes:

```
{
  "status": "COMPLETED",
  "progress": 100,
  "total_chunks": 25
}
```

The numbers above are illustrative. Your actual chunk count depends on the uploaded document.

### **Step 4: Test your document listing endpoint**

```
GET /documents?limit=20&offset=0
```

Verify that your uploaded document appears with its correct ID, filename, processing status, and chunk count.

### **Step 5: Test semantic search**

Use `POST /search`:

```
{
  "question": "What is the main topic of this document?",
  "top_k": 5,
  "document_ids": [
    "YOUR_DOCUMENT_ID"
  ]
}
```

You should receive relevant chunks with source information and similarity scores.

### **Step 6: Test RAG question answering**

Use `POST /ask`:

```
{
  "question": "Summarize the main points of this document.",
  "top_k": 5,
  "document_ids": [
    "YOUR_DOCUMENT_ID"
  ]
}
```

Verify that:

- Gemini returns an answer.
- Retrieved chunks are relevant.
- Citations contain the correct document information.
- No unrelated tenant's document is returned.

# **8. Useful monitoring commands**

Keep these commands handy for regular development.

| What you want to check | Command                           |
| ---------------------- | --------------------------------- |
| All containers         | `docker compose ps`               |
| All service logs       | `docker compose logs --tail=100`  |
| API logs               | `docker compose logs -f api`      |
| Worker logs            | `docker compose logs -f worker`   |
| Beat logs              | `docker compose logs -f beat`     |
| PostgreSQL logs        | `docker compose logs -f postgres` |
| Redis logs             | `docker compose logs -f redis`    |
| Resource consumption   | `docker stats`                    |
| Restart all services   | `docker compose restart`          |

You can also check whether the containers are consuming excessive CPU or memory with:

```
docker stats
```

# **9. My recommended verification checklist**

**Service and workflow verification**

0/11 completed

API starts successfully and /health returns ok

PostgreSQL accepts connections and application tables exist

Redis returns PONG

Celery worker responds with pong

Celery worker has process_document registered

Celery Beat starts without errors

Document upload returns HTTP 202

Worker successfully processes an uploaded document

Document status reaches COMPLETED

Pinecone search returns relevant chunks

Gemini generates an answer with citations

Reset checklist

## **# 10. What I recommend you do next**

Your current Docker status is a good starting point. I would not change the Docker configuration right now.

Instead, execute these five commands first and share their outputs:

```
docker compose logs --tail=80 api
```

```
docker compose exec worker celery -A src.celery_app.celery_app inspect ping
```

```
docker compose exec worker celery -A src.celery_app.celery_app inspect registered
```

```
docker compose exec redis redis-cli ping
```

```
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json
```

**My technical recommendation:** Once these checks pass, perform one real document upload and monitor its complete ingestion lifecycle. That is the strongest practical test of your API, PostgreSQL, Redis, Celery Worker, Pinecone, and Gemini integration.

One final distinction: Celery Beat being healthy is not the same as having a working scheduled task. We should verify its actual schedule separately when reviewing your Celery configuration.


# Assignment 12 API testing : 
I have reviewed all four files against the `main.py` implementation you shared earlier. We now have enough information to build a proper endpoint-testing plan with the **actual request schemas, database models, filter rules, and Celery ingestion workflow**.

Your API structure is good for a RAG document-Q&A application. However, I want you to test more than whether endpoints return `200` or `202`. We need to verify the complete workflow:

```
Upload / URL Ingestion
        |
        v
Document Record Created in PostgreSQL
        |
        v
Celery Job Created
        |
        v
Worker Processes Document
        |
        v
Document Loaded and Chunked
        |
        v
Vectors Stored in Pinecone
        |
        v
Chunk Records Stored in PostgreSQL
        |
        v
Document Status = COMPLETED
        |
        v
Search / Ask / Streaming Ask
        |
        v
Soft Delete + Pinecone Vector Removal
```

# **1. Important findings from your code**

Before testing, there are some important implementation details to understand.

## **1.1 Your actual API request schemas**

These are the request bodies confirmed by `schemas.py`.

| Endpoint                              | Request body        |
| ------------------------------------- | ------------------- |
| `POST /documents/upload`              | Multipart form data |
| `POST /documents/url`                 | `UrlIngestRequest`  |
| `GET /documents`                      | Query parameters    |
| `GET /documents/{document_id}/status` | Path parameter      |
| `DELETE /documents/{document_id}`     | Path parameter      |
| `POST /search`                        | `SearchRequest`     |
| `POST /ask`                           | `AskRequest`        |
| `POST /ask/stream`                    | `AskRequest`        |
| `GET /health`                         | No request body     |

Your `SearchRequest` inherits from `AskRequest`, so it supports:

- `question`
- `top_k`
- `document_ids`
- `filters`
- `rewrite_query`
- `mode`

Your search modes are:

- `dense`
- `keyword`
- `hybrid`
- `hybrid_rerank`

## **1.2 Important security issue: access-level filtering**

Your `filters.py` allows callers to submit:

```
{
  "filters": {
    "access_level": "confidential"
  }
}
```

The filter builder checks whether the value is valid, but it does not check whether the current user is authorized to access confidential documents.

This is important.

**A valid access-level value does not mean the user is authorized to access it.**

For production, you should derive the user's role and permissions from a verified authentication token and enforce document-level access rules on the server.

Your current `identity()` function uses `X-Tenant-ID` and `X-User-ID` headers. These are useful for development and testing, but users can supply arbitrary values. Do not treat these headers as secure authentication.

## **1.3 Document ownership inconsistency**

Your earlier `main.py` implementation has different ownership rules:

| Operation           | Current filtering      |
| ------------------- | ---------------------- |
| List documents      | Tenant + uploaded user |
| Get document status | Tenant                 |
| Delete document     | Tenant                 |
| Search/Ask          | Tenant metadata filter |

This means users in the same tenant may be able to inspect or delete each other's documents if they know the document ID.

You should decide whether your application supports:

1. **User-owned documents:** users can access only their own documents.
2. **Tenant-shared documents:** authorized users can access documents across their tenant, subject to access-level and department permissions.

For a multi-tenant business RAG application, I recommend implementing tenant-level isolation combined with role-based and document-level access control.

## **1.4 Your deletion logic is consistent with your chunking design**

Your `tasks.py` stores parent chunks in PostgreSQL but embeds only child or standalone chunks in Pinecone.

This means the following condition is appropriate for deleting Pinecone vectors:

```
DocumentChunk.chunk_type != "parent"
```

Your deletion endpoint and `purge_deleted_vectors` task use the same condition.

One additional point: `DocumentChunk.chunk_type` is non-nullable, which is good. Your current model defaults it to `"standalone"`.

# **2. Prepare your testing environment**

Open PowerShell in your backend directory:

```
cd "D:\Shopify_Tex\Subhash_RAG_Practice_2026\Day1_Basic_RAG_Pipeline_QA_assistance\Backend"
```

Start your Docker services:

```
docker compose up -d --build
```

Check the services:

```
docker compose ps
```

You should see these five services running:

| Service    | Purpose                                 |
| ---------- | --------------------------------------- |
| API        | FastAPI application                     |
| Worker     | Celery document processing              |
| Beat       | Scheduled Celery tasks                  |
| PostgreSQL | Document, chunk, job, and usage records |
| Redis      | Celery message broker                   |

Check logs:

```
docker compose logs --tail=100 api
docker compose logs --tail=100 worker
docker compose logs --tail=100 beat
```

Check Redis:

```
docker compose exec redis redis-cli ping
```

Expected:

```
PONG
```

Check PostgreSQL:

```
docker compose exec postgres pg_isready -U postgres -d rag_db
```

Check whether Celery workers are responding:

```
docker compose exec worker celery -A src.celery_app.celery_app inspect ping
```

Check registered Celery tasks:

```
docker compose exec worker celery -A src.celery_app.celery_app inspect registered
```

You should find tasks such as:

- `src.tasks.process_document`
- `src.tasks.requeue_stuck_documents`
- `src.tasks.purge_deleted_vectors`

If your Celery application registers tasks under different names, use the names displayed by the command.

# **3. Configure common testing headers**

Your identity dependency requires these two headers for all endpoints that use it.

Use the same values throughout the initial testing:

```
X-Tenant-ID: tenant_001
X-User-ID: user_001
```

In Swagger:

1. Open `http://localhost:8000/docs`.
2. Expand an endpoint.
3. Select **Try it out**.
4. Enter the required headers.
5. Provide the request body or query parameters.
6. Execute the request.

Note: Swagger may display the headers as endpoint parameters rather than providing a global authentication button.

# **4. Test API 1 — Health Check**

Start here

**Endpoint:** `GET /health`

This endpoint checks PostgreSQL, Redis, and Pinecone.

### **Request**

```
Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/health" |
    ConvertTo-Json -Depth 5
```

### **Expected response**

```
{
  "status": "ok",
  "database": true,
  "redis": true,
  "pinecone": true,
  "model": "your-configured-model"
}
```

The actual model name depends on your configuration.

### **What to verify**

| Check        | Expected |
| ------------ | -------- |
| API responds | Yes      |
| Database     | `true`   |
| Redis        | `true`   |
| Pinecone     | `true`   |
| Status       | `ok`     |

If any dependency fails, your endpoint should report a degraded status according to your `main.py` implementation.

**Important:** This health endpoint does not currently verify Gemini connectivity or whether the Celery worker is processing jobs. A successful health response does not guarantee that ingestion and answer generation will work.

# **5. Test API 2 — Upload Document**

**Endpoint:** `POST /documents/upload`

This is the most important test because it starts your ingestion pipeline.

### **Step 1: Prepare a test document**

Create a small text-rich PDF or DOCX file.

For example:

`rag_testing_guide.pdf`

Include some distinctive content:

```
ShvetVarni RAG Testing Document

ShvetVarni is a software development company.

The company provides custom software development,
Python backend development, AI integration,
and web application development.

The RAG application uses FastAPI, PostgreSQL,
Redis, Celery, and Pinecone.

Pinecone is responsible for storing and searching
document vectors.
```

Use a real PDF file, not a renamed `.txt` file. Your application validates PDF magic bytes.

### **Step 2: Upload through Swagger**

Open:

`POST /documents/upload`

Provide:

| Field        | Value                   |
| ------------ | ----------------------- |
| file         | `rag_testing_guide.pdf` |
| department   | `Engineering`           |
| access_level | `internal`              |
| X-Tenant-ID  | `tenant_001`            |
| X-User-ID    | `user_001`              |

### **Expected response**

```
{
  "document_id": "doc_xxxxxxxxxxxx",
  "filename": "rag_testing_guide.pdf",
  "status": "PENDING",
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "duplicate": false,
  "message": "..."
}
```

The exact message depends on your implementation.

**Save these two values:**

- `document_id`
- `task_id`

You will use them in subsequent tests.

### **Step 3: Monitor the Celery worker**

Open another PowerShell window:

```
docker compose logs -f worker
```

You should see the ingestion task start.

Your task implementation should progress through:

```
PENDING
  |
  v
PROCESSING
  |
  v
Loading document
  |
  v
Chunking document
  |
  v
Pinecone upsert
  |
  v
Saving PostgreSQL chunk records
  |
  v
COMPLETED
```

### **Step 4: Check PostgreSQL**

Run:

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT document_id, filename, status, total_chunks, tenant_id, uploaded_by FROM documents ORDER BY id DESC LIMIT 5;"
```

Check ingestion jobs:

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT task_id, document_id, status, progress, retries, error_message FROM ingestion_jobs ORDER BY id DESC LIMIT 5;"
```

Check chunk records:

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT document_id, COUNT(*) AS total_chunks FROM document_chunks GROUP BY document_id ORDER BY document_id DESC;"
```

### **Success conditions**

- API returns `202 Accepted`.
- Document record is created.
- Ingestion job is created.
- Worker processes the task.
- Pinecone receives vectors.
- Document status eventually becomes `COMPLETED`.
- `total_chunks` is greater than zero.

Do not move to the RAG search tests until your document has completed ingestion.

# **6. Test API 3 — List Documents**

**Endpoint:** `GET /documents`

Your implementation supports pagination and filters.

### **Request**

```
$headers = @{
    "X-Tenant-ID" = "tenant_001"
    "X-User-ID" = "user_001"
}

Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/documents?limit=20&offset=0" `
    -Headers $headers |
    ConvertTo-Json -Depth 8
```

### **Expected response structure**

```
{
  "total": 1,
  "limit": 20,
  "offset": 0,
  "documents": [
    {
      "document_id": "doc_xxxxxxxxxxxx",
      "filename": "rag_testing_guide.pdf",
      "source_type": "pdf",
      "source_url": null,
      "uploaded_by": "user_001",
      "department": "Engineering",
      "access_level": "internal",
      "status": "COMPLETED",
      "total_chunks": 5,
      "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "progress": 100,
      "retries": 0,
      "error_message": null,
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

The example is illustrative. Your actual number of chunks and timestamps will differ.

### **Test pagination**

```
GET /documents?limit=5&offset=0
GET /documents?limit=5&offset=5
GET /documents?limit=10&offset=0
```

### **Test filters**

```
GET /documents?status=COMPLETED
GET /documents?status=PENDING
GET /documents?source_type=pdf
GET /documents?department=Engineering
```

### **Negative tests**

| Input                    | Expected |
| ------------------------ | -------- |
| `limit=0`                | 422      |
| `limit=101`              | 422      |
| `offset=-1`              | 422      |
| `status=INVALID`         | 422      |
| Missing identity headers | 401      |

# **7. Test API 4 — Document Status**

**Endpoint:** `GET /documents/{document_id}/status`

Use the document ID from your upload response.

### **Request**

```
$documentId = "doc_xxxxxxxxxxxx"

Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/documents/$documentId/status" `
    -Headers $headers |
    ConvertTo-Json -Depth 5
```

### **Expected response**

```
{
  "document_id": "doc_xxxxxxxxxxxx",
  "filename": "rag_testing_guide.pdf",
  "source_type": "pdf",
  "status": "COMPLETED",
  "progress": 100,
  "total_chunks": 5,
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "retries": 0,
  "error_message": null,
  "created_at": "...",
  "updated_at": "..."
}
```

### **What to verify**

- Correct document ID.
- Status changes from `PENDING` to `PROCESSING` to `COMPLETED`.
- Progress eventually reaches `100`.
- Total chunks is greater than zero.
- Error message is null on success.

If the document remains in `PROCESSING`, inspect the worker logs and ingestion job record before continuing.

# **8. Test API 5 — URL Document Ingestion**

**Endpoint:** `POST /documents/url`

Your actual schema is:

```
class UrlIngestRequest(BaseModel):    url: HttpUrl    department: str | None = None    access_level: Literal[        "public",        "internal",        "confidential"    ] = "internal"
```

### **Request body**

```
{
  "url": "https://example.com/",
  "department": "Engineering",
  "access_level": "internal"
}
```

Use the required identity headers.

### **PowerShell request**

```
$urlBody = @{
    url = "https://example.com/"
    department = "Engineering"
    access_level = "internal"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/documents/url" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $urlBody |
    ConvertTo-Json -Depth 5
```

### **Expected response**

```
{
  "document_id": "doc_xxxxxxxxxxxx",
  "filename": "...",
  "status": "PENDING",
  "task_id": "...",
  "duplicate": false,
  "message": "..."
}
```

### **Verify ingestion**

Use the returned `document_id` in:

```
GET /documents/{document_id}/status
```

Wait for `COMPLETED`.

### **Negative tests**

Try these cases:

| Test                     | Expected behavior                  |
| ------------------------ | ---------------------------------- |
| Invalid URL syntax       | 422                                |
| Missing URL              | 422                                |
| Unsupported access level | 422                                |
| Private/internal IP URL  | 400                                |
| Localhost URL            | 400                                |
| Public reachable URL     | Accepted if the loader supports it |

**Security note:** Your `_assert_public_url()` validation is important because URL ingestion can create SSRF vulnerabilities. It should also protect against DNS rebinding and redirects to private or reserved IP addresses. Ensure redirects are revalidated, not just the initial URL.

# **9. Test API 6 — Search Documents**

Core RAG retrieval test

**Endpoint:** `POST /search`

Your `SearchRequest` supports all the fields inherited from `AskRequest`, plus `mode`.

### **Request body**

```
{
  "question": "Which technologies are used in the ShvetVarni RAG application?",
  "top_k": 5,
  "document_ids": [
    "doc_xxxxxxxxxxxx"
  ],
  "filters": {
    "department": "Engineering",
    "access_level": "internal"
  },
  "rewrite_query": false,
  "mode": "hybrid_rerank"
}
```

Replace the document ID with your actual ID.

### **PowerShell request**

```
$searchBody = @{
    question = "Which technologies are used in the ShvetVarni RAG application?"
    top_k = 5
    document_ids = @("doc_xxxxxxxxxxxx")
    filters = @{
        department = "Engineering"
        access_level = "internal"
    }
    rewrite_query = $false
    mode = "hybrid_rerank"
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/search" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $searchBody |
    ConvertTo-Json -Depth 10
```

### **Expected response structure**

```
{
  "query": "Which technologies are used in the ShvetVarni RAG application?",
  "mode": "hybrid_rerank",
  "results": [
    {
      "source_id": 1,
      "chunk_id": "...",
      "score": 0.92,
      "document_id": "doc_xxxxxxxxxxxx",
      "document_name": "rag_testing_guide.pdf",
      "page_number": 1,
      "section": null,
      "text": "The RAG application uses FastAPI, PostgreSQL, Redis, Celery, and Pinecone.",
      "rerank_score": 0.92,
      "rrf_score": 0.03,
      "dense_score": 0.85,
      "keyword_score": 0.02,
      "dense_rank": 1,
      "keyword_rank": 1
    }
  ],
  "stats": {}
}
```

Scores and statistics are illustrative. Your actual response depends on retrieval and reranking.

### **Test all four retrieval modes**

Use the same question and document ID, changing only `mode`.

| Mode            | What you are testing         |
| --------------- | ---------------------------- |
| `dense`         | Semantic vector retrieval    |
| `keyword`       | Keyword/BM25 retrieval       |
| `hybrid`        | Dense + keyword fusion       |
| `hybrid_rerank` | Hybrid retrieval + reranking |

Test each mode separately.

**What I want you to check:**

1. Does the API return results?
2. Does the correct document appear?
3. Is the returned text relevant to the question?
4. Are the scores present?
5. Does `hybrid_rerank` return reranking information?
6. Do the results respect the document ID and tenant filters?

Also test a question whose answer is not in the document. Your search endpoint may still return semantically related chunks; that is not automatically an error. Evaluate relevance rather than expecting an empty result every time.

# **10. Test API 7 — Ask Question**

**Endpoint:** `POST /ask`

This endpoint executes the retrieval and LLM answer-generation pipeline.

Unlike `/search`, it does not accept a retrieval mode in the request body.

### **Request body**

```
{
  "question": "Which technologies are used in the ShvetVarni RAG application?",
  "top_k": 5,
  "document_ids": [
    "doc_xxxxxxxxxxxx"
  ],
  "filters": {
    "department": "Engineering",
    "access_level": "internal"
  },
  "rewrite_query": false
}
```

### **PowerShell request**

```
$askBody = @{
    question = "Which technologies are used in the ShvetVarni RAG application?"
    top_k = 5
    document_ids = @("doc_xxxxxxxxxxxx")
    filters = @{
        department = "Engineering"
        access_level = "internal"
    }
    rewrite_query = $false
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/ask" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $askBody |
    ConvertTo-Json -Depth 10
```

### **Expected response structure**

```
{
  "question": "Which technologies are used in the ShvetVarni RAG application?",
  "answer": "The ShvetVarni RAG application uses FastAPI, PostgreSQL, Redis, Celery, and Pinecone.",
  "citations": [
    {
      "source_id": 1,
      "document_name": "rag_testing_guide.pdf",
      "page_number": 1,
      "section": null,
      "chunk_id": "...",
      "score": 0.92,
      "rerank_score": 0.92,
      "rrf_score": 0.03,
      "dense_score": 0.85,
      "keyword_score": 0.02,
      "text": "The RAG application uses FastAPI, PostgreSQL, Redis, Celery, and Pinecone."
    }
  ],
  "model": "your-configured-model",
  "retrieved_chunks": 1,
  "retrieval": {},
  "usage": {}
}
```

The actual answer and values will depend on Gemini and retrieval.

### **What to verify**

- Answer is based on uploaded document content.
- Citations reference the correct document.
- Citation text supports the generated answer.
- Page number and section are correct when available.
- Retrieved chunk count is sensible.
- Usage information is recorded when applicable.

### **Test query rewriting**

Repeat with:

```
{
  "question": "What technologies does it use?",
  "top_k": 5,
  "document_ids": [
    "doc_xxxxxxxxxxxx"
  ],
  "rewrite_query": true
}
```

Compare the results with `rewrite_query: false`.

This tests whether query rewriting improves retrieval for short or ambiguous questions.

### **Test usage records**

Your `LLMUsage` model stores request-level and model-call information.

Run:

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT request_id, tenant_id, user_id, endpoint, purpose, model_requested, model_used, status, prompt_tokens, completion_tokens, total_tokens, latency_ms FROM llm_usage ORDER BY id DESC LIMIT 20;"
```

Verify that the expected `ask` and query-rewrite calls are recorded when those calls occur.

# **11. Test API 8 — Streaming Ask**

**Endpoint:** `POST /ask/stream`

Your endpoint uses Server-Sent Events (SSE).

This endpoint needs a slightly different testing method because the response is a continuous event stream.

### **Request body**

Use the same request body as `/ask`:

```
{
  "question": "Explain the document ingestion workflow.",
  "top_k": 5,
  "document_ids": [
    "doc_xxxxxxxxxxxx"
  ],
  "rewrite_query": false
}
```

### **PowerShell testing**

Use `curl.exe`, not PowerShell's historical `curl` alias.

```
curl.exe -N `
  -X POST "http://localhost:8000/ask/stream" `
  -H "Content-Type: application/json" `
  -H "X-Tenant-ID: tenant_001" `
  -H "X-User-ID: user_001" `
  -d "{\"question\":\"Explain the document ingestion workflow.\",\"top_k\":5,\"document_ids\":[\"doc_xxxxxxxxxxxx\"],\"rewrite_query\":false}"
```

### **Expected event sequence**

Your implementation can emit events such as:

```
event: retrieval
data: {...}

event: rerank
data: {...}

event: sources
data: [...]

event: token
data: {"text":"The "}

event: token
data: {"text":"document "}

event: token
data: {"text":"ingestion..."}

event: done
data: {...}
```

The `query_rewrite` event is conditional. The exact event payloads depend on your `stream_rag_answer()` implementation.

### **What to verify**

| Test                   | Expected                      |
| ---------------------- | ----------------------------- |
| Connection established | Yes                           |
| Retrieval event        | When emitted by the pipeline  |
| Sources event          | When sources are available    |
| Token events           | Answer arrives incrementally  |
| Done event             | Successful completion         |
| Error event            | Appropriate failure reporting |

Do not judge streaming only by whether the endpoint returns `200`. Confirm that the client receives incremental events rather than waiting for one completed response.

Also verify that a disconnected client does not leave unnecessary LLM generation or usage records running indefinitely.

# **12. Test API 9 — Delete Document**

Run this after other tests

**Endpoint:** `DELETE /documents/{document_id}`

Use a disposable test document. Do not delete your only test document until all search and ask tests are complete.

### **Request**

```
$documentId = "doc_xxxxxxxxxxxx"

Invoke-RestMethod `
    -Method Delete `
    -Uri "http://localhost:8000/documents/$documentId" `
    -Headers $headers |
    ConvertTo-Json -Depth 5
```

### **Expected response structure**

```
{
  "document_id": "doc_xxxxxxxxxxxx",
  "deleted": true,
  "vectors_removed": true,
  "purge_queued": false,
  "chunks": 5
}
```

If inline Pinecone deletion fails, your code may queue a background purge. In that case, `vectors_removed` can be false and `purge_queued` may be true.

### **Verify soft deletion**

Run:

```
docker compose exec postgres psql -U postgres -d rag_db -c "SELECT document_id, is_deleted, deleted_at FROM documents WHERE document_id = 'doc_xxxxxxxxxxxx';"
```

Expected:

- `is_deleted = true`
- `deleted_at` contains a timestamp.

### **Verify the document is no longer listed**

```
Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/documents?limit=20&offset=0" `
    -Headers $headers |
    ConvertTo-Json -Depth 8
```

The deleted document should not appear.

### **Verify the status endpoint**

```
Invoke-RestMethod `
    -Method Get `
    -Uri "http://localhost:8000/documents/$documentId/status" `
    -Headers $headers
```

Expected: `404 Not Found`.

### **Verify Pinecone deletion**

Check your worker/API logs for any purge errors. Also perform a search constrained to the deleted document ID.

Expected: no matching document chunks are returned.

Remember that PostgreSQL retains the document and chunk rows because this is a soft delete. The goal is to hide the document and remove its searchable vectors, not necessarily to physically delete all database records.

# **13. Complete negative testing checklist**

After all successful endpoint tests, run these negative tests.

| Endpoint                 | Test case                          | Expected |
| ------------------------ | ---------------------------------- | -------- |
| `/health`                | Normal request                     | 200      |
| `/documents/upload`      | Missing identity headers           | 401      |
| `/documents/upload`      | Unsupported extension              | 415      |
| `/documents/upload`      | Empty file                         | 400      |
| `/documents/upload`      | Oversized file                     | 413      |
| `/documents/upload`      | Invalid PDF/DOCX magic bytes       | 400      |
| `/documents/upload`      | Invalid access level               | 400      |
| `/documents/url`         | Invalid URL schema                 | 422      |
| `/documents/url`         | Private URL                        | 400      |
| `/documents`             | Missing identity headers           | 401      |
| `/documents`             | Invalid status                     | 422      |
| `/documents`             | Invalid limit                      | 422      |
| `/documents/{id}/status` | Unknown document ID                | 404      |
| `/search`                | Question shorter than 3 characters | 422      |
| `/search`                | `top_k=0`                          | 422      |
| `/search`                | `top_k=11`                         | 422      |
| `/search`                | Invalid retrieval mode             | 422      |
| `/search`                | Unsupported filter key             | 400      |
| `/search`                | Invalid date filter                | 400      |
| `/ask`                   | Missing identity headers           | 401      |
| `/ask`                   | Invalid request body               | 422      |
| `/ask/stream`            | Missing identity headers           | 401      |
| `/ask/stream`            | Invalid request body               | 422      |
| `/documents/{id}` DELETE | Unknown document ID                | 404      |

One important distinction: `document_ids` is limited to a maximum of 20 entries by your Pydantic schema, but the schema does not currently enforce a minimum length or validate each ID's format.

# **14. Additional database and Celery verification**

Your application is asynchronous at the ingestion level, so checking only HTTP responses is insufficient.

## **14.1 Check document ingestion status**

```
SELECT
    document_id,
    filename,
    status,
    total_chunks,
    is_deleted,
    created_at,
    updated_at
FROM documents
ORDER BY id DESC;
```

## **14.2 Check ingestion job progress**

```
SELECT
    task_id,
    document_id,
    status,
    progress,
    retries,
    error_message,
    started_at,
    completed_at
FROM ingestion_jobs
ORDER BY id DESC;
```

## **14.3 Check chunk type distribution**

```
SELECT
    document_id,
    chunk_type,
    embedding_status,
    COUNT(*) AS total
FROM document_chunks
GROUP BY document_id, chunk_type, embedding_status
ORDER BY document_id;
```

This is particularly important because your ingestion pipeline stores parent chunks in PostgreSQL without embedding them.

## **14.4 Check document-to-chunk consistency**

```
SELECT
    d.document_id,
    d.status,
    d.total_chunks,
    COUNT(c.id) FILTER (
        WHERE c.chunk_type != 'parent'
    ) AS embedded_chunk_records,
    COUNT(c.id) FILTER (
        WHERE c.chunk_type = 'parent'
    ) AS parent_chunk_records
FROM documents d
LEFT JOIN document_chunks c
    ON c.document_id = d.document_id
GROUP BY d.document_id, d.status, d.total_chunks
ORDER BY d.created_at DESC;
```

For completed documents, `total_chunks` should match the number of non-parent chunk records.

# **15. Issues I recommend fixing before production**

Your code has a good foundation, but I would not consider it production-ready until these issues are addressed.

| Priority | Issue                                                     | Recommendation                                                 |
| -------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| Critical | Identity is based on caller-supplied headers              | Implement verified JWT authentication                          |
| Critical | Access-level filtering does not enforce authorization     | Enforce role and document permissions server-side              |
| High     | Status and delete use tenant-only access                  | Apply consistent ownership/permission rules                    |
| High     | Duplicate document IDs are based on tenant + content      | Decide whether identical documents are shared or user-specific |
| High     | URL ingestion can be an SSRF entry point                  | Revalidate redirects and resolved IP addresses                 |
| Medium   | Health does not check Gemini or Celery readiness          | Add separate readiness checks                                  |
| Medium   | Celery retry and stuck-job behavior needs failure testing | Test worker crashes and retries                                |
| Medium   | Soft deletion and vector deletion are separate operations | Verify purge retries and eventual consistency                  |

**One particularly important issue with duplicate uploads:** Your document ID is derived from the tenant ID and document content. Two users uploading identical content within the same tenant can receive the same document ID. However, your list endpoint filters by `uploaded_by`. That can create confusing behavior where the second user receives a duplicate response but cannot see the existing document in their own list.

You should resolve this according to your intended document-sharing policy.

# **16. Recommended execution order**

Do not test all endpoints randomly. Follow this sequence so that each test has the data it needs.

**API testing execution checklist**

0 of 15 completed

0%

Start Docker services and verify all containers

Test GET /health

Upload a PDF using POST /documents/upload

Verify PostgreSQL document and job records

Monitor Celery until document reaches COMPLETED

Test GET /documents

Test GET /documents/{document_id}/status

Ingest a public URL using POST /documents/url

Test all four POST /search retrieval modes

Test POST /ask and verify citations

Test POST /ask/stream and verify SSE events

Test negative inputs and missing headers

Delete a disposable document

Verify soft deletion and Pinecone vector removal

Review LLM usage and database consistency

Reset checklist

# **17. My recommendation: let's test your actual application step by step**

I recommend that we do not try to execute all nine APIs at once. First, we should establish that the infrastructure and ingestion pipeline work correctly. Otherwise, later RAG failures will be difficult to diagnose.

**Your first assignment:**

1. Start Docker Compose.
2. Verify all five containers.
3. Test `GET /health`.
4. Upload `rag_testing_guide.pdf`.
5. Wait until the document status becomes `COMPLETED`.
6. Share the following outputs:

```
docker compose ps

docker compose logs --tail=100 worker

docker compose exec postgres psql -U postgres -d rag_db -c "SELECT document_id, filename, status, total_chunks FROM documents ORDER BY id DESC LIMIT 5;"

docker compose exec postgres psql -U postgres -d rag_db -c "SELECT task_id, document_id, status, progress, retries, error_message FROM ingestion_jobs ORDER BY id DESC LIMIT 5;"
```

Also share the upload API response.

I will then check whether the **API → PostgreSQL → Redis → Celery → Document Loader → Chunking → Pinecone** workflow is genuinely successful before we move to `/search`, `/ask`, and streaming.