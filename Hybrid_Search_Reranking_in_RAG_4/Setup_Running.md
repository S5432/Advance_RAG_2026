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