-- Initial schema (PostgreSQL). Equivalent to Base.metadata.create_all().
-- Apply with: psql "$DATABASE_URL_PLAIN" -f migrations/001_init.sql
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    document_id   VARCHAR(64)  NOT NULL UNIQUE,
    filename      VARCHAR(512) NOT NULL,
    source_type   VARCHAR(32)  NOT NULL,
    source_url    VARCHAR(2048),
    file_path     VARCHAR(1024),
    tenant_id     VARCHAR(128) NOT NULL,
    uploaded_by   VARCHAR(128) NOT NULL,
    status        VARCHAR(16)  NOT NULL DEFAULT 'PENDING'
                  CHECK (status IN ('PENDING','PROCESSING','COMPLETED','FAILED')),
    error_message TEXT,
    total_chunks  INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_documents_tenant_status ON documents (tenant_id, status);

CREATE TABLE IF NOT EXISTS document_chunks (
    id               SERIAL PRIMARY KEY,
    document_id      VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_id         VARCHAR(96) NOT NULL UNIQUE,
    chunk_index      INTEGER NOT NULL,
    content          TEXT NOT NULL,
    page_number      INTEGER,
    section          VARCHAR(256),
    token_count      INTEGER NOT NULL DEFAULT 0,
    embedding_status VARCHAR(16) NOT NULL DEFAULT 'PENDING',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_chunk_doc_index UNIQUE (document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS ix_chunks_document ON document_chunks (document_id);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id            SERIAL PRIMARY KEY,
    task_id       VARCHAR(64) NOT NULL UNIQUE,
    document_id   VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    status        VARCHAR(16) NOT NULL DEFAULT 'PENDING',
    progress      INTEGER NOT NULL DEFAULT 0,
    retries       INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_jobs_document ON ingestion_jobs (document_id);
