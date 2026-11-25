-- LLM usage log: one row per Gemini call. Fresh docker volumes apply this automatically;
-- init_db() (create_all) also creates the table on an existing database.
CREATE TABLE IF NOT EXISTS llm_usage (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64),
    tenant_id VARCHAR(128),
    user_id VARCHAR(128),
    endpoint VARCHAR(32),
    purpose VARCHAR(32),
    model_requested VARCHAR(128),
    model_used VARCHAR(128),
    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,
    attempts INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(16),
    stream BOOLEAN NOT NULL DEFAULT FALSE,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    thinking_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    usage_source VARCHAR(16),
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_llm_usage_request ON llm_usage (request_id);
CREATE INDEX IF NOT EXISTS ix_llm_usage_tenant_created ON llm_usage (tenant_id, created_at);
