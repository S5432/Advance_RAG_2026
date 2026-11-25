-- Phase 2: parent-child chunks. Run once on an existing database (fresh docker volumes: auto-applied).
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS parent_chunk_id VARCHAR(96);
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_type VARCHAR(16) NOT NULL DEFAULT 'standalone';
ALTER TABLE document_chunks DROP CONSTRAINT IF EXISTS uq_chunk_doc_index;
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON document_chunks (parent_chunk_id);
