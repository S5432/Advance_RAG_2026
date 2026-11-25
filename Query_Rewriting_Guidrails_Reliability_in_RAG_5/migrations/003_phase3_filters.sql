-- Phase 3: department / access level / soft delete. Fresh docker volumes apply this automatically.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS department VARCHAR(128);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS access_level VARCHAR(16) NOT NULL DEFAULT 'internal';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS ix_documents_department ON documents (department);
CREATE INDEX IF NOT EXISTS ix_documents_is_deleted ON documents (is_deleted);
