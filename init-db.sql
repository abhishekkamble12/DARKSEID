-- =============================================================================
-- PostgreSQL Initialization Script for LangGraph Checkpointer
-- =============================================================================

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Note: The LangGraph checkpointer will create its own tables automatically
-- when PostgresSaver.setup() is called. This script is for any additional
-- custom tables you might need.

-- Example: Custom table for storing document metadata
CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    file_name VARCHAR(500) NOT NULL,
    file_type VARCHAR(50),
    chunks_count INTEGER,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Index for faster lookups by session
CREATE INDEX IF NOT EXISTS idx_doc_session ON document_metadata(session_id);

-- Example: Custom table for user sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Grant permissions (if needed for different users)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO user;
