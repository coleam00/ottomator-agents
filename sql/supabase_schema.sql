-- =====================================================
-- Medical RAG Agent - Supabase PostgreSQL Schema
-- =====================================================
-- This schema sets up the complete database structure for a Medical RAG agent
-- with vector search, knowledge graph support, and multi-tenant capabilities.

-- =====================================================
-- 1. ENABLE REQUIRED EXTENSIONS
-- =====================================================

-- Enable pgvector for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable trigram matching for full-text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =====================================================
-- 2. DROP EXISTING OBJECTS (Clean Slate)
-- =====================================================

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;

-- Drop existing indexes explicitly
DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_chunks_document_id;
DROP INDEX IF EXISTS idx_chunks_chunk_index;
DROP INDEX IF EXISTS idx_chunks_content_trgm;
DROP INDEX IF EXISTS idx_documents_metadata;
DROP INDEX IF EXISTS idx_documents_created_at;
DROP INDEX IF EXISTS idx_sessions_user_id;
DROP INDEX IF EXISTS idx_sessions_expires_at;
DROP INDEX IF EXISTS idx_messages_session_id;

-- Drop existing functions
DROP FUNCTION IF EXISTS match_chunks(vector, integer);
DROP FUNCTION IF EXISTS hybrid_search(vector, text, integer, float);
DROP FUNCTION IF EXISTS get_document_chunks(uuid);
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- Drop existing views
DROP VIEW IF EXISTS document_summaries;

-- =====================================================
-- 3. CREATE CORE TABLES
-- =====================================================

-- Documents table: Stores raw documents and metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create indexes for documents table
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_documents_created_at ON documents (created_at DESC);
CREATE INDEX idx_documents_source ON documents (source);
CREATE INDEX idx_documents_title_trgm ON documents USING GIN (title gin_trgm_ops);

-- Chunks table: Stores document chunks with embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768), -- Standardized to 768 to meet Supabase limits
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}' NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Constraints
    CONSTRAINT chunks_chunk_index_positive CHECK (chunk_index >= 0),
    CONSTRAINT chunks_token_count_positive CHECK (token_count IS NULL OR token_count > 0)
);

-- Create indexes for chunks table
-- IVFFlat index for vector similarity search (adjust lists based on data size)
CREATE INDEX idx_chunks_embedding ON chunks 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100); -- Adjust lists = sqrt(row_count) for optimal performance

CREATE INDEX idx_chunks_document_id ON chunks (document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks (document_id, chunk_index);
CREATE INDEX idx_chunks_content_trgm ON chunks USING GIN (content gin_trgm_ops);
CREATE INDEX idx_chunks_created_at ON chunks (created_at DESC);

-- Sessions table: Manages user sessions and conversations
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT,
    metadata JSONB DEFAULT '{}' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT sessions_expires_after_created CHECK (expires_at IS NULL OR expires_at > created_at)
);

-- Create indexes for sessions table
CREATE INDEX idx_sessions_user_id ON sessions (user_id);
CREATE INDEX idx_sessions_expires_at ON sessions (expires_at);
CREATE INDEX idx_sessions_created_at ON sessions (created_at DESC);

-- Messages table: Stores conversation messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create indexes for messages table
CREATE INDEX idx_messages_session_id ON messages (session_id, created_at);
CREATE INDEX idx_messages_role ON messages (role);

-- =====================================================
-- 4. CREATE DATABASE FUNCTIONS
-- =====================================================

-- Function: Vector similarity search
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(768),
    match_count INT DEFAULT 10,
    similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT,
    chunk_index INTEGER,
    token_count INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.document_id,
        c.content,
        1 - (c.embedding <=> query_embedding) AS similarity,
        c.metadata,
        d.title AS document_title,
        d.source AS document_source,
        c.chunk_index,
        c.token_count
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE 
        c.embedding IS NOT NULL 
        AND (1 - (c.embedding <=> query_embedding)) >= similarity_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Function: Hybrid search combining vector and text search
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(768),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3,
    similarity_threshold FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    vector_similarity FLOAT,
    text_similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT,
    chunk_index INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            1 - (c.embedding <=> query_embedding) AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source,
            c.chunk_index
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            ts_rank_cd(
                to_tsvector('english', c.content), 
                plainto_tsquery('english', query_text)
            ) AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source,
            c.chunk_index
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE 
            query_text IS NOT NULL 
            AND query_text != ''
            AND to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT 
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (COALESCE(v.vector_sim, 0) * (1 - text_weight) + COALESCE(t.text_sim, 0) * text_weight) AS combined_score,
        COALESCE(v.vector_sim, 0) AS vector_similarity,
        COALESCE(t.text_sim, 0) AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source,
        COALESCE(v.chunk_index, t.chunk_index) AS chunk_index
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    WHERE (COALESCE(v.vector_sim, 0) * (1 - text_weight) + COALESCE(t.text_sim, 0) * text_weight) >= similarity_threshold
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Function: Get all chunks for a specific document
CREATE OR REPLACE FUNCTION get_document_chunks(doc_id UUID)
RETURNS TABLE (
    chunk_id UUID,
    content TEXT,
    chunk_index INTEGER,
    metadata JSONB,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id AS chunk_id,
        c.content,
        c.chunk_index,
        c.metadata,
        c.token_count,
        c.created_at
    FROM chunks c
    WHERE c.document_id = doc_id
    ORDER BY c.chunk_index;
END;
$$;

-- Function: Automatic timestamp update trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Clean expired sessions
CREATE OR REPLACE FUNCTION clean_expired_sessions()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sessions 
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- =====================================================
-- 5. CREATE TRIGGERS
-- =====================================================

-- Trigger to auto-update updated_at on documents
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger to auto-update updated_at on sessions
CREATE TRIGGER update_sessions_updated_at 
    BEFORE UPDATE ON sessions
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 6. CREATE VIEWS
-- =====================================================

-- View: Document summaries with statistics
CREATE OR REPLACE VIEW document_summaries AS
SELECT 
    d.id,
    d.title,
    d.source,
    d.created_at,
    d.updated_at,
    d.metadata,
    COUNT(c.id) AS chunk_count,
    COALESCE(AVG(c.token_count), 0) AS avg_tokens_per_chunk,
    COALESCE(SUM(c.token_count), 0) AS total_tokens,
    COUNT(CASE WHEN c.embedding IS NOT NULL THEN 1 END) AS chunks_with_embeddings,
    ROUND(
        COUNT(CASE WHEN c.embedding IS NOT NULL THEN 1 END)::NUMERIC / 
        GREATEST(COUNT(c.id), 1) * 100, 2
    ) AS embedding_completion_percentage
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
GROUP BY d.id, d.title, d.source, d.created_at, d.updated_at, d.metadata;

-- View: Session summaries with message counts
CREATE OR REPLACE VIEW session_summaries AS
SELECT 
    s.id,
    s.user_id,
    s.created_at,
    s.updated_at,
    s.expires_at,
    s.metadata,
    COUNT(m.id) AS message_count,
    MAX(m.created_at) AS last_message_at,
    CASE 
        WHEN s.expires_at IS NULL THEN false
        WHEN s.expires_at < CURRENT_TIMESTAMP THEN true
        ELSE false
    END AS is_expired
FROM sessions s
LEFT JOIN messages m ON s.id = m.session_id
GROUP BY s.id, s.user_id, s.created_at, s.updated_at, s.expires_at, s.metadata;

-- =====================================================
-- 7. GRANT PERMISSIONS (For Supabase RLS)
-- =====================================================

-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- Grant permissions on tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO anon;

-- Grant permissions on sequences (for UUID generation)
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;

-- Grant execute permissions on functions
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;

-- =====================================================
-- 8. ENABLE ROW LEVEL SECURITY (Optional - for multi-tenant)
-- =====================================================

-- Enable RLS on all tables (uncomment if needed for multi-tenant setup)
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Example RLS policies (uncomment and modify as needed)
-- CREATE POLICY "Users can view their own sessions" ON sessions
--     FOR ALL USING (auth.uid()::text = user_id);

-- CREATE POLICY "Users can view messages from their sessions" ON messages
--     FOR ALL USING (session_id IN (
--         SELECT id FROM sessions WHERE user_id = auth.uid()::text
--     ));

-- =====================================================
-- 9. CREATE HELPFUL UTILITY FUNCTIONS
-- =====================================================

-- Function: Get database statistics
CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE (
    total_documents BIGINT,
    total_chunks BIGINT,
    total_sessions BIGINT,
    total_messages BIGINT,
    chunks_with_embeddings BIGINT,
    avg_chunks_per_document NUMERIC,
    database_size TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM documents),
        (SELECT COUNT(*) FROM chunks),
        (SELECT COUNT(*) FROM sessions),
        (SELECT COUNT(*) FROM messages),
        (SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL),
        (SELECT ROUND(COUNT(c.*)::NUMERIC / GREATEST(COUNT(DISTINCT c.document_id), 1), 2) 
         FROM chunks c),
        pg_size_pretty(pg_database_size(current_database()));
END;
$$;

-- Function: Search documents by title or content
CREATE OR REPLACE FUNCTION search_documents(
    search_query TEXT,
    limit_count INT DEFAULT 10
)
RETURNS TABLE (
    document_id UUID,
    title TEXT,
    source TEXT,
    content_preview TEXT,
    relevance_score REAL
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id AS document_id,
        d.title,
        d.source,
        LEFT(d.content, 500) AS content_preview,
        ts_rank_cd(
            to_tsvector('english', d.title || ' ' || d.content),
            plainto_tsquery('english', search_query)
        ) AS relevance_score
    FROM documents d
    WHERE 
        to_tsvector('english', d.title || ' ' || d.content) @@ plainto_tsquery('english', search_query)
    ORDER BY relevance_score DESC
    LIMIT limit_count;
END;
$$;

-- =====================================================
-- SETUP COMPLETE
-- =====================================================

-- Insert a test document to verify setup
INSERT INTO documents (title, source, content, metadata) VALUES 
(
    'Database Setup Test',
    'system',
    'This is a test document to verify that the Medical RAG database schema has been set up correctly. It includes vector search capabilities, full-text search, and session management.',
    '{"type": "test", "version": "1.0", "setup_date": "2025-01-22"}'
);

-- Display setup completion message
DO $$
BEGIN
    RAISE NOTICE '=== Medical RAG Database Setup Complete ===';
    RAISE NOTICE 'Extensions enabled: vector, uuid-ossp, pg_trgm';
    RAISE NOTICE 'Tables created: documents, chunks, sessions, messages';
    RAISE NOTICE 'Functions created: match_chunks, hybrid_search, get_document_chunks, and utilities';
    RAISE NOTICE 'Views created: document_summaries, session_summaries';
    RAISE NOTICE 'Triggers created: automatic timestamp updates';
    RAISE NOTICE 'Test document inserted successfully';
    RAISE NOTICE '================================================';
END $$;