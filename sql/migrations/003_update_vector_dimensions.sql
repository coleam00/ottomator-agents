-- Migration: 003_update_vector_dimensions.sql
-- Description: Update vector dimensions from 3072 to 1536 for Supabase IVFFlat compatibility
-- Author: MaryPause AI Team
-- Date: 2025
-- 
-- IMPORTANT: This migration will:
-- 1. Drop the existing vector column
-- 2. Recreate it with 1536 dimensions
-- 3. Drop and recreate vector-related functions
-- 4. Existing embeddings will be lost and need to be regenerated

BEGIN;

-- Drop existing vector index
DROP INDEX IF EXISTS idx_chunks_embedding;
DROP INDEX IF EXISTS idx_episodes_embedding;

-- Drop existing functions that use the old vector dimension
DROP FUNCTION IF EXISTS match_chunks(vector, integer);
DROP FUNCTION IF EXISTS hybrid_search(vector, text, integer, float);

-- Alter chunks table to use new dimension
ALTER TABLE chunks 
DROP COLUMN IF EXISTS embedding;

ALTER TABLE chunks 
ADD COLUMN embedding vector(1536);

-- Alter episodes table to use new dimension (if it exists)
DO $$ 
BEGIN 
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'episodes') THEN
        ALTER TABLE episodes DROP COLUMN IF EXISTS embedding;
        ALTER TABLE episodes ADD COLUMN embedding vector(1536);
    END IF;
END $$;

-- Recreate the match_chunks function with new dimension
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1536),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
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
        d.source AS document_source
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Recreate the hybrid_search function with new dimension
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    combined_score FLOAT,
    metadata JSONB,
    document_title TEXT,
    document_source TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_scores AS (
        SELECT 
            c.id AS chunk_id,
            c.document_id,
            c.content,
            c.metadata,
            (1 - (c.embedding <=> query_embedding)) AS vector_score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    text_scores AS (
        SELECT 
            c.id AS chunk_id,
            similarity(c.content, query_text) AS text_score
        FROM chunks c
        WHERE c.content % query_text
        ORDER BY similarity(c.content, query_text) DESC
        LIMIT match_count * 2
    )
    SELECT 
        vs.chunk_id,
        vs.document_id,
        vs.content,
        ((1 - text_weight) * COALESCE(vs.vector_score, 0) + 
         text_weight * COALESCE(ts.text_score, 0)) AS combined_score,
        vs.metadata,
        d.title AS document_title,
        d.source AS document_source
    FROM vector_scores vs
    LEFT JOIN text_scores ts ON vs.chunk_id = ts.chunk_id
    JOIN documents d ON vs.document_id = d.id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Recreate vector indexes with new dimension
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Create index for episodes if table exists
DO $$ 
BEGIN 
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'episodes') THEN
        CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
    END IF;
END $$;

-- Add a configuration table to track vector dimensions
CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert or update the vector dimension configuration
INSERT INTO system_config (key, value) 
VALUES ('vector_dimension', '1536')
ON CONFLICT (key) DO UPDATE 
SET value = '1536', updated_at = CURRENT_TIMESTAMP;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Vector dimension migration completed successfully!';
    RAISE NOTICE 'Updated from 3072 to 1536 dimensions for Supabase IVFFlat compatibility';
    RAISE NOTICE 'IMPORTANT: You will need to regenerate all embeddings by running the ingestion script again';
END $$;

COMMIT;