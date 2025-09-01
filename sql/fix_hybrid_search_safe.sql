-- Safe migration to fix hybrid_search type mismatch
-- This handles cases where multiple versions of the function exist

-- First, check what hybrid_search functions exist
DO $$ 
DECLARE
    func_record RECORD;
BEGIN
    -- Drop all variants of hybrid_search function
    -- We use a loop to handle all possible signatures
    FOR func_record IN 
        SELECT pg_proc.proname, pg_proc.oid::regprocedure as func_signature
        FROM pg_proc 
        WHERE proname = 'hybrid_search'
    LOOP
        EXECUTE 'DROP FUNCTION IF EXISTS ' || func_record.func_signature || ' CASCADE';
    END LOOP;
END $$;

-- Also drop match_chunks variants if they exist
DO $$ 
DECLARE
    func_record RECORD;
BEGIN
    FOR func_record IN 
        SELECT pg_proc.proname, pg_proc.oid::regprocedure as func_signature
        FROM pg_proc 
        WHERE proname = 'match_chunks'
    LOOP
        EXECUTE 'DROP FUNCTION IF EXISTS ' || func_record.func_signature || ' CASCADE';
    END LOOP;
END $$;

-- Now recreate hybrid_search with consistent FLOAT types
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(768),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
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
    document_source TEXT
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
            (1 - (c.embedding <=> query_embedding))::FLOAT AS vector_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
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
            )::FLOAT AS text_sim,
            c.metadata,
            d.title AS doc_title,
            d.source AS doc_source
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT 
        COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
        COALESCE(v.document_id, t.document_id) AS document_id,
        COALESCE(v.content, t.content) AS content,
        (
            COALESCE(v.vector_sim, 0::FLOAT) * (1 - text_weight) + 
            COALESCE(t.text_sim, 0::FLOAT) * text_weight
        )::FLOAT AS combined_score,
        COALESCE(v.vector_sim, 0::FLOAT)::FLOAT AS vector_similarity,
        COALESCE(t.text_sim, 0::FLOAT)::FLOAT AS text_similarity,
        COALESCE(v.metadata, t.metadata) AS metadata,
        COALESCE(v.doc_title, t.doc_title) AS document_title,
        COALESCE(v.doc_source, t.doc_source) AS document_source
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION hybrid_search(vector(768), TEXT, INT, FLOAT) TO anon;
GRANT EXECUTE ON FUNCTION hybrid_search(vector(768), TEXT, INT, FLOAT) TO authenticated;
GRANT EXECUTE ON FUNCTION hybrid_search(vector(768), TEXT, INT, FLOAT) TO service_role;

-- Recreate match_chunks with consistent FLOAT types
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(768),
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
        (1 - (c.embedding <=> query_embedding))::FLOAT AS similarity,
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

-- Grant permissions for match_chunks
GRANT EXECUTE ON FUNCTION match_chunks(vector(768), INT) TO anon;
GRANT EXECUTE ON FUNCTION match_chunks(vector(768), INT) TO authenticated;
GRANT EXECUTE ON FUNCTION match_chunks(vector(768), INT) TO service_role;

-- Add helpful comments
COMMENT ON FUNCTION hybrid_search IS 'Hybrid search combining vector similarity and text search with consistent FLOAT types';
COMMENT ON FUNCTION match_chunks IS 'Vector similarity search with consistent FLOAT types';

-- Verify the functions were created
SELECT 
    proname AS function_name,
    pg_get_function_identity_arguments(oid) AS arguments
FROM pg_proc 
WHERE proname IN ('hybrid_search', 'match_chunks')
ORDER BY proname;