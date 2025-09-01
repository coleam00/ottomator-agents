-- Fix for hybrid_search type mismatch error
-- Error: "Returned type real does not match expected type double precision in column 6"
-- This migration ensures all numeric types are consistent

-- Drop ALL existing hybrid_search functions (handle multiple signatures)
DROP FUNCTION IF EXISTS hybrid_search(vector(768), TEXT, INT, FLOAT);
DROP FUNCTION IF EXISTS hybrid_search(vector(768), TEXT, INT);
DROP FUNCTION IF EXISTS hybrid_search(vector(768), TEXT);
DROP FUNCTION IF EXISTS hybrid_search CASCADE;

-- Recreate with consistent type handling
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
            (1 - (c.embedding <=> query_embedding))::FLOAT AS vector_sim,  -- Cast to FLOAT
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
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::FLOAT AS text_sim,  -- Cast to FLOAT
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
        (COALESCE(v.vector_sim, 0::FLOAT) * (1 - text_weight) + 
         COALESCE(t.text_sim, 0::FLOAT) * text_weight)::FLOAT AS combined_score,  -- Ensure FLOAT
        COALESCE(v.vector_sim, 0::FLOAT)::FLOAT AS vector_similarity,  -- Cast to FLOAT
        COALESCE(t.text_sim, 0::FLOAT)::FLOAT AS text_similarity,  -- Cast to FLOAT
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

-- Also fix match_chunks if it has similar issues
DROP FUNCTION IF EXISTS match_chunks(vector(768), INT);

CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(768),
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    similarity FLOAT,  -- Use FLOAT consistently
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
        (1 - (c.embedding <=> query_embedding))::FLOAT AS similarity,  -- Cast to FLOAT
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

-- Add helpful comment
COMMENT ON FUNCTION hybrid_search IS 'Performs hybrid search combining vector similarity and text search with consistent FLOAT types';
COMMENT ON FUNCTION match_chunks IS 'Performs vector similarity search with consistent FLOAT types';