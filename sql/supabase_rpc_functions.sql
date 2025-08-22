-- Additional RPC functions for Supabase API operations
-- These functions should be executed in your Supabase SQL editor

-- Function to list documents with chunk count (replaces complex JOIN in REST API)
CREATE OR REPLACE FUNCTION list_documents_with_chunk_count(
    doc_limit INT DEFAULT 100,
    doc_offset INT DEFAULT 0,
    metadata_filter JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    source TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    chunk_count BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.source,
        d.metadata,
        d.created_at,
        d.updated_at,
        COUNT(c.id) AS chunk_count
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    WHERE (metadata_filter = '{}'::jsonb OR d.metadata @> metadata_filter)
    GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
    ORDER BY d.created_at DESC
    LIMIT doc_limit OFFSET doc_offset;
END;
$$;

-- Function to search documents by metadata with full-text search
CREATE OR REPLACE FUNCTION search_documents(
    search_text TEXT DEFAULT '',
    metadata_filter JSONB DEFAULT '{}'::jsonb,
    doc_limit INT DEFAULT 50
)
RETURNS TABLE (
    id UUID,
    title TEXT,
    source TEXT,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    chunk_count BIGINT,
    search_rank FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.id,
        d.title,
        d.source,
        d.content,
        d.metadata,
        d.created_at,
        COUNT(c.id) AS chunk_count,
        CASE 
            WHEN search_text = '' THEN 0.0
            ELSE ts_rank_cd(to_tsvector('english', d.title || ' ' || d.content), plainto_tsquery('english', search_text))
        END AS search_rank
    FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    WHERE (
        search_text = '' OR 
        to_tsvector('english', d.title || ' ' || d.content) @@ plainto_tsquery('english', search_text)
    )
    AND (metadata_filter = '{}'::jsonb OR d.metadata @> metadata_filter)
    GROUP BY d.id, d.title, d.source, d.content, d.metadata, d.created_at
    ORDER BY search_rank DESC, d.created_at DESC
    LIMIT doc_limit;
END;
$$;

-- Function to get session statistics
CREATE OR REPLACE FUNCTION get_session_stats(session_id UUID)
RETURNS TABLE (
    session_id UUID,
    message_count BIGINT,
    first_message TIMESTAMP WITH TIME ZONE,
    last_message TIMESTAMP WITH TIME ZONE,
    total_characters BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        s.id AS session_id,
        COUNT(m.id) AS message_count,
        MIN(m.created_at) AS first_message,
        MAX(m.created_at) AS last_message,
        SUM(LENGTH(m.content)) AS total_characters
    FROM sessions s
    LEFT JOIN messages m ON s.id = m.session_id
    WHERE s.id = get_session_stats.session_id
    GROUP BY s.id;
END;
$$;

-- Function to clean up expired sessions and their messages
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS TABLE (
    deleted_sessions INT,
    deleted_messages INT
)
LANGUAGE plpgsql
AS $$
DECLARE
    session_count INT;
    message_count INT;
BEGIN
    -- Count messages that will be deleted
    SELECT COUNT(*) INTO message_count
    FROM messages m
    JOIN sessions s ON m.session_id = s.id
    WHERE s.expires_at IS NOT NULL AND s.expires_at < CURRENT_TIMESTAMP;
    
    -- Count sessions that will be deleted
    SELECT COUNT(*) INTO session_count
    FROM sessions s
    WHERE s.expires_at IS NOT NULL AND s.expires_at < CURRENT_TIMESTAMP;
    
    -- Delete expired sessions (messages will be cascade deleted)
    DELETE FROM sessions 
    WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
    
    RETURN QUERY SELECT session_count, message_count;
END;
$$;

-- Function to get vector search statistics
CREATE OR REPLACE FUNCTION get_vector_stats()
RETURNS TABLE (
    total_chunks BIGINT,
    chunks_with_embeddings BIGINT,
    avg_embedding_similarity FLOAT,
    embedding_dimensions INT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) AS total_chunks,
        COUNT(embedding) AS chunks_with_embeddings,
        0.0 AS avg_embedding_similarity, -- Placeholder - would need specific calculation
        CASE 
            WHEN COUNT(embedding) > 0 THEN array_length(string_to_array(substring(embedding::text, 2, length(embedding::text)-2), ','), 1)
            ELSE 0
        END AS embedding_dimensions
    FROM chunks
    WHERE embedding IS NOT NULL
    LIMIT 1; -- We only need one record for dimensions
END;
$$;

-- Function to batch delete documents and their chunks
CREATE OR REPLACE FUNCTION delete_documents_by_source(source_pattern TEXT)
RETURNS TABLE (
    deleted_documents INT,
    deleted_chunks INT
)
LANGUAGE plpgsql
AS $$
DECLARE
    doc_count INT;
    chunk_count INT;
BEGIN
    -- Count chunks that will be deleted
    SELECT COUNT(*) INTO chunk_count
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE d.source LIKE source_pattern;
    
    -- Count documents that will be deleted
    SELECT COUNT(*) INTO doc_count
    FROM documents d
    WHERE d.source LIKE source_pattern;
    
    -- Delete documents (chunks will be cascade deleted)
    DELETE FROM documents 
    WHERE source LIKE source_pattern;
    
    RETURN QUERY SELECT doc_count, chunk_count;
END;
$$;

-- Function to update document metadata
CREATE OR REPLACE FUNCTION update_document_metadata(
    doc_id UUID,
    new_metadata JSONB
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    updated_count INT;
BEGIN
    UPDATE documents 
    SET metadata = metadata || new_metadata
    WHERE id = doc_id;
    
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count > 0;
END;
$$;

-- Function to get recent activity summary
CREATE OR REPLACE FUNCTION get_recent_activity(days_back INT DEFAULT 7)
RETURNS TABLE (
    activity_date DATE,
    new_documents BIGINT,
    new_chunks BIGINT,
    new_sessions BIGINT,
    new_messages BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH date_range AS (
        SELECT generate_series(
            CURRENT_DATE - (days_back - 1),
            CURRENT_DATE,
            '1 day'::interval
        )::date AS activity_date
    )
    SELECT 
        dr.activity_date,
        COALESCE(doc_counts.new_documents, 0) AS new_documents,
        COALESCE(chunk_counts.new_chunks, 0) AS new_chunks,
        COALESCE(session_counts.new_sessions, 0) AS new_sessions,
        COALESCE(message_counts.new_messages, 0) AS new_messages
    FROM date_range dr
    LEFT JOIN (
        SELECT DATE(created_at) AS date, COUNT(*) AS new_documents
        FROM documents
        WHERE created_at >= CURRENT_DATE - days_back
        GROUP BY DATE(created_at)
    ) doc_counts ON dr.activity_date = doc_counts.date
    LEFT JOIN (
        SELECT DATE(created_at) AS date, COUNT(*) AS new_chunks
        FROM chunks
        WHERE created_at >= CURRENT_DATE - days_back
        GROUP BY DATE(created_at)
    ) chunk_counts ON dr.activity_date = chunk_counts.date
    LEFT JOIN (
        SELECT DATE(created_at) AS date, COUNT(*) AS new_sessions
        FROM sessions
        WHERE created_at >= CURRENT_DATE - days_back
        GROUP BY DATE(created_at)
    ) session_counts ON dr.activity_date = session_counts.date
    LEFT JOIN (
        SELECT DATE(created_at) AS date, COUNT(*) AS new_messages
        FROM messages
        WHERE created_at >= CURRENT_DATE - days_back
        GROUP BY DATE(created_at)
    ) message_counts ON dr.activity_date = message_counts.date
    ORDER BY dr.activity_date DESC;
END;
$$;