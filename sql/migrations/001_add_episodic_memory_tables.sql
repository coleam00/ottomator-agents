-- Migration: 001_add_episodic_memory_tables.sql
-- Description: Add episodic memory tables for Graphiti conversation tracking
-- Author: MaryPause AI Team
-- Date: 2025

BEGIN;

-- Enable required extensions if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Episodes table: Core episodic memory storage
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id TEXT UNIQUE NOT NULL, -- Graphiti episode ID
    user_id TEXT, -- User who created the episode
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    
    -- Episode content
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Episode metadata
    metadata JSONB DEFAULT '{}',
    entities JSONB DEFAULT '[]', -- Extracted entities
    relationships JSONB DEFAULT '[]', -- Extracted relationships
    
    -- Vector embedding for episode content (3072 dimensions for Gemini)
    embedding vector(3072),
    
    -- Episode scoring and importance
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ, -- For memory decay/cleanup
    
    -- Status tracking
    status TEXT CHECK (status IN ('active', 'archived', 'expired', 'deleted')) DEFAULT 'active',
    processing_status TEXT CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')) DEFAULT 'completed'
);

-- Episode references: Links episodes to documents/chunks
CREATE TABLE IF NOT EXISTS episode_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    reference_type TEXT CHECK (reference_type IN ('source', 'derived', 'related')) DEFAULT 'source',
    relevance_score FLOAT DEFAULT 1.0 CHECK (relevance_score >= 0 AND relevance_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(episode_id, document_id, chunk_id)
);

-- Episode relationships: Temporal and causal relationships between episodes
CREATE TABLE IF NOT EXISTS episode_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    target_episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL CHECK (relationship_type IN (
        'temporal_before', 'temporal_after', 'caused_by', 'leads_to', 
        'similar_to', 'contradicts', 'supports', 'references'
    )),
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(source_episode_id, target_episode_id, relationship_type)
);

-- Episode facts: Extracted facts from episodes
CREATE TABLE IF NOT EXISTS episode_facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    fact_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Episode context: Contextual information for episodes
CREATE TABLE IF NOT EXISTS episode_context (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    context_type TEXT NOT NULL CHECK (context_type IN ('temporal', 'spatial', 'topical', 'emotional', 'user')),
    context_value JSONB NOT NULL,
    relevance_score FLOAT DEFAULT 0.5 CHECK (relevance_score >= 0 AND relevance_score <= 1),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(episode_id, context_type)
);

-- Performance indexes for episodes
CREATE INDEX IF NOT EXISTS idx_episodes_episode_id ON episodes(episode_id);
CREATE INDEX IF NOT EXISTS idx_episodes_user_id ON episodes(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance_score DESC) WHERE status = 'active';

-- Vector similarity search index (using IVFFlat)
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_episodes_content_trgm ON episodes USING GIN (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_episodes_source_trgm ON episodes USING GIN (source gin_trgm_ops);

-- JSONB indexes for metadata and entities
CREATE INDEX IF NOT EXISTS idx_episodes_metadata ON episodes USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_episodes_entities ON episodes USING GIN (entities);
CREATE INDEX IF NOT EXISTS idx_episodes_relationships ON episodes USING GIN (relationships);

-- Reference indexes
CREATE INDEX IF NOT EXISTS idx_episode_references_episode_id ON episode_references(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_references_document_id ON episode_references(document_id) WHERE document_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_episode_references_chunk_id ON episode_references(chunk_id) WHERE chunk_id IS NOT NULL;

-- Relationship indexes
CREATE INDEX IF NOT EXISTS idx_episode_relationships_source ON episode_relationships(source_episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_relationships_target ON episode_relationships(target_episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_relationships_type ON episode_relationships(relationship_type);

-- Fact indexes
CREATE INDEX IF NOT EXISTS idx_episode_facts_episode_id ON episode_facts(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_facts_subject ON episode_facts(subject);
CREATE INDEX IF NOT EXISTS idx_episode_facts_predicate ON episode_facts(predicate);
CREATE INDEX IF NOT EXISTS idx_episode_facts_temporal ON episode_facts(valid_from, valid_until);

-- Context indexes
CREATE INDEX IF NOT EXISTS idx_episode_context_episode_id ON episode_context(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_context_type ON episode_context(context_type);

-- Function to calculate episode importance score
CREATE OR REPLACE FUNCTION calculate_episode_importance(
    p_episode_id UUID
) RETURNS FLOAT AS $$
DECLARE
    v_score FLOAT := 0.5;
    v_entity_count INTEGER;
    v_relationship_count INTEGER;
    v_reference_count INTEGER;
    v_fact_count INTEGER;
    v_access_frequency FLOAT;
    v_recency_factor FLOAT;
BEGIN
    -- Get entity and relationship counts
    SELECT 
        jsonb_array_length(entities),
        jsonb_array_length(relationships)
    INTO v_entity_count, v_relationship_count
    FROM episodes
    WHERE id = p_episode_id;
    
    -- Get reference count
    SELECT COUNT(*) INTO v_reference_count
    FROM episode_references
    WHERE episode_id = p_episode_id;
    
    -- Get fact count
    SELECT COUNT(*) INTO v_fact_count
    FROM episode_facts
    WHERE episode_id = p_episode_id;
    
    -- Calculate access frequency (accesses per day since creation)
    SELECT 
        CASE 
            WHEN created_at < NOW() - INTERVAL '1 day' THEN
                access_count::FLOAT / EXTRACT(EPOCH FROM (NOW() - created_at)) * 86400
            ELSE access_count::FLOAT
        END
    INTO v_access_frequency
    FROM episodes
    WHERE id = p_episode_id;
    
    -- Calculate recency factor (exponential decay)
    SELECT 
        EXP(-EXTRACT(EPOCH FROM (NOW() - COALESCE(last_accessed_at, created_at))) / (86400 * 7))
    INTO v_recency_factor
    FROM episodes
    WHERE id = p_episode_id;
    
    -- Composite importance score
    v_score := (
        0.2 * LEAST(COALESCE(v_entity_count, 0) / 10.0, 1.0) +
        0.2 * LEAST(COALESCE(v_relationship_count, 0) / 5.0, 1.0) +
        0.15 * LEAST(COALESCE(v_reference_count, 0) / 3.0, 1.0) +
        0.15 * LEAST(COALESCE(v_fact_count, 0) / 5.0, 1.0) +
        0.15 * LEAST(COALESCE(v_access_frequency, 0) / 10.0, 1.0) +
        0.15 * COALESCE(v_recency_factor, 0.5)
    );
    
    RETURN LEAST(GREATEST(v_score, 0.0), 1.0);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to update episode timestamps
CREATE OR REPLACE FUNCTION update_episode_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE TRIGGER update_episodes_timestamp
    BEFORE UPDATE ON episodes
    FOR EACH ROW
    EXECUTE FUNCTION update_episode_timestamp();

-- Function to automatically create episodes from messages
CREATE OR REPLACE FUNCTION create_interaction_episode()
RETURNS TRIGGER AS $$
DECLARE
    v_episode_id UUID;
    v_content TEXT;
    v_user_id TEXT;
BEGIN
    -- Only create episodes for user and assistant messages
    IF NEW.role NOT IN ('user', 'assistant') THEN
        RETURN NEW;
    END IF;
    
    -- Get user_id from session
    SELECT user_id INTO v_user_id FROM sessions WHERE id = NEW.session_id;
    
    -- Create episode content
    v_content := json_build_object(
        'role', NEW.role,
        'content', NEW.content,
        'session_id', NEW.session_id::TEXT,
        'message_id', NEW.id::TEXT
    )::TEXT;
    
    -- Insert episode
    INSERT INTO episodes (
        episode_id,
        user_id,
        session_id,
        content,
        source,
        timestamp,
        metadata,
        status
    ) VALUES (
        'msg_' || NEW.id::TEXT,
        v_user_id,
        NEW.session_id,
        v_content,
        'message_' || NEW.role,
        NEW.created_at,
        json_build_object(
            'message_id', NEW.id,
            'role', NEW.role,
            'auto_generated', true
        ),
        'active'
    ) RETURNING id INTO v_episode_id;
    
    -- Update message metadata with episode reference
    NEW.metadata = jsonb_set(
        COALESCE(NEW.metadata, '{}'::jsonb),
        '{episode_id}',
        to_jsonb(v_episode_id)
    );
    
    RETURN NEW;
EXCEPTION
    WHEN OTHERS THEN
        -- Log error but don't fail the message insertion
        RAISE WARNING 'Failed to create episode for message %: %', NEW.id, SQLERRM;
        RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Optional: Enable trigger for automatic episode creation
-- Uncomment to automatically create episodes from messages
-- CREATE TRIGGER create_episode_on_message
--     BEFORE INSERT ON messages
--     FOR EACH ROW
--     WHEN (NEW.role IN ('user', 'assistant'))
--     EXECUTE FUNCTION create_interaction_episode();

-- Migrate existing session data to episodes (optional, run once)
-- This preserves historical data in the knowledge graph
INSERT INTO episodes (
    episode_id,
    user_id,
    session_id,
    content,
    source,
    timestamp,
    metadata,
    status
)
SELECT 
    'session_migration_' || s.id::TEXT,
    s.user_id,
    s.id,
    json_build_object(
        'session_metadata', s.metadata,
        'created_at', s.created_at,
        'expires_at', s.expires_at
    )::TEXT,
    'session_migration',
    s.created_at,
    s.metadata,
    'active'
FROM sessions s
WHERE NOT EXISTS (
    SELECT 1 FROM episodes e 
    WHERE e.episode_id = 'session_migration_' || s.id::TEXT
)
ON CONFLICT (episode_id) DO NOTHING;

-- Migrate existing messages to episodes (optional, run once)
INSERT INTO episodes (
    episode_id,
    user_id,
    session_id,
    content,
    source,
    timestamp,
    metadata,
    status
)
SELECT 
    'msg_migration_' || m.id::TEXT,
    s.user_id,
    m.session_id,
    m.content,
    'message_migration_' || m.role,
    m.created_at,
    json_build_object(
        'original_message_id', m.id,
        'role', m.role,
        'original_metadata', m.metadata,
        'migrated_at', CURRENT_TIMESTAMP
    ),
    'active'
FROM messages m
JOIN sessions s ON s.id = m.session_id
WHERE m.role IN ('user', 'assistant')
AND NOT EXISTS (
    SELECT 1 FROM episodes e 
    WHERE e.episode_id = 'msg_migration_' || m.id::TEXT
)
ON CONFLICT (episode_id) DO NOTHING;

COMMIT;