-- Combined Episodic Memory Migration for Medical RAG Agent
-- This file combines migrations 001 and 002 for easier execution in Supabase
-- Fixed: Using 1536 dimensions for OpenAI embeddings (or adjust based on your provider)
-- =========================================================================

-- =========================================================================
-- PART 1: Migration 001 - Add episodic memory tables
-- =========================================================================

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
    
    -- Vector embedding for episode content
    -- Using 1536 dimensions for OpenAI text-embedding-3-small
    -- Change this to match your embedding model:
    -- OpenAI text-embedding-3-small: 1536
    -- Gemini gemini-embedding-001: 3072 (but needs HNSW index instead of IVFFlat)
    -- Ollama nomic-embed-text: 768
    embedding vector(1536),
    
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

-- Vector similarity search index
-- Note: If you're using Gemini with 3072 dimensions, comment out this IVFFlat index
-- and use HNSW instead (if available) or no index for now
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

-- =========================================================================
-- PART 2: Migration 002 - Enhance episodic memory with medical tracking
-- =========================================================================

-- Medical entities table for tracking extracted entities
CREATE TABLE IF NOT EXISTS medical_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL CHECK (entity_type IN (
        'Patient', 'Symptom', 'Condition', 'Treatment', 'Medication', 'TestResult'
    )),
    entity_name TEXT NOT NULL,
    entity_attributes JSONB DEFAULT '{}',
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    extracted_from TEXT CHECK (extracted_from IN ('user_message', 'assistant_response', 'both')),
    context_text TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Index for fast lookups
    UNIQUE(episode_id, entity_type, entity_name)
);

-- Symptom timeline table for tracking symptom progression
CREATE TABLE IF NOT EXISTS symptom_timeline (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id TEXT,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    symptom_name TEXT NOT NULL,
    severity TEXT CHECK (severity IN ('mild', 'moderate', 'severe', 'critical')),
    location TEXT,
    onset_date TIMESTAMPTZ,
    reported_date TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    duration TEXT,
    frequency TEXT,
    triggers TEXT[],
    relieving_factors TEXT[],
    associated_symptoms TEXT[],
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Treatment outcomes table for tracking treatment effectiveness
CREATE TABLE IF NOT EXISTS treatment_outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id TEXT,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    treatment_name TEXT NOT NULL,
    treatment_type TEXT CHECK (treatment_type IN (
        'medication', 'therapy', 'surgery', 'lifestyle', 'alternative'
    )),
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    effectiveness_score FLOAT CHECK (effectiveness_score >= 0 AND effectiveness_score <= 1),
    side_effects TEXT[],
    adherence_level FLOAT CHECK (adherence_level >= 0 AND adherence_level <= 1),
    outcome TEXT CHECK (outcome IN (
        'effective', 'partially_effective', 'ineffective', 'discontinued', 'ongoing'
    )),
    notes TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Episode medical facts table for storing extracted fact triples
CREATE TABLE IF NOT EXISTS episode_medical_facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence_score FLOAT DEFAULT 0.5 CHECK (confidence_score >= 0 AND confidence_score <= 1),
    fact_type TEXT CHECK (fact_type IN (
        'symptom', 'condition', 'treatment', 'temporal', 'causal', 'relationship'
    )),
    source TEXT CHECK (source IN (
        'pattern_extraction', 'llm_extraction', 'user_stated', 'inferred'
    )),
    valid_from TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    valid_until TIMESTAMPTZ,
    contradicts UUID REFERENCES episode_medical_facts(id),
    supports UUID REFERENCES episode_medical_facts(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicate facts
    UNIQUE(episode_id, subject, predicate, object)
);

-- Patient profile table for aggregated patient information
CREATE TABLE IF NOT EXISTS patient_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT UNIQUE NOT NULL,
    age INTEGER,
    gender TEXT,
    medical_history TEXT[],
    current_medications TEXT[],
    allergies TEXT[],
    chronic_conditions TEXT[],
    lifestyle_factors JSONB DEFAULT '{}',
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Memory importance scores table
CREATE TABLE IF NOT EXISTS memory_importance_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID UNIQUE NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    base_score FLOAT DEFAULT 0.5,
    entity_score FLOAT DEFAULT 0.0,
    fact_score FLOAT DEFAULT 0.0,
    temporal_score FLOAT DEFAULT 0.0,
    medical_relevance_score FLOAT DEFAULT 0.0,
    final_score FLOAT GENERATED ALWAYS AS (
        (base_score + entity_score + fact_score + temporal_score + medical_relevance_score) / 5
    ) STORED,
    calculation_metadata JSONB DEFAULT '{}',
    calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for medical entities
CREATE INDEX IF NOT EXISTS idx_medical_entities_episode_id ON medical_entities(episode_id);
CREATE INDEX IF NOT EXISTS idx_medical_entities_entity_type ON medical_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_medical_entities_entity_name ON medical_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_medical_entities_confidence ON medical_entities(confidence_score DESC);

-- Indexes for symptom timeline
CREATE INDEX IF NOT EXISTS idx_symptom_timeline_patient_id ON symptom_timeline(patient_id);
CREATE INDEX IF NOT EXISTS idx_symptom_timeline_session_id ON symptom_timeline(session_id);
CREATE INDEX IF NOT EXISTS idx_symptom_timeline_symptom ON symptom_timeline(symptom_name);
CREATE INDEX IF NOT EXISTS idx_symptom_timeline_severity ON symptom_timeline(severity);
CREATE INDEX IF NOT EXISTS idx_symptom_timeline_reported_date ON symptom_timeline(reported_date DESC);

-- Indexes for treatment outcomes
CREATE INDEX IF NOT EXISTS idx_treatment_outcomes_patient_id ON treatment_outcomes(patient_id);
CREATE INDEX IF NOT EXISTS idx_treatment_outcomes_treatment ON treatment_outcomes(treatment_name);
CREATE INDEX IF NOT EXISTS idx_treatment_outcomes_effectiveness ON treatment_outcomes(effectiveness_score DESC);
CREATE INDEX IF NOT EXISTS idx_treatment_outcomes_outcome ON treatment_outcomes(outcome);

-- Indexes for medical facts
CREATE INDEX IF NOT EXISTS idx_medical_facts_episode_id ON episode_medical_facts(episode_id);
CREATE INDEX IF NOT EXISTS idx_medical_facts_subject ON episode_medical_facts(subject);
CREATE INDEX IF NOT EXISTS idx_medical_facts_predicate ON episode_medical_facts(predicate);
CREATE INDEX IF NOT EXISTS idx_medical_facts_confidence ON episode_medical_facts(confidence_score DESC);
CREATE INDEX IF NOT EXISTS idx_medical_facts_fact_type ON episode_medical_facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_medical_facts_temporal ON episode_medical_facts(valid_from, valid_until);

-- Indexes for patient profiles
CREATE INDEX IF NOT EXISTS idx_patient_profiles_user_id ON patient_profiles(user_id);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_medical_entities_context_trgm ON medical_entities USING GIN (context_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_symptom_timeline_notes_trgm ON symptom_timeline USING GIN (notes gin_trgm_ops);

-- Function to update patient profile from episodes
CREATE OR REPLACE FUNCTION update_patient_profile()
RETURNS TRIGGER AS $$
DECLARE
    v_user_id TEXT;
BEGIN
    -- Get user_id from session
    SELECT s.user_id INTO v_user_id
    FROM sessions s
    WHERE s.id = NEW.session_id;
    
    IF v_user_id IS NOT NULL THEN
        -- Update or insert patient profile
        INSERT INTO patient_profiles (user_id, last_updated)
        VALUES (v_user_id, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id) DO UPDATE
        SET last_updated = CURRENT_TIMESTAMP;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update patient profile on new episodes
CREATE TRIGGER update_patient_profile_on_episode
    AFTER INSERT ON episodes
    FOR EACH ROW
    WHEN (NEW.user_id IS NOT NULL)
    EXECUTE FUNCTION update_patient_profile();

-- Function to calculate and update memory importance
CREATE OR REPLACE FUNCTION calculate_memory_importance()
RETURNS TRIGGER AS $$
DECLARE
    v_entity_count INTEGER;
    v_fact_count INTEGER;
    v_entity_score FLOAT;
    v_fact_score FLOAT;
    v_medical_relevance FLOAT;
BEGIN
    -- Count entities
    SELECT COUNT(*) INTO v_entity_count
    FROM medical_entities
    WHERE episode_id = NEW.id;
    
    -- Count facts
    SELECT COUNT(*) INTO v_fact_count
    FROM episode_medical_facts
    WHERE episode_id = NEW.id;
    
    -- Calculate scores (normalized to 0-1)
    v_entity_score := LEAST(v_entity_count / 10.0, 1.0);
    v_fact_score := LEAST(v_fact_count / 10.0, 1.0);
    
    -- Calculate medical relevance based on entity types
    SELECT 
        CASE 
            WHEN COUNT(DISTINCT entity_type) >= 3 THEN 1.0
            WHEN COUNT(DISTINCT entity_type) = 2 THEN 0.7
            WHEN COUNT(DISTINCT entity_type) = 1 THEN 0.4
            ELSE 0.2
        END INTO v_medical_relevance
    FROM medical_entities
    WHERE episode_id = NEW.id;
    
    -- Insert or update importance score
    INSERT INTO memory_importance_scores (
        episode_id,
        base_score,
        entity_score,
        fact_score,
        medical_relevance_score
    ) VALUES (
        NEW.id,
        0.5,
        v_entity_score,
        v_fact_score,
        v_medical_relevance
    )
    ON CONFLICT (episode_id) DO UPDATE
    SET 
        entity_score = v_entity_score,
        fact_score = v_fact_score,
        medical_relevance_score = v_medical_relevance,
        calculated_at = CURRENT_TIMESTAMP;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to calculate importance after episode updates
CREATE TRIGGER calculate_importance_on_episode
    AFTER INSERT OR UPDATE ON episodes
    FOR EACH ROW
    EXECUTE FUNCTION calculate_memory_importance();

-- View for comprehensive episode information
CREATE OR REPLACE VIEW episode_summary AS
SELECT 
    e.id,
    e.episode_id,
    e.user_id,
    e.session_id,
    e.timestamp,
    e.status,
    e.importance_score,
    COUNT(DISTINCT me.id) as entity_count,
    COUNT(DISTINCT mf.id) as fact_count,
    COUNT(DISTINCT st.id) as symptom_entries,
    mis.final_score as calculated_importance,
    e.metadata
FROM episodes e
LEFT JOIN medical_entities me ON me.episode_id = e.id
LEFT JOIN episode_medical_facts mf ON mf.episode_id = e.id
LEFT JOIN symptom_timeline st ON st.episode_id = e.id
LEFT JOIN memory_importance_scores mis ON mis.episode_id = e.id
GROUP BY e.id, mis.final_score;

-- View for patient medical history
CREATE OR REPLACE VIEW patient_medical_history AS
SELECT 
    pp.user_id,
    pp.age,
    pp.gender,
    pp.medical_history,
    pp.current_medications,
    pp.allergies,
    pp.chronic_conditions,
    COUNT(DISTINCT st.symptom_name) as unique_symptoms,
    COUNT(DISTINCT to2.treatment_name) as treatments_tried,
    MAX(e.timestamp) as last_interaction
FROM patient_profiles pp
LEFT JOIN sessions s ON s.user_id = pp.user_id
LEFT JOIN episodes e ON e.session_id = s.id
LEFT JOIN symptom_timeline st ON st.session_id = s.id
LEFT JOIN treatment_outcomes to2 ON to2.session_id = s.id
GROUP BY pp.user_id, pp.age, pp.gender, pp.medical_history, 
         pp.current_medications, pp.allergies, pp.chronic_conditions;

COMMIT;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Episodic memory migration completed successfully!';
    RAISE NOTICE 'Created 10 new tables, 2 views, and multiple functions/triggers';
    RAISE NOTICE 'Your Medical RAG agent now has enhanced episodic memory capabilities';
END $$;