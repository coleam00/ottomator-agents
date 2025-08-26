-- Migration: 002_enhance_episodic_memory.sql
-- Description: Enhance episodic memory with medical entity tracking and fact storage
-- Author: MaryPause AI Team
-- Date: 2025

BEGIN;

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