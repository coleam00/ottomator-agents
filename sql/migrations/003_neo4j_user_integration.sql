-- Migration: 003_neo4j_user_integration.sql
-- Description: Add Neo4j user registration tracking and automatic sync
-- Author: MaryPause AI Team
-- Date: 2025

BEGIN;

-- =====================================================
-- 1. Create tracking table for Neo4j user registration
-- =====================================================

CREATE TABLE IF NOT EXISTS neo4j_users (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    registration_status TEXT CHECK (registration_status IN ('pending', 'registered', 'failed', 'retry')) DEFAULT 'pending',
    registered_at TIMESTAMPTZ,
    last_attempt TIMESTAMPTZ,
    attempt_count INTEGER DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_neo4j_users_status ON neo4j_users(registration_status);
CREATE INDEX IF NOT EXISTS idx_neo4j_users_registered_at ON neo4j_users(registered_at);

-- =====================================================
-- 2. Create function to trigger Neo4j user registration
-- =====================================================

CREATE OR REPLACE FUNCTION register_user_in_neo4j()
RETURNS TRIGGER AS $$
DECLARE
    v_user_id UUID;
BEGIN
    -- Get the user ID from the new user record
    v_user_id := NEW.id;
    
    -- Insert tracking record with pending status
    INSERT INTO neo4j_users (
        user_id,
        registration_status,
        last_attempt,
        attempt_count
    ) VALUES (
        v_user_id,
        'pending',
        CURRENT_TIMESTAMP,
        1
    ) ON CONFLICT (user_id) DO NOTHING;
    
    -- Note: The actual HTTP call to the backend API will be handled by a Supabase Edge Function
    -- This trigger just creates the tracking record and marks it as pending
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 3. Create trigger for new user registration
-- =====================================================

-- Drop existing trigger if exists
DROP TRIGGER IF EXISTS trigger_register_neo4j_user ON auth.users;

-- Create trigger that fires when a new user is created
CREATE TRIGGER trigger_register_neo4j_user
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION register_user_in_neo4j();

-- =====================================================
-- 4. Create function to update Neo4j registration status
-- =====================================================

CREATE OR REPLACE FUNCTION update_neo4j_registration_status(
    p_user_id UUID,
    p_status TEXT,
    p_error TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    UPDATE neo4j_users
    SET 
        registration_status = p_status,
        registered_at = CASE 
            WHEN p_status = 'registered' THEN CURRENT_TIMESTAMP 
            ELSE registered_at 
        END,
        last_attempt = CURRENT_TIMESTAMP,
        attempt_count = attempt_count + 1,
        last_error = p_error,
        updated_at = CURRENT_TIMESTAMP
    WHERE user_id = p_user_id;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 5. Create function to get pending registrations
-- =====================================================

CREATE OR REPLACE FUNCTION get_pending_neo4j_registrations()
RETURNS TABLE (
    user_id UUID,
    attempt_count INTEGER,
    last_attempt TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        nu.user_id,
        nu.attempt_count,
        nu.last_attempt
    FROM neo4j_users nu
    WHERE nu.registration_status IN ('pending', 'retry')
        AND nu.attempt_count < 5  -- Max 5 attempts
        AND (
            nu.last_attempt IS NULL 
            OR nu.last_attempt < CURRENT_TIMESTAMP - INTERVAL '5 minutes' * nu.attempt_count  -- Exponential backoff
        )
    ORDER BY nu.last_attempt ASC NULLS FIRST
    LIMIT 10;  -- Process max 10 at a time
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 6. Create function to call from Edge Function
-- =====================================================

CREATE OR REPLACE FUNCTION process_neo4j_registration(p_user_id UUID)
RETURNS JSONB AS $$
DECLARE
    v_result JSONB;
    v_backend_url TEXT;
BEGIN
    -- Get the backend URL from environment or use default
    v_backend_url := current_setting('app.backend_url', true);
    IF v_backend_url IS NULL THEN
        v_backend_url := 'http://localhost:8058';  -- Default for local development
    END IF;
    
    -- Return the data needed for the Edge Function to make the API call
    v_result := jsonb_build_object(
        'user_id', p_user_id,
        'backend_url', v_backend_url,
        'endpoint', '/api/users/register-neo4j',
        'method', 'POST'
    );
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 7. Handle existing users (optional backfill)
-- =====================================================

-- Insert records for existing users who don't have Neo4j registration
INSERT INTO neo4j_users (user_id, registration_status, created_at)
SELECT 
    u.id,
    'pending',
    CURRENT_TIMESTAMP
FROM auth.users u
LEFT JOIN neo4j_users nu ON u.id = nu.user_id
WHERE nu.user_id IS NULL
ON CONFLICT (user_id) DO NOTHING;

-- =====================================================
-- 8. Add RLS policies for neo4j_users table
-- =====================================================

-- Enable RLS
ALTER TABLE neo4j_users ENABLE ROW LEVEL SECURITY;

-- Service role can do everything
CREATE POLICY "Service role can manage neo4j_users" ON neo4j_users
    FOR ALL
    USING (auth.role() = 'service_role');

-- Users can view their own registration status
CREATE POLICY "Users can view own neo4j registration" ON neo4j_users
    FOR SELECT
    USING (auth.uid() = user_id);

-- =====================================================
-- 9. Create updated_at trigger
-- =====================================================

CREATE OR REPLACE FUNCTION update_neo4j_users_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_neo4j_users_updated_at
    BEFORE UPDATE ON neo4j_users
    FOR EACH ROW
    EXECUTE FUNCTION update_neo4j_users_updated_at();

-- =====================================================
-- 10. Grant necessary permissions
-- =====================================================

-- Grant permissions to authenticated users
GRANT SELECT ON neo4j_users TO authenticated;
GRANT EXECUTE ON FUNCTION get_pending_neo4j_registrations() TO authenticated;
GRANT EXECUTE ON FUNCTION process_neo4j_registration(UUID) TO authenticated;

-- Grant permissions to service role
GRANT ALL ON neo4j_users TO service_role;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- =====================================================
-- Comments for documentation
-- =====================================================

COMMENT ON TABLE neo4j_users IS 'Tracks Neo4j/Graphiti user registration status for knowledge graph isolation';
COMMENT ON COLUMN neo4j_users.user_id IS 'Supabase user UUID used as Neo4j user_id for isolation';
COMMENT ON COLUMN neo4j_users.registration_status IS 'Current registration status: pending, registered, failed, retry';
COMMENT ON COLUMN neo4j_users.attempt_count IS 'Number of registration attempts made';
COMMENT ON FUNCTION register_user_in_neo4j() IS 'Trigger function to initiate Neo4j user registration';
COMMENT ON FUNCTION update_neo4j_registration_status(UUID, TEXT, TEXT) IS 'Update Neo4j registration status after attempt';
COMMENT ON FUNCTION get_pending_neo4j_registrations() IS 'Get users pending Neo4j registration with exponential backoff';
COMMENT ON FUNCTION process_neo4j_registration(UUID) IS 'Get data needed for Edge Function to call backend API';

COMMIT;

-- =====================================================
-- Rollback Script (save separately)
-- =====================================================
-- BEGIN;
-- DROP TRIGGER IF EXISTS trigger_register_neo4j_user ON auth.users;
-- DROP TRIGGER IF EXISTS update_neo4j_users_updated_at ON neo4j_users;
-- DROP FUNCTION IF EXISTS register_user_in_neo4j() CASCADE;
-- DROP FUNCTION IF EXISTS update_neo4j_registration_status(UUID, TEXT, TEXT) CASCADE;
-- DROP FUNCTION IF EXISTS get_pending_neo4j_registrations() CASCADE;
-- DROP FUNCTION IF EXISTS process_neo4j_registration(UUID) CASCADE;
-- DROP FUNCTION IF EXISTS update_neo4j_users_updated_at() CASCADE;
-- DROP TABLE IF EXISTS neo4j_users CASCADE;
-- COMMIT;