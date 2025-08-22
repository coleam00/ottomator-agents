-- =====================================================
-- Medical RAG Database Setup Verification Script
-- =====================================================
-- This script verifies that all components have been created successfully

-- =====================================================
-- 1. VERIFY EXTENSIONS
-- =====================================================

SELECT 'Checking Extensions...' as status;

SELECT 
    extname as extension_name,
    extversion as version
FROM pg_extension 
WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')
ORDER BY extname;

-- =====================================================
-- 2. VERIFY TABLES
-- =====================================================

SELECT 'Checking Tables...' as status;

SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE tablename IN ('documents', 'chunks', 'sessions', 'messages')
ORDER BY tablename;

-- =====================================================
-- 3. VERIFY TABLE STRUCTURES
-- =====================================================

SELECT 'Checking Table Structures...' as status;

-- Documents table structure
SELECT 'documents' as table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'documents'
ORDER BY ordinal_position;

-- Chunks table structure  
SELECT 'chunks' as table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'chunks'
ORDER BY ordinal_position;

-- Sessions table structure
SELECT 'sessions' as table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'sessions'
ORDER BY ordinal_position;

-- Messages table structure
SELECT 'messages' as table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'messages'
ORDER BY ordinal_position;

-- =====================================================
-- 4. VERIFY INDEXES
-- =====================================================

SELECT 'Checking Indexes...' as status;

SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename IN ('documents', 'chunks', 'sessions', 'messages')
ORDER BY tablename, indexname;

-- =====================================================
-- 5. VERIFY FUNCTIONS
-- =====================================================

SELECT 'Checking Functions...' as status;

SELECT 
    routine_name,
    routine_type,
    data_type as return_type
FROM information_schema.routines 
WHERE routine_name IN (
    'match_chunks',
    'hybrid_search', 
    'get_document_chunks',
    'update_updated_at_column',
    'clean_expired_sessions',
    'get_database_stats',
    'search_documents'
)
ORDER BY routine_name;

-- =====================================================
-- 6. VERIFY TRIGGERS
-- =====================================================

SELECT 'Checking Triggers...' as status;

SELECT 
    event_object_table as table_name,
    trigger_name,
    event_manipulation as event,
    action_timing as timing
FROM information_schema.triggers 
WHERE event_object_table IN ('documents', 'sessions')
ORDER BY event_object_table, trigger_name;

-- =====================================================
-- 7. VERIFY VIEWS
-- =====================================================

SELECT 'Checking Views...' as status;

SELECT 
    schemaname,
    viewname,
    viewowner
FROM pg_views 
WHERE viewname IN ('document_summaries', 'session_summaries')
ORDER BY viewname;

-- =====================================================
-- 8. VERIFY FOREIGN KEY CONSTRAINTS
-- =====================================================

SELECT 'Checking Foreign Key Constraints...' as status;

SELECT 
    tc.table_name,
    tc.constraint_name,
    tc.constraint_type,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc 
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
    AND ccu.table_schema = tc.table_schema
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_name IN ('chunks', 'messages')
ORDER BY tc.table_name, tc.constraint_name;

-- =====================================================
-- 9. VERIFY CHECK CONSTRAINTS
-- =====================================================

SELECT 'Checking Check Constraints...' as status;

SELECT 
    table_name,
    constraint_name,
    check_clause
FROM information_schema.check_constraints
WHERE constraint_name LIKE '%chunks%' OR constraint_name LIKE '%sessions%' OR constraint_name LIKE '%messages%'
ORDER BY table_name, constraint_name;

-- =====================================================
-- 10. TEST BASIC FUNCTIONALITY
-- =====================================================

SELECT 'Testing Basic Functionality...' as status;

-- Test document insertion and retrieval
SELECT 'Testing document operations...' as test;
SELECT COUNT(*) as document_count FROM documents;

-- Test that test document was inserted
SELECT title, source, created_at 
FROM documents 
WHERE source = 'system' 
LIMIT 1;

-- Test views
SELECT 'Testing views...' as test;
SELECT * FROM document_summaries LIMIT 1;

-- Test utility function
SELECT 'Testing utility functions...' as test;
SELECT * FROM get_database_stats();

-- =====================================================
-- 11. VECTOR SEARCH CAPABILITY TEST
-- =====================================================

SELECT 'Testing Vector Search Capability...' as status;

-- Check if we can create a dummy vector for testing
SELECT 'Testing vector operations...' as test;

-- Test vector dimension and operations (this should work if pgvector is properly installed)
WITH test_vector AS (
    SELECT array_fill(0.1, ARRAY[1536])::vector as test_embedding
)
SELECT 
    'Vector operations working' as status,
    array_length(test_embedding::float[], 1) as dimensions
FROM test_vector;

-- =====================================================
-- VERIFICATION COMPLETE
-- =====================================================

SELECT 'Database setup verification completed successfully!' as final_status;