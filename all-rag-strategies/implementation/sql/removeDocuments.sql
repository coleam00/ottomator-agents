-- Remove all documents and chunks from the database
-- This script clears all data without dropping the schema/tables
-- Useful for resetting the knowledge base before re-ingesting documents

-- ==============================================================================
-- REMOVE ALL DOCUMENTS AND CHUNKS
-- ==============================================================================

-- Delete all chunks first (foreign key constraint)
-- This will automatically cascade if you have ON DELETE CASCADE
DELETE FROM chunks;

-- Delete all documents
DELETE FROM documents;

-- ==============================================================================
-- RESET SEQUENCES (Optional - ensures clean IDs if you were using sequences)
-- ==============================================================================
-- Note: We're using UUID, so no sequences to reset
-- If you were using SERIAL IDs, you would reset sequences here

-- ==============================================================================
-- VACUUM (Optional - reclaim space and update statistics)
-- ==============================================================================
-- Uncomment the lines below if you want to reclaim disk space and update stats
-- This is recommended after deleting large amounts of data

-- VACUUM ANALYZE chunks;
-- VACUUM ANALYZE documents;

-- ==============================================================================
-- VERIFICATION
-- ==============================================================================
-- Check that tables are empty
SELECT 'Documents remaining:' as status, COUNT(*) as count FROM documents
UNION ALL
SELECT 'Chunks remaining:' as status, COUNT(*) as count FROM chunks;

-- ==============================================================================
-- SUMMARY
-- ==============================================================================
-- This script has:
-- 1. Deleted all chunks
-- 2. Deleted all documents
-- 3. Verified tables are empty
--
-- The schema (tables, indexes, functions) remains intact.
-- You can now re-run the ingestion pipeline to add new documents.
--
-- Usage:
--   psql $DATABASE_URL < sql/removeDocuments.sql
--
--   Or from psql:
--   \i sql/removeDocuments.sql
-- ==============================================================================
