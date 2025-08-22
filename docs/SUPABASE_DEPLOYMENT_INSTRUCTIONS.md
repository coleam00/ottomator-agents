# Supabase Database Schema Deployment Instructions

## Issue Summary

The API server is connecting successfully to Supabase, but the database tables (`documents`, `chunks`, `sessions`, `messages`) do not exist. The error `"Could not find the table 'public.documents' in the schema cache"` indicates that the database schema hasn't been deployed yet.

## Solution Options

### Option 1: Manual Deployment via Supabase SQL Editor (RECOMMENDED)

This is the easiest and most reliable method:

1. **Open Supabase Dashboard**
   - Go to: https://supabase.com/dashboard/project/bpopugzfbokjzgawshov
   - Navigate to SQL Editor

2. **Deploy the Schema**
   - Copy the entire contents of `deploy_supabase_manual.sql`
   - Paste into the Supabase SQL Editor
   - Click "Run" to execute the script

3. **Verify Deployment**
   - The script includes verification queries at the end
   - Look for the success message: "🎉 Medical RAG Database Setup Complete for Supabase! 🎉"

### Option 2: Python Deployment Script (If you have the database password)

If you have the Supabase database password:

1. **Update Environment Variables**
   ```bash
   # Get your database password from Supabase dashboard settings
   # Update this line in .env:
   DATABASE_URL=postgresql://postgres:YOUR_ACTUAL_PASSWORD@db.bpopugzfbokjzgawshov.supabase.co:5432/postgres
   ```

2. **Run Deployment Script**
   ```bash
   source venv/bin/activate
   python sql/deploy_to_supabase.py
   ```

## What the Deployment Script Does

### 1. **Extensions Setup**
- Enables `vector` extension for pgvector support
- Enables `uuid-ossp` for UUID generation
- Enables `pg_trgm` for text search

### 2. **Core Tables Created**
- `documents` - Stores medical documents and metadata
- `chunks` - Document chunks with vector embeddings (768 dimensions for Gemini)
- `sessions` - User session management
- `messages` - Conversation message storage

### 3. **Database Functions**
- `match_chunks()` - Vector similarity search
- `hybrid_search()` - Combined vector + text search
- `get_document_chunks()` - Get chunks for a document
- `list_documents_with_chunk_count()` - List documents with statistics
- Utility functions for database stats and cleanup

### 4. **Indexes and Performance**
- IVFFlat index on embeddings for fast vector search
- GIN indexes for full-text search
- Composite indexes for efficient queries

### 5. **Views and Monitoring**
- `document_summaries` - Document statistics view
- `session_summaries` - Session information view

## Troubleshooting

### Issue: "Could not find the table 'public.documents'"
**Solution**: The schema hasn't been deployed. Use Option 1 or 2 above.

### Issue: "Vector extension not available"
**Solution**: Contact Supabase support to enable pgvector extension (usually enabled by default).

### Issue: RLS Permission Denied
**Solution**: The script grants appropriate permissions. If issues persist, temporarily disable RLS:
```sql
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
ALTER TABLE chunks DISABLE ROW LEVEL SECURITY;
ALTER TABLE sessions DISABLE ROW LEVEL SECURITY;
ALTER TABLE messages DISABLE ROW LEVEL SECURITY;
```

### Issue: Dimension Mismatch
**Solution**: The script is configured for 768 dimensions (Gemini embeddings). If using different embeddings:
1. Update the vector dimensions in the SQL script
2. Update `VECTOR_DIMENSION` in .env file

## Verification Steps

After deployment, verify the setup:

1. **Check Tables Exist**
   ```sql
   SELECT tablename FROM pg_tables WHERE tablename IN ('documents', 'chunks', 'sessions', 'messages');
   ```

2. **Check Functions Exist**
   ```sql
   SELECT routine_name FROM information_schema.routines WHERE routine_name IN ('match_chunks', 'hybrid_search');
   ```

3. **Test Vector Functionality**
   ```sql
   SELECT array_length(array_fill(0.1, ARRAY[768])::vector::float[], 1) as dimensions;
   ```

4. **Check Database Stats**
   ```sql
   SELECT * FROM get_database_stats();
   ```

## API Integration

Once the schema is deployed:

1. **The API will automatically work** - No code changes needed
2. **Run document ingestion** to populate the database:
   ```bash
   python -m ingestion.ingest
   ```
3. **Test API endpoints** - Health check should pass, chat should work

## Next Steps After Deployment

1. **Ingest Medical Documents**
   ```bash
   source venv/bin/activate
   python -m ingestion.ingest --verbose
   ```

2. **Test the System**
   ```bash
   # Terminal 1: API server (should already be running)
   python -m agent.api

   # Terminal 2: Test client
   python cli.py
   ```

3. **Monitor Performance**
   - Use `SELECT * FROM get_database_stats();` to monitor growth
   - Watch for slow queries in Supabase dashboard

## Configuration Notes

- **Embedding Model**: Configured for Gemini gemini-embedding-001 (768 dimensions)
- **Vector Index**: Uses IVFFlat with 100 lists (optimal for ~10K chunks)
- **Session Timeout**: 60 minutes by default
- **Permissions**: Service role has full access, anonymous has read access

The database is now ready for production use with your Medical RAG agent! 🎉