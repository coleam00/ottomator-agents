# Medical RAG Database - Supabase Deployment Guide

This guide provides step-by-step instructions for deploying the Medical RAG database schema to Supabase PostgreSQL.

## 🚀 Quick Deployment Options

### Option 1: Automated Python Deployment (Recommended)

```bash
# Install dependencies
pip install asyncpg python-dotenv

# Set environment variables
export DATABASE_URL="postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"

# Run deployment
python sql/deploy_to_supabase.py

# Or verify existing deployment
python sql/deploy_to_supabase.py --verify-only
```

### Option 2: Manual Supabase SQL Editor Deployment

1. **Open Supabase Dashboard**
   - Go to your Supabase project
   - Navigate to SQL Editor

2. **Execute Schema**
   - Copy the contents of `sql/supabase_schema.sql`
   - Paste into SQL Editor
   - Click "Run" to execute

3. **Verify Deployment**
   - Copy the contents of `sql/verify_setup.sql`
   - Paste into SQL Editor
   - Click "Run" to verify

## 📋 Database Schema Overview

### Core Tables

1. **`documents`** - Stores raw documents and metadata
   - Primary key: `id` (UUID)
   - Full-text search on title and content
   - JSONB metadata storage

2. **`chunks`** - Document chunks with vector embeddings
   - Foreign key: `document_id` → `documents(id)`
   - Vector field: `embedding` (3072 dimensions for Gemini)
   - IVFFlat index for fast vector search

3. **`sessions`** - User session management
   - Supports session expiration
   - JSONB metadata for extensibility

4. **`messages`** - Conversation message storage
   - Foreign key: `session_id` → `sessions(id)`
   - Role constraints: 'user', 'assistant', 'system'

### Key Features

- **Vector Search**: pgvector extension with IVFFlat indexing
- **Hybrid Search**: Combines vector similarity + full-text search
- **Automatic Timestamps**: Triggers for `updated_at` fields
- **Performance Optimized**: Strategic indexes for common queries
- **Multi-tenant Ready**: Optional RLS policies included
- **Utility Functions**: Database stats, search helpers, cleanup

## 🔧 Configuration Requirements

### Environment Variables

```bash
# Required: Database connection
DATABASE_URL="postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres"

# Alternative: Individual components
DB_HOST="db.[project-ref].supabase.co"
DB_PORT="5432"
DB_USER="postgres"
DB_PASSWORD="[your-password]"
DB_NAME="postgres"

# Application settings
VECTOR_DIMENSION=3072  # Gemini gemini-embedding-001
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

### Embedding Model Compatibility

The schema is configured for **3072-dimensional vectors** (Gemini gemini-embedding-001).

**To change dimensions:**

1. Update `vector(3072)` to your dimension in:
   - Line 31: `chunks.embedding`
   - Line 67: `match_chunks` function parameter
   - Line 100: `hybrid_search` function parameter

2. Common dimensions:
   - OpenAI text-embedding-3-small: **1536**
   - OpenAI text-embedding-3-large: **3072**
   - Ollama nomic-embed-text: **768**

## 🛡️ Security Features

### Row Level Security (Optional)

The schema includes commented RLS policies for multi-tenant setups:

```sql
-- Enable RLS (uncomment if needed)
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;

-- Example policies (customize as needed)
CREATE POLICY "Users can view their own sessions" ON sessions
    FOR ALL USING (auth.uid()::text = user_id);
```

### Permissions

The schema grants appropriate permissions:
- `authenticated` role: Full CRUD access
- `anon` role: Read-only access
- Function execution permissions for both roles

## 📊 Post-Deployment Verification

### 1. Extension Check
```sql
SELECT extname, extversion 
FROM pg_extension 
WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm');
```

### 2. Table Check
```sql
SELECT tablename FROM pg_tables 
WHERE tablename IN ('documents', 'chunks', 'sessions', 'messages');
```

### 3. Function Check
```sql
SELECT routine_name FROM information_schema.routines 
WHERE routine_name IN ('match_chunks', 'hybrid_search', 'get_document_chunks');
```

### 4. Test Vector Operations
```sql
-- Test vector functionality
WITH test_vector AS (
    SELECT array_fill(0.1, ARRAY[3072])::vector as test_embedding
)
SELECT array_length(test_embedding::float[], 1) as dimensions
FROM test_vector;
```

### 5. Database Statistics
```sql
SELECT * FROM get_database_stats();
```

## 🔍 Available Database Functions

### Search Functions

1. **`match_chunks(query_embedding, match_count, similarity_threshold)`**
   - Vector similarity search
   - Returns chunks with similarity scores

2. **`hybrid_search(query_embedding, query_text, match_count, text_weight, similarity_threshold)`**
   - Combined vector + text search
   - Weighted scoring system

3. **`get_document_chunks(doc_id)`**
   - Retrieve all chunks for a document
   - Ordered by chunk index

4. **`search_documents(search_query, limit_count)`**
   - Full-text search across documents
   - Returns relevance-ranked results

### Utility Functions

1. **`get_database_stats()`**
   - Database statistics and metrics
   - Document/chunk counts, embeddings status

2. **`clean_expired_sessions()`**
   - Cleanup function for expired sessions
   - Returns count of deleted sessions

## 📈 Performance Optimization

### Index Strategy

- **Vector Index**: IVFFlat with 100 lists (optimal for ~10K chunks)
- **Text Index**: GIN trigram for fuzzy text search
- **Composite Index**: (document_id, chunk_index) for chunk retrieval
- **Timestamp Index**: Descending order for recent-first queries

### Scaling Considerations

1. **Vector Index Tuning**:
   ```sql
   -- Adjust lists parameter: approximately sqrt(row_count)
   CREATE INDEX idx_chunks_embedding ON chunks 
   USING ivfflat (embedding vector_cosine_ops) 
   WITH (lists = 316); -- For ~100K chunks
   ```

2. **Connection Pooling**: Use Supabase connection pooler for high concurrency

3. **Query Optimization**: Use `EXPLAIN ANALYZE` to optimize slow queries

## 🧪 Testing the Setup

### 1. Insert Test Document
```sql
INSERT INTO documents (title, source, content, metadata) VALUES 
('Medical Test Document', 'test', 'This document tests medical terminology and concepts.', '{"type": "test"}');
```

### 2. Insert Test Chunk with Embedding
```sql
INSERT INTO chunks (document_id, content, embedding, chunk_index, token_count) 
SELECT 
    id as document_id,
    'Test medical content for vector search',
    array_fill(random(), ARRAY[3072])::vector,
    0,
    10
FROM documents WHERE source = 'test' LIMIT 1;
```

### 3. Test Vector Search
```sql
SELECT * FROM match_chunks(
    (SELECT embedding FROM chunks LIMIT 1),
    5,
    0.0
);
```

## 🔧 Troubleshooting

### Common Issues

1. **Vector Extension Not Found**
   ```
   ERROR: extension "vector" does not exist
   ```
   - Solution: Contact Supabase support to enable pgvector

2. **Dimension Mismatch**
   ```
   ERROR: vector dimension mismatch
   ```
   - Solution: Ensure embedding dimensions match schema definition

3. **Permission Denied**
   ```
   ERROR: permission denied for function
   ```
   - Solution: Check function permissions and RLS policies

4. **Index Creation Fails**
   ```
   ERROR: index method "ivfflat" does not exist
   ```
   - Solution: Ensure pgvector extension is properly installed

### Performance Issues

1. **Slow Vector Search**
   - Check if IVFFlat index exists: `\d+ chunks`
   - Consider adjusting `lists` parameter
   - Ensure `embedding` field is not null

2. **High Memory Usage**
   - Monitor connection count
   - Use Supabase connection pooler
   - Optimize query patterns

## 📝 Next Steps

After successful deployment:

1. **Configure Application**
   - Update `DATABASE_URL` in your application
   - Test database connectivity
   - Run document ingestion

2. **Enable Monitoring**
   - Set up Supabase monitoring
   - Configure query performance tracking
   - Set up alerting for errors

3. **Production Considerations**
   - Enable Row Level Security if needed
   - Set up automated backups
   - Configure connection pooling
   - Monitor database metrics

## 📞 Support

- **Supabase Documentation**: https://supabase.com/docs
- **pgvector Documentation**: https://github.com/pgvector/pgvector
- **PostgreSQL Documentation**: https://www.postgresql.org/docs/

Your Medical RAG database is now ready for production use with Supabase! 🎉