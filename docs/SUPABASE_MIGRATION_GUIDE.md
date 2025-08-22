# Supabase Migration Guide

This guide walks you through migrating the Medical RAG agent system from direct PostgreSQL connections to using Supabase's API/SDK.

## Overview

The system now supports **two database providers**:

1. **PostgreSQL (asyncpg)** - Direct database connection (original method)
2. **Supabase** - API-based connection using Supabase client (new method)

You can switch between providers using the `DB_PROVIDER` environment variable.

## Prerequisites

### For Supabase Setup
1. **Supabase Project**: Already created at `https://bpopugzfbokjzgawshov.supabase.co`
2. **Database Schema**: Already applied (tables, functions, indexes)
3. **API Keys**: Available in Supabase dashboard

### For PostgreSQL Setup
1. **Direct Database Access**: PostgreSQL connection string with credentials
2. **pgvector Extension**: Already enabled
3. **Database Schema**: Applied via `sql/schema.sql`

## Configuration

### Environment Variables

Update your `.env` file with the following configuration:

```bash
# Database Provider Selection
# Set to 'supabase' to use Supabase API, 'postgres' for direct connection
DB_PROVIDER=supabase

# Supabase Configuration (for DB_PROVIDER=supabase)
SUPABASE_URL=https://bpopugzfbokjzgawshov.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# PostgreSQL Configuration (for DB_PROVIDER=postgres)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Other configurations remain the same...
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=google
# etc...
```

### API Keys Setup

1. **Get Supabase Keys**:
   - Go to your Supabase project dashboard
   - Navigate to Settings > API
   - Copy the `URL`, `anon key`, and `service_role key`

2. **Key Usage**:
   - **Service Role Key**: Used for admin operations (recommended for server-side)
   - **Anon Key**: Used for client-side operations with RLS policies

## Installation

### 1. Install Dependencies

```bash
# Install Supabase Python client
pip install supabase==2.10.0

# Or update requirements
pip install -r requirements.txt
```

### 2. Database Setup (Supabase)

If you haven't already set up your Supabase database, run these SQL commands in the Supabase SQL editor:

```sql
-- Apply the main schema
-- (Run the contents of sql/schema.sql)

-- Apply additional RPC functions for Supabase
-- (Run the contents of sql/supabase_rpc_functions.sql)
```

**Note**: Your database schema is already set up with:
- ✅ Documents, chunks, sessions, messages tables
- ✅ Vector embeddings (3072 dimensions for Gemini)
- ✅ Custom functions: `match_chunks`, `hybrid_search`, `get_document_chunks`
- ✅ Proper indexes and constraints

## Usage

### 1. Start the System

The system automatically detects the provider based on `DB_PROVIDER`:

```bash
# For Supabase
export DB_PROVIDER=supabase
python -m agent.api

# For PostgreSQL
export DB_PROVIDER=postgres
python -m agent.api
```

### 2. Verify Configuration

Check the health endpoint to verify your setup:

```bash
curl http://localhost:8058/health
```

Expected response with Supabase:
```json
{
  "status": "healthy",
  "database": true,
  "graph_database": true,
  "llm_connection": true,
  "version": "0.1.0",
  "timestamp": "2025-01-22T10:30:00Z",
  "provider": "supabase",
  "stats": {
    "documents": 10,
    "chunks": 150,
    "sessions": 5,
    "messages": 25
  }
}
```

### 3. Provider Information

Get detailed provider information:

```bash
# Check current provider configuration
curl http://localhost:8058/provider/info

# Validate configuration
curl http://localhost:8058/provider/validate

# Get database statistics
curl http://localhost:8058/database/stats
```

## Key Differences: Supabase vs PostgreSQL

### Vector Search Operations

**PostgreSQL (asyncpg)**:
```python
# Direct SQL execution
embedding_str = '[' + ','.join(map(str, embedding)) + ']'
results = await conn.fetch(
    "SELECT * FROM match_chunks($1::vector, $2)",
    embedding_str, limit
)
```

**Supabase**:
```python
# RPC function call
embedding_str = '[' + ','.join(map(str, embedding)) + ']'
response = client.rpc("match_chunks", {
    "query_embedding": embedding_str,
    "match_count": limit
}).execute()
results = response.data
```

### Session Management

**PostgreSQL (asyncpg)**:
```python
# Direct SQL with connection pool
async with db_pool.acquire() as conn:
    result = await conn.fetchrow("INSERT INTO sessions...", ...)
```

**Supabase**:
```python
# REST API call
response = client.table("sessions").insert({...}).execute()
```

### Connection Handling

**PostgreSQL (asyncpg)**:
- Connection pooling with min/max connections
- Explicit connection lifecycle management
- Transaction control

**Supabase**:
- HTTP-based API calls
- Built-in connection management
- Automatic retries and timeouts

## Performance Considerations

### Supabase Advantages
- ✅ No connection pool management
- ✅ Built-in retry logic
- ✅ Automatic scaling
- ✅ Built-in monitoring
- ✅ Row Level Security integration

### Supabase Limitations
- ⚠️ HTTP overhead vs direct connection
- ⚠️ API rate limits (generous for most use cases)
- ⚠️ Less control over query optimization
- ⚠️ Potential latency for complex operations

### Performance Tips
1. **Batch Operations**: Use `bulk_insert_chunks()` for multiple inserts
2. **RPC Functions**: Use custom RPC functions for complex queries
3. **Caching**: Implement application-level caching for frequently accessed data
4. **Connection Reuse**: The Supabase client handles connection reuse automatically

## Troubleshooting

### Common Issues

**1. Authentication Errors**
```
Error: {"code":401,"details":null,"hint":null,"message":"Invalid API key"}
```
- Check that `SUPABASE_SERVICE_ROLE_KEY` is correctly set
- Verify the key has not expired

**2. Vector Dimension Mismatch**
```
Error: vector has wrong dimensions
```
- Ensure embedding dimensions match schema (3072 for Gemini)
- Check `VECTOR_DIMENSION` in your `.env`

**3. RPC Function Not Found**
```
Error: function "match_chunks" does not exist
```
- Run `sql/supabase_rpc_functions.sql` in Supabase SQL editor
- Verify functions are created in the public schema

**4. Connection Timeout**
```
Error: Request timeout
```
- Check your internet connection
- Verify Supabase project is not paused
- Consider increasing timeout in client options

### Debug Mode

Enable debug logging to troubleshoot issues:

```bash
export LOG_LEVEL=DEBUG
python -m agent.api
```

### Health Check Commands

```bash
# Test database connection
curl http://localhost:8058/health

# Validate configuration
curl http://localhost:8058/provider/validate

# Check database stats
curl http://localhost:8058/database/stats
```

## Migration Between Providers

### From PostgreSQL to Supabase

1. **Export Data** (if needed):
   ```bash
   # Export your data from PostgreSQL
   pg_dump $DATABASE_URL > backup.sql
   ```

2. **Update Configuration**:
   ```bash
   # Change provider
   DB_PROVIDER=supabase
   ```

3. **Verify Migration**:
   ```bash
   # Check health and stats
   curl http://localhost:8058/health
   curl http://localhost:8058/database/stats
   ```

### From Supabase to PostgreSQL

1. **Set up PostgreSQL** with the schema from `sql/schema.sql`
2. **Update Configuration**:
   ```bash
   DB_PROVIDER=postgres
   DATABASE_URL=postgresql://user:pass@host:port/db
   ```
3. **Migrate Data** if needed using custom scripts

## API Reference

### Vector Search

```python
# Both providers support the same interface
from agent.unified_db_utils import vector_search

results = await vector_search(
    embedding=[0.1, 0.2, ...],  # 3072 dimensions
    limit=10
)
```

### Hybrid Search

```python
from agent.unified_db_utils import hybrid_search

results = await hybrid_search(
    embedding=[0.1, 0.2, ...],
    query_text="medical condition",
    limit=10,
    text_weight=0.3
)
```

### Document Operations

```python
from agent.unified_db_utils import (
    insert_document,
    get_document,
    list_documents
)

# Insert document
doc_id = await insert_document(
    title="Medical Paper",
    source="pubmed_12345",
    content="Full paper content...",
    metadata={"category": "research"}
)

# Get document
doc = await get_document(doc_id)

# List documents
docs = await list_documents(limit=20, offset=0)
```

## Next Steps

1. **Test Your Setup**: Run the health check endpoints
2. **Run Ingestion**: Process your medical documents
3. **Test Agent**: Use the CLI or API to query your data
4. **Monitor Performance**: Check the stats endpoint regularly
5. **Scale as Needed**: Supabase automatically handles scaling

## Support

For issues specific to:
- **Supabase**: Check [Supabase documentation](https://supabase.com/docs)
- **Vector Operations**: Ensure pgvector is properly configured
- **Agent System**: Check the application logs with `LOG_LEVEL=DEBUG`

The unified database layer ensures your application works seamlessly with either provider!