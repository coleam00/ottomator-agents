# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Medical RAG (Retrieval-Augmented Generation) agent system that combines traditional vector search with knowledge graph capabilities. Built with Pydantic AI, FastAPI, PostgreSQL (with pgvector), and Neo4j (via Graphiti), it provides an intelligent agent that can search across both vector embeddings and knowledge graph relationships to answer questions about medical documents.

## Key Commands

### Development Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Edit with your database credentials
```

### Database Setup

**For Supabase (recommended):**
```bash
# Schema is already applied to your Supabase project
# Just configure your environment variables:
DB_PROVIDER=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Test your configuration
python test_supabase_connection.py
```

**For Direct PostgreSQL:**
```bash
# Execute schema creation (adjust embedding dimensions based on your model)
# OpenAI text-embedding-3-small: 1536 dimensions
# Gemini gemini-embedding-001: 3072 dimensions
# Ollama nomic-embed-text: 768 dimensions
DB_PROVIDER=postgres
psql -d "$DATABASE_URL" -f sql/schema.sql
```

### Running the System
```bash
# 1. First ingest documents (REQUIRED before using the agent)
python -m ingestion.ingest                          # Basic ingestion
python -m ingestion.ingest --clean                  # Clean and re-ingest
python -m ingestion.ingest --no-semantic --verbose  # Fast mode without semantic chunking

# 2. Start the API server (Terminal 1)
python -m agent.api

# 3. Use the CLI interface (Terminal 2)
python cli.py                                       # Connect to default localhost:8058
python cli.py --url http://localhost:8058          # Specify URL
python cli.py --port 8080                          # Specify port only
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=agent --cov=ingestion --cov-report=html

# Run specific test categories
pytest tests/agent/
pytest tests/ingestion/

# Run a single test file
pytest tests/agent/test_models.py

# Run a single test function
pytest tests/agent/test_models.py::test_chat_request_model
```

## Architecture

## Database Provider Support

The system supports **two database providers** that can be switched via environment configuration:

1. **Supabase** (recommended) - API-based connection using Supabase client
   - ✅ No database password required
   - ✅ Built-in connection management and scaling
   - ✅ Integrated monitoring and backups
   - ✅ Row Level Security support

2. **PostgreSQL** (direct) - Direct database connection using asyncpg
   - ✅ Full control over connection pooling
   - ✅ Direct SQL execution for complex queries
   - ✅ Lower latency for high-frequency operations

**Switch providers** by setting `DB_PROVIDER=supabase` or `DB_PROVIDER=postgres` in your `.env` file.

### Component Organization

The system is organized into three main layers:

1. **Agent Layer** (`/agent`) - Core AI agent with Pydantic AI
   - `agent.py`: Main agent with system prompts and tool registration
   - `tools.py`: RAG tools (vector_search, graph_search, hybrid_search)
   - `api.py`: FastAPI endpoints with SSE streaming support
   - `providers.py`: Flexible LLM provider abstraction (OpenAI, Ollama, OpenRouter, Gemini)
   - `db_utils.py`: PostgreSQL/pgvector database operations
   - `graph_utils.py`: Neo4j/Graphiti graph operations with OpenAI-compatible clients

2. **Ingestion Layer** (`/ingestion`) - Document processing pipeline
   - `ingest.py`: Main orchestration of document ingestion
   - `chunker.py`: Semantic and standard text chunking
   - `embedder.py`: Embedding generation with flexible providers
   - `graph_builder.py`: Knowledge graph construction via Graphiti

3. **Storage Layer** - Dual database architecture
   - PostgreSQL with pgvector: Vector embeddings and document storage
   - Neo4j with Graphiti: Temporal knowledge graph relationships

### Key Design Patterns

- **Flexible Provider System**: Environment-based LLM switching without code changes
- **Tool Transparency**: Agent tools are tracked and displayed in responses
- **Async-First**: All database operations are async for performance
- **Type Safety**: Comprehensive Pydantic models for validation
- **Streaming Responses**: Server-Sent Events for real-time interaction

## Configuration

### Critical Environment Variables

```bash
# Database Provider Selection
DB_PROVIDER=supabase  # or "postgres" for direct connection

# Option 1: Supabase (recommended)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# Option 2: Direct PostgreSQL (alternative)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Neo4j (for knowledge graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Provider (openai, ollama, openrouter, gemini)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_CHOICE=gpt-4o-mini  # Must support tool/function calling

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=sk-...

# Ingestion (can use different/faster model)
INGESTION_LLM_CHOICE=gpt-4o-nano

# Application
APP_PORT=8058
LOG_LEVEL=INFO
```

### Embedding Dimensions

When setting up the database, ensure vector dimensions match your embedding model:
- **Lines to modify in `sql/schema.sql`**: 31, 67, 100
- OpenAI text-embedding-3-small: 1536
- Ollama nomic-embed-text: 768

## Working with the Agent

### Agent System Prompt

The agent's behavior is controlled by the system prompt in `agent/prompts.py`. This determines:
- When to use vector search vs knowledge graph
- How to combine results from different tools
- The agent's reasoning strategy

### Available Tools

1. **vector_search**: Semantic similarity search in pgvector
2. **graph_search**: Knowledge graph queries via Graphiti
3. **hybrid_search**: Combined vector + text search with weighting
4. **get_entity_relationships**: Direct entity relationship queries
5. **list_documents**: List available documents
6. **get_document**: Retrieve full document content

### Tool Usage Tracking

The API and CLI expose which tools the agent used via the `tools_used` field in responses. This provides transparency into the agent's reasoning process.

## Testing Strategy

### Test Organization
- Unit tests for models and utilities
- Integration tests for database operations
- Mocked external dependencies (LLMs, embeddings)
- Async test support with pytest-asyncio

### Key Test Files
- `tests/conftest.py`: Shared fixtures and mocks
- `tests/agent/test_models.py`: Pydantic model validation
- `tests/agent/test_db_utils.py`: Database operation tests
- `tests/ingestion/test_chunker.py`: Document chunking tests

## Common Development Tasks

### Adding a New LLM Provider

1. Add provider logic to `agent/providers.py`
2. Update environment variable handling
3. Test with different models that support tool calling

### Modifying Agent Behavior

1. Edit system prompt in `agent/prompts.py`
2. Adjust tool selection logic if needed
3. Test with various query types

### Processing New Document Types

1. Add documents to `medical_docs/` folder
2. Run ingestion with appropriate settings
3. Knowledge graph building can be slow (30+ minutes for large datasets)

### Debugging Database Issues

```bash
# Test PostgreSQL connection
psql -d "$DATABASE_URL" -c "SELECT 1;"

# Test Neo4j connection
curl -u neo4j:password http://localhost:7474/db/data/

# Check if ingestion completed
psql -d "$DATABASE_URL" -c "SELECT COUNT(*) FROM chunks;"
```

## Important Considerations

### Performance
- Knowledge graph operations are computationally expensive
- Use `--no-semantic` flag for faster ingestion during development
- Consider using faster models for ingestion (INGESTION_LLM_CHOICE)

### Database Management
- Schema drops all tables before recreating (be careful in production)
- Vector index uses IVFFlat - rebuild after significant data changes
- Neo4j requires manual cleanup if graph becomes corrupted

### Session Management
- Sessions expire after SESSION_TIMEOUT_MINUTES (default: 60)
- Messages are stored per session for context
- CLI maintains session across the conversation

## API Endpoints

- `GET /health` - Health check
- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - SSE streaming chat
- `POST /search` - Direct search operations
- `GET /documents` - List documents
- `GET /sessions/{session_id}` - Get session info

Interactive API docs available at `http://localhost:8058/docs` when server is running.