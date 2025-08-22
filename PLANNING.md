# Agentic RAG with Knowledge Graph - Project Plan

## Project Overview

This project builds an AI agent system that combines traditional RAG (Retrieval Augmented Generation) with knowledge graph capabilities to analyze and provide insights about big tech companies and their AI initiatives. The system uses PostgreSQL with pgvector for vector search and Neo4j (via Graphiti) for knowledge graph operations.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer                             │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │   FastAPI       │        │   Streaming SSE    │     │
│  │   Endpoints     │        │   Responses        │     │
│  └────────┬────────┘        └────────────────────┘     │
│           │                                              │
├───────────┴──────────────────────────────────────────────┤
│                    Agent Layer                           │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │  Pydantic AI    │        │   Agent Tools      │     │
│  │    Agent        │◄──────►│  - Vector Search   │     │
│  └────────┬────────┘        │  - Graph Search    │     │
│           │                 │  - Doc Retrieval   │     │
│           │                 └────────────────────┘     │
├───────────┴──────────────────────────────────────────────┤
│                  Storage Layer                           │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │   PostgreSQL    │        │      Neo4j         │     │
│  │   + pgvector    │        │   (via Graphiti)   │     │
│  └─────────────────┘        └────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Agent System (`/agent`)
- **agent.py**: Main Pydantic AI agent with system prompts and configuration
- **tools.py**: All agent tools for RAG and knowledge graph operations
- **prompts.py**: System prompts controlling agent tool selection behavior
- **api.py**: FastAPI endpoints with streaming support and tool usage extraction
- **db_utils.py**: PostgreSQL database utilities and connection management
- **graph_utils.py**: Neo4j/Graphiti utilities with OpenAI-compatible client configuration
- **models.py**: Pydantic models for data validation including ToolCall tracking
- **providers.py**: Flexible LLM provider abstraction supporting multiple backends

### 2. Ingestion System (`/ingestion`)
- **ingest.py**: Main ingestion script to process markdown files
- **chunker.py**: Semantic chunking implementation
- **embedder.py**: Document embedding generation
- **graph_builder.py**: Knowledge graph construction from documents
- **cleaner.py**: Database cleanup utilities

### 3. Database Schema (`/sql`)
- **schema.sql**: PostgreSQL schema with pgvector
- **migrations/**: Database migration scripts

### 4. Tests (`/tests`)
- Comprehensive unit and integration tests
- Mocked external dependencies
- Test fixtures and utilities

### 5. CLI Interface (`/cli.py`)
- Interactive command-line interface for the agent
- Real-time streaming with Server-Sent Events
- Tool usage visibility showing agent reasoning
- Session management and conversation context

## Technical Stack

### Core Technologies
- **Python 3.11+**: Primary language
- **Pydantic AI**: Agent framework
- **FastAPI**: API framework
- **PostgreSQL + pgvector**: Vector database
- **Neo4j + Graphiti**: Knowledge graph
- **Flexible LLM Providers**: OpenAI, Ollama, OpenRouter, Gemini

### Key Libraries
- **asyncpg**: PostgreSQL async driver
- **httpx**: Async HTTP client
- **python-dotenv**: Environment management
- **pytest + pytest-asyncio**: Testing
- **black + ruff**: Code formatting/linting

## Design Principles

### 1. Modularity
- Clear separation of concerns
- Reusable components
- Clean dependency injection

### 2. Type Safety
- Comprehensive type hints
- Pydantic models for validation
- Dataclasses for dependencies

### 3. Async-First
- All database operations async
- Concurrent processing where applicable
- Proper resource management

### 4. Error Handling
- Graceful degradation
- Comprehensive logging
- User-friendly error messages

### 5. Testing
- Unit tests for all components
- Integration tests for workflows
- Mocked external dependencies

## Key Features

### 1. Hybrid Search
- Vector similarity search for semantic queries
- Knowledge graph traversal for relationship queries
- Combined results with intelligent ranking

### 2. Document Management
- Semantic chunking for optimal retrieval
- Metadata preservation
- Full document retrieval capability

### 3. Knowledge Graph
- Entity and relationship extraction
- Temporal data handling
- Graph-based reasoning

### 4. API Capabilities
- Streaming responses (SSE)
- Session management
- File attachment support

### 5. Flexible Provider System
- Multiple LLM providers (OpenAI, Ollama, OpenRouter, Gemini)
- Environment-based provider switching
- Separate models for different tasks (chat vs ingestion)
- OpenAI-compatible API interface
- Graphiti with custom OpenAI-compatible clients (OpenAIClient, OpenAIEmbedder)

### 6. Agent Transparency
- Tool usage tracking and display in API responses
- CLI with real-time tool visibility
- Configurable agent behavior via system prompt
- Clear reasoning process exposure

## Implementation Strategy

### Phase 1: Foundation
1. Set up project structure
2. Configure PostgreSQL and Neo4j
3. Implement database utilities
4. Create base models

### Phase 2: Core Agent
1. Build Pydantic AI agent
2. Implement RAG tools
3. Implement knowledge graph tools
4. Create prompts and configurations

### Phase 3: API Layer
1. Set up FastAPI application
2. Implement streaming endpoints
3. Add error handling
4. Create health checks

### Phase 4: Ingestion System
1. Build semantic chunker
2. Implement document processor
3. Create knowledge graph builder
4. Add cleanup utilities

### Phase 5: Testing & Documentation
1. Write comprehensive tests
2. Create detailed README
3. Generate API documentation
4. Add usage examples

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration  
LLM_PROVIDER=openai  # openai, ollama, openrouter, gemini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_CHOICE=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
INGESTION_LLM_CHOICE=gpt-4.1-nano

# Application
APP_ENV=development
LOG_LEVEL=INFO
APP_PORT=8058
```

### Database Schema
- **documents**: Store document metadata
- **chunks**: Store document chunks with embeddings
- **sessions**: Manage conversation sessions
- **messages**: Store conversation history

## Security Considerations
- Environment-based configuration
- No hardcoded credentials
- Input validation at all layers
- SQL injection prevention
- Rate limiting on API

## Performance Optimizations
- Connection pooling for databases
- Embedding caching
- Batch processing for ingestion
- Indexed vector searches
- Async operations throughout

## Monitoring & Logging
- Structured logging with context
- Performance metrics
- Error tracking
- Usage analytics

## Future Enhancements
- ✅ ~~Multi-model support~~ (Completed - Flexible provider system)
- Advanced reranking algorithms
- Real-time document updates
- GraphQL API option
- Web UI for exploration
- Additional LLM providers (Anthropic Claude direct, Cohere, etc.)
- Embedding provider diversity (Voyage, Cohere embeddings)
- Model performance optimization and caching