# MaryPause AI - Medical RAG Agent with Knowledge Graph

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced medical knowledge retrieval system specializing in menopause and women's health. This system combines vector similarity search with knowledge graph relationships to provide comprehensive, context-aware medical information.

## 🚀 Key Features

- **Dual Search Architecture**: Combines vector embeddings (Supabase/pgvector) with knowledge graphs (Neo4j)
- **Medical Knowledge Base**: Pre-ingested with 11 comprehensive medical documents on menopause
- **768-Dimensional Embeddings**: Optimized for performance and compatibility
- **Streaming API**: Real-time responses with Server-Sent Events
- **Multi-Provider Support**: Works with OpenAI, Gemini, Ollama, and OpenRouter

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (CLI / Web Application)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                     FastAPI Server                           │
│                  (Streaming SSE Responses)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Pydantic AI Agent                          │
│              (Tool orchestration & reasoning)                │
└──────┬──────────────────────────────────────┬───────────────┘
       │                                      │
┌──────▼──────────┐                    ┌─────▼────────────────┐
│  Vector Search  │                    │  Knowledge Graph     │
│   (Supabase)    │                    │     (Neo4j)          │
├─────────────────┤                    ├──────────────────────┤
│ 89 chunks       │                    │ 105 entities         │
│ 768-dim vectors │                    │ 661 relationships    │
└─────────────────┘                    └──────────────────────┘
```

## 📊 Current Database Status

### Supabase (Vector Database)
- **Documents**: 11 medical documents
- **Chunks**: 89 text segments
- **Embeddings**: 768-dimensional vectors (normalized from various models)
- **Search**: Semantic similarity with pgvector

### Neo4j (Knowledge Graph)
- **Entities**: 105 unique medical concepts
- **Relationships**: 661 connections between entities
- **Structure**: Documents → Chunks → Entities → Relationships
- **Search**: Direct graph queries for concept relationships

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- PostgreSQL with pgvector extension (or Supabase account)
- Neo4j database (cloud or local)
- API keys for LLM provider (OpenAI, Gemini, etc.)

### Quick Start

1. **Clone and setup environment**
```bash
git clone https://github.com/marypause/marypause_ai.git
cd ottomator-agents
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your credentials:
# - SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
# - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
# - LLM_API_KEY (OpenAI, Gemini, etc.)
```

3. **Run the system**
```bash
# Start the API server
python -m agent.api

# In another terminal, use the CLI
python cli.py
```

## 📁 Project Structure

```
ottomator-agents/
├── agent/                    # Core agent system
│   ├── agent.py             # Pydantic AI agent definition
│   ├── api.py               # FastAPI server
│   ├── tools.py             # Search and retrieval tools
│   ├── neo4j_direct.py      # Direct Neo4j queries
│   ├── embedding_config.py  # Centralized embedding configuration
│   └── graphiti_patch.py    # Graphiti dimension normalization
├── ingestion/               # Document processing pipeline
│   ├── ingest.py           # Main ingestion orchestrator
│   ├── chunker.py          # Text chunking strategies
│   ├── embedder.py         # Embedding generation
│   └── graph_builder.py    # Knowledge graph construction
├── scripts/                 # Utility scripts
│   ├── ingestion/          # Ingestion scripts
│   ├── database/           # Database management
│   └── monitoring/         # System monitoring
├── tests/                   # Test suites
│   ├── integration/        # Integration tests
│   └── system/             # System tests
├── medical_docs/           # Medical knowledge base
└── cli.py                  # Command-line interface
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
DB_PROVIDER=supabase              # or "postgres" for direct connection
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=xxx
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=xxx

# LLM Configuration
LLM_PROVIDER=gemini               # openai, ollama, openrouter
LLM_API_KEY=xxx
LLM_CHOICE=gemini-2.5-flash      # Model selection

# Embedding Configuration
EMBEDDING_PROVIDER=gemini         # openai, ollama
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_API_KEY=xxx
VECTOR_DIMENSION=768              # Target dimension (fixed)

# Application
APP_PORT=8058
LOG_LEVEL=INFO
```

## 🚀 Usage

### CLI Interface

```bash
# Basic usage
python cli.py

# Example queries
> What are the symptoms of menopause?
> How does hormone therapy work?
> What's the difference between perimenopause and menopause?
```

### API Endpoints

```python
# Health check
GET /health

# Chat (streaming)
POST /chat/stream
{
    "message": "What are hot flash treatments?",
    "session_id": "optional-session-id"
}

# Direct search
POST /search
{
    "query": "estrogen therapy",
    "search_type": "vector|graph|hybrid"
}
```

### Python Client

```python
import httpx
import json

# Streaming chat
with httpx.stream(
    "POST",
    "http://localhost:8058/chat/stream",
    json={"message": "Tell me about menopause symptoms"}
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            print(data["content"], end="")
```

## 📝 Scripts

### Ingestion Scripts

```bash
# Complete Neo4j ingestion
python scripts/ingestion/complete_neo4j_ingestion.py

# Build knowledge graph
python scripts/ingestion/build_knowledge_graph.py
```

### Monitoring Scripts

```bash
# Check system status
python scripts/monitoring/check_neo4j_status.py
python scripts/monitoring/check_ingestion_status.py

# Verify database state
python scripts/database/verify_supabase.py
python scripts/database/check_db_state.py
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/integration/
pytest tests/system/

# Run with coverage
pytest --cov=agent --cov=ingestion
```

## 🔍 How It Works

### 1. Document Ingestion
- Documents are chunked using semantic boundaries
- Each chunk gets a 768-dimensional embedding
- Entities are extracted and linked in Neo4j
- Relationships between entities are mapped

### 2. Dual Search Strategy
- **Vector Search**: Find semantically similar content
- **Graph Search**: Explore entity relationships
- **Hybrid Search**: Combine both approaches

### 3. Agent Reasoning
- Pydantic AI agent orchestrates tool usage
- Determines optimal search strategy
- Combines results from multiple sources
- Generates contextual responses

## 🐛 Troubleshooting

### Vector Dimension Errors
- System expects 768-dimensional vectors
- All embeddings are automatically normalized
- Check `VECTOR_DIMENSION` environment variable

### Neo4j Connection Issues
- Verify Neo4j credentials and URI
- Ensure database is running
- Check network connectivity

### Ingestion Problems
- Run `python scripts/database/clean_all_databases.py` for fresh start
- Check logs in monitoring scripts
- Verify document format in `medical_docs/`

## 📚 Documentation

- [CLAUDE.md](CLAUDE.md) - Claude Code assistant instructions
- [docs/reports/](docs/reports/) - Technical reports and guides
- [API Documentation](http://localhost:8058/docs) - Interactive API docs (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built with [Pydantic AI](https://github.com/pydantic/pydantic-ai)
- Knowledge graphs powered by [Graphiti](https://github.com/getzep/graphiti)
- Vector search by [Supabase](https://supabase.com) and pgvector
- Graph database by [Neo4j](https://neo4j.com)

## 📞 Support

For issues and questions:
- Open an issue on [GitHub](https://github.com/marypause/marypause_ai/issues)
- Check existing documentation in `/docs`
- Review test files for usage examples

---

**Current Version**: 1.0.0  
**Last Updated**: August 30, 2025  
**Status**: Production Ready ✅