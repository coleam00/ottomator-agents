# RAG Strategy Agent Pseudocode Examples

This folder contains simplified, practical pseudocode examples for each RAG strategy using **Pydantic AI** agents and **PG Vector** (PostgreSQL with pgvector).

## Framework Overview

- **Pydantic AI**: Python agent framework with `@agent.tool` decorators for function calling
- **PG Vector**: PostgreSQL extension for vector similarity search with `<=>` operator
- All examples are under 50 lines and show the core concept in action

## Scripts

### 01_query_expansion.py
**Strategy**: Generate multiple query variations to improve recall
- Shows: Expanding a single query into 3+ variations
- Tool: `expand_query()` and `search_knowledge_base()`
- Key: Searches with multiple perspectives and deduplicates results

### 02_reranking.py
**Strategy**: Two-stage retrieval with cross-encoder refinement
- Shows: Fast vector search (20 candidates) → accurate re-ranking (top 5)
- Tool: `search_with_reranking()`
- Key: Balance between retrieval speed and precision

### 03_agentic_rag.py
**Strategy**: Agent autonomously chooses tools (vector, SQL, web)
- Shows: Multiple tools for different data types
- Tools: `vector_search()`, `sql_query()`, `web_search()`
- Key: Agent decides which tool(s) to use based on query

### 04_multi_query_rag.py
**Strategy**: Parallel searches with reformulated queries
- Shows: Multiple query perspectives executed in parallel
- Tool: `multi_query_search()`
- Key: Unique union of all results from different query angles

### 05_context_aware_chunking.py
**Strategy**: Semantic chunking based on embedding similarity
- Shows: `semantic_chunk()` function that groups similar sentences
- Key: Chunk boundaries determined by semantic similarity, not fixed size
- Ingestion: Compares consecutive sentence embeddings

### 06_late_chunking.py
**Strategy**: Embed full document before chunking (Jina AI approach)
- Shows: `late_chunk()` processes entire document through transformer first
- Key: Token-level embeddings capture full context, then pooled per chunk
- Ingestion: `transformer_embed()` → chunk boundaries → mean pooling

### 07_hierarchical_rag.py
**Strategy**: Parent-child relationships with metadata
- Shows: Two tables (`parent_chunks`, `child_chunks`) with foreign keys
- Tool: `search_knowledge_base()` searches children, returns parents
- Key: Small chunks for matching, large parents for context

### 08_contextual_retrieval.py
**Strategy**: Add document context to chunks (Anthropic method)
- Shows: `add_context_to_chunk()` prepends LLM-generated context
- Key: Each chunk gets document-level context before embedding
- Ingestion: Original chunk → contextualized → embedded

### 09_self_reflective_rag.py
**Strategy**: Iterative refinement with self-assessment
- Shows: `search_and_grade()`, `refine_query()`, `answer_with_verification()`
- Tools: Grade relevance, refine queries, verify answers
- Key: Multiple LLM calls for reflection and improvement

### 10_knowledge_graphs.py
**Strategy**: Combine vector search with graph relationships
- Shows: Two tables (`entities`, `relationships`) forming a graph
- Tool: `search_knowledge_graph()` does hybrid vector + graph traversal
- Ingestion: Extract entities and relationships, store in graph structure

### 11_fine_tuned_embeddings.py
**Strategy**: Custom embedding model trained on domain data
- Shows: `fine_tune_model()` trains on query-document pairs
- Key: Domain-specific embeddings (medical, legal, financial)
- Ingestion: Uses fine-tuned model instead of generic embeddings

## Common Patterns

All scripts follow this structure:
```python
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

# Initialize agent
agent = Agent('openai:gpt-4o', system_prompt='...')

# Database connection
conn = psycopg2.connect("dbname=rag_db")
register_vector(conn)

# Ingestion function (strategy-specific)
def ingest_document(text: str):
    # ... chunking logic varies by strategy
    pass

# Agent tools (strategy-specific)
@agent.tool
def search_knowledge_base(query: str) -> str:
    # ... search logic varies by strategy
    pass

# Run agent
result = agent.run_sync("query")
print(result.data)
```

## Notes

- Functions like `get_embedding()`, `llm_generate()`, etc. are placeholders for clarity
- Database schemas are simplified; production would need proper table creation
- Each example focuses on demonstrating the core RAG strategy concept
- All scripts use pgvector's `<=>` operator for cosine distance similarity search

## Database Schema Examples

**Basic chunks table**:
```sql
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(768)
);
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops);
```

**Hierarchical (parent-child)**:
```sql
CREATE TABLE parent_chunks (id INT PRIMARY KEY, content TEXT);
CREATE TABLE child_chunks (id SERIAL PRIMARY KEY, content TEXT, embedding vector(768), parent_id INT);
```

**Knowledge graph**:
```sql
CREATE TABLE entities (name TEXT PRIMARY KEY, embedding vector(768));
CREATE TABLE relationships (source TEXT, relation TEXT, target TEXT);
```
