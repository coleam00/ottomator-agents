# Advanced RAG Strategies - Complete Guide

**A comprehensive resource for understanding and implementing advanced Retrieval-Augmented Generation strategies.**

This repository demonstrates 11 RAG strategies with:
- 📖 Detailed theory and research ([docs/](docs/))
- 💻 Simple pseudocode examples ([examples/](examples/))
- 🔧 Full code examples ([implementation/](implementation/))

Perfect for: AI engineers, ML practitioners, and anyone building RAG systems.

---

## 📚 Table of Contents

1. [Strategy Overview](#-strategy-overview)
2. [Quick Start](#-quick-start)
3. [Pseudocode Examples](#-pseudocode-examples)
4. [Code Examples (Not Production)](#-code-examples-not-production)
5. [Detailed Strategy Guide](#-detailed-strategy-guide)
6. [Repository Structure](#-repository-structure)

---

## 🎯 Strategy Overview

| # | Strategy | Status | Use Case | Pros | Cons |
|---|----------|--------|----------|------|------|
| 1 | [Query Expansion](#1-query-expansion) | ✅ Code Example | Ambiguous queries | Better recall, multiple perspectives | Extra LLM call, higher cost |
| 2 | [Re-ranking](#2-re-ranking) | ✅ Code Example | Precision-critical | Highly accurate results | Slower, more compute |
| 3 | [Agentic RAG](#3-agentic-rag) | ✅ Code Example | Flexible retrieval needs | Autonomous tool selection | More complex logic |
| 4 | [Multi-Query RAG](#4-multi-query-rag) | ✅ Code Example | Broad searches | Comprehensive coverage | Multiple API calls |
| 5 | [Context-Aware Chunking](#5-context-aware-chunking) | ✅ Code Example | All documents | Semantic coherence | Slightly slower ingestion |
| 6 | [Late Chunking](#6-late-chunking) | 📝 Pseudocode Only | Context preservation | Full document context | Requires long-context models |
| 7 | [Hierarchical RAG](#7-hierarchical-rag) | 📝 Pseudocode Only | Complex documents | Precision + context | Complex setup |
| 8 | [Contextual Retrieval](#8-contextual-retrieval) | ✅ Code Example | Critical documents | 35-49% better accuracy | High ingestion cost |
| 9 | [Self-Reflective RAG](#9-self-reflective-rag) | ✅ Code Example | Research queries | Self-correcting | Highest latency |
| 10 | [Knowledge Graphs](#10-knowledge-graphs) | 📝 Pseudocode Only | Relationship-heavy | Captures connections | Infrastructure overhead |
| 11 | [Fine-tuned Embeddings](#11-fine-tuned-embeddings) | 📝 Pseudocode Only | Domain-specific | Best accuracy | Training required |

### Legend
- ✅ **Code Example**: Full code in `implementation/` (educational, not production-ready)
- 📝 **Pseudocode Only**: Conceptual examples in `examples/`

---

## 🚀 Quick Start

### View Pseudocode Examples

```bash
cd examples
# Browse simple, < 50 line examples for each strategy
cat 01_query_expansion.py
```

### Run the Code Examples (Educational)

> **Note**: These are educational examples to show how strategies work in real code. Not guaranteed to be fully functional or production-ready.

```bash
cd implementation

# Install dependencies
pip install -r requirements-advanced.txt

# Setup environment
cp .env.example .env
# Edit .env: Add DATABASE_URL and OPENAI_API_KEY

# Ingest documents (with optional contextual enrichment)
python -m ingestion.ingest --documents ./documents --contextual

# Run the advanced agent
python rag_agent_advanced.py
```

---

## 💻 Pseudocode Examples

All strategies have simple, working pseudocode examples in [`examples/`](examples/).

Each file is **< 50 lines** and demonstrates:
- Core concept
- How to implement with Pydantic AI
- Integration with PG Vector

**Example** (`01_query_expansion.py`):
```python
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector

agent = Agent('openai:gpt-4o', system_prompt='RAG assistant with query expansion')

@agent.tool
def expand_query(query: str) -> list[str]:
    """Expand single query into multiple variations"""
    expansion_prompt = f"Generate 3 variations of: '{query}'"
    variations = llm_generate(expansion_prompt)
    return [query] + variations

@agent.tool
def search_knowledge_base(queries: list[str]) -> str:
    """Search vector DB with multiple queries"""
    all_results = []
    for query in queries:
        query_embedding = get_embedding(query)
        results = db.query('SELECT * FROM chunks ORDER BY embedding <=> %s', query_embedding)
        all_results.extend(results)
    return deduplicate(all_results)
```

**Browse all pseudocode**: [examples/README.md](examples/README.md)

---

## 🏗️ Code Examples

> **⚠️ Important Note**: The `implementation/` folder contains **educational code examples** based on a real implementation, not production-ready. These strategies are added to demonstrate concepts and show how they work in real code. They are **not guaranteed to be fully working** and it's **not ideal to have all strategies in one codebase** (which is why I haven't refined this specifically for production use). Use these as learning references and starting points for your own implementations.
> Think of this as an "off-the-shelf RAG implementation" with strategies added for demonstration purposes. Use as inspiration for your own production systems.

### Architecture

```
implementation/
├── rag_agent_advanced.py          # Agent with all strategy examples
├── ingestion/
│   ├── ingest.py                  # Document ingestion pipeline
│   ├── chunker.py                 # Context-aware chunking (Docling)
│   ├── embedder.py                # OpenAI embeddings
│   └── contextual_enrichment.py   # Anthropic's contextual retrieval
├── utils/
│   ├── db_utils.py                # Database utilities
│   └── models.py                  # Pydantic models
└── IMPLEMENTATION_GUIDE.md        # Detailed implementation reference
```

**Tech Stack**:
- **Pydantic AI** - Agent framework
- **PostgreSQL + pgvector** - Vector search
- **Docling** - Hybrid chunking
- **OpenAI** - Embeddings and LLM

---

## 📖 Detailed Strategy Guide

### ✅ Code Examples (Educational)

---

## 1️⃣ Query Expansion

**Status**: ✅ Code Example
**File**: `rag_agent_advanced.py` (Lines 72-107)

### What It Is
Generates multiple variations of a user query to improve retrieval recall. Takes 1 query → returns 4 queries (original + 3 LLM-generated variations).

### Pros & Cons
✅ Better recall, captures multiple perspectives
❌ Extra LLM call adds latency and cost

### Code Example
```python
# Lines 72-107 in rag_agent_advanced.py
async def expand_query_variations(ctx: RunContext[None], query: str) -> List[str]:
    """Generate multiple variations of a query for better retrieval."""
    expansion_prompt = f"""Generate 3 different variations of this search query.
Each variation should capture a different perspective or phrasing.

Original query: {query}

Return only the 3 variations, one per line."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": expansion_prompt}],
        temperature=0.7
    )

    variations = response.choices[0].message.content.strip().split('\n')
    return [query] + variations[:3]  # Original + 3 variations
```

**Used by**: Multi-Query RAG strategy

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#2-query-expansion)
- Pseudocode: [01_query_expansion.py](examples/01_query_expansion.py)
- Research: [docs/01-query-expansion.md](docs/01-query-expansion.md)

---

## 2️⃣ Re-ranking

**Status**: ✅ Code Example
**File**: `rag_agent_advanced.py` (Lines 194-256)

### What It Is
Two-stage retrieval: Fast vector search (20 candidates) → Precise cross-encoder re-ranking (top 5).

### Pros & Cons
✅ Significantly better precision, ideal for critical queries
❌ Slower than pure vector search, uses more compute

### Code Example
```python
# Lines 194-256 in rag_agent_advanced.py
async def search_with_reranking(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """Two-stage retrieval with cross-encoder re-ranking."""
    initialize_reranker()  # Loads cross-encoder/ms-marco-MiniLM-L-6-v2

    # Stage 1: Fast vector retrieval (retrieve 20 candidates)
    candidate_limit = min(limit * 4, 20)
    results = await vector_search(query, candidate_limit)

    # Stage 2: Re-rank with cross-encoder
    pairs = [[query, row['content']] for row in results]
    scores = reranker.predict(pairs)

    # Sort by new scores and return top N
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:limit]
    return format_results(reranked)
```

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#4-re-ranking)
- Pseudocode: [02_reranking.py](examples/02_reranking.py)
- Research: [docs/02-reranking.md](docs/02-reranking.md)

---

## 3️⃣ Agentic RAG

**Status**: ✅ Code Example
**Files**: `rag_agent_advanced.py` (Lines 263-354)

### What It Is
Agent autonomously chooses between multiple retrieval tools:
1. `search_knowledge_base()` - Semantic search over chunks
2. `retrieve_full_document()` - Pull entire documents when chunks aren't enough

### Pros & Cons
✅ Flexible, adapts to query needs automatically
❌ More complex, less predictable behavior

### Code Example
```python
# Tool 1: Semantic search (Lines 263-305)
@agent.tool
async def search_knowledge_base(query: str, limit: int = 5) -> str:
    """Standard semantic search over document chunks."""
    query_embedding = await embedder.embed_query(query)
    results = await db.match_chunks(query_embedding, limit)
    return format_results(results)

# Tool 2: Full document retrieval (Lines 308-354)
@agent.tool
async def retrieve_full_document(document_title: str) -> str:
    """Retrieve complete document when chunks lack context."""
    result = await db.query(
        "SELECT title, content FROM documents WHERE title ILIKE %s",
        f"%{document_title}%"
    )
    return f"**{result['title']}**\n\n{result['content']}"
```

**Example Flow**:
```
User: "What's the full refund policy?"
Agent:
  1. Calls search_knowledge_base("refund policy")
  2. Finds chunks mentioning "refund_policy.pdf"
  3. Calls retrieve_full_document("refund policy")
  4. Returns complete document
```

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#5-agentic-rag)
- Pseudocode: [03_agentic_rag.py](examples/03_agentic_rag.py)
- Research: [docs/03-agentic-rag.md](docs/03-agentic-rag.md)

---

## 4️⃣ Multi-Query RAG

**Status**: ✅ Code Example
**File**: `rag_agent_advanced.py` (Lines 114-187)

### What It Is
Combines query expansion with parallel execution. Generates 4 query variations, runs all searches concurrently, deduplicates results.

### Pros & Cons
✅ Comprehensive coverage, better recall on ambiguous queries
❌ 4x database queries (though parallelized), higher cost

### Code Example
```python
# Lines 114-187 in rag_agent_advanced.py
async def search_with_multi_query(query: str, limit: int = 5) -> str:
    """Search using multiple query variations in parallel."""
    # Generate variations
    queries = await expand_query_variations(query)  # Returns 4 queries

    # Execute all searches in parallel
    search_tasks = []
    for q in queries:
        query_embedding = await embedder.embed_query(q)
        task = db.fetch("SELECT * FROM match_chunks($1::vector, $2)", query_embedding, limit)
        search_tasks.append(task)

    results_lists = await asyncio.gather(*search_tasks)

    # Deduplicate by chunk ID, keep highest similarity
    seen = {}
    for results in results_lists:
        for row in results:
            if row['chunk_id'] not in seen or row['similarity'] > seen[row['chunk_id']]['similarity']:
                seen[row['chunk_id']] = row

    return format_results(sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit])
```

**Key Features**:
- Parallel execution with `asyncio.gather()`
- Smart deduplication (keeps best score per chunk)

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#3-multi-query-rag)
- Pseudocode: [04_multi_query_rag.py](examples/04_multi_query_rag.py)
- Research: [docs/04-multi-query-rag.md](docs/04-multi-query-rag.md)

---

## 5️⃣ Context-Aware Chunking

**Status**: ✅ Code Example (Default)
**File**: `ingestion/chunker.py` (Lines 70-102)

### What It Is
Uses Docling's HybridChunker for intelligent document splitting that:
- Respects document structure (headings, sections, tables)
- Is token-aware (uses actual tokenizer, not estimates)
- Preserves semantic coherence
- Includes heading context in chunks

### Pros & Cons
✅ Free, fast, maintains document structure
❌ Slightly more complex than naive chunking

### Code Example
```python
# Lines 70-102 in chunker.py
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

class DoclingHybridChunker:
    def __init__(self, config: ChunkingConfig):
        # Initialize tokenizer for token-aware chunking
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Create HybridChunker
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True  # Merge small adjacent chunks
        )

    async def chunk_document(self, docling_doc: DoclingDocument) -> List[DocumentChunk]:
        # Use HybridChunker to chunk the DoclingDocument
        chunks = list(self.chunker.chunk(dl_doc=docling_doc))

        # Contextualize each chunk (includes heading hierarchy)
        for chunk in chunks:
            contextualized_text = self.chunker.contextualize(chunk=chunk)
            # Store contextualized text as chunk content
```

**Enabled by default during ingestion**

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#1-context-aware-chunking)
- Pseudocode: [05_context_aware_chunking.py](examples/05_context_aware_chunking.py)
- Research: [docs/05-context-aware-chunking.md](docs/05-context-aware-chunking.md)

---

## 8️⃣ Contextual Retrieval

**Status**: ✅ Code Example (Optional)
**File**: `ingestion/contextual_enrichment.py` (Lines 41-89)

### What It Is
Anthropic's method: Adds document-level context to each chunk before embedding. LLM generates 1-2 sentences explaining what the chunk discusses in relation to the whole document.

### Pros & Cons
✅ 35-49% reduction in retrieval failures, chunks are self-contained
❌ Expensive (1 LLM call per chunk), slower ingestion

### Before/After Example
```
BEFORE:
"Clean data is essential. Remove duplicates, handle missing values..."

AFTER:
"This chunk from 'ML Best Practices' discusses data preparation techniques
for machine learning workflows.

Clean data is essential. Remove duplicates, handle missing values..."
```

### Code Example
```python
# Lines 41-89 in contextual_enrichment.py
async def enrich_chunk(chunk: str, document: str, title: str) -> str:
    """Add contextual prefix to a chunk."""
    prompt = f"""<document>
Title: {title}
{document[:4000]}
</document>

<chunk>
{chunk}
</chunk>

Provide brief context explaining what this chunk discusses.
Format: "This chunk from [title] discusses [explanation]." """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    context = response.choices[0].message.content.strip()
    return f"{context}\n\n{chunk}"
```

**Enable with**: `python -m ingestion.ingest --documents ./docs --contextual`

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#7-contextual-retrieval)
- Pseudocode: [08_contextual_retrieval.py](examples/08_contextual_retrieval.py)
- Research: [docs/08-contextual-retrieval.md](docs/08-contextual-retrieval.md)

---

## 9️⃣ Self-Reflective RAG

**Status**: ✅ Code Example
**File**: `rag_agent_advanced.py` (Lines 361-482)

### What It Is
Self-correcting search loop:
1. Perform initial search
2. LLM grades relevance (1-5 scale)
3. If score < 3, refine query and search again

### Pros & Cons
✅ Self-correcting, improves over time
❌ Highest latency (2-3 LLM calls), most expensive

### Code Example
```python
# Lines 361-482 in rag_agent_advanced.py
async def search_with_self_reflection(query: str, limit: int = 5) -> str:
    """Self-reflective search: evaluate and refine if needed."""
    # Initial search
    results = await vector_search(query, limit)

    # Grade relevance
    grade_prompt = f"""Query: {query}
Retrieved: {results[:200]}...

Grade relevance 1-5. Respond with number only."""

    grade_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": grade_prompt}],
        temperature=0
    )
    grade_score = int(grade_response.choices[0].message.content.split()[0])

    # If low relevance, refine and re-search
    if grade_score < 3:
        refine_prompt = f"""Query "{query}" returned low-relevance results.
Suggest improved query. Respond with query only."""

        refined_query = await client.chat.completions.create(...)
        results = await vector_search(refined_query, limit)
        note = f"[Refined from '{query}' to '{refined_query}']"

    return format_results(results, note)
```

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#6-self-reflective-rag)
- Pseudocode: [09_self_reflective_rag.py](examples/09_self_reflective_rag.py)
- Research: [docs/09-self-reflective-rag.md](docs/09-self-reflective-rag.md)

---

### 📝 Pseudocode-Only Strategies

---

## 6️⃣ Late Chunking

**Status**: 📝 Pseudocode Only
**Why not in code examples**: Docling HybridChunker provides similar benefits

### What It Is
Embed the full document through transformer first, then chunk the token embeddings (not the text). Preserves full document context in each chunk's embedding.

### Pros & Cons
✅ Maintains full document context, leverages long-context models
❌ Requires 8K+ token models, more complex than standard chunking

### Pseudocode Concept
```python
# From 06_late_chunking.py
def late_chunk(text: str, chunk_size=512) -> list:
    """Process full document through transformer BEFORE chunking."""
    # Step 1: Embed entire document (up to 8192 tokens)
    full_doc_token_embeddings = transformer_embed(text)  # Token-level embeddings

    # Step 2: Define chunk boundaries
    tokens = text.split()
    chunk_boundaries = range(0, len(tokens), chunk_size)

    # Step 3: Pool token embeddings for each chunk
    chunks_with_embeddings = []
    for start in chunk_boundaries:
        end = start + chunk_size
        chunk_text = ' '.join(tokens[start:end])

        # Mean pool the token embeddings (preserves full doc context!)
        chunk_embedding = mean_pool(full_doc_token_embeddings[start:end])
        chunks_with_embeddings.append((chunk_text, chunk_embedding))

    return chunks_with_embeddings
```

**Alternative**: Use Context-Aware Chunking (Docling) + Contextual Retrieval for similar benefits

**See**:
- Pseudocode: [06_late_chunking.py](examples/06_late_chunking.py)
- Research: [docs/06-late-chunking.md](docs/06-late-chunking.md)

---

## 7️⃣ Hierarchical RAG

**Status**: 📝 Pseudocode Only
**Why not in code examples**: Agentic RAG achieves similar goals more flexibly

### What It Is
Parent-child chunk relationships: Search small chunks for precision, but return large parent chunks for context.

### Pros & Cons
✅ Balances precision (small chunks) with context (large chunks)
❌ Complex database schema, requires metadata management

### Pseudocode Concept
```python
# From 07_hierarchical_rag.py
def ingest_hierarchical(document: str):
    """Create parent-child chunk structure."""
    # Parent: Large sections (2000 chars)
    parent_chunks = [document[i:i+2000] for i in range(0, len(document), 2000)]

    for parent_id, parent in enumerate(parent_chunks):
        # Store parent (not embedded)
        db.execute("INSERT INTO parent_chunks (id, content) VALUES (%s, %s)",
                   (parent_id, parent))

        # Children: Small chunks (500 chars) from parent
        child_chunks = [parent[j:j+500] for j in range(0, len(parent), 500)]
        for child in child_chunks:
            embedding = get_embedding(child)
            # Store child with parent_id reference
            db.execute(
                "INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)",
                (child, embedding, parent_id)
            )

@agent.tool
def hierarchical_search(query: str) -> str:
    """Search children, return parents."""
    query_emb = get_embedding(query)

    # Find matching child chunks
    parent_ids = db.query(
        "SELECT parent_id FROM child_chunks ORDER BY embedding <=> %s LIMIT 3",
        query_emb
    )

    # Return full parent chunks
    parents = db.query("SELECT content FROM parent_chunks WHERE id = ANY(%s)", parent_ids)
    return "\n\n".join(parents)
```

**Alternative**: Use Agentic RAG (semantic search + full document retrieval) for similar flexibility

**See**:
- Pseudocode: [07_hierarchical_rag.py](examples/07_hierarchical_rag.py)
- Research: [docs/07-hierarchical-rag.md](docs/07-hierarchical-rag.md)

---

## 🔟 Knowledge Graphs

**Status**: 📝 Pseudocode Only (Graphiti)
**Why not in code examples**: Requires Neo4j infrastructure, entity extraction

### What It Is
Combines vector search with graph database (Neo4j) to capture entity relationships. Uses **Graphiti from Zep** for temporal knowledge graphs.

### Pros & Cons
✅ Captures relationships vectors miss, great for interconnected data
❌ Requires Neo4j setup, entity extraction, graph maintenance, slower and more expensive

### Pseudocode Concept (Graphiti)
```python
# From 10_knowledge_graphs.py (with Graphiti)
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Initialize Graphiti (connects to Neo4j)
graphiti = Graphiti("neo4j://localhost:7687", "neo4j", "password")

async def ingest_document(text: str, source: str):
    """Ingest document into Graphiti knowledge graph."""
    # Graphiti automatically extracts entities and relationships
    await graphiti.add_episode(
        name=source,
        episode_body=text,
        source=EpisodeType.text,
        source_description=f"Document: {source}"
    )

@agent.tool
async def search_knowledge_graph(query: str) -> str:
    """Hybrid search: semantic + keyword + graph traversal."""
    # Graphiti combines:
    # - Semantic similarity (embeddings)
    # - BM25 keyword search
    # - Graph structure traversal
    # - Temporal context (when was this true?)

    results = await graphiti.search(query=query, num_results=5)

    return format_graph_results(results)
```

**Framework**: [Graphiti from Zep](https://github.com/getzep/graphiti) - Temporal knowledge graphs for agents

**See**:
- Pseudocode: [10_knowledge_graphs.py](examples/10_knowledge_graphs.py)
- Research: [docs/10-knowledge-graphs.md](docs/10-knowledge-graphs.md)

---

## 1️⃣1️⃣ Fine-tuned Embeddings

**Status**: 📝 Pseudocode Only
**Why not in code examples**: Requires domain-specific training data and infrastructure

### What It Is
Train embedding models on domain-specific query-document pairs to improve retrieval accuracy for specialized domains (medical, legal, financial, etc.).

### Pros & Cons
✅ 5-10% accuracy gains, smaller models can outperform larger generic ones
❌ Requires training data, infrastructure, ongoing maintenance

### Pseudocode Concept
```python
# From 11_fine_tuned_embeddings.py
from sentence_transformers import SentenceTransformer

def prepare_training_data():
    """Create domain-specific query-document pairs."""
    return [
        ("What is EBITDA?", "financial_doc_about_ebitda.txt"),
        ("Explain capital expenditure", "capex_explanation.txt"),
        # ... thousands more domain-specific pairs
    ]

def fine_tune_model():
    """Fine-tune on domain data (one-time process)."""
    base_model = SentenceTransformer('all-MiniLM-L6-v2')
    training_data = prepare_training_data()

    # Train with MultipleNegativesRankingLoss
    fine_tuned_model = base_model.fit(
        training_data,
        epochs=3,
        loss=MultipleNegativesRankingLoss()
    )

    fine_tuned_model.save('./fine_tuned_model')

# Load fine-tuned model for embeddings
embedding_model = SentenceTransformer('./fine_tuned_model')

def get_embedding(text: str):
    """Use fine-tuned model for embeddings."""
    return embedding_model.encode(text)
```

**Alternative**: Use high-quality generic models (OpenAI text-embedding-3-small) and Contextual Retrieval

**See**:
- Pseudocode: [11_fine_tuned_embeddings.py](examples/11_fine_tuned_embeddings.py)
- Research: [docs/11-fine-tuned-embeddings.md](docs/11-fine-tuned-embeddings.md)

---

## 📊 Performance Comparison

### Ingestion Strategies

| Strategy | Speed | Cost | Quality | Status |
|----------|-------|------|---------|--------|
| Simple Chunking | ⚡⚡⚡ | $ | ⭐⭐ | ✅ Available |
| Context-Aware (Docling) | ⚡⚡ | $ | ⭐⭐⭐⭐ | ✅ Default |
| Contextual Enrichment | ⚡ | $$$ | ⭐⭐⭐⭐⭐ | ✅ Optional |
| Late Chunking | ⚡⚡ | $ | ⭐⭐⭐⭐ | 📝 Pseudocode |
| Hierarchical | ⚡⚡ | $ | ⭐⭐⭐⭐ | 📝 Pseudocode |

### Query Strategies

| Strategy | Latency | Cost | Precision | Recall | Status |
|----------|---------|------|-----------|--------|--------|
| Standard Search | ⚡⚡⚡ | $ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ Default |
| Query Expansion | ⚡⚡ | $$ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Multi-Query |
| Multi-Query | ⚡⚡ | $$ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ Code Example |
| Re-ranking | ⚡⚡ | $$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ Code Example |
| Agentic | ⚡⚡ | $$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Code Example |
| Self-Reflective | ⚡ | $$$ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Code Example |
| Knowledge Graphs | ⚡⚡ | $$$ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 📝 Pseudocode |

---

## 📂 Repository Structure

```
all-rag-strategies/
├── README.md                           # This file
├── docs/                               # Detailed research (theory + use cases)
│   ├── 01-query-expansion.md
│   ├── 02-reranking.md
│   ├── ... (all 11 strategies)
│   └── 11-fine-tuned-embeddings.md
│
├── examples/                           # Simple < 50 line examples
│   ├── 01_query_expansion.py
│   ├── 02_reranking.py
│   ├── ... (all 11 strategies)
│   ├── 11_fine_tuned_embeddings.py
│   └── README.md
│
└── implementation/                     # Educational code examples (NOT production)
    ├── rag_agent.py                    # Basic agent (single tool)
    ├── rag_agent_advanced.py           # Advanced agent (all strategies)
    ├── ingestion/
    │   ├── ingest.py                   # Main ingestion pipeline
    │   ├── chunker.py                  # Docling HybridChunker
    │   ├── embedder.py                 # OpenAI embeddings
    │   └── contextual_enrichment.py    # Anthropic's contextual retrieval
    ├── utils/
    │   ├── db_utils.py
    │   └── models.py
    ├── IMPLEMENTATION_GUIDE.md         # Exact line numbers + code
    ├── STRATEGIES.md                   # Detailed strategy documentation
    └── requirements-advanced.txt
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Framework | [Pydantic AI](https://ai.pydantic.dev/) | Type-safe agents with tool calling |
| Vector Database | PostgreSQL + [pgvector](https://github.com/pgvector/pgvector) via [Neon](https://neon.tech/) | Vector similarity search (Neon used for demonstrations) |
| Document Processing | [Docling](https://github.com/DS4SD/docling) | Hybrid chunking + multi-format |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim embeddings |
| Re-ranking | sentence-transformers | Cross-encoder for precision |
| LLM | OpenAI GPT-4o-mini | Query expansion, grading, refinement |

---

## 📚 Additional Resources

- **Implementation Details**: [implementation/IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md)
- **Strategy Theory**: [docs/](docs/) (11 detailed docs)
- **Code Examples**: [examples/README.md](examples/README.md)
- **Anthropic's Contextual Retrieval**: https://www.anthropic.com/news/contextual-retrieval
- **Graphiti (Knowledge Graphs)**: https://github.com/getzep/graphiti
- **Pydantic AI Docs**: https://ai.pydantic.dev/

---

## 🤝 Contributing

This is a demonstration/education project. Feel free to:
- Fork and adapt for your use case
- Report issues or suggestions
- Share your own RAG strategy implementations

---

## 🙏 Acknowledgments

- **Anthropic** - Contextual Retrieval methodology
- **Docling Team** - HybridChunker implementation
- **Jina AI** - Late chunking concept
- **Pydantic Team** - Pydantic AI framework
- **Zep** - Graphiti knowledge graph framework
- **Sentence Transformers** - Cross-encoder models
