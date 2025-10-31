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
4. [Code Examples](#-code-examples)
5. [Detailed Strategy Guide](#-detailed-strategy-guide)
6. [Repository Structure](#-repository-structure)

---

## 🎯 Strategy Overview

| # | Strategy | Status | Use Case | Pros | Cons |
|---|----------|--------|----------|------|------|
| 1 | [Re-ranking](#1-re-ranking) | ✅ Code Example | Precision-critical | Highly accurate results | Slower, more compute |
| 2 | [Agentic RAG](#2-agentic-rag) | ✅ Code Example | Flexible retrieval needs | Autonomous tool selection | More complex logic |
| 3 | [Knowledge Graphs](#3-knowledge-graphs) | 📝 Pseudocode Only | Relationship-heavy | Captures connections | Infrastructure overhead |
| 4 | [Contextual Retrieval](#4-contextual-retrieval) | ✅ Code Example | Critical documents | 35-49% better accuracy | High ingestion cost |
| 5 | [Query Expansion](#5-query-expansion) | ✅ Code Example | Ambiguous queries | Better recall, multiple perspectives | Extra LLM call, higher cost |
| 6 | [Multi-Query RAG](#6-multi-query-rag) | ✅ Code Example | Broad searches | Comprehensive coverage | Multiple API calls |
| 7 | [Context-Aware Chunking](#7-context-aware-chunking) | ✅ Code Example | All documents | Semantic coherence | Slightly slower ingestion |
| 8 | [Late Chunking](#8-late-chunking) | 📝 Pseudocode Only | Context preservation | Full document context | Requires long-context models |
| 9 | [Hierarchical RAG](#9-hierarchical-rag) | 📝 Pseudocode Only | Complex documents | Precision + context | Complex setup |
| 10 | [Self-Reflective RAG](#10-self-reflective-rag) | ✅ Code Example | Research queries | Self-correcting | Highest latency |
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
cat 01_reranking.py
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

**Example** (`05_query_expansion.py`):
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

## 1. Re-ranking

**Status**: ✅ Code Example

**File**: `rag_agent_advanced.py` (Lines 194-256)

### What It Is
Two-stage retrieval: Vector search (20-50+ candidates) → Reranking model to filter (top 5).

### Pros & Cons
✅ Significantly better precision, more knowledge considered without overwhelming LLM

❌ Slightly slower than pure vector search, uses more compute

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
- Pseudocode: [01_reranking.py](examples/01_reranking.py)
- Research: [docs/01-reranking.md](docs/01-reranking.md)

---

## 2. Agentic RAG

**Status**: ✅ Code Example

**Files**: `rag_agent_advanced.py` (Lines 263-354)

### What It Is
Agent autonomously chooses between multiple retrieval tools, example:
1. `search_knowledge_base()` - Semantic search over chunks (can include **hybrid search**: dense vector + sparse keyword/BM25)
2. `retrieve_full_document()` - Pull entire documents when chunks aren't enough

**Note**: Hybrid search (combining dense vector embeddings with sparse keyword search like BM25) is typically implemented as part of the agentic retrieval strategy, giving the agent access to both semantic similarity and keyword matching.

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
- Pseudocode: [02_agentic_rag.py](examples/02_agentic_rag.py)
- Research: [docs/02-agentic-rag.md](docs/02-agentic-rag.md)

---

## 3. Knowledge Graphs

**Status**: 📝 Pseudocode Only (Graphiti)

**Why not in code examples**: Requires Neo4j infrastructure, entity extraction

### What It Is
Combines vector search with graph databases (such as Neo4j/FalkorDB) to capture entity relationships.

### Pros & Cons
✅ Captures relationships vectors miss, great for interconnected data

❌ Requires Neo4j setup, entity extraction, graph maintenance, slower and more expensive

### Pseudocode Concept (Graphiti)
```python
# From 03_knowledge_graphs.py (with Graphiti)
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
- Pseudocode: [03_knowledge_graphs.py](examples/03_knowledge_graphs.py)
- Research: [docs/03-knowledge-graphs.md](docs/03-knowledge-graphs.md)

---

## 4. Contextual Retrieval

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
- Pseudocode: [04_contextual_retrieval.py](examples/04_contextual_retrieval.py)
- Research: [docs/04-contextual-retrieval.md](docs/04-contextual-retrieval.md)

---

## 5. Query Expansion

**Status**: ✅ Code Example

**File**: `rag_agent_advanced.py` (Lines 72-107)

### What It Is
Expands a single brief query into a more detailed, comprehensive version by adding context, related terms, and clarifying intent. Uses an LLM with a system prompt that describes how to enrich the query while maintaining the original intent.

**Example:**
- **Input:** "What is RAG?"
- **Output:** "What is Retrieval-Augmented Generation (RAG), how does it combine information retrieval with language generation, what are its key components and architecture, and what advantages does it provide for question-answering systems?"

### Pros & Cons
✅ Improved retrieval precision by adding relevant context and specificity

❌ Extra LLM call adds latency, may over-specify simple queries

### Code Example
```python
# Query expansion using system prompt to guide enrichment
async def expand_query(ctx: RunContext[None], query: str) -> str:
    """Expand a brief query into a more detailed, comprehensive version."""
    system_prompt = """You are a query expansion assistant. Take brief user queries and expand them into more detailed, comprehensive versions that:
1. Add relevant context and clarifications
2. Include related terminology and concepts
3. Specify what aspects should be covered
4. Maintain the original intent
5. Keep it as a single, coherent question

Expand the query to be 2-3x more detailed while staying focused."""

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Expand this query: {query}"}
        ],
        temperature=0.3
    )

    expanded_query = response.choices[0].message.content.strip()
    return expanded_query  # Returns ONE enhanced query
```

**Note**: This strategy returns ONE enriched query. For generating multiple query variations, see Multi-Query RAG (Strategy 6).

**See**:
- Full guide: [IMPLEMENTATION_GUIDE.md](implementation/IMPLEMENTATION_GUIDE.md#2-query-expansion)
- Pseudocode: [05_query_expansion.py](examples/05_query_expansion.py)
- Research: [docs/05-query-expansion.md](docs/05-query-expansion.md)

---

## 6. Multi-Query RAG

**Status**: ✅ Code Example

**File**: `rag_agent_advanced.py` (Lines 114-187)

### What It Is
Generates multiple different query variations/perspectives with an LLM (e.g., 3-4 variations), runs all searches concurrently, and deduplicates results. Unlike Query Expansion which enriches ONE query, this creates MULTIPLE distinct phrasings to capture different angles.

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
- Pseudocode: [06_multi_query_rag.py](examples/06_multi_query_rag.py)
- Research: [docs/06-multi-query-rag.md](docs/06-multi-query-rag.md)

---

## 7. Context-Aware Chunking

**Status**: ✅ Code Example (Default)

**File**: `ingestion/chunker.py` (Lines 70-102)

### What It Is
Intelligent document splitting that uses semantic similarity and document structure analysis to find natural chunk boundaries, rather than naive fixed-size splitting. This approach:
- Analyzes document structure (headings, sections, paragraphs, tables)
- Uses semantic analysis to identify topic boundaries
- Respects linguistic coherence within chunks
- Preserves hierarchical context (e.g., heading information)

**Implementation Example**: Docling's HybridChunker demonstrates this strategy through:
- Token-aware chunking (uses actual tokenizer, not estimates)
- Document structure preservation
- Semantic coherence
- Heading context inclusion

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
- Pseudocode: [07_context_aware_chunking.py](examples/07_context_aware_chunking.py)
- Research: [docs/07-context-aware-chunking.md](docs/07-context-aware-chunking.md)

---

## 8. Late Chunking

**Status**: 📝 Pseudocode Only

**Why not in code examples**: Docling HybridChunker provides similar benefits

### What It Is
Embed the full document through transformer first, then chunk the token embeddings (not the text). Preserves full document context in each chunk's embedding.

### Pros & Cons
✅ Maintains full document context, leverages long-context models

❌ More complex than standard chunking

### Pseudocode Concept
```python
# From 08_late_chunking.py
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
- Pseudocode: [08_late_chunking.py](examples/08_late_chunking.py)
- Research: [docs/08-late-chunking.md](docs/08-late-chunking.md)

---

## 9. Hierarchical RAG

**Status**: 📝 Pseudocode Only

**Why not in code examples**: Agentic RAG achieves similar goals for this demo

### What It Is
Parent-child chunk relationships: Search small chunks for precision, return large parent chunks for context.

**Metadata Enhancement**: Can store metadata like `section_type` ("summary", "table", "detail") and `heading_path` to intelligently decide when to return just the child vs. the parent, or to include heading context.

### Pros & Cons
✅ Balances precision (search small) with context (return big)

❌ Requires parent-child database schema

### Pseudocode Concept
```python
# From 09_hierarchical_rag.py
def ingest_hierarchical(document: str, doc_title: str):
    """Create parent-child chunk structure with simple metadata."""
    parent_chunks = [document[i:i+2000] for i in range(0, len(document), 2000)]

    for parent_id, parent in enumerate(parent_chunks):
        # Store parent with metadata (section type, heading)
        metadata = {"heading": f"{doc_title} - Section {parent_id}", "type": "detail"}
        db.execute("INSERT INTO parent_chunks (id, content, metadata) VALUES (%s, %s, %s)",
                   (parent_id, parent, metadata))

        # Children: Small chunks with parent_id
        child_chunks = [parent[j:j+500] for j in range(0, len(parent), 500)]
        for child in child_chunks:
            embedding = get_embedding(child)
            db.execute(
                "INSERT INTO child_chunks (content, embedding, parent_id) VALUES (%s, %s, %s)",
                (child, embedding, parent_id)
            )

@agent.tool
def hierarchical_search(query: str) -> str:
    """Search children, return parents with heading context."""
    query_emb = get_embedding(query)

    # Find matching children and their parent metadata
    results = db.query(
        """SELECT p.content, p.metadata
           FROM child_chunks c
           JOIN parent_chunks p ON c.parent_id = p.id
           ORDER BY c.embedding <=> %s LIMIT 3""",
        query_emb
    )

    # Return parents with heading context
    return "\n\n".join([f"[{r['metadata']['heading']}]\n{r['content']}" for r in results])
```

**Alternative**: Use Agentic RAG (semantic search + full document retrieval) for similar flexibility

**See**:
- Pseudocode: [09_hierarchical_rag.py](examples/09_hierarchical_rag.py)
- Research: [docs/09-hierarchical-rag.md](docs/09-hierarchical-rag.md)

---

## 10. Self-Reflective RAG

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
- Pseudocode: [10_self_reflective_rag.py](examples/10_self_reflective_rag.py)
- Research: [docs/10-self-reflective-rag.md](docs/10-self-reflective-rag.md)

---

## 11. Fine-tuned Embeddings

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
│   ├── 01-reranking.md
│   ├── 02-agentic-rag.md
│   ├── ... (all 11 strategies)
│   └── 11-fine-tuned-embeddings.md
│
├── examples/                           # Simple < 50 line examples
│   ├── 01_reranking.py
│   ├── 02_agentic_rag.py
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
