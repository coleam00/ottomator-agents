# Advanced RAG Strategies Implementation

This document describes all the advanced RAG strategies implemented in this system.

## Implemented Strategies

### 1. Context-Aware Chunking ✅
**Status**: Already implemented via Docling HybridChunker
**Location**: `ingestion/chunker.py`

Uses Docling's built-in HybridChunker which:
- Respects document structure (headings, sections, tables)
- Is token-aware (uses actual tokenizer, not estimates)
- Preserves semantic coherence
- Includes heading context in chunks

**How to use**:
```bash
python -m ingestion.ingest --documents ./documents
# Hybrid chunking is enabled by default
```

### 2. Contextual Retrieval (Anthropic Method) ✅
**Status**: Newly implemented
**Location**: `ingestion/contextual_enrichment.py`

Adds document-level context to each chunk before embedding, reducing retrieval failures by 35-49%.

**How to use**:
```bash
python -m ingestion.ingest --documents ./documents --contextual
```

Example output:
```
Original chunk:
"Clean data is essential. Remove duplicates, handle missing values..."

Enriched chunk:
"This chunk from 'ML Best Practices' discusses data preparation techniques
for machine learning workflows.

Clean data is essential. Remove duplicates, handle missing values..."
```

### 3. Query Expansion ✅
**Status**: Implemented in agent
**Location**: `rag_agent_advanced.py` - `expand_query_variations()`

Generates multiple variations of a user query to improve recall.

**How it works**:
1. Takes original query
2. LLM generates 3 variations with different perspectives
3. Returns original + variations (4 total queries)

**Used by**: Multi-Query RAG strategy

### 4. Multi-Query RAG ✅
**Status**: Implemented in agent
**Location**: `rag_agent_advanced.py` - `search_with_multi_query()`

Combines query expansion with parallel execution for better recall.

**How it works**:
1. Generate query variations (4 queries)
2. Execute all searches in parallel
3. Deduplicate results by chunk ID
4. Return top results by similarity

**When to use**: Ambiguous queries or queries with multiple interpretations

### 5. Re-ranking ✅
**Status**: Implemented in agent
**Location**: `rag_agent_advanced.py` - `search_with_reranking()`

Two-stage retrieval: fast vector search → precise cross-encoder reranking.

**How it works**:
1. Stage 1: Fast vector search retrieves 20 candidates
2. Stage 2: Cross-encoder (ms-marco-MiniLM) scores each query-doc pair
3. Return top 5 after reranking

**When to use**: Precision-critical queries (legal, medical, financial)

**Model used**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### 6. Agentic RAG ✅
**Status**: Implemented in agent
**Location**: `rag_agent_advanced.py`

Agent autonomously chooses between multiple retrieval strategies.

**Available tools**:
- `search_knowledge_base()` - Standard semantic search over chunks
- `retrieve_full_document()` - Get complete document by title

**How it works**: Agent analyzes the query and decides which tool(s) to use:
- For most queries → semantic search
- When chunks lack context → full document retrieval
- Can chain tools together as needed

**Example flow**:
```
User: "What's the full policy on remote work?"
Agent:
  1. search_knowledge_base("remote work policy")
  2. Finds relevant chunks mentioning "remote work policy.pdf"
  3. retrieve_full_document("remote work policy")
  4. Returns complete document
```

### 7. Self-Reflective RAG ✅
**Status**: Implemented in agent
**Location**: `rag_agent_advanced.py` - `search_with_self_reflection()`

Evaluates retrieval quality and refines if needed.

**How it works**:
1. Perform initial search
2. LLM grades relevance (1-5 scale)
3. If score < 3, refine query and search again
4. Return final results with reflection metadata

**When to use**: Complex research questions where initial results may miss the mark

**Example**:
```
Query: "AI ethics"
Initial results: Grade 2/5 (too broad)
Refined query: "AI ethics in healthcare applications"
Final results: Grade 4/5
```

## Agent Tool Selection Guide

The advanced agent has multiple tools and automatically selects the best one:

| Query Type | Recommended Tool | Rationale |
|------------|------------------|-----------|
| General factual | `search_knowledge_base` | Fast, reliable |
| Ambiguous/broad | `search_with_multi_query` | Multiple perspectives |
| Precision-critical | `search_with_reranking` | Highest accuracy |
| Complex research | `search_with_self_reflection` | Self-correcting |
| Need full context | `retrieve_full_document` | Complete document |

## Usage

### Running the Advanced Agent

```bash
# Basic usage
python rag_agent_advanced.py

# The agent will automatically choose the best strategy
# You can also request specific strategies:
User: "Search for machine learning but use multi-query approach"
User: "Find the full document about our AI policy"
```

### Running Ingestion with Strategies

```bash
# Standard ingestion (hybrid chunking only)
python -m ingestion.ingest --documents ./documents

# With contextual enrichment (Anthropic method)
python -m ingestion.ingest --documents ./documents --contextual

# Custom chunk sizes
python -m ingestion.ingest --documents ./documents --chunk-size 500 --chunk-overlap 100

# Simple chunking (no Docling hybrid)
python -m ingestion.ingest --documents ./documents --no-semantic
```

## Performance Notes

### Context-Aware Chunking (Docling)
- ✅ Fast (no LLM calls)
- ✅ Token-precise
- ✅ Maintains document structure
- ✅ Free to use

### Contextual Enrichment
- ⚠️ Adds LLM cost (1 API call per chunk)
- ⚠️ Increases ingestion time
- ✅ 35-49% reduction in retrieval failures
- ✅ Especially valuable for technical documents

### Multi-Query RAG
- ⚠️ 1 LLM call for query expansion
- ⚠️ 4x database queries (but parallel)
- ✅ Better recall
- ✅ Good for ambiguous queries

### Re-ranking
- ⚠️ Requires cross-encoder model in memory
- ⚠️ Slower than pure vector search
- ✅ Significantly better precision
- ✅ Worth it for critical queries

### Self-Reflective RAG
- ⚠️ 2-3 LLM calls (grading + optional refinement)
- ⚠️ Highest latency
- ✅ Self-correcting
- ✅ Best for complex research

## Strategies NOT Implemented (By Design)

### Knowledge Graphs
**Reason**: Adds significant complexity (Neo4j setup, entity extraction, graph maintenance)
**Alternative**: Use Graphiti from Zep if needed (see pseudocode example)

### Fine-tuned Embeddings
**Reason**: Requires domain-specific training data and additional infrastructure
**Alternative**: Use high-quality pre-trained models (OpenAI text-embedding-3-small)

### Late Chunking
**Reason**: Docling's HybridChunker already provides context-aware chunking
**Alternative**: Current hybrid chunking + contextual enrichment achieves similar benefits

### Hierarchical RAG (Parent-Child)
**Reason**: Agentic RAG with full document retrieval provides similar benefits more flexibly
**Alternative**: Use `search_knowledge_base()` + `retrieve_full_document()` in sequence

## Dependencies

```bash
# Core dependencies
pip install pydantic-ai asyncpg pgvector openai python-dotenv

# For hybrid chunking
pip install docling transformers

# For re-ranking
pip install sentence-transformers

# For contextual enrichment
# (uses OpenAI API, no additional install needed)
```

## Configuration

All strategies are configured via:
1. **Ingestion time**: `IngestionConfig` in `utils/models.py`
2. **Query time**: Agent automatically selects appropriate tool

```python
# Ingestion config
config = IngestionConfig(
    chunk_size=1000,
    chunk_overlap=200,
    use_semantic_chunking=True,  # Docling HybridChunker
    use_contextual_enrichment=False  # Anthropic method (expensive)
)
```

## Examples

### Example 1: Simple Query
```
You: What is machine learning?
Agent: [Uses search_knowledge_base - fast and reliable]
```

### Example 2: Ambiguous Query
```
You: Tell me about Python
Agent: [Uses search_with_multi_query]
# Generates variations:
# - "Python programming language"
# - "Python snake species"
# - "Python in data science"
```

### Example 3: Needs Full Context
```
You: What's our complete refund policy?
Agent: [Uses search_knowledge_base first]
Agent: Found relevant chunks mentioning "refund_policy.pdf"
Agent: [Uses retrieve_full_document]
Agent: Here's the complete refund policy...
```

### Example 4: Complex Research
```
You: What are the ethical implications of our AI system?
Agent: [Uses search_with_self_reflection]
Initial query: "ethical implications AI system"
Grade: 2/5
Refined: "AI ethics guidelines data privacy bias"
Grade: 4/5
```

## Troubleshooting

### Contextual enrichment is slow
- Reduce `max_concurrent` in `enrich_chunks_batch()` (default: 5)
- Use for critical documents only
- Cache enriched chunks

### Re-ranking uses too much memory
- Cross-encoder model is ~100MB
- Consider using smaller model or disable re-ranking for low-memory environments

### Multi-query returns duplicates
- System deduplicates by chunk ID
- If seeing similar (not identical) results, this is expected behavior
- Adjust `limit` parameter to get more diverse results

## Future Enhancements

Potential additions (not currently planned):
1. **Hybrid Search**: Combine semantic + BM25 keyword search
2. **Query Routing**: Automatically route to best strategy based on query analysis
3. **Result Fusion**: Combine results from multiple strategies with weighted scoring
4. **Caching**: Cache query embeddings and search results for faster responses
5. **Batch Processing**: Process multiple queries efficiently

## References

- Docling HybridChunker: https://github.com/DS4SD/docling
- Anthropic Contextual Retrieval: https://www.anthropic.com/news/contextual-retrieval
- Cross-Encoders: https://www.sbert.net/examples/applications/cross-encoder/README.html
- Pydantic AI: https://ai.pydantic.dev/
