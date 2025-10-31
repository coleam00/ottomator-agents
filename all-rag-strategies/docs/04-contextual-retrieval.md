# Contextual Retrieval

## Resource
**Introducing Contextual Retrieval | Anthropic**
https://www.anthropic.com/news/contextual-retrieval

## What It Is
Contextual Retrieval, introduced by Anthropic, prepends chunk-specific explanatory context to each chunk before embedding and indexing. An LLM generates a brief description explaining what each chunk is about in relation to the entire document. This technique includes Contextual Embeddings and Contextual BM25, reducing retrieval failures by 49% alone and 67% when combined with re-ranking.

## Simple Example
```python
# Original chunk (lacks context)
chunk = "The company's revenue grew by 3% over the previous quarter."

# Generate contextual prefix with LLM
context = llm.generate(
    f"Document: {full_document}\n\nChunk: {chunk}\n\n"
    "Provide brief context for this chunk:"
)
# Returns: "This chunk is from ACME Corp's Q2 2023 SEC filing;
# previous quarter revenue was $314M."

# Embed with context
contextualized_chunk = context + " " + chunk
embedding = embed_model.encode(contextualized_chunk)

# Also create contextual BM25 index with the same contextualized chunks
bm25_index.add(contextualized_chunk)
```

## Pros
Dramatically improves retrieval accuracy by adding document context to chunks. Works with both vector embeddings and BM25 keyword search.

## Cons
Significantly increases indexing time and cost due to LLM calls for every chunk. Larger index size due to additional context text.

## When to Use It
Use when chunks lack standalone meaning without document context. Ideal for technical documents, financial reports, or dense reference materials.

## When NOT to Use It
Avoid when chunks are already self-contained and clear. Skip if indexing budget or time constraints are tight, or corpus updates frequently.
