# Multi-Query RAG

## Resource
**Advanced RAG: Multi-Query Retriever Approach | Medium**
https://medium.com/@kbdhunga/advanced-rag-multi-query-retriever-approach-ad8cd0ea0f5b

## What It Is
Multi-Query RAG generates multiple reformulations of the original query, executes parallel searches, and aggregates results. An LLM creates diverse perspectives of the same question to overcome the limitations of distance-based retrieval. Results from all queries are combined (typically taking the unique union) to create a richer, more comprehensive result set.

## Simple Example
```python
# Generate multiple query perspectives
original_query = "How do I deploy a model?"

reformulated_queries = llm.generate([
    "What are model deployment steps?",
    "Best practices for deploying ML models",
    "Model deployment infrastructure options"
])

# Execute searches in parallel
all_results = []
for query in reformulated_queries:
    results = vector_search(query, top_k=20)
    all_results.extend(results)

# Deduplicate and return unique results
final_results = deduplicate(all_results)
```

## Pros
Mitigates single query bias and improves result diversity. Increases recall by capturing different interpretations of user intent.

## Cons
Higher computational cost due to multiple retrievals. May retrieve redundant or less relevant documents if queries overlap poorly.

## When to Use It
Use when user queries may have multiple valid interpretations. Ideal for improving recall on ambiguous or broad questions.

## When NOT to Use It
Avoid for very specific queries with clear intent. Skip when latency and cost are major constraints, or retrieval corpus is small.
