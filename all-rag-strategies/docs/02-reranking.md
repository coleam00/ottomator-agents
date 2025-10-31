# Re-ranking

## Resource
**Rerankers and Two-Stage Retrieval | Pinecone**
https://www.pinecone.io/learn/series/rag/rerankers/

## What It Is
Re-ranking uses a cross-encoder model to refine initial retrieval results by scoring query-document pairs more accurately. After a fast retriever (like vector search) returns candidate documents, the re-ranker evaluates each candidate with the query simultaneously, capturing richer semantic interactions. This two-stage approach balances speed and accuracy.

## Simple Example
```python
# Stage 1: Fast retrieval
candidates = vector_search(query, top_k=100)

# Stage 2: Re-rank with cross-encoder
reranker = CrossEncoder('ms-marco-MiniLM')
scored_results = []
for doc in candidates:
    score = reranker.predict([query, doc])
    scored_results.append((doc, score))

# Return top re-ranked results
final_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:10]
```

## Pros
Significantly improves retrieval precision by understanding query-document relationships. Works well as a refinement layer on top of existing systems.

## Cons
Computationally expensive compared to embedding models. Adds latency as each document must be processed with the query.

## When to Use It
Use when accuracy is more important than speed. Ideal for narrowing down a large candidate set to the most relevant documents.

## When NOT to Use It
Avoid when real-time performance is critical. Skip if you have limited compute resources or very large result sets to re-rank.
