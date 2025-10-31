# Query Expansion

## Resource
**Advanced RAG: Query Expansion | Haystack**
https://haystack.deepset.ai/blog/query-expansion

## What It Is
Query Expansion enhances user queries by generating multiple variations or adding related terms before retrieval. An LLM automatically generates additional queries from different perspectives, capturing various aspects of the user's intent. This addresses vague or poorly formed queries and helps cover synonyms and similar meanings.

## Simple Example
```python
# Original query
user_query = "What is RAG?"

# LLM generates expanded queries
expanded_queries = [
    "What is Retrieval Augmented Generation?",
    "How does RAG work in AI systems?",
    "Explain RAG architecture and components"
]

# Retrieve documents for all queries
for query in expanded_queries:
    results = vector_search(query)
```

## Pros
Improves retrieval recall by capturing multiple interpretations of the query. Handles vague queries and terminology variations effectively.

## Cons
Increases latency and cost due to multiple LLM calls and retrievals. May introduce noise if expanded queries drift from original intent.

## When to Use It
Use when users provide short, ambiguous, or poorly-worded queries. Ideal for keyword-based retrieval systems that need semantic variations.

## When NOT to Use It
Avoid when queries are already specific and well-formed. Skip if latency is critical or when operating under strict cost constraints.
