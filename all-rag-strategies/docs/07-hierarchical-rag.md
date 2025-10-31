# Hierarchical RAG

## Resource
**Document Hierarchy in RAG: Enhancing AI Efficiency | Medium**
https://medium.com/@nay1228/document-hierarchy-in-rag-boosting-ai-retrieval-efficiency-aa23f21b5fb9

## What It Is
Hierarchical RAG organizes documents in parent-child relationships, retrieving small chunks for accurate matching while providing larger parent contexts for generation. Child chunks are embedded and searched, but when a match is found, the system returns the parent chunk (containing broader context) to the LLM. Metadata maintains relationships between chunks, enabling efficient navigation of the hierarchy.

## Simple Example
```python
# Index structure
document = {
    "parent": "Q2 Financial Report - Full Section",
    "children": [
        "Revenue increased 3% to $314M",
        "Operating costs decreased 5%",
        "Net profit margin improved to 12%"
    ]
}

# Embed only child chunks
for child in document["children"]:
    index.add(embed(child), metadata={"parent_id": document["parent"]})

# Retrieval
query = "What was Q2 revenue?"
child_match = vector_search(query)  # Finds "Revenue increased 3%..."

# Return parent context instead of just the child
full_context = get_parent(child_match.metadata["parent_id"])
# LLM sees entire Q2 section for better reasoning
```

## Pros
Balances retrieval precision with generation context. Reduces noise in search while providing sufficient context for reasoning.

## Cons
Requires careful design of parent-child relationships. Adds complexity to indexing and retrieval logic.

## When to Use It
Use when small chunks match better but lack sufficient context for answers. Ideal for structured documents with natural hierarchies (sections, chapters).

## When NOT to Use It
Avoid when documents lack clear hierarchical structure. Skip if simple flat chunking provides adequate context for your use case.
