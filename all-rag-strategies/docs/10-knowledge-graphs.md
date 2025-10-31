# Knowledge Graphs

## Resource
**RAG Tutorial: How to Build a RAG System on a Knowledge Graph | Neo4j**
https://neo4j.com/blog/developer/rag-tutorial/

## What It Is
Knowledge Graph RAG (GraphRAG) combines vector search with graph databases to capture both semantic meaning and explicit relationships between entities. Instead of just retrieving similar text chunks, the system queries a graph of interconnected entities (nodes) and relationships (edges), providing structured, contextual information. This grounds LLM responses in factual relationships and prevents hallucinations.

## Simple Example
```python
# Knowledge graph structure
graph = {
    "ACME Corp": {
        "type": "Company",
        "relationships": {
            "HAS_CEO": "Jane Smith",
            "REPORTED_REVENUE": "$314M",
            "LOCATED_IN": "California"
        }
    }
}

# Query combining vector + graph
query = "Who runs ACME Corp?"

# 1. Vector search finds relevant entity
entity = vector_search(query)  # Returns "ACME Corp"

# 2. Traverse graph for relationships
result = graph.query(
    "MATCH (c:Company {name: 'ACME Corp'})-[:HAS_CEO]->(ceo) RETURN ceo"
)  # Returns "Jane Smith"

# 3. LLM generates answer with structured facts
answer = llm.generate(query, context=result)
```

## Pros
Captures explicit relationships that vectors miss. Reduces hallucinations by providing structured, factual connections.

## Cons
Requires building and maintaining a knowledge graph. Complex setup and querying compared to simple vector search.

## When to Use It
Use when relationships between entities are crucial to answers. Ideal for domains with complex interconnections (healthcare, finance, research).

## When NOT to Use It
Avoid when data lacks clear entities and relationships. Skip if you need rapid prototyping without graph infrastructure investment.
