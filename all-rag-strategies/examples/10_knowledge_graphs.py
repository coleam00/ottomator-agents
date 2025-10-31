"""Knowledge Graphs RAG - Using Graphiti by Zep for temporal knowledge graphs"""
from pydantic_ai import Agent
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

agent = Agent('openai:gpt-4o', system_prompt='You are a GraphRAG assistant with Graphiti.')

# Initialize Graphiti (connects to Neo4j)
graphiti = Graphiti("neo4j://localhost:7687", "neo4j", "password")

async def ingest_document(text: str, source: str):
    """Ingest document into Graphiti knowledge graph"""
    # Graphiti automatically extracts entities and relationships
    await graphiti.add_episode(
        name=source,
        episode_body=text,
        source=EpisodeType.text,
        source_description=f"Document: {source}"
    )
    # Graphiti builds the graph incrementally with temporal awareness

@agent.tool
async def search_knowledge_graph(query: str) -> str:
    """Hybrid search: semantic + keyword + graph traversal"""
    # Graphiti's search combines:
    # - Semantic similarity (embeddings)
    # - BM25 keyword search
    # - Graph structure traversal
    # - Temporal context (when was this true?)

    results = await graphiti.search(
        query=query,
        num_results=5
    )

    # Format results from graph
    response_parts = []
    for result in results:
        response_parts.append(
            f"Entity: {result.node.name}\n"
            f"Type: {result.node.type}\n"
            f"Context: {result.context}\n"
            f"Relationships: {result.relationships}"
        )

    return "\n---\n".join(response_parts)

# Run agent
result = await agent.run("Who runs ACME Corp and what changed in Q2?")
print(result.data)
