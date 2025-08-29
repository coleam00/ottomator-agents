"""
Main Pydantic AI agent for agentic RAG with knowledge graph.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    get_entity_relationships_tool,
    get_entity_timeline_tool,
    episodic_memory_search_tool,
    knowledge_base_search_tool,
    find_entity_paths_tool,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
    EntityRelationshipInput,
    EntityTimelineInput,
    EpisodicSearchInput,
    KnowledgeBaseSearchInput,
    EntityPathInput
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """Dependencies for the agent."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate session_id is a proper UUID."""
        if self.session_id:
            try:
                from uuid import UUID
                UUID(self.session_id)  # Validate UUID format
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid session_id format in AgentDependencies: {self.session_id}")
                # Don't raise here, just log - let the system handle it gracefully
        
        if self.search_preferences is None:
            self.search_preferences = {}
    
    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "use_graph": True,
                "default_limit": 10
            }


# Initialize the agent with flexible model configuration
rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT
)


# Register tools with proper docstrings (no description parameter)
@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search for relevant information using semantic similarity.
    
    This tool performs vector similarity search across document chunks
    to find semantically related content. Returns the most relevant results
    regardless of similarity score.
    
    Args:
        query: Search query to find similar content
        limit: Maximum number of results to return (1-50)
    
    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    input_data = VectorSearchInput(
        query=query,
        limit=limit
    )
    
    results = await vector_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]


@rag_agent.tool
async def knowledge_base_search(
    ctx: RunContext[AgentDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Search the medical knowledge base for facts and relationships.
    
    This tool queries the medical knowledge graph that was ingested directly
    into Neo4j. It finds entities, relationships, symptoms, treatments, and
    medical facts. Best for finding specific medical information, relationships
    between symptoms and treatments, and evidence-based medical knowledge.
    
    Args:
        query: Search query to find medical facts and relationships
    
    Returns:
        List of medical facts, entities, and relationships
    """
    input_data = KnowledgeBaseSearchInput(query=query, limit=20)
    
    # Search knowledge base directly
    results = await knowledge_base_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "source": "medical_knowledge_base"
        }
        for r in results
    ]


@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform both vector and keyword search for comprehensive results.
    
    This tool combines semantic similarity search with keyword matching
    for the best coverage. It ranks results using both vector similarity
    and text matching scores. Best for combining semantic and exact matching.
    
    Args:
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-50)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)
    
    Returns:
        List of chunks ranked by combined relevance score
    """
    input_data = HybridSearchInput(
        query=query,
        limit=limit,
        text_weight=text_weight
    )
    
    results = await hybrid_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "content": r.content,
            "score": r.score,
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id
        }
        for r in results
    ]


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete content of a specific document.
    
    This tool fetches the full document content along with all its chunks
    and metadata. Best for getting comprehensive information from a specific
    source when you need the complete context.
    
    Args:
        document_id: UUID of the document to retrieve
    
    Returns:
        Complete document data with content and metadata, or None if not found
    """
    input_data = DocumentInput(document_id=document_id)
    
    document = await get_document_tool(input_data)
    
    if document:
        # Format for agent consumption
        return {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": document["content"],
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"]
        }
    
    return None


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    List available documents with their metadata.
    
    This tool provides an overview of all documents in the knowledge base,
    including titles, sources, and chunk counts. Best for understanding
    what information sources are available.
    
    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination
    
    Returns:
        List of documents with metadata and chunk counts
    """
    input_data = DocumentListInput(limit=limit, offset=offset)
    
    documents = await list_documents_tool(input_data)
    
    # Convert to dict for agent
    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat()
        }
        for d in documents
    ]


@rag_agent.tool
async def get_entity_relationships(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Get all relationships for a specific entity in the medical knowledge base.
    
    This tool explores the medical knowledge graph to find how a specific entity
    (symptom, treatment, condition, hormone) relates to other medical entities.
    Best for understanding how symptoms relate to treatments, how hormones affect
    conditions, and mapping medical relationships.
    
    Args:
        entity_name: Name of the medical entity (e.g., "hot flashes", "estrogen", "HRT")
        depth: Maximum traversal depth for relationships (1-3)
    
    Returns:
        Entity relationships and connected entities with relationship types
    """
    input_data = EntityRelationshipInput(
        entity_name=entity_name,
        depth=depth
    )
    
    return await get_entity_relationships_tool(input_data)


@rag_agent.tool
async def get_entity_timeline(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get the timeline of facts for a specific entity.
    
    This tool retrieves chronological information about an entity,
    showing how information has evolved over time. Best for understanding
    how information about an entity has developed or changed.
    
    Args:
        entity_name: Name of the entity (e.g., "Microsoft", "AI")
        start_date: Start date in ISO format (YYYY-MM-DD), optional
        end_date: End date in ISO format (YYYY-MM-DD), optional
    
    Returns:
        Chronological list of facts about the entity with timestamps
    """
    input_data = EntityTimelineInput(
        entity_name=entity_name,
        start_date=start_date,
        end_date=end_date
    )
    
    return await get_entity_timeline_tool(input_data)


@rag_agent.tool
async def find_entity_paths(
    ctx: RunContext[AgentDependencies],
    entity1: str,
    entity2: str,
    max_depth: int = 3
) -> List[Dict[str, Any]]:
    """
    Find paths connecting two entities in the medical knowledge base.
    
    This tool discovers how two medical entities are connected through
    intermediate relationships. Useful for understanding indirect connections
    between symptoms and treatments, or how different conditions relate.
    
    Args:
        entity1: First medical entity (e.g., "menopause")
        entity2: Second medical entity (e.g., "bone density")
        max_depth: Maximum path length to search (1-4)
    
    Returns:
        List of paths showing how entities are connected
    """
    input_data = EntityPathInput(
        entity1=entity1,
        entity2=entity2,
        max_depth=max_depth
    )
    
    return await find_entity_paths_tool(input_data)


@rag_agent.tool
async def episodic_memory(
    ctx: RunContext[AgentDependencies],
    query: str
) -> List[Dict[str, Any]]:
    """
    Search episodic memory from previous conversations.
    
    This tool searches the conversation history stored via Graphiti
    to find relevant information from past interactions. It helps maintain
    context across sessions and remember important facts discussed previously.
    Use this to recall what was discussed in earlier conversations, especially
    for personalized information about the user's health concerns or symptoms.
    
    Args:
        query: What to search for in conversation history
    
    Returns:
        List of relevant episodic memories with facts and timestamps
    """
    # Use session and user context from dependencies
    input_data = EpisodicSearchInput(
        query=query,
        session_id=ctx.deps.session_id if ctx.deps else None,
        user_id=ctx.deps.user_id if ctx.deps else None,
        limit=10
    )
    
    results = await episodic_memory_search_tool(input_data)
    
    # Convert results to dict for agent
    return [
        {
            "fact": r.fact,
            "uuid": r.uuid,
            "valid_at": r.valid_at,
            "source": "conversation_history"
        }
        for r in results
    ]