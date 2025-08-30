"""
Tools for the Pydantic AI agent.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .unified_db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks
)
from .graph_utils import (
    search_knowledge_graph,
    get_entity_relationships,
    graph_client
)
from .neo4j_direct import (
    search_knowledge_base_direct,
    get_entity_relationships_direct,
    find_entity_paths,
    neo4j_direct_client
)
from .episodic_memory import episodic_memory_service
from .models import ChunkResult, GraphSearchResult, DocumentMetadata, _safe_parse_int
from .providers import get_embedding_client, get_embedding_model

# Import centralized embedding normalization
from ingestion.embedding_truncator import normalize_embedding_dimension
from agent.embedding_config import EmbeddingConfig

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize embedding client with flexible provider
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


# The normalize_embedding_dimension function is now imported from the centralized module
# No need for a local implementation


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using configured provider.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector normalized to target dimension
    """
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        
        # Get target dimension from centralized configuration
        target_dim = EmbeddingConfig.get_target_dimension()
        
        # Normalize embedding to target dimension with model info for logging
        embedding = response.data[0].embedding
        normalized_embedding = normalize_embedding_dimension(
            embedding, 
            target_dim,
            model_name=EMBEDDING_MODEL
        )
        
        logger.debug(
            f"Generated embedding: model={EMBEDDING_MODEL}, "
            f"native dim={len(embedding)}, normalized dim={len(normalized_embedding)}"
        )
        
        return normalized_embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


# Tool Input Models
class VectorSearchInput(BaseModel):
    """Input for vector search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")


class GraphSearchInput(BaseModel):
    """Input for graph search tool."""
    query: str = Field(..., description="Search query")


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(default=0.3, description="Weight for text similarity (0-1)")


class DocumentInput(BaseModel):
    """Input for document retrieval."""
    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    """Input for listing documents."""
    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")


class EntityRelationshipInput(BaseModel):
    """Input for entity relationship query."""
    entity_name: str = Field(..., description="Name of the entity")
    depth: int = Field(default=2, description="Maximum traversal depth")


class EntityTimelineInput(BaseModel):
    """Input for entity timeline query."""
    entity_name: str = Field(..., description="Name of the entity")
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")


class KnowledgeBaseSearchInput(BaseModel):
    """Input for direct knowledge base search."""
    query: str = Field(..., description="Search query for knowledge base")
    limit: int = Field(default=20, description="Maximum number of results")


class EntityPathInput(BaseModel):
    """Input for finding paths between entities."""
    entity1: str = Field(..., description="First entity name")
    entity2: str = Field(..., description="Second entity name")
    max_depth: int = Field(default=3, description="Maximum path length")


class EpisodicSearchInput(BaseModel):
    """Input for episodic memory search."""
    query: str = Field(..., description="Search query for episodic memories")
    session_id: Optional[str] = Field(None, description="Filter by session ID")
    user_id: Optional[str] = Field(None, description="Filter by user ID")
    limit: int = Field(default=10, description="Maximum number of results")


# Tool Implementation Functions
async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    """
    Perform vector similarity search.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)
        
        # Perform vector search
        results = await vector_search(
            embedding=embedding,
            limit=input_data.limit
        )

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def graph_search_tool(input_data: GraphSearchInput, group_ids: Optional[List[str]] = None) -> List[GraphSearchResult]:
    """
    Search the knowledge graph.
    
    This tool now intelligently routes between:
    - Direct Neo4j queries for medical knowledge base (group_id="0" or None)
    - Graphiti search for user interaction history (user-specific group_id)
    
    Args:
        input_data: Search parameters
        group_ids: Optional list of group IDs to filter (e.g., ["0"] for shared, [user_id] for personal)
    
    Returns:
        List of graph search results
    """
    try:
        # Determine which search strategy to use
        use_direct_neo4j = False
        
        # Use direct Neo4j for knowledge base queries (shared data)
        if group_ids is None or group_ids == ["0"] or "0" in group_ids:
            use_direct_neo4j = True
        
        if use_direct_neo4j:
            # Use direct Neo4j for knowledge base
            logger.debug(f"Using direct Neo4j search for knowledge base: {input_data.query}")
            results = await search_knowledge_base_direct(
                query=input_data.query,
                limit=20  # Get more results from knowledge base
            )
            
            # Convert direct Neo4j results to GraphSearchResult format
            graph_results = []
            for r in results:
                if r["type"] == "fact":
                    graph_results.append(
                        GraphSearchResult(
                            fact=r["fact"],
                            uuid=f"kb_{r.get('subject', '')}_{r.get('object', '')}",
                            valid_at=None,
                            invalid_at=None,
                            source_node_uuid=None
                        )
                    )
                elif r["type"] == "entity":
                    # Convert entity to fact format
                    entity_fact = f"{r['entity']}: {r.get('description', 'Entity in knowledge base')}"
                    graph_results.append(
                        GraphSearchResult(
                            fact=entity_fact,
                            uuid=f"kb_entity_{r['entity']}",
                            valid_at=None,
                            invalid_at=None,
                            source_node_uuid=None
                        )
                    )
                    # Add relationships as separate facts
                    for rel in r.get("relationships", []):
                        if rel.get("related_entity"):
                            rel_fact = f"{r['entity']} {rel.get('type', 'relates to')} {rel['related_entity']}"
                            graph_results.append(
                                GraphSearchResult(
                                    fact=rel_fact,
                                    uuid=f"kb_rel_{r['entity']}_{rel['related_entity']}",
                                    valid_at=None,
                                    invalid_at=None,
                                    source_node_uuid=None
                                )
                            )
            
            return graph_results
            
        else:
            # Use Graphiti for user interaction history
            logger.debug(f"Using Graphiti search for user history: {input_data.query}")
            results = await search_knowledge_graph(
                query=input_data.query,
                group_ids=group_ids
            )
            
            # Convert to GraphSearchResult models
            return [
                GraphSearchResult(
                    fact=r["fact"],
                    uuid=r["uuid"],
                    valid_at=r.get("valid_at"),
                    invalid_at=r.get("invalid_at"),
                    source_node_uuid=r.get("source_node_uuid")
                )
                for r in results
            ]
        
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return []


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    """
    Perform hybrid search (vector + keyword).
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)
        
        # Perform hybrid search
        results = await hybrid_search(
            embedding=embedding,
            query_text=input_data.query,
            limit=input_data.limit,
            text_weight=input_data.text_weight
        )
        
        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"]
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def get_document_tool(input_data: DocumentInput) -> Optional[Dict[str, Any]]:
    """
    Retrieve a complete document.
    
    Args:
        input_data: Document retrieval parameters
    
    Returns:
        Document data or None
    """
    try:
        document = await get_document(input_data.document_id)
        
        if document:
            # Also get all chunks for the document
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks
        
        return document
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return None


async def list_documents_tool(input_data: DocumentListInput) -> List[DocumentMetadata]:
    """
    List available documents.
    
    Args:
        input_data: Listing parameters
    
    Returns:
        List of document metadata
    """
    try:
        documents = await list_documents(
            limit=input_data.limit,
            offset=input_data.offset
        )
        
        # Convert to DocumentMetadata models
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count")
            )
            for d in documents
        ]
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        return []


async def get_entity_relationships_tool(input_data: EntityRelationshipInput) -> Dict[str, Any]:
    """
    Get relationships for an entity.
    
    Uses direct Neo4j queries for knowledge base entities.
    
    Args:
        input_data: Entity relationship parameters
    
    Returns:
        Entity relationships
    """
    try:
        # Use direct Neo4j for knowledge base entities
        logger.debug(f"Getting entity relationships via direct Neo4j: {input_data.entity_name}")
        return await get_entity_relationships_direct(
            entity=input_data.entity_name,
            depth=input_data.depth
        )
        
    except Exception as e:
        logger.error(f"Entity relationship query failed: {e}")
        return {
            "central_entity": input_data.entity_name,
            "connections": [],
            "depth": input_data.depth,
            "error": str(e)
        }


async def get_entity_timeline_tool(input_data: EntityTimelineInput) -> List[Dict[str, Any]]:
    """
    Get timeline of facts for an entity.
    
    Args:
        input_data: Timeline query parameters
    
    Returns:
        Timeline of facts
    """
    try:
        # Parse dates if provided
        start_date = None
        end_date = None
        
        if input_data.start_date:
            start_date = datetime.fromisoformat(input_data.start_date)
        if input_data.end_date:
            end_date = datetime.fromisoformat(input_data.end_date)
        
        # Get timeline from graph
        timeline = await graph_client.get_entity_timeline(
            entity_name=input_data.entity_name,
            start_date=start_date,
            end_date=end_date
        )
        
        return timeline
        
    except Exception as e:
        logger.error(f"Entity timeline query failed: {e}")
        return []


async def knowledge_base_search_tool(input_data: KnowledgeBaseSearchInput) -> List[GraphSearchResult]:
    """
    Search the medical knowledge base directly.
    
    This tool performs direct Neo4j queries on the medical knowledge base
    that was ingested directly into Neo4j. It searches for entities,
    relationships, and facts in the medical domain.
    
    Args:
        input_data: Search parameters for knowledge base
    
    Returns:
        List of knowledge base facts and entities
    """
    try:
        # Search knowledge base directly
        results = await search_knowledge_base_direct(
            query=input_data.query,
            limit=input_data.limit
        )
        
        # Convert to GraphSearchResult format
        graph_results = []
        for r in results:
            if r["type"] == "fact":
                graph_results.append(
                    GraphSearchResult(
                        fact=r["fact"],
                        uuid=f"kb_fact_{hash(r['fact'])}",
                        valid_at=None,
                        invalid_at=None,
                        source_node_uuid=None
                    )
                )
            elif r["type"] == "entity":
                # Include entity as a fact
                entity_fact = f"{r['entity']}: {r.get('description', 'Medical entity')}"
                graph_results.append(
                    GraphSearchResult(
                        fact=entity_fact,
                        uuid=f"kb_entity_{r['entity']}",
                        valid_at=None,
                        invalid_at=None,
                        source_node_uuid=None
                    )
                )
        
        return graph_results
        
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}")
        return []


async def find_entity_paths_tool(input_data: EntityPathInput) -> List[Dict[str, Any]]:
    """
    Find paths between two entities in the knowledge base.
    
    This tool finds all paths connecting two entities in the medical
    knowledge graph, showing how they are related through other entities.
    
    Args:
        input_data: Path finding parameters
    
    Returns:
        List of paths between entities
    """
    try:
        paths = await find_entity_paths(
            entity1=input_data.entity1,
            entity2=input_data.entity2,
            max_depth=input_data.max_depth
        )
        
        return paths
        
    except Exception as e:
        logger.error(f"Entity path finding failed: {e}")
        return []


async def episodic_memory_search_tool(input_data: EpisodicSearchInput) -> List[GraphSearchResult]:
    """
    Search episodic memory from previous conversations.
    
    This tool searches the conversation history stored in the knowledge graph
    to find relevant information from past interactions. Useful for maintaining
    context across sessions and remembering important facts discussed previously.
    
    Args:
        input_data: Search parameters for episodic memory
    
    Returns:
        List of relevant episodic memories
    """
    try:
        # Search episodic memories
        results = await episodic_memory_service.search_episodic_memories(
            query=input_data.query,
            session_id=input_data.session_id,
            user_id=input_data.user_id,
            limit=input_data.limit
        )
        
        # Convert to GraphSearchResult format for consistency
        return [
            GraphSearchResult(
                fact=r.get("fact", r.get("content", "")),
                uuid=r.get("uuid", ""),
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid")
            )
            for r in results
        ]
        
    except Exception as e:
        logger.error(f"Episodic memory search failed: {e}")
        return []


# Combined search function for agent use
async def perform_comprehensive_search(
    query: str,
    use_vector: bool = True,
    use_graph: bool = True,
    use_knowledge_base: bool = True,
    use_episodic: bool = False,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Perform a comprehensive search using multiple methods.
    
    This function intelligently combines different search strategies:
    - Vector search: Semantic similarity in documents
    - Knowledge base: Direct Neo4j queries on medical knowledge
    - Episodic memory: User conversation history via Graphiti
    
    Args:
        query: Search query
        use_vector: Whether to use vector search
        use_graph: Whether to use graph search (deprecated, use use_knowledge_base)
        use_knowledge_base: Whether to search medical knowledge base
        use_episodic: Whether to search user conversation history
        session_id: Session ID for episodic search
        user_id: User ID for episodic search
        limit: Maximum results per search type
    
    Returns:
        Combined search results from all sources
    """
    results = {
        "query": query,
        "vector_results": [],
        "knowledge_base_results": [],
        "episodic_results": [],
        "total_results": 0
    }
    
    tasks = []
    task_types = []
    
    # Vector search for semantic similarity
    if use_vector:
        tasks.append(vector_search_tool(VectorSearchInput(query=query, limit=limit)))
        task_types.append("vector")
    
    # Knowledge base search (direct Neo4j)
    if use_knowledge_base or use_graph:
        tasks.append(knowledge_base_search_tool(KnowledgeBaseSearchInput(query=query, limit=limit)))
        task_types.append("knowledge_base")
    
    # Episodic memory search (Graphiti)
    if use_episodic and (session_id or user_id):
        tasks.append(episodic_memory_search_tool(
            EpisodicSearchInput(
                query=query,
                session_id=session_id,
                user_id=user_id,
                limit=limit
            )
        ))
        task_types.append("episodic")
    
    if tasks:
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (result, task_type) in enumerate(zip(search_results, task_types)):
            if not isinstance(result, Exception):
                if task_type == "vector":
                    results["vector_results"] = result
                elif task_type == "knowledge_base":
                    results["knowledge_base_results"] = result
                elif task_type == "episodic":
                    results["episodic_results"] = result
            else:
                logger.error(f"{task_type} search failed: {result}")
    
    # Calculate total results
    results["total_results"] = (
        len(results["vector_results"]) + 
        len(results["knowledge_base_results"]) + 
        len(results["episodic_results"])
    )
    
    # Add metadata about search sources
    results["search_sources"] = {
        "vector": use_vector,
        "knowledge_base": use_knowledge_base or use_graph,
        "episodic": use_episodic
    }
    
    return results