"""
Optimized tools for the Pydantic AI agent with caching and performance improvements.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .unified_db_utils import (
    vector_search as db_vector_search,
    hybrid_search as db_hybrid_search,
    get_document,
    list_documents,
    get_document_chunks
)
from .graph_utils import (
    search_knowledge_graph,
    get_entity_relationships,
    graph_client
)
from .cache_manager import cache_manager, cached
from .performance_optimizer import (
    track_performance,
    embedding_batcher,
    QueryOptimizer,
    metrics
)
from .models import ChunkResult, GraphSearchResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

# Import centralized embedding normalization
from ingestion.embedding_truncator import normalize_embedding_dimension
from agent.embedding_config import EmbeddingConfig

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize embedding client
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


# Tool Input Models (same as before)
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


@track_performance("embedding_generation")
async def generate_embedding_optimized(text: str) -> List[float]:
    """
    Generate embedding with caching and batching support.
    
    Args:
        text: Text to embed
    
    Returns:
        Embedding vector normalized to target dimension
    """
    # Check cache first
    cached_embedding = await cache_manager.cache_embedding(text)
    if cached_embedding:
        metrics.cache_hits += 1
        logger.debug(f"Using cached embedding for text: {text[:50]}...")
        return cached_embedding
    
    metrics.cache_misses += 1
    
    try:
        # Use batching for better performance
        if hasattr(embedding_batcher, 'get_embedding'):
            embedding = await embedding_batcher.get_embedding(
                text,
                lambda texts: generate_embeddings_batch(texts)
            )
        else:
            # Fallback to single embedding
            response = await embedding_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = response.data[0].embedding
        
        # Normalize embedding
        target_dim = EmbeddingConfig.get_target_dimension()
        normalized_embedding = normalize_embedding_dimension(
            embedding,
            target_dim,
            model_name=EMBEDDING_MODEL
        )
        
        # Cache the result
        await cache_manager.set_embedding(text, normalized_embedding, ttl=7200)
        
        metrics.embeddings_generated += 1
        
        return normalized_embedding
        
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts in a single API call."""
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        
        target_dim = EmbeddingConfig.get_target_dimension()
        embeddings = []
        
        for data in response.data:
            normalized = normalize_embedding_dimension(
                data.embedding,
                target_dim,
                model_name=EMBEDDING_MODEL
            )
            embeddings.append(normalized)
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise


@track_performance("vector_search")
async def vector_search_tool_optimized(input_data: VectorSearchInput) -> List[ChunkResult]:
    """
    Optimized vector search with caching.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of search results
    """
    # Generate embedding
    embedding = await generate_embedding_optimized(input_data.query)
    
    # Check cache
    cached_results = await cache_manager.cache_vector_search(
        input_data.query,
        embedding,
        input_data.limit
    )
    
    if cached_results:
        metrics.cache_hits += 1
        logger.info(f"Vector search cache hit for query: {input_data.query[:50]}...")
        return [ChunkResult(**r) for r in cached_results]
    
    metrics.cache_misses += 1
    
    # Perform search
    results = await db_vector_search(
        embedding=embedding,
        limit=input_data.limit
    )
    
    # Convert to ChunkResult objects
    chunk_results = []
    for row in results:
        chunk_results.append(ChunkResult(
            chunk_id=str(row.get("id")),
            content=row.get("content", ""),
            document_id=str(row.get("document_id")),
            similarity_score=1.0 - float(row.get("distance", 0)),
            metadata=row.get("metadata", {}),
            document_title=row.get("document_title", ""),
            document_source=row.get("document_source", "")
        ))
    
    # Cache results
    cache_data = [r.dict() for r in chunk_results]
    await cache_manager.set_vector_search(
        input_data.query,
        embedding,
        cache_data,
        input_data.limit,
        ttl=1800
    )
    
    logger.info(f"Vector search found {len(chunk_results)} results for query: {input_data.query[:50]}...")
    return chunk_results


@track_performance("graph_search")
async def graph_search_tool_optimized(input_data: GraphSearchInput) -> List[GraphSearchResult]:
    """
    Optimized knowledge graph search with caching.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of graph search results
    """
    # Check cache
    cached_results = await cache_manager.cache_graph_search(
        input_data.query,
        "similarity"
    )
    
    if cached_results:
        metrics.cache_hits += 1
        logger.info(f"Graph search cache hit for query: {input_data.query[:50]}...")
        return [GraphSearchResult(**r) for r in cached_results.get("results", [])]
    
    metrics.cache_misses += 1
    
    # Perform search
    results = await search_knowledge_graph(input_data.query)
    
    # Convert to GraphSearchResult objects
    graph_results = []
    for item in results:
        graph_results.append(GraphSearchResult(
            entity_name=item.get("name", ""),
            entity_type=item.get("type", ""),
            fact=item.get("fact", ""),
            confidence_score=item.get("score", 0.0),
            relationships=item.get("relationships", []),
            source_episode_id=item.get("source_id"),
            metadata=item.get("metadata", {})
        ))
    
    # Cache results
    cache_data = {
        "results": [r.dict() for r in graph_results]
    }
    await cache_manager.set_graph_search(
        input_data.query,
        cache_data,
        "similarity",
        ttl=1800
    )
    
    logger.info(f"Graph search found {len(graph_results)} results for query: {input_data.query[:50]}...")
    return graph_results


@track_performance("hybrid_search")
async def hybrid_search_tool_optimized(input_data: HybridSearchInput) -> List[ChunkResult]:
    """
    Optimized hybrid search combining vector and text search.
    
    Args:
        input_data: Search parameters
    
    Returns:
        List of search results
    """
    # Generate cache key
    cache_key = cache_manager._generate_key("hybrid_search", {
        "query": input_data.query,
        "limit": input_data.limit,
        "text_weight": input_data.text_weight
    })
    
    # Check cache
    cached_results = await cache_manager.get(cache_key)
    if cached_results:
        metrics.cache_hits += 1
        return [ChunkResult(**r) for r in cached_results]
    
    metrics.cache_misses += 1
    
    # Generate embedding
    embedding = await generate_embedding_optimized(input_data.query)
    
    # Perform hybrid search
    results = await db_hybrid_search(
        query=input_data.query,
        embedding=embedding,
        limit=input_data.limit,
        text_weight=input_data.text_weight
    )
    
    # Convert to ChunkResult objects
    chunk_results = []
    for row in results:
        chunk_results.append(ChunkResult(
            chunk_id=str(row.get("id")),
            content=row.get("content", ""),
            document_id=str(row.get("document_id")),
            similarity_score=float(row.get("combined_score", 0)),
            metadata=row.get("metadata", {}),
            document_title=row.get("document_title", ""),
            document_source=row.get("document_source", "")
        ))
    
    # Cache results
    cache_data = [r.dict() for r in chunk_results]
    await cache_manager.set(cache_key, cache_data, ttl=1800)
    
    logger.info(f"Hybrid search found {len(chunk_results)} results")
    return chunk_results


@cached(ttl=3600, key_prefix="list_documents")
@track_performance("list_documents")
async def list_documents_tool_optimized(input_data: DocumentListInput) -> List[DocumentMetadata]:
    """
    Optimized document listing with caching.
    
    Args:
        input_data: Listing parameters
    
    Returns:
        List of document metadata
    """
    documents = await list_documents(
        limit=input_data.limit,
        offset=input_data.offset
    )
    
    doc_list = []
    for doc in documents:
        doc_list.append(DocumentMetadata(
            document_id=str(doc.get("id")),
            title=doc.get("title", ""),
            source=doc.get("source", ""),
            created_at=doc.get("created_at"),
            chunk_count=doc.get("chunk_count", 0),
            metadata=doc.get("metadata", {})
        ))
    
    return doc_list


@cached(ttl=3600, key_prefix="get_document")
@track_performance("get_document")
async def get_document_tool_optimized(input_data: DocumentInput) -> Dict[str, Any]:
    """
    Optimized document retrieval with caching.
    
    Args:
        input_data: Document parameters
    
    Returns:
        Document data with chunks
    """
    # Get document
    document = await get_document(input_data.document_id)
    if not document:
        return {"error": "Document not found"}
    
    # Get chunks
    chunks = await get_document_chunks(input_data.document_id)
    
    return {
        "document": {
            "id": str(document.get("id")),
            "title": document.get("title"),
            "source": document.get("source"),
            "content": document.get("content"),
            "created_at": document.get("created_at"),
            "metadata": document.get("metadata", {})
        },
        "chunks": [
            {
                "id": str(chunk.get("id")),
                "content": chunk.get("content"),
                "chunk_index": chunk.get("chunk_index"),
                "metadata": chunk.get("metadata", {})
            }
            for chunk in chunks
        ]
    }


@track_performance("entity_relationships")
async def get_entity_relationships_optimized(
    input_data: EntityRelationshipInput
) -> Dict[str, Any]:
    """
    Optimized entity relationship query with caching.
    
    Args:
        input_data: Entity parameters
    
    Returns:
        Entity relationships data
    """
    # Generate cache key
    cache_key = cache_manager._generate_key("entity_relationships", {
        "entity": input_data.entity_name,
        "depth": input_data.depth
    })
    
    # Check cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        metrics.cache_hits += 1
        return cached_result
    
    metrics.cache_misses += 1
    
    # Get relationships
    relationships = await get_entity_relationships(
        input_data.entity_name,
        input_data.depth
    )
    
    # Cache result
    await cache_manager.set(cache_key, relationships, ttl=1800)
    
    return relationships


# Export optimized tools
__all__ = [
    'vector_search_tool_optimized',
    'graph_search_tool_optimized',
    'hybrid_search_tool_optimized',
    'list_documents_tool_optimized',
    'get_document_tool_optimized',
    'get_entity_relationships_optimized',
    'generate_embedding_optimized',
    'VectorSearchInput',
    'GraphSearchInput',
    'HybridSearchInput',
    'DocumentInput',
    'DocumentListInput',
    'EntityRelationshipInput'
]