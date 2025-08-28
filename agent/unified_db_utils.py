"""
Unified database utilities that can use either asyncpg or Supabase based on configuration.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Determine which database provider to use
DB_PROVIDER = os.getenv("DB_PROVIDER", "postgres").lower()

if DB_PROVIDER == "supabase":
    from .supabase_db_utils import (
        initialize_database,
        close_database,
        create_session,
        get_session,
        update_session,
        add_message,
        get_session_messages,
        get_document,
        list_documents,
        vector_search,
        hybrid_search,
        get_document_chunks,
        execute_query,
        test_connection,
        # Supabase-specific functions
        insert_document,
        insert_chunk,
        bulk_insert_chunks,
        get_database_stats
    )
    logger.info("Using Supabase database provider")
else:
    from .db_utils import (
        initialize_database,
        close_database,
        create_session,
        get_session,
        update_session,
        add_message,
        get_session_messages,
        get_document,
        list_documents,
        vector_search,
        hybrid_search,
        get_document_chunks,
        execute_query,
        test_connection
    )
    
    # For asyncpg, we need to implement the additional functions
    from .db_utils import db_pool
    import json
    
    async def insert_document(
        title: str,
        source: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Insert a new document using asyncpg."""
        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                INSERT INTO documents (title, source, content, metadata)
                VALUES ($1, $2, $3, $4)
                RETURNING id::text
                """,
                title,
                source,
                content,
                json.dumps(metadata or {})
            )
            return result["id"]
    
    async def insert_chunk(
        document_id: str,
        content: str,
        embedding: List[float],
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> str:
        """Insert a new chunk using asyncpg."""
        async with db_pool.acquire() as conn:
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            result = await conn.fetchrow(
                """
                INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                RETURNING id::text
                """,
                document_id,
                content,
                embedding_str,
                chunk_index,
                json.dumps(metadata or {}),
                token_count
            )
            return result["id"]
    
    async def bulk_insert_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple chunks using asyncpg."""
        async with db_pool.acquire() as conn:
            chunk_ids = []
            for chunk in chunks:
                embedding_str = '[' + ','.join(map(str, chunk["embedding"])) + ']'
                
                result = await conn.fetchrow(
                    """
                    INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                    VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)
                    RETURNING id::text
                    """,
                    chunk["document_id"],
                    chunk["content"],
                    embedding_str,
                    chunk["chunk_index"],
                    json.dumps(chunk.get("metadata", {})),
                    chunk.get("token_count")
                )
                chunk_ids.append(result["id"])
            return chunk_ids
    
    async def get_database_stats() -> Dict[str, Any]:
        """Get database statistics using asyncpg."""
        async with db_pool.acquire() as conn:
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
            chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions")
            message_count = await conn.fetchval("SELECT COUNT(*) FROM messages")
            
            return {
                "documents": doc_count,
                "chunks": chunk_count,
                "sessions": session_count,
                "messages": message_count
            }
    
    logger.info("Using PostgreSQL (asyncpg) database provider")


# Additional utility functions that work with both providers
async def get_provider_info() -> Dict[str, Any]:
    """Get information about the current database provider."""
    return {
        "provider": DB_PROVIDER,
        "supabase_configured": all([
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        ]),
        "postgres_configured": bool(os.getenv("DATABASE_URL"))
    }


async def validate_configuration() -> Dict[str, Any]:
    """Validate the current database configuration."""
    provider_info = await get_provider_info()
    issues = []
    
    if DB_PROVIDER == "supabase":
        if not provider_info["supabase_configured"]:
            issues.append("Supabase provider selected but SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not configured")
    else:
        if not provider_info["postgres_configured"]:
            issues.append("PostgreSQL provider selected but DATABASE_URL not configured")
    
    # Test connection
    connection_ok = await test_connection()
    
    return {
        "provider": DB_PROVIDER,
        "configuration_valid": len(issues) == 0,
        "connection_ok": connection_ok,
        "issues": issues,
        **provider_info
    }


# Enhanced search functions that work with both providers
async def comprehensive_search(
    query: str,
    search_type: str = "hybrid",
    limit: int = 10,
    text_weight: float = 0.3,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive search with multiple options.
    
    Args:
        query: Search query
        search_type: "vector", "hybrid", or "text"
        limit: Maximum number of results
        text_weight: Weight for text similarity in hybrid search
        metadata_filter: Optional metadata filter
    
    Returns:
        Search results with metadata
    """
    from .providers import get_embedding_client, get_embedding_model
    
    results = {
        "query": query,
        "search_type": search_type,
        "results": [],
        "total_results": 0,
        "provider": DB_PROVIDER
    }
    
    try:
        if search_type in ["vector", "hybrid"]:
            # Generate embedding
            embedding_client = get_embedding_client()
            embedding_model = get_embedding_model()
            
            response = await embedding_client.embeddings.create(
                model=embedding_model,
                input=query
            )
            embedding = response.data[0].embedding
            
            # Normalize embedding to target dimension (768)
            from agent.models import _safe_parse_int
            target_dim = _safe_parse_int("VECTOR_DIMENSION", 768, min_value=1, max_value=10000)
            
            # Import the normalization function
            from ingestion.embedding_truncator import normalize_embedding_dimension
            embedding = normalize_embedding_dimension(embedding, target_dim)
            
            if search_type == "vector":
                search_results = await vector_search(embedding, limit)
            else:  # hybrid
                search_results = await hybrid_search(embedding, query, limit, text_weight)
            
            results["results"] = search_results
            results["total_results"] = len(search_results)
        
        elif search_type == "text":
            # For text-only search, we'd need to implement a text search function
            # This is a placeholder
            results["results"] = []
            results["total_results"] = 0
            results["error"] = "Text-only search not implemented yet"
    
    except Exception as e:
        logger.error(f"Comprehensive search failed: {e}")
        results["error"] = str(e)
    
    return results


# Migration and maintenance functions
async def migrate_to_supabase(
    supabase_url: str,
    supabase_service_key: str,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Migrate data from PostgreSQL to Supabase.
    Note: This is a placeholder for a full migration function.
    """
    if DB_PROVIDER != "postgres":
        return {"error": "Migration only available from PostgreSQL provider"}
    
    migration_log = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "documents_migrated": 0,
        "chunks_migrated": 0,
        "sessions_migrated": 0,
        "messages_migrated": 0,
        "errors": []
    }
    
    try:
        # This would implement the actual migration logic
        # For now, it's just a placeholder
        migration_log["status"] = "not_implemented"
        migration_log["message"] = "Migration function needs to be implemented"
        
    except Exception as e:
        migration_log["errors"].append(str(e))
        migration_log["status"] = "failed"
    
    migration_log["completed_at"] = datetime.now(timezone.utc).isoformat()
    return migration_log


# Health check and monitoring
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check for the database."""
    health = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": DB_PROVIDER,
        "status": "unknown"
    }
    
    try:
        # Test basic connection
        connection_ok = await test_connection()
        health["connection"] = "ok" if connection_ok else "failed"
        
        if connection_ok:
            # Get basic stats
            stats = await get_database_stats()
            health["stats"] = stats
            
            # Check if we have data
            has_data = any(count > 0 for count in stats.values())
            health["has_data"] = has_data
            
            health["status"] = "healthy"
        else:
            health["status"] = "connection_failed"
    
    except Exception as e:
        health["status"] = "error"
        health["error"] = str(e)
    
    return health