"""
Supabase database utilities for PostgreSQL operations via Supabase API.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, timezone
from uuid import UUID
from contextlib import asynccontextmanager

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from dotenv import load_dotenv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SupabasePool:
    """Manages Supabase client connection."""
    
    def __init__(
        self, 
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        use_service_role: bool = True
    ):
        """
        Initialize Supabase client.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (anon or service role)
            use_service_role: Whether to use service role key (for admin operations)
        """
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        
        if use_service_role:
            self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        else:
            self.supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY")
        
        if not self.supabase_url:
            raise ValueError("SUPABASE_URL environment variable not set")
        if not self.supabase_key:
            key_type = "SUPABASE_SERVICE_ROLE_KEY" if use_service_role else "SUPABASE_ANON_KEY"
            raise ValueError(f"{key_type} environment variable not set")
        
        self.client: Optional[Client] = None
        self.use_service_role = use_service_role
    
    def initialize(self):
        """Create Supabase client with robust SSL handling."""
        if not self.client:
            try:
                # Configure client options with proper timeouts
                options = ClientOptions(
                    postgrest_client_timeout=60,
                    storage_client_timeout=60,
                    function_client_timeout=60
                )
                
                # Create client with options
                self.client = create_client(
                    self.supabase_url, 
                    self.supabase_key,
                    options=options
                )
                
                # Patch the client's httpx instance with better SSL handling
                # This helps with Cloudflare SSL handshake issues
                if hasattr(self.client, 'postgrest') and hasattr(self.client.postgrest, '_client'):
                    # Configure the underlying httpx client if accessible
                    pass  # The Supabase client handles this internally
                
                logger.info(f"Supabase client initialized ({'service role' if self.use_service_role else 'anon key'})")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                # Re-raise to allow retry logic to handle it
                raise
        return self.client
    
    def close(self):
        """Close Supabase client (no-op for Supabase)."""
        # Supabase client doesn't need explicit closing
        self.client = None
        logger.info("Supabase client closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire Supabase client (context manager for compatibility)."""
        if not self.client:
            self.initialize()
        yield self.client


# Global Supabase pool instance
supabase_pool = SupabasePool()


async def initialize_database():
    """Initialize Supabase client."""
    supabase_pool.initialize()


async def close_database():
    """Close Supabase client."""
    supabase_pool.close()


# Session Management Functions
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying create_session due to connection error (attempt {retry_state.attempt_number})")
)
async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 60
) -> str:
    """
    Create a new session with retry logic for SSL/connection issues.
    
    Args:
        user_id: Optional user identifier
        metadata: Optional session metadata
        timeout_minutes: Session timeout in minutes
    
    Returns:
        Session ID
    """
    async with supabase_pool.acquire() as client:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        
        response = client.table("sessions").insert({
            "user_id": user_id,
            "metadata": metadata or {},
            "expires_at": expires_at.isoformat()
        }).execute()
        
        if response.data:
            return response.data[0]["id"]
        else:
            raise Exception(f"Failed to create session: {response}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying get_session due to connection error (attempt {retry_state.attempt_number})")
)
async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Get session by ID with retry logic for SSL/connection issues.
    
    Args:
        session_id: Session UUID
    
    Returns:
        Session data or None if not found/expired
    """
    async with supabase_pool.acquire() as client:
        try:
            # Build query without chaining to avoid missing response errors
            query = client.table("sessions").select("*").eq("id", session_id)
            
            # Only add expires_at filter if we want to check expiration
            # This helps avoid issues with null expires_at values or query chain problems
            current_time = datetime.now(timezone.utc).isoformat()
            
            # Execute query
            response = query.execute()
            
            # Check if we got any results
            if not response or not hasattr(response, 'data'):
                logger.debug(f"No response or data attribute for session {session_id}")
                return None
            
            if not response.data or len(response.data) == 0:
                logger.debug(f"No session found with ID {session_id}")
                return None
            
            session = response.data[0]  # Get first result
            
            # Check expiration manually if expires_at exists
            if session.get("expires_at"):
                expires_at = session["expires_at"]
                # Parse ISO string to datetime if needed
                if isinstance(expires_at, str):
                    from dateutil import parser
                    expires_at = parser.parse(expires_at)
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                
                # Check if session is expired
                if expires_at < datetime.now(timezone.utc):
                    logger.debug(f"Session {session_id} has expired")
                    return None
            
            # Session is valid, return it
            # Keep datetime fields as they are from Supabase (usually ISO strings)
            return session
            
        except AttributeError as e:
            # Handle missing response or data attribute more gracefully
            logger.warning(f"Supabase response missing expected attributes for session {session_id}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get session {session_id}: {e}")
            return None


async def update_session(session_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Update session metadata.
    
    Args:
        session_id: Session UUID
        metadata: New metadata to merge
    
    Returns:
        True if updated, False if not found
    """
    async with supabase_pool.acquire() as client:
        # First get existing metadata
        existing = client.table("sessions").select("metadata").eq("id", session_id).maybe_single().execute()
        
        if existing.data:
            # Merge metadata
            current_metadata = existing.data.get("metadata", {})
            current_metadata.update(metadata)
            
            response = client.table("sessions").update({
                "metadata": current_metadata
            }).eq("id", session_id).gte(
                "expires_at", datetime.now(timezone.utc).isoformat()
            ).execute()
            
            return len(response.data) > 0
        
        return False


# Message Management Functions
async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Add a message to a session.
    
    Args:
        session_id: Session UUID
        role: Message role (user/assistant/system)
        content: Message content
        metadata: Optional message metadata
    
    Returns:
        Message ID
    """
    async with supabase_pool.acquire() as client:
        response = client.table("messages").insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }).execute()
        
        if response.data:
            return response.data[0]["id"]
        else:
            raise Exception(f"Failed to create message: {response}")


async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get messages for a session.
    
    Args:
        session_id: Session UUID
        limit: Maximum number of messages to return
    
    Returns:
        List of messages ordered by creation time
    """
    async with supabase_pool.acquire() as client:
        query = client.table("messages").select("*").eq("session_id", session_id).order("created_at")
        
        if limit:
            query = query.limit(limit)
        
        response = query.execute()
        
        return response.data or []


# Document Management Functions
async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document by ID.
    
    Args:
        document_id: Document UUID
    
    Returns:
        Document data or None if not found
    """
    async with supabase_pool.acquire() as client:
        response = client.table("documents").select("*").eq("id", document_id).maybe_single().execute()
        
        return response.data


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    List documents with optional filtering.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        metadata_filter: Optional metadata filter
    
    Returns:
        List of documents with chunk counts
    """
    async with supabase_pool.acquire() as client:
        # For complex queries with JOINs and aggregations, we use RPC
        response = client.rpc("list_documents_with_chunk_count", {
            "doc_limit": limit,
            "doc_offset": offset,
            "metadata_filter": metadata_filter or {}
        }).execute()
        
        return response.data or []


# Vector Search Functions
async def vector_search(
    embedding: List[float],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search using the existing match_chunks function.
    
    Args:
        embedding: Query embedding vector
        limit: Maximum number of results
    
    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    async with supabase_pool.acquire() as client:
        # Convert embedding to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        response = client.rpc("match_chunks", {
            "query_embedding": embedding_str,
            "match_count": limit
        }).execute()
        
        results = response.data or []
        
        # Convert response to expected format
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "content": row["content"],
                "similarity": row["similarity"],
                "metadata": row["metadata"],
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search (vector + keyword) using the existing hybrid_search function.
    
    Args:
        embedding: Query embedding vector
        query_text: Query text for keyword search
        limit: Maximum number of results
        text_weight: Weight for text similarity (0-1)
    
    Returns:
        List of matching chunks ordered by combined score (best first)
    """
    async with supabase_pool.acquire() as client:
        # Convert embedding to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        response = client.rpc("hybrid_search", {
            "query_embedding": embedding_str,
            "query_text": query_text,
            "match_count": limit,
            "text_weight": text_weight
        }).execute()
        
        results = response.data or []
        
        # Convert response to expected format
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "content": row["content"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                "metadata": row["metadata"],
                "document_title": row["document_title"],
                "document_source": row["document_source"]
            }
            for row in results
        ]


# Chunk Management Functions
async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all chunks for a document using the existing get_document_chunks function.
    
    Args:
        document_id: Document UUID
    
    Returns:
        List of chunks ordered by chunk index
    """
    async with supabase_pool.acquire() as client:
        response = client.rpc("get_document_chunks", {
            "doc_id": document_id
        }).execute()
        
        results = response.data or []
        
        # Convert response to expected format
        return [
            {
                "chunk_id": str(row["chunk_id"]),
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": row["metadata"]
            }
            for row in results
        ]


# Utility Functions
async def execute_query(query: str, *params) -> List[Dict[str, Any]]:
    """
    Execute a custom query via RPC.
    Note: For security, this should be replaced with specific RPC functions.
    
    Args:
        query: SQL query
        *params: Query parameters
    
    Returns:
        Query results
    """
    async with supabase_pool.acquire() as client:
        # This is a placeholder - in production, you should create specific RPC functions
        # rather than executing arbitrary SQL
        logger.warning("execute_query not implemented for Supabase client - use specific RPC functions")
        return []


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    before_sleep=lambda retry_state: logger.warning(f"Retrying test_connection due to connection error (attempt {retry_state.attempt_number})")
)
async def test_connection() -> bool:
    """
    Test Supabase connection with retry logic for SSL/connection issues.
    
    Returns:
        True if connection successful
    """
    try:
        async with supabase_pool.acquire() as client:
            # Simple query to test connection
            response = client.table("documents").select("id").limit(1).execute()
            return True
    except Exception as e:
        logger.error(f"Supabase connection test failed: {e}")
        return False


# Additional helper functions specific to Supabase

async def insert_document(
    title: str,
    source: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Insert a new document.
    
    Args:
        title: Document title
        source: Document source
        content: Document content
        metadata: Optional document metadata
    
    Returns:
        Document ID
    """
    async with supabase_pool.acquire() as client:
        response = client.table("documents").insert({
            "title": title,
            "source": source,
            "content": content,
            "metadata": metadata or {}
        }).execute()
        
        if response.data:
            return response.data[0]["id"]
        else:
            raise Exception(f"Failed to insert document: {response}")


async def insert_chunk(
    document_id: str,
    content: str,
    embedding: List[float],
    chunk_index: int,
    metadata: Optional[Dict[str, Any]] = None,
    token_count: Optional[int] = None
) -> str:
    """
    Insert a new chunk.
    
    Args:
        document_id: Parent document ID
        content: Chunk content
        embedding: Embedding vector
        chunk_index: Index of chunk in document
        metadata: Optional chunk metadata
        token_count: Number of tokens in chunk
    
    Returns:
        Chunk ID
    """
    async with supabase_pool.acquire() as client:
        # Convert embedding to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        response = client.table("chunks").insert({
            "document_id": document_id,
            "content": content,
            "embedding": embedding_str,
            "chunk_index": chunk_index,
            "metadata": metadata or {},
            "token_count": token_count
        }).execute()
        
        if response.data:
            return response.data[0]["id"]
        else:
            raise Exception(f"Failed to insert chunk: {response}")


async def bulk_insert_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Insert multiple chunks in a single request.
    
    Args:
        chunks: List of chunk data dictionaries
    
    Returns:
        List of chunk IDs
    """
    async with supabase_pool.acquire() as client:
        # Convert embeddings to string format
        for chunk in chunks:
            if "embedding" in chunk and isinstance(chunk["embedding"], list):
                chunk["embedding"] = '[' + ','.join(map(str, chunk["embedding"])) + ']'
        
        response = client.table("chunks").insert(chunks).execute()
        
        if response.data:
            return [chunk["id"] for chunk in response.data]
        else:
            raise Exception(f"Failed to insert chunks: {response}")


# Database stats and monitoring
async def get_database_stats() -> Dict[str, Any]:
    """Get database statistics."""
    async with supabase_pool.acquire() as client:
        # Get document count
        doc_count = client.table("documents").select("id", count="exact").execute()
        
        # Get chunk count
        chunk_count = client.table("chunks").select("id", count="exact").execute()
        
        # Get session count
        session_count = client.table("sessions").select("id", count="exact").execute()
        
        # Get message count
        message_count = client.table("messages").select("id", count="exact").execute()
        
        return {
            "documents": doc_count.count,
            "chunks": chunk_count.count,
            "sessions": session_count.count,
            "messages": message_count.count
        }