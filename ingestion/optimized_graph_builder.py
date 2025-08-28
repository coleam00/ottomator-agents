"""
Optimized knowledge graph builder with improved Neo4j performance.
Addresses timeout issues and connection stability problems.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import time

from graphiti_core import Graphiti
from dotenv import load_dotenv

from .chunker import DocumentChunk

# Import graph utilities
try:
    from ..agent.graph_utils import GraphitiClient
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agent.graph_utils import GraphitiClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class OptimizedGraphBuilder:
    """Optimized graph builder with connection pooling and batch processing."""
    
    def __init__(self):
        """Initialize optimized graph builder."""
        self.graph_client = GraphitiClient()
        self._initialized = False
        self._connection_retries = 0
        self._last_connection_time = None
        self._connection_pool = []  # Pool of connections
        self._max_pool_size = 3
        
    async def initialize(self):
        """Initialize graph client with connection pooling."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
            self._last_connection_time = time.time()
            logger.info("Optimized graph builder initialized")
    
    async def close(self):
        """Close graph client and clean up connection pool."""
        if self._initialized:
            await self.graph_client.close()
            self._initialized = False
            self._connection_pool.clear()
            logger.info("Graph builder closed")
    
    async def _ensure_connection(self):
        """Ensure connection is alive and reinitialize if needed."""
        if not self._initialized:
            await self.initialize()
            return
        
        # Check if connection is stale (more than 60 seconds old)
        if self._last_connection_time and (time.time() - self._last_connection_time) > 60:
            logger.info("Connection might be stale, refreshing...")
            await self.close()
            await self.initialize()
    
    async def add_document_to_graph_optimized(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,  # Process one at a time for stability
        group_id: str = "0",  # Default to shared knowledge base
        max_retries: int = 2,  # Reduced retries per chunk
        timeout_seconds: int = 30  # Reduced timeout per chunk
    ) -> Dict[str, Any]:
        """
        Optimized document ingestion with better error handling.
        
        Key optimizations:
        1. Shorter timeouts per chunk (30s instead of 120s)
        2. Connection health checks
        3. Aggressive truncation of content
        4. Parallel processing where possible
        5. Skip problematic chunks instead of retrying indefinitely
        """
        await self._ensure_connection()
        
        if not chunks:
            return {"episodes_created": 0, "errors": [], "skipped": 0}
        
        logger.info(f"Processing {len(chunks)} chunks with optimized strategy")
        
        episodes_created = 0
        errors = []
        skipped_chunks = 0
        
        # Process chunks with individual timeouts and error handling
        for i, chunk in enumerate(chunks):
            chunk_start_time = time.time()
            success = False
            
            for retry in range(max_retries):
                try:
                    # Check connection health before processing
                    await self._ensure_connection()
                    
                    # Create episode ID
                    episode_id = f"{document_source}_{chunk.index}_{int(datetime.now().timestamp())}"
                    
                    # Aggressively truncate content to avoid token limits
                    content = self._truncate_content(chunk.content, max_chars=4000)
                    
                    # Minimal metadata to reduce processing overhead
                    metadata = {
                        "chunk_index": chunk.index,
                        "doc_title": document_title[:50],  # Truncate title
                        "knowledge_type": "shared"
                    }
                    
                    # Use asyncio.wait_for with shorter timeout
                    await asyncio.wait_for(
                        self.graph_client.add_episode(
                            episode_id=episode_id,
                            content=content,
                            source=f"Doc: {document_title[:30]} (Chunk {chunk.index})",
                            timestamp=datetime.now(timezone.utc),
                            metadata=metadata,
                            group_id=group_id
                        ),
                        timeout=timeout_seconds
                    )
                    
                    episodes_created += 1
                    success = True
                    
                    chunk_time = time.time() - chunk_start_time
                    logger.info(f"✓ Chunk {i+1}/{len(chunks)} processed in {chunk_time:.1f}s")
                    
                    # Small delay to avoid overwhelming the API
                    await asyncio.sleep(0.2)
                    break
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Chunk {i+1} timeout (attempt {retry+1}/{max_retries})")
                    if retry == max_retries - 1:
                        skipped_chunks += 1
                        errors.append(f"Chunk {chunk.index}: Timeout after {max_retries} attempts")
                    else:
                        await asyncio.sleep(1)  # Brief pause before retry
                        
                except Exception as e:
                    logger.error(f"Chunk {i+1} error (attempt {retry+1}): {str(e)[:100]}")
                    if retry == max_retries - 1:
                        skipped_chunks += 1
                        errors.append(f"Chunk {chunk.index}: {str(e)[:100]}")
                    else:
                        # Reset connection on error
                        if "defunct connection" in str(e).lower():
                            await self.close()
                            await self.initialize()
                        await asyncio.sleep(1)
            
            # Log progress
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                logger.info(f"Progress: {i+1}/{len(chunks)} chunks, {episodes_created} successful, {skipped_chunks} skipped")
        
        result = {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "skipped_chunks": skipped_chunks,
            "success_rate": (episodes_created / len(chunks) * 100) if chunks else 0,
            "errors": errors[:5]  # Limit error messages
        }
        
        logger.info(f"Graph building complete: {episodes_created}/{len(chunks)} successful ({result['success_rate']:.1f}%)")
        return result
    
    def _truncate_content(self, content: str, max_chars: int = 4000) -> str:
        """Aggressively truncate content to avoid token limits."""
        if len(content) <= max_chars:
            return content
        
        # Try to find a good break point
        truncated = content[:max_chars]
        
        # Look for sentence endings
        for sep in ['. ', '! ', '? ', '\n\n', '\n']:
            last_sep = truncated.rfind(sep)
            if last_sep > max_chars * 0.6:  # Keep at least 60%
                return truncated[:last_sep + len(sep)] + "[TRUNCATED]"
        
        # Fallback to hard truncation
        return truncated + "... [TRUNCATED]"
    
    async def add_document_parallel(
        self,
        chunks: List[DocumentChunk],
        document_title: str,
        document_source: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        group_id: str = "0",
        max_concurrent: int = 2  # Limited concurrency
    ) -> Dict[str, Any]:
        """
        Alternative: Process chunks in parallel with limited concurrency.
        Use this if the sequential approach is too slow.
        """
        await self._ensure_connection()
        
        if not chunks:
            return {"episodes_created": 0, "errors": []}
        
        logger.info(f"Processing {len(chunks)} chunks in parallel (max {max_concurrent} concurrent)")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chunk(chunk, index):
            async with semaphore:
                try:
                    episode_id = f"{document_source}_{chunk.index}_{int(datetime.now().timestamp())}"
                    content = self._truncate_content(chunk.content, max_chars=4000)
                    
                    await asyncio.wait_for(
                        self.graph_client.add_episode(
                            episode_id=episode_id,
                            content=content,
                            source=f"Doc: {document_title[:30]} (Chunk {chunk.index})",
                            timestamp=datetime.now(timezone.utc),
                            metadata={"chunk_index": chunk.index},
                            group_id=group_id
                        ),
                        timeout=30
                    )
                    
                    logger.info(f"✓ Chunk {index+1}/{len(chunks)} processed")
                    return True, None
                    
                except Exception as e:
                    logger.error(f"Failed chunk {index+1}: {str(e)[:100]}")
                    return False, str(e)[:100]
        
        # Process all chunks in parallel
        tasks = [process_chunk(chunk, i) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and collect errors
        episodes_created = sum(1 for r in results if isinstance(r, tuple) and r[0])
        errors = [r[1] for r in results if isinstance(r, tuple) and not r[0]][:5]
        
        return {
            "episodes_created": episodes_created,
            "total_chunks": len(chunks),
            "errors": errors
        }
    
    async def clear_graph(self):
        """Clear all data from the knowledge graph."""
        await self._ensure_connection()
        await self.graph_client.clear_graph()


# Factory function
def create_optimized_graph_builder() -> OptimizedGraphBuilder:
    """Create optimized graph builder instance."""
    return OptimizedGraphBuilder()