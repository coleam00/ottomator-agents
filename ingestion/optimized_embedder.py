"""
Optimized embedding generation with performance enhancements.

Key optimizations:
- Parallel batch processing with semaphore control
- Persistent Redis caching for embeddings
- Automatic dimension normalization
- Connection pooling and retry logic
- Memory-efficient streaming for large datasets
"""

import os
import asyncio
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta
import numpy as np

from openai import RateLimitError, APIError
from dotenv import load_dotenv

from .chunker import DocumentChunk
from .embedding_truncator import normalize_embedding_dimension, get_target_dimension

# Import flexible providers
from ..agent.providers import get_embedding_client, get_embedding_model

# Optional Redis support for persistent caching
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize client with flexible provider
embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()


class OptimizedEmbeddingGenerator:
    """High-performance embedding generator with caching and parallel processing."""
    
    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        parallel_workers: int = 4,
        use_redis_cache: bool = True,
        redis_url: str = "redis://localhost:6379",
        cache_ttl: int = 86400  # 24 hours
    ):
        """
        Initialize optimized embedding generator.
        
        Args:
            model: Embedding model to use
            batch_size: Number of texts per batch
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
            parallel_workers: Number of parallel workers for batch processing
            use_redis_cache: Whether to use Redis for persistent caching
            redis_url: Redis connection URL
            cache_ttl: Cache time-to-live in seconds
        """
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.parallel_workers = parallel_workers
        self.cache_ttl = cache_ttl
        
        # Model configurations with native dimensions
        self.model_configs = {
            "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
            "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
            "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191},
            "gemini-embedding-001": {"dimensions": 3072, "max_tokens": 8191},
            "nomic-embed-text": {"dimensions": 768, "max_tokens": 8192}  # Native 768
        }
        
        self.config = self.model_configs.get(
            model, 
            {"dimensions": 768, "max_tokens": 8191}
        )
        
        # Get target dimension from environment
        self.target_dimension = get_target_dimension()
        
        # Initialize caching
        self.redis_client = None
        self.use_redis = use_redis_cache and REDIS_AVAILABLE
        if self.use_redis:
            self._init_redis(redis_url)
        
        # In-memory LRU cache as fallback
        self.memory_cache: Dict[str, tuple[List[float], datetime]] = {}
        self.cache_max_size = 1000
        
        # Performance metrics
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "embeddings_generated": 0,
            "batches_processed": 0,
            "errors": 0,
            "total_time": 0
        }
    
    def _init_redis(self, redis_url: str):
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False  # We'll handle encoding/decoding
            )
            logger.info("Redis cache initialized for embeddings")
        except Exception as e:
            logger.warning(f"Redis initialization failed, using memory cache only: {e}")
            self.use_redis = False
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Include model and dimension in key for cache invalidation
        key_data = f"{self.model}:{self.target_dimension}:{text}"
        return f"emb:{hashlib.sha256(key_data.encode()).hexdigest()}"
    
    async def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache (Redis or memory)."""
        cache_key = self._get_cache_key(text)
        
        # Try Redis first
        if self.use_redis and self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    self.stats["cache_hits"] += 1
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            embedding, timestamp = self.memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                self.stats["cache_hits"] += 1
                # Update timestamp for true LRU behavior
                self.memory_cache[cache_key] = (embedding, datetime.now())
                return embedding
            else:
                # Expired
                del self.memory_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        return None
    
    async def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache (Redis and memory)."""
        cache_key = self._get_cache_key(text)
        
        # Save to Redis
        if self.use_redis and self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(embedding)
                )
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")
        
        # Save to memory cache (LRU eviction based on access time)
        if len(self.memory_cache) >= self.cache_max_size:
            # Evict least recently used (oldest access time)
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k][1]
            )
            del self.memory_cache[lru_key]
        
        self.memory_cache[cache_key] = (embedding, datetime.now())
    
    async def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[float]:
        """
        Generate embedding for a single text with caching.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
        
        Returns:
            Normalized embedding vector
        """
        # Check cache first
        if use_cache:
            cached = await self._get_from_cache(text)
            if cached is not None:
                return cached
        
        # Truncate text if needed
        if len(text) > self.config["max_tokens"] * 4:
            text = text[:self.config["max_tokens"] * 4]
        
        start_time = datetime.now()
        
        for attempt in range(self.max_retries):
            try:
                # Generate embedding with timeout
                response = await asyncio.wait_for(
                    embedding_client.embeddings.create(
                        model=self.model,
                        input=text
                    ),
                    timeout=30.0
                )
                
                # Normalize to target dimension
                embedding = response.data[0].embedding
                normalized = normalize_embedding_dimension(embedding, self.target_dimension)
                
                # Update stats
                self.stats["embeddings_generated"] += 1
                self.stats["total_time"] += (datetime.now() - start_time).total_seconds()
                
                # Cache the result
                if use_cache:
                    await self._save_to_cache(text, normalized)
                
                return normalized
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    self.stats["errors"] += 1
                    raise Exception("Embedding generation timed out")
                    
            except (RateLimitError, APIError) as e:
                if attempt == self.max_retries - 1:
                    self.stats["errors"] += 1
                    raise
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"API error, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == self.max_retries - 1:
                    self.stats["errors"] += 1
                    raise
                await asyncio.sleep(self.retry_delay * (2 ** attempt))
    
    async def generate_embeddings_parallel(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings in parallel with worker pool.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
        
        Returns:
            List of normalized embeddings
        """
        # Semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.parallel_workers)
        
        async def process_with_limit(text: str) -> List[float]:
            async with semaphore:
                return await self.generate_embedding(text, use_cache)
        
        # Process all texts in parallel with limited concurrency
        tasks = [process_with_limit(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        results = []
        for i, result in enumerate(embeddings):
            if isinstance(result, Exception):
                logger.error(f"Failed to embed text {i}: {result}")
                # Use zero vector as fallback
                results.append([0.0] * self.target_dimension)
            else:
                results.append(result)
        
        return results
    
    async def embed_chunks_streaming(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None
    ) -> AsyncIterator[DocumentChunk]:
        """
        Stream embedded chunks as they're processed (memory efficient).
        
        Args:
            chunks: List of document chunks
            progress_callback: Optional progress callback
        
        Yields:
            Embedded chunks as they're processed
        """
        total_chunks = len(chunks)
        processed = 0
        
        # Process in batches
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk.content for chunk in batch]
            
            # Generate embeddings in parallel
            embeddings = await self.generate_embeddings_parallel(texts)
            
            # Yield embedded chunks
            for chunk, embedding in zip(batch, embeddings):
                embedded_chunk = DocumentChunk(
                    content=chunk.content,
                    index=chunk.index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        **(chunk.metadata or {}),  # Safe default for None metadata
                        "embedding_model": self.model,
                        "embedding_dimension": self.target_dimension,
                        "embedding_generated_at": datetime.now().isoformat()
                    },
                    token_count=chunk.token_count
                )
                embedded_chunk.embedding = embedding
                
                processed += 1
                yield embedded_chunk
            
            # Progress callback
            if progress_callback:
                progress_callback(processed, total_chunks)
            
            self.stats["batches_processed"] += 1
    
    async def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        progress_callback: Optional[callable] = None
    ) -> List[DocumentChunk]:
        """
        Embed all chunks with optimized batch processing.
        
        Args:
            chunks: List of document chunks
            progress_callback: Optional progress callback
        
        Returns:
            List of embedded chunks
        """
        embedded_chunks = []
        
        async for chunk in self.embed_chunks_streaming(chunks, progress_callback):
            embedded_chunks.append(chunk)
        
        logger.info(f"Embedded {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests * 100
            if total_requests > 0 else 0
        )
        
        avg_time = (
            self.stats["total_time"] / self.stats["embeddings_generated"]
            if self.stats["embeddings_generated"] > 0 else 0
        )
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_embedding_time": f"{avg_time:.3f}s",
            "target_dimension": self.target_dimension,
            "native_dimension": self.config["dimensions"],
            "dimension_reduction": f"{(1 - self.target_dimension/self.config['dimensions']) * 100:.1f}%"
        }
    
    async def close(self):
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()


# Factory function with optimization defaults
def create_optimized_embedder(
    model: str = EMBEDDING_MODEL,
    enable_redis: bool = False,  # Disabled by default for simplicity
    parallel_workers: int = 4,
    **kwargs
) -> OptimizedEmbeddingGenerator:
    """
    Create optimized embedding generator.
    
    Args:
        model: Embedding model to use
        enable_redis: Whether to enable Redis caching
        parallel_workers: Number of parallel workers
        **kwargs: Additional arguments
    
    Returns:
        OptimizedEmbeddingGenerator instance
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    return OptimizedEmbeddingGenerator(
        model=model,
        use_redis_cache=enable_redis,
        redis_url=redis_url,
        parallel_workers=parallel_workers,
        **kwargs
    )


# Example usage
async def main():
    """Example usage of optimized embedder."""
    from .chunker import ChunkingConfig, create_chunker
    
    # Create chunker and optimized embedder
    config = ChunkingConfig(chunk_size=500, use_semantic_splitting=False)
    chunker = create_chunker(config)
    embedder = create_optimized_embedder(parallel_workers=4)
    
    sample_text = """
    Artificial Intelligence has revolutionized healthcare through advanced diagnostics,
    personalized treatment plans, and drug discovery acceleration. Machine learning models
    can now detect diseases earlier than traditional methods, potentially saving millions
    of lives through early intervention.
    """ * 10  # Create larger sample for testing
    
    # Chunk the document
    chunks = chunker.chunk_document(
        content=sample_text,
        title="AI in Healthcare",
        source="example.md"
    )
    
    print(f"Created {len(chunks)} chunks")
    
    # Progress callback
    def progress(current, total):
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
    
    # Generate embeddings with streaming
    print("\nStreaming embedded chunks...")
    embedded_chunks = []
    async for chunk in embedder.embed_chunks_streaming(chunks, progress):
        embedded_chunks.append(chunk)
    
    # Display stats
    print("\nPerformance Statistics:")
    stats = embedder.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test query embedding with caching
    print("\nTesting query embedding with cache...")
    query = "AI healthcare diagnostics"
    
    # First call (cache miss)
    start = datetime.now()
    embedding1 = await embedder.generate_embedding(query)
    time1 = (datetime.now() - start).total_seconds()
    
    # Second call (cache hit)
    start = datetime.now()
    embedding2 = await embedder.generate_embedding(query)
    time2 = (datetime.now() - start).total_seconds()
    
    print(f"  First call (cache miss): {time1:.3f}s")
    print(f"  Second call (cache hit): {time2:.3f}s")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Clean up
    await embedder.close()


if __name__ == "__main__":
    asyncio.run(main())