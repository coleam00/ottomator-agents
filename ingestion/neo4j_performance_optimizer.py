"""
Advanced Neo4j Performance Optimizer
Implements multiple optimization strategies to eliminate bottlenecks during ingestion.
"""

import os
import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
import json
from collections import deque
from contextlib import asynccontextmanager

from graphiti_core import Graphiti
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing optimization."""
    max_batch_size: int = 10
    batch_timeout: float = 30.0  # Per-chunk timeout
    document_timeout: float = 300.0  # Per-document timeout
    max_retries: int = 3  # Increased for better resilience
    retry_delay: float = 2.0  # Increased base delay
    enable_parallel: bool = False
    max_concurrent: int = 3
    content_truncation_limit: int = 2000  # Aggressive truncation for Graphiti
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_cooldown: float = 30.0
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.max_batch_size < 1:
            self.max_batch_size = 1
        if self.batch_timeout < 5:
            self.batch_timeout = 5.0
        if self.max_concurrent < 1:
            self.max_concurrent = 1


@dataclass
class PerformanceMetrics:
    """Track performance metrics for optimization."""
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_chunks: int = 0
    total_time: float = 0
    avg_chunk_time: float = 0
    min_chunk_time: float = float('inf')
    max_chunk_time: float = 0
    timeouts: int = 0
    connection_errors: int = 0
    processing_times: List[float] = field(default_factory=list)
    
    def update(self, success: bool, processing_time: float, error_type: Optional[str] = None):
        """Update metrics with new processing result."""
        self.total_chunks += 1
        if success:
            self.successful_chunks += 1
        else:
            self.failed_chunks += 1
            if error_type == "timeout":
                self.timeouts += 1
            elif error_type == "connection":
                self.connection_errors += 1
        
        self.processing_times.append(processing_time)
        self.total_time += processing_time
        self.avg_chunk_time = self.total_time / self.total_chunks
        self.min_chunk_time = min(self.min_chunk_time, processing_time)
        self.max_chunk_time = max(self.max_chunk_time, processing_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "total_chunks": self.total_chunks,
            "successful_chunks": self.successful_chunks,
            "failed_chunks": self.failed_chunks,
            "success_rate": (self.successful_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0,
            "avg_processing_time": self.avg_chunk_time,
            "min_processing_time": self.min_chunk_time if self.min_chunk_time != float('inf') else 0,
            "max_processing_time": self.max_chunk_time,
            "total_time": self.total_time,
            "timeouts": self.timeouts,
            "connection_errors": self.connection_errors
        }


class ConnectionPool:
    """Manage Neo4j connection pooling with health checks."""
    
    def __init__(self, uri: str, auth: Tuple[str, str], pool_size: int = 5):
        self.uri = uri
        self.auth = auth
        self.pool_size = pool_size
        self.drivers = []
        self.available = deque()
        self.in_use = set()
        self._lock = asyncio.Lock()
        self._closed = False
        
    async def initialize(self):
        """Initialize connection pool."""
        config = {
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
            "connection_acquisition_timeout": 30,
            "connection_timeout": 10,
            "keep_alive": True,
        }
        
        for _ in range(self.pool_size):
            driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth, **config)
            try:
                await driver.verify_connectivity()
                self.drivers.append(driver)
                self.available.append(driver)
            except Exception as e:
                logger.warning(f"Failed to create driver: {e}")
        
        if not self.drivers:
            raise RuntimeError("Failed to initialize any connections")
        
        logger.info(f"Connection pool initialized with {len(self.drivers)} connections")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        async with self._lock:
            while not self.available:
                await asyncio.sleep(0.1)
            
            driver = self.available.popleft()
            self.in_use.add(driver)
        
        try:
            # Verify connection is still alive
            try:
                await driver.verify_connectivity()
            except:
                # Replace dead connection
                await driver.close()
                config = {
                    "max_connection_lifetime": 3600,
                    "max_connection_pool_size": 50,
                    "connection_acquisition_timeout": 30,
                    "connection_timeout": 10,
                    "keep_alive": True,
                }
                driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth, **config)
                await driver.verify_connectivity()
            
            yield driver
            
        finally:
            async with self._lock:
                self.in_use.discard(driver)
                self.available.append(driver)
    
    async def close(self):
        """Close all connections in the pool."""
        self._closed = True
        for driver in self.drivers:
            try:
                await driver.close()
            except:
                pass
        self.drivers.clear()
        self.available.clear()
        self.in_use.clear()


class OptimizedNeo4jProcessor:
    """Optimized Neo4j processor with multiple performance enhancements."""
    
    def __init__(self, batch_config: Optional[BatchConfig] = None):
        """Initialize optimized processor."""
        self.config = batch_config or BatchConfig()
        self.metrics = PerformanceMetrics()
        
        # Neo4j configuration
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        if not all([self.neo4j_uri, self.neo4j_password]):
            raise ValueError("Neo4j credentials not configured")
        
        # Connection management
        self.connection_pool: Optional[ConnectionPool] = None
        self._cache = {}  # Simple cache for deduplication
        self._initialized = False
        
        # Circuit breaker pattern with configurable settings
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.config.circuit_breaker_threshold
        self.circuit_open = False
        self.circuit_open_until = 0
        self.circuit_breaker_cooldown = self.config.circuit_breaker_cooldown
        
    async def initialize(self):
        """Initialize processor with connection pool."""
        if self._initialized:
            return
        
        try:
            # Initialize connection pool
            self.connection_pool = ConnectionPool(
                self.neo4j_uri,
                (self.neo4j_user, self.neo4j_password),
                pool_size=self.config.max_concurrent
            )
            await self.connection_pool.initialize()
            
            self._initialized = True
            logger.info("Neo4j processor initialized with optimizations")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j processor: {e}")
            raise
    
    async def close(self):
        """Close processor and cleanup resources."""
        if self.connection_pool:
            await self.connection_pool.close()
        self._cache.clear()
        self._initialized = False
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_open:
            return False
        
        # Check if cooldown period has passed
        if time.time() > self.circuit_open_until:
            self.circuit_open = False
            self.consecutive_failures = 0
            logger.info("Circuit breaker reset")
            return False
        
        return True
    
    def _trip_circuit_breaker(self):
        """Trip the circuit breaker after too many failures."""
        if not self.config.enable_circuit_breaker:
            return
        self.circuit_open = True
        self.circuit_open_until = time.time() + self.circuit_breaker_cooldown
        logger.warning(f"Circuit breaker tripped after {self.consecutive_failures} failures, cooldown: {self.circuit_breaker_cooldown}s")
    
    async def process_chunk_optimized(
        self,
        chunk_content: str,
        chunk_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a single chunk with optimizations.
        
        Returns:
            Tuple of (success, error_message)
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            return False, "Circuit breaker is open"
        
        # Check cache for duplicate content
        content_hash = hashlib.md5(chunk_content.encode()).hexdigest()
        if content_hash in self._cache:
            logger.debug(f"Chunk {chunk_id} already processed (cached)")
            return True, None
        
        start_time = time.time()
        error_type = None
        
        try:
            # Use connection from pool
            async with self.connection_pool.acquire() as driver:
                async with driver.session() as session:
                    # Optimized Cypher query with minimal operations
                    query = """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.content = $content,
                        c.processed_at = timestamp(),
                        c.metadata = $metadata
                    RETURN c.id as id
                    """
                    
                    # Aggressive content truncation for Graphiti optimization
                    truncated_content = chunk_content
                    truncation_limit = self.config.content_truncation_limit
                    if len(chunk_content) > truncation_limit:
                        original_length = len(chunk_content)
                        truncated_content = chunk_content[:truncation_limit]
                        logger.info(
                            f"Truncating chunk {chunk_id} from {original_length} to {truncation_limit} chars "
                            f"(reduced by {original_length - truncation_limit} chars) for optimal Graphiti performance"
                        )
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        session.run(
                            query,
                            chunk_id=chunk_id,
                            content=truncated_content,  # Use truncated content
                            metadata=json.dumps(metadata) if metadata else "{}"
                        ),
                        timeout=self.config.batch_timeout
                    )
                    
                    # Consume result to complete transaction
                    await result.single()
                    
                    # Cache successful processing
                    self._cache[content_hash] = chunk_id
                    
                    # Update metrics
                    processing_time = time.time() - start_time
                    self.metrics.update(True, processing_time)
                    
                    # Reset failure counter on success
                    self.consecutive_failures = 0
                    
                    return True, None
                    
        except asyncio.TimeoutError:
            error_type = "timeout"
            error_msg = f"Timeout processing chunk {chunk_id}"
            logger.warning(error_msg)
            
        except (ServiceUnavailable, SessionExpired, TransientError) as e:
            error_type = "connection"
            error_msg = f"Connection error for chunk {chunk_id}: {str(e)[:100]}"
            logger.warning(error_msg)
            
        except Exception as e:
            error_type = "general"
            error_msg = f"Error processing chunk {chunk_id}: {str(e)[:100]}"
            logger.error(error_msg)
        
        # Update metrics and failure tracking
        processing_time = time.time() - start_time
        self.metrics.update(False, processing_time, error_type)
        
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self._trip_circuit_breaker()
        
        return False, error_msg
    
    async def process_batch_parallel(
        self,
        chunks: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Process chunks in parallel with concurrency control.
        
        Args:
            chunks: List of (content, id, metadata) tuples
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"processed": 0, "failed": 0, "errors": []}
        
        logger.info(f"Processing {len(chunks)} chunks in parallel (max {self.config.max_concurrent})")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_with_semaphore(chunk_data):
            async with semaphore:
                content, chunk_id, metadata = chunk_data
                return await self.process_chunk_optimized(content, chunk_id, metadata)
        
        # Process all chunks in parallel
        tasks = [process_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        processed = 0
        failed = 0
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                errors.append(f"Chunk {i}: {str(result)[:100]}")
            elif isinstance(result, tuple) and result[0]:
                processed += 1
            else:
                failed += 1
                if isinstance(result, tuple) and result[1]:
                    errors.append(result[1])
        
        return {
            "processed": processed,
            "failed": failed,
            "total": len(chunks),
            "success_rate": (processed / len(chunks) * 100) if chunks else 0,
            "errors": errors[:10],  # Limit error messages
            "metrics": self.metrics.get_summary()
        }
    
    async def process_batch_sequential(
        self,
        chunks: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """
        Process chunks sequentially with optimizations.
        
        Args:
            chunks: List of (content, id, metadata) tuples
        
        Returns:
            Processing results
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return {"processed": 0, "failed": 0, "errors": []}
        
        logger.info(f"Processing {len(chunks)} chunks sequentially")
        
        processed = 0
        failed = 0
        errors = []
        
        for i, (content, chunk_id, metadata) in enumerate(chunks):
            # Check circuit breaker
            if self._check_circuit_breaker():
                logger.warning("Circuit breaker open, stopping processing")
                break
            
            success, error = await self.process_chunk_optimized(content, chunk_id, metadata)
            
            if success:
                processed += 1
            else:
                failed += 1
                if error:
                    errors.append(error)
            
            # Progress update
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(chunks)} chunks, {processed} successful")
            
            # Small delay to avoid overwhelming
            if not success:
                await asyncio.sleep(self.config.retry_delay)
        
        return {
            "processed": processed,
            "failed": failed,
            "total": len(chunks),
            "success_rate": (processed / len(chunks) * 100) if chunks else 0,
            "errors": errors[:10],
            "metrics": self.metrics.get_summary()
        }
    
    async def optimize_indices(self):
        """Create optimal indices for performance."""
        if not self._initialized:
            await self.initialize()
        
        indices = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.processed_at)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)",
        ]
        
        async with self.connection_pool.acquire() as driver:
            async with driver.session() as session:
                for index_query in indices:
                    try:
                        await session.run(index_query)
                        logger.info(f"Created index: {index_query[:50]}...")
                    except Exception as e:
                        logger.warning(f"Index creation failed: {e}")
    
    async def warm_up_connection_pool(self):
        """Warm up connection pool with test queries."""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Warming up connection pool...")
        
        # Run simple queries on each connection
        warm_up_tasks = []
        for _ in range(self.config.max_concurrent):
            async def warm_up():
                async with self.connection_pool.acquire() as driver:
                    async with driver.session() as session:
                        await session.run("RETURN 1")
            
            warm_up_tasks.append(warm_up())
        
        await asyncio.gather(*warm_up_tasks, return_exceptions=True)
        logger.info("Connection pool warmed up")


class GraphitiBatchProcessor:
    """Batch processor specifically for Graphiti framework."""
    
    def __init__(self):
        """Initialize Graphiti batch processor."""
        try:
            # Try absolute import first
            from agent.graph_utils import GraphitiClient
        except ImportError:
            # Fall back to relative import if absolute fails
            try:
                from ..agent.graph_utils import GraphitiClient
            except ImportError:
                logger.error("Failed to import GraphitiClient from agent.graph_utils")
                raise ImportError("Could not import GraphitiClient. Please ensure agent.graph_utils is available.")
        
        self.graph_client = GraphitiClient()
        self._initialized = False
        self.batch_queue = []
        self.batch_size = 5
        self.flush_interval = 10  # seconds
        self._last_flush = time.time()
        
    async def initialize(self):
        """Initialize Graphiti client."""
        if not self._initialized:
            await self.graph_client.initialize()
            self._initialized = True
    
    async def close(self):
        """Close Graphiti client and flush remaining episodes."""
        if self._initialized:
            # Final flush with sequential processing for safety
            await self.flush_batch(force_sequential=True)
            await self.graph_client.close()
            self._initialized = False
            
            # Log final statistics
            if self.processing_stats["total_episodes"] > 0:
                logger.info(
                    f"GraphitiBatchProcessor final stats: "
                    f"Total: {self.processing_stats['total_episodes']}, "
                    f"Success: {self.processing_stats['successful_episodes']}, "
                    f"Failed: {self.processing_stats['failed_episodes']}, "
                    f"Total time: {self.processing_stats['total_processing_time']:.2f}s"
                )
    
    async def add_to_batch(
        self,
        episode_id: str,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add episode to batch queue."""
        # Aggressive content truncation for Graphiti optimization (2000 chars)
        truncated_content = content
        truncation_limit = 2000  # Aggressive limit for Graphiti
        if len(content) > truncation_limit:
            original_length = len(content)
            truncated_content = content[:truncation_limit]
            logger.info(
                f"Truncating episode {episode_id} from {original_length} to {truncation_limit} chars "
                f"for optimal Graphiti performance"
            )
        
        self.batch_queue.append({
            "episode_id": episode_id,
            "content": truncated_content,  # Use truncated content
            "source": source,
            "timestamp": datetime.now(timezone.utc),
            "metadata": metadata or {}
        })
        
        # Check if batch should be flushed
        if len(self.batch_queue) >= self.batch_size or \
           (time.time() - self._last_flush) > self.flush_interval:
            await self.flush_batch()
    
    async def flush_batch(self):
        """Flush batch to Graphiti."""
        if not self.batch_queue:
            return
        
        if not self._initialized:
            await self.initialize()
        
        batch = self.batch_queue.copy()
        self.batch_queue.clear()
        self._last_flush = time.time()
        
        logger.info(f"Flushing batch of {len(batch)} episodes")
        
        successful = 0
        failed = 0
        
        for episode in batch:
            try:
                await asyncio.wait_for(
                    self.graph_client.add_episode(
                        episode_id=episode["episode_id"],
                        content=episode["content"],
                        source=episode["source"],
                        timestamp=episode["timestamp"],
                        metadata=episode["metadata"],
                        group_id="0"
                    ),
                    timeout=30
                )
                successful += 1
            except Exception as e:
                logger.warning(f"Failed to add episode {episode['episode_id']}: {str(e)[:100]}")
                failed += 1
        
        logger.info(f"Batch flush complete: {successful} successful, {failed} failed")
        
        return {"successful": successful, "failed": failed}


async def benchmark_neo4j_operations():
    """Benchmark different Neo4j optimization strategies."""
    logger.info("Starting Neo4j optimization benchmark...")
    
    # Create test data
    test_chunks = [
        (f"Test content {i} with some medical information about treatment protocols and patient care guidelines that might be relevant for healthcare providers.", 
         f"chunk_{i}",
         {"index": i, "type": "test"})
        for i in range(20)
    ]
    
    results = {}
    
    # Test 1: Sequential processing with default config
    processor1 = OptimizedNeo4jProcessor(BatchConfig(enable_parallel=False))
    await processor1.initialize()
    
    start = time.time()
    result1 = await processor1.process_batch_sequential(test_chunks[:10])
    results["sequential_default"] = {
        "time": time.time() - start,
        "result": result1
    }
    await processor1.close()
    
    # Test 2: Parallel processing with concurrency
    processor2 = OptimizedNeo4jProcessor(BatchConfig(enable_parallel=True, max_concurrent=3))
    await processor2.initialize()
    
    start = time.time()
    result2 = await processor2.process_batch_parallel(test_chunks[10:])
    results["parallel_3_concurrent"] = {
        "time": time.time() - start,
        "result": result2
    }
    await processor2.close()
    
    # Test 3: Optimized with warm-up
    processor3 = OptimizedNeo4jProcessor(BatchConfig(enable_parallel=True, max_concurrent=2))
    await processor3.initialize()
    await processor3.warm_up_connection_pool()
    await processor3.optimize_indices()
    
    start = time.time()
    result3 = await processor3.process_batch_parallel(test_chunks)
    results["optimized_with_warmup"] = {
        "time": time.time() - start,
        "result": result3
    }
    await processor3.close()
    
    # Print benchmark results
    print("\n" + "="*60)
    print("NEO4J OPTIMIZATION BENCHMARK RESULTS")
    print("="*60)
    
    for strategy, data in results.items():
        print(f"\nStrategy: {strategy}")
        print(f"  Total Time: {data['time']:.2f} seconds")
        print(f"  Processed: {data['result']['processed']}/{data['result']['total']}")
        print(f"  Success Rate: {data['result']['success_rate']:.1f}%")
        if "metrics" in data["result"]:
            metrics = data["result"]["metrics"]
            print(f"  Avg Chunk Time: {metrics['avg_processing_time']:.3f}s")
            print(f"  Min/Max Time: {metrics['min_processing_time']:.3f}s / {metrics['max_processing_time']:.3f}s")
    
    # Save detailed results with proper error handling
    output_file = f"neo4j_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        # Ensure directory exists (in case we're in a subdirectory)
        import os
        output_dir = os.path.dirname(output_file) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # Write to temporary file first for atomic operation
        temp_file = f"{output_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Move temp file to final location (atomic on most systems)
        os.replace(temp_file, output_file)
        
    except (OSError, IOError) as e:
        logger.error(f"Failed to save benchmark results to {output_file}: {e}")
        # Try to save to a fallback location
        fallback_file = f"/tmp/neo4j_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(fallback_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved benchmark results to fallback location: {fallback_file}")
            output_file = fallback_file
        except Exception as fallback_error:
            logger.error(f"Also failed to save to fallback location: {fallback_error}")
            output_file = None
    
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run benchmark when executed directly
    asyncio.run(benchmark_neo4j_operations())