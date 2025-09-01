"""
Comprehensive performance optimization module for the Medical RAG system.

This module provides optimizations for:
1. Database connection pooling and query optimization
2. Caching strategies for embeddings and search results
3. Batch processing for embeddings
4. Response compression
5. Performance monitoring and metrics
"""

import os
import asyncio
import logging
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import wraps
from contextlib import asynccontextmanager
import gzip
import zlib

from dotenv import load_dotenv
import asyncpg
from neo4j import AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track performance metrics across the system."""
    
    # Query metrics
    total_queries: int = 0
    query_times: List[float] = field(default_factory=list)
    slow_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Embedding metrics
    embeddings_generated: int = 0
    embeddings_batched: int = 0
    embedding_times: List[float] = field(default_factory=list)
    
    # API metrics
    request_count: int = 0
    response_times: List[float] = field(default_factory=list)
    compressed_responses: int = 0
    
    # Database metrics
    db_pool_size: int = 0
    db_active_connections: int = 0
    neo4j_pool_size: int = 0
    
    def add_query_time(self, query_type: str, duration: float, query: str = ""):
        """Record query execution time."""
        self.total_queries += 1
        self.query_times.append(duration)
        
        # Track slow queries (> 100ms)
        if duration > 0.1:
            self.slow_queries.append({
                "type": query_type,
                "duration": duration,
                "query": query[:200],  # Truncate for storage
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        avg_query_time = sum(self.query_times) / len(self.query_times) if self.query_times else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        avg_embedding_time = sum(self.embedding_times) / len(self.embedding_times) if self.embedding_times else 0
        cache_hit_rate = (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        return {
            "queries": {
                "total": self.total_queries,
                "avg_time_ms": avg_query_time * 1000,
                "slow_queries_count": len(self.slow_queries)
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": f"{cache_hit_rate:.1f}%"
            },
            "embeddings": {
                "total_generated": self.embeddings_generated,
                "batched": self.embeddings_batched,
                "avg_time_ms": avg_embedding_time * 1000
            },
            "api": {
                "requests": self.request_count,
                "avg_response_time_ms": avg_response_time * 1000,
                "compressed_responses": self.compressed_responses
            },
            "database": {
                "pg_pool_size": self.db_pool_size,
                "pg_active": self.db_active_connections,
                "neo4j_pool_size": self.neo4j_pool_size
            }
        }


# Global metrics instance
metrics = PerformanceMetrics()


class OptimizedPostgresPool:
    """Optimized PostgreSQL connection pool with monitoring."""
    
    def __init__(self, database_url: str, min_size: int = 10, max_size: int = 20):
        """Initialize optimized connection pool."""
        self.database_url = database_url
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the connection pool with optimized settings."""
        if self._initialized:
            return
        
        try:
            # Parse database URL to extract components
            import urllib.parse
            parsed = urllib.parse.urlparse(self.database_url)
            
            self.pool = await asyncpg.create_pool(
                dsn=self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=10,
                server_settings={
                    'jit': 'off',  # Disable JIT for consistent performance
                    'random_page_cost': '1.1',  # Optimize for SSDs
                    'effective_cache_size': '4GB',
                    'shared_buffers': '256MB',
                    'work_mem': '4MB',
                    'maintenance_work_mem': '64MB'
                },
                # Connection lifecycle settings
                max_inactive_connection_lifetime=300,
                max_queries=50000,
                max_cached_statement_lifetime=300,
                # Enable statement caching
                statement_cache_size=1024
            )
            
            # Create optimized indexes if they don't exist
            async with self.pool.acquire() as conn:
                await self._create_performance_indexes(conn)
            
            self._initialized = True
            metrics.db_pool_size = self.max_size
            logger.info(f"PostgreSQL pool initialized with {self.min_size}-{self.max_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def _create_performance_indexes(self, conn: asyncpg.Connection):
        """Create performance-optimized indexes."""
        indexes = [
            # Vector search optimization
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
            
            # Text search optimization
            "CREATE INDEX IF NOT EXISTS idx_chunks_content_gin ON chunks USING gin(to_tsvector('english', content))",
            
            # Document lookups
            "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(chunk_index)",
            
            # Session queries
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id_created ON messages(session_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
            
            # Metadata queries (JSONB)
            "CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin ON chunks USING gin(metadata)",
            "CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING gin(metadata)"
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation skipped (may already exist): {e}")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool with monitoring."""
        if not self.pool:
            await self.initialize()
        
        start_time = time.time()
        async with self.pool.acquire() as conn:
            metrics.db_active_connections += 1
            try:
                yield conn
            finally:
                metrics.db_active_connections -= 1
                duration = time.time() - start_time
                if duration > 0.1:  # Log slow acquisitions
                    logger.warning(f"Slow connection acquisition: {duration:.3f}s")
    
    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False


class OptimizedNeo4jPool:
    """Optimized Neo4j connection pool with better error handling."""
    
    def __init__(self, uri: str, auth: Tuple[str, str], pool_size: int = 10):
        """Initialize Neo4j connection pool."""
        self.uri = uri
        self.auth = auth
        self.pool_size = pool_size
        self.driver: Optional[AsyncGraphDatabase.driver] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Neo4j driver with optimized settings."""
        if self._initialized:
            return
        
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=self.auth,
                max_connection_lifetime=3600,
                max_connection_pool_size=self.pool_size,
                connection_acquisition_timeout=30,
                connection_timeout=10,
                keep_alive=True,
                # Optimized settings
                fetch_size=1000,  # Batch fetch size
                database="neo4j",
                default_access_mode="READ"
            )
            
            # Verify connectivity
            await self.driver.verify_connectivity()
            
            # Create performance indexes
            await self._create_performance_indexes()
            
            self._initialized = True
            metrics.neo4j_pool_size = self.pool_size
            logger.info(f"Neo4j pool initialized with {self.pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j pool: {e}")
            raise
    
    async def _create_performance_indexes(self):
        """Create Neo4j performance indexes."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Episode) ON (e.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)",
            "CREATE TEXT INDEX IF NOT EXISTS FOR (n:Entity) ON (n.description)",
            # Vector index for similarity search
            "CREATE VECTOR INDEX IF NOT EXISTS entity_embeddings FOR (n:Entity) ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}"
        ]
        
        async with self.driver.session() as session:
            for index_query in indexes:
                try:
                    await session.run(index_query)
                except Exception as e:
                    logger.debug(f"Index creation skipped: {e}")
    
    @asynccontextmanager
    async def session(self, access_mode: str = "READ"):
        """Get a Neo4j session with monitoring."""
        if not self.driver:
            await self.initialize()
        
        start_time = time.time()
        async with self.driver.session(default_access_mode=access_mode) as session:
            try:
                yield session
            finally:
                duration = time.time() - start_time
                metrics.add_query_time("neo4j", duration)
    
    async def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            await self.driver.close()
            self._initialized = False


class EmbeddingBatcher:
    """Batch embedding generation for improved performance."""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        """Initialize embedding batcher."""
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[Tuple[str, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
    
    async def get_embedding(self, text: str, embedding_func: Callable) -> List[float]:
        """Get embedding with batching."""
        future = asyncio.Future()
        
        async with self._lock:
            self.pending_requests.append((text, future))
            
            # Start batch processor if not running
            if not self._batch_task or self._batch_task.done():
                self._batch_task = asyncio.create_task(
                    self._process_batch(embedding_func)
                )
        
        return await future
    
    async def _process_batch(self, embedding_func: Callable):
        """Process a batch of embedding requests."""
        await asyncio.sleep(self.batch_timeout)  # Wait for batch to fill
        
        async with self._lock:
            batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
        
        if not batch:
            return
        
        texts = [text for text, _ in batch]
        start_time = time.time()
        
        try:
            # Generate embeddings in batch
            embeddings = await embedding_func(texts)
            
            # Resolve futures
            for (text, future), embedding in zip(batch, embeddings):
                future.set_result(embedding)
            
            metrics.embeddings_batched += len(batch)
            metrics.embedding_times.append(time.time() - start_time)
            
        except Exception as e:
            # Reject all futures in batch
            for _, future in batch:
                future.set_exception(e)


class ResponseCompressor:
    """Compress API responses for better network performance."""
    
    @staticmethod
    def compress_response(data: Any, encoding: str = "gzip") -> bytes:
        """Compress response data."""
        json_str = json.dumps(data) if not isinstance(data, str) else data
        json_bytes = json_str.encode('utf-8')
        
        if encoding == "gzip":
            compressed = gzip.compress(json_bytes, compresslevel=6)
        elif encoding == "deflate":
            compressed = zlib.compress(json_bytes, level=6)
        else:
            compressed = json_bytes
        
        compression_ratio = len(compressed) / len(json_bytes) * 100
        logger.debug(f"Compressed response: {len(json_bytes)} -> {len(compressed)} bytes ({compression_ratio:.1f}%)")
        
        if compressed != json_bytes:
            metrics.compressed_responses += 1
        
        return compressed
    
    @staticmethod
    def should_compress(data_size: int, min_size: int = 1000) -> bool:
        """Determine if response should be compressed."""
        return data_size > min_size


def track_performance(operation_type: str = "unknown"):
    """Decorator to track performance of functions."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.add_query_time(operation_type, duration, func.__name__)
                
                if duration > 1.0:  # Log slow operations
                    logger.warning(f"Slow {operation_type} operation: {func.__name__} took {duration:.2f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_type} operation failed after {duration:.2f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.add_query_time(operation_type, duration, func.__name__)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_type} operation failed after {duration:.2f}s: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class QueryOptimizer:
    """Optimize database queries for better performance."""
    
    @staticmethod
    async def optimize_vector_search(
        conn: asyncpg.Connection,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Optimized vector similarity search."""
        # Use prepared statement for better performance
        query = """
        WITH vector_search AS (
            SELECT 
                c.id,
                c.content,
                c.document_id,
                c.chunk_index,
                c.metadata,
                c.embedding <=> $1::vector as distance,
                d.title as document_title,
                d.source as document_source
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.embedding <=> $1::vector < $3
            ORDER BY distance
            LIMIT $2
        )
        SELECT * FROM vector_search
        """
        
        # Convert embedding to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        start_time = time.time()
        rows = await conn.fetch(query, embedding_str, limit, 1 - threshold)
        duration = time.time() - start_time
        
        metrics.add_query_time("vector_search", duration)
        
        return [dict(row) for row in rows]
    
    @staticmethod
    async def optimize_neo4j_query(
        session,
        query: str,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute optimized Neo4j query with profiling."""
        # Add query hints for better performance
        optimized_query = f"""
        CYPHER runtime=parallel
        CYPHER planner=cost
        {query}
        """
        
        start_time = time.time()
        result = await session.run(optimized_query, parameters)
        records = await result.data()
        duration = time.time() - start_time
        
        metrics.add_query_time("neo4j_query", duration, query[:50])
        
        return records


class PerformanceMonitor:
    """Monitor and report system performance."""
    
    def __init__(self, report_interval: int = 60):
        """Initialize performance monitor."""
        self.report_interval = report_interval
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start performance monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.report_interval)
                
                # Get performance summary
                summary = metrics.get_summary()
                
                # Log performance report
                logger.info(f"Performance Report: {json.dumps(summary, indent=2)}")
                
                # Check for performance issues
                self._check_performance_issues(summary)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
    
    def _check_performance_issues(self, summary: Dict[str, Any]):
        """Check for performance issues and log warnings."""
        # Check query performance
        if summary["queries"]["avg_time_ms"] > 100:
            logger.warning(f"High average query time: {summary['queries']['avg_time_ms']:.1f}ms")
        
        # Check cache performance
        cache_hit_rate = float(summary["cache"]["hit_rate"].rstrip('%'))
        if cache_hit_rate < 50 and summary["cache"]["hits"] + summary["cache"]["misses"] > 100:
            logger.warning(f"Low cache hit rate: {cache_hit_rate:.1f}%")
        
        # Check slow queries
        if summary["queries"]["slow_queries_count"] > 10:
            logger.warning(f"Many slow queries detected: {summary['queries']['slow_queries_count']}")
        
        # Check API response times
        if summary["api"]["avg_response_time_ms"] > 500:
            logger.warning(f"High API response time: {summary['api']['avg_response_time_ms']:.1f}ms")


# Global instances
postgres_pool = None
neo4j_pool = None
embedding_batcher = EmbeddingBatcher()
response_compressor = ResponseCompressor()
performance_monitor = PerformanceMonitor()


async def initialize_performance_optimizations():
    """Initialize all performance optimizations."""
    global postgres_pool, neo4j_pool
    
    # Initialize PostgreSQL pool
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        postgres_pool = OptimizedPostgresPool(database_url)
        await postgres_pool.initialize()
    
    # Initialize Neo4j pool
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    if neo4j_uri and neo4j_password:
        neo4j_pool = OptimizedNeo4jPool(
            neo4j_uri,
            (os.getenv("NEO4J_USER", "neo4j"), neo4j_password)
        )
        await neo4j_pool.initialize()
    
    # Start performance monitoring
    await performance_monitor.start()
    
    logger.info("Performance optimizations initialized")


async def cleanup_performance_optimizations():
    """Cleanup performance optimization resources."""
    # Stop monitoring
    await performance_monitor.stop()
    
    # Close connection pools
    if postgres_pool:
        await postgres_pool.close()
    
    if neo4j_pool:
        await neo4j_pool.close()
    
    logger.info("Performance optimizations cleaned up")


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    report = metrics.get_summary()
    
    # Add slow query details
    if metrics.slow_queries:
        report["slow_queries"] = metrics.slow_queries[-10:]  # Last 10 slow queries
    
    return report