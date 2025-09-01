"""
Connection pooling and optimization for database connections.

Provides efficient connection pooling for PostgreSQL and Neo4j
with health checks, retries, and automatic reconnection.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import asyncpg
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class PostgreSQLPool:
    """Optimized PostgreSQL connection pool."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        min_size: int = 5,
        max_size: int = 20,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0
    ):
        """
        Initialize PostgreSQL connection pool.
        
        Args:
            database_url: PostgreSQL connection URL
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_queries: Maximum queries per connection before recycling
            max_inactive_connection_lifetime: Max idle time for connections
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not configured")
        
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        
        # Statistics
        self.total_queries = 0
        self.failed_queries = 0
        self.connection_errors = 0
    
    async def initialize(self):
        """Initialize connection pool."""
        if self._initialized:
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_size,
                max_size=self.max_size,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
                command_timeout=60,
                server_settings={
                    'application_name': 'medical_rag_optimized',
                    'jit': 'off'  # Disable JIT for more consistent performance
                }
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            self._initialized = True
            logger.info(f"PostgreSQL pool initialized with {self.min_size}-{self.max_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def close(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("PostgreSQL pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.pool.acquire() as conn:
                self.total_queries += 1
                yield conn
                
        except asyncpg.PostgresError as e:
            self.failed_queries += 1
            logger.error(f"PostgreSQL query error: {e}")
            raise
        except Exception as e:
            self.connection_errors += 1
            logger.error(f"PostgreSQL connection error: {e}")
            raise
    
    async def execute_query(self, query: str, *args) -> List[asyncpg.Record]:
        """Execute query with automatic retry."""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                async with self.acquire() as conn:
                    return await conn.fetch(query, *args)
                    
            except asyncpg.PostgresConnectionError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                raise
    
    async def execute_many(self, query: str, args_list: List[tuple]):
        """Execute many queries efficiently."""
        async with self.acquire() as conn:
            await conn.executemany(query, args_list)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if self.pool:
            return {
                "size": self.pool.get_size(),
                "free_size": self.pool.get_idle_size(),
                "total_queries": self.total_queries,
                "failed_queries": self.failed_queries,
                "connection_errors": self.connection_errors,
                "success_rate": ((self.total_queries - self.failed_queries) / self.total_queries * 100) 
                               if self.total_queries > 0 else 0
            }
        return {}


class Neo4jPool:
    """Optimized Neo4j connection pool."""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60
    ):
        """
        Initialize Neo4j connection pool.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            max_connection_pool_size: Maximum pool size
            connection_acquisition_timeout: Timeout for acquiring connections
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        
        if not self.password:
            raise ValueError("NEO4J_PASSWORD not configured")
        
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout
        
        self.driver: Optional[AsyncGraphDatabase.driver] = None
        self._initialized = False
        
        # Statistics
        self.total_queries = 0
        self.failed_queries = 0
        self.slow_queries = 0
    
    async def initialize(self):
        """Initialize Neo4j driver."""
        if self._initialized:
            return
        
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_pool_size=self.max_connection_pool_size,
                connection_acquisition_timeout=self.connection_acquisition_timeout,
                max_connection_lifetime=3600,
                keep_alive=True,
                encrypted=False  # Set to True in production with proper certificates
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            
            # Create indices for better performance
            await self._create_indices()
            
            self._initialized = True
            logger.info(f"Neo4j pool initialized with max {self.max_connection_pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j pool: {e}")
            raise
    
    async def close(self):
        """Close Neo4j driver."""
        if self.driver:
            await self.driver.close()
            self._initialized = False
            logger.info("Neo4j pool closed")
    
    async def _create_indices(self):
        """Create performance indices."""
        indices = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Episode) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Episode) ON (n.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.created_at)",
        ]
        
        async with self.driver.session() as session:
            for index_query in indices:
                try:
                    await session.run(index_query)
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
    
    @asynccontextmanager
    async def session(self, database: str = "neo4j"):
        """Get Neo4j session."""
        if not self._initialized:
            await self.initialize()
        
        async with self.driver.session(database=database) as session:
            yield session
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> List[Dict[str, Any]]:
        """
        Execute Cypher query with timeout and retry.
        
        Args:
            query: Cypher query
            parameters: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query results
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                start_time = asyncio.get_event_loop().time()
                
                async with self.session() as session:
                    result = await asyncio.wait_for(
                        session.run(query, parameters or {}),
                        timeout=timeout
                    )
                    records = [record.data() async for record in result]
                    
                    # Track statistics
                    elapsed = asyncio.get_event_loop().time() - start_time
                    self.total_queries += 1
                    if elapsed > 5.0:
                        self.slow_queries += 1
                        logger.warning(f"Slow Neo4j query ({elapsed:.2f}s): {query[:100]}")
                    
                    return records
                    
            except asyncio.TimeoutError:
                self.failed_queries += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                raise
            except Exception as e:
                self.failed_queries += 1
                logger.error(f"Neo4j query error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                raise
    
    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute write transaction."""
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                result = await tx.run(query, parameters or {})
                summary = await result.consume()
                await tx.commit()
                
                return {
                    "nodes_created": summary.counters.nodes_created,
                    "relationships_created": summary.counters.relationships_created,
                    "properties_set": summary.counters.properties_set
                }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_queries": self.total_queries,
            "failed_queries": self.failed_queries,
            "slow_queries": self.slow_queries,
            "success_rate": ((self.total_queries - self.failed_queries) / self.total_queries * 100)
                           if self.total_queries > 0 else 0,
            "slow_query_rate": (self.slow_queries / self.total_queries * 100)
                              if self.total_queries > 0 else 0
        }


class ConnectionManager:
    """Unified connection manager for all databases."""
    
    def __init__(self):
        """Initialize connection manager."""
        self.postgres_pool: Optional[PostgreSQLPool] = None
        self.neo4j_pool: Optional[Neo4jPool] = None
        self._initialized = False
        
        # Configuration
        self.enable_postgres = bool(os.getenv("DATABASE_URL"))
        self.enable_neo4j = bool(os.getenv("NEO4J_PASSWORD"))
    
    async def initialize(self):
        """Initialize all connection pools."""
        if self._initialized:
            return
        
        tasks = []
        
        # Initialize PostgreSQL pool
        if self.enable_postgres:
            self.postgres_pool = PostgreSQLPool()
            tasks.append(self.postgres_pool.initialize())
        
        # Initialize Neo4j pool
        if self.enable_neo4j:
            self.neo4j_pool = Neo4jPool()
            tasks.append(self.neo4j_pool.initialize())
        
        # Initialize all pools concurrently
        if tasks:
            await asyncio.gather(*tasks)
        
        self._initialized = True
        logger.info("Connection manager initialized")
    
    async def close(self):
        """Close all connection pools."""
        tasks = []
        
        if self.postgres_pool:
            tasks.append(self.postgres_pool.close())
        
        if self.neo4j_pool:
            tasks.append(self.neo4j_pool.close())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        self._initialized = False
        logger.info("Connection manager closed")
    
    def get_postgres(self) -> PostgreSQLPool:
        """Get PostgreSQL pool."""
        if not self.postgres_pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        return self.postgres_pool
    
    def get_neo4j(self) -> Neo4jPool:
        """Get Neo4j pool."""
        if not self.neo4j_pool:
            raise RuntimeError("Neo4j pool not initialized")
        return self.neo4j_pool
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        stats = {}
        
        if self.postgres_pool:
            stats["postgresql"] = self.postgres_pool.get_stats()
        
        if self.neo4j_pool:
            stats["neo4j"] = self.neo4j_pool.get_stats()
        
        return stats


# Global connection manager instance
connection_manager = ConnectionManager()