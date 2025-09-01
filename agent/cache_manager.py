"""
Advanced caching layer for the Medical RAG system.

Provides in-memory and Redis-based caching with TTL, LRU eviction,
and automatic cache invalidation.
"""

import os
import json
import hashlib
import pickle
import asyncio
import logging
from typing import Any, Dict, Optional, Union, List, Callable
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        async with self._lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                
                # Check if expired
                if datetime.now() > expiry:
                    del self.cache[key]
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in cache."""
        async with self._lock:
            # Use provided TTL or default
            ttl_seconds = ttl or self.ttl_seconds
            expiry = datetime.now() + timedelta(seconds=ttl_seconds)
            
            # Remove oldest items if cache is full
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = (value, expiry)
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self):
        """Clear all items from cache."""
        async with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": self.ttl_seconds
        }


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: Optional[str] = None, prefix: str = "rag:", ttl_seconds: int = 3600):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all cache entries
            ttl_seconds: Default TTL for cache entries
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self):
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=False)
            await self.client.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache only.")
            self._connected = False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client and self._connected:
            await self.client.close()
            self._connected = False
    
    def _make_key(self, key: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from Redis cache."""
        if not self._connected:
            return None
        
        try:
            redis_key = self._make_key(key)
            value = await self.client.get(redis_key)
            
            if value:
                return pickle.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set item in Redis cache."""
        if not self._connected:
            return
        
        try:
            redis_key = self._make_key(key)
            ttl_seconds = ttl or self.ttl_seconds
            
            serialized = pickle.dumps(value)
            await self.client.setex(redis_key, ttl_seconds, serialized)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """Delete item from Redis cache."""
        if not self._connected:
            return False
        
        try:
            redis_key = self._make_key(key)
            result = await self.client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern."""
        if not self._connected:
            return
        
        try:
            pattern_key = self._make_key(pattern)
            cursor = 0
            
            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern_key, count=100)
                if keys:
                    await self.client.delete(*keys)
                if cursor == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")


class CacheManager:
    """Hybrid cache manager with memory and Redis backends."""
    
    def __init__(self):
        """Initialize cache manager."""
        # Configuration
        self.enable_redis = os.getenv("ENABLE_REDIS_CACHE", "false").lower() == "true"
        self.memory_cache_size = int(os.getenv("MEMORY_CACHE_SIZE", "1000"))
        self.default_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        
        # Initialize caches
        self.memory_cache = LRUCache(max_size=self.memory_cache_size, ttl_seconds=self.default_ttl)
        self.redis_cache = RedisCache(ttl_seconds=self.default_ttl) if self.enable_redis else None
        
        # Cache statistics
        self.total_hits = 0
        self.total_misses = 0
    
    async def initialize(self):
        """Initialize cache connections."""
        if self.redis_cache:
            await self.redis_cache.connect()
    
    async def close(self):
        """Close cache connections."""
        if self.redis_cache:
            await self.redis_cache.disconnect()
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prefix and parameters."""
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(sorted_params.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_digest}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache (memory first, then Redis).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            self.total_hits += 1
            return value
        
        # Try Redis cache
        if self.redis_cache:
            value = await self.redis_cache.get(key)
            if value is not None:
                # Populate memory cache
                await self.memory_cache.set(key, value)
                self.total_hits += 1
                return value
        
        self.total_misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set item in cache (both memory and Redis).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        # Set in memory cache
        await self.memory_cache.set(key, value, ttl)
        
        # Set in Redis cache
        if self.redis_cache:
            await self.redis_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        memory_deleted = await self.memory_cache.delete(key)
        redis_deleted = False
        
        if self.redis_cache:
            redis_deleted = await self.redis_cache.delete(key)
        
        return memory_deleted or redis_deleted
    
    async def clear_prefix(self, prefix: str):
        """
        Clear all cache entries with given prefix.
        
        Args:
            prefix: Key prefix to clear
        """
        # Clear from memory cache (inefficient but works)
        keys_to_delete = [k for k in self.memory_cache.cache.keys() if k.startswith(prefix)]
        for key in keys_to_delete:
            await self.memory_cache.delete(key)
        
        # Clear from Redis
        if self.redis_cache:
            await self.redis_cache.clear_pattern(f"{prefix}*")
    
    async def cache_vector_search(
        self,
        query: str,
        embedding: List[float],
        limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Cache vector search results."""
        key = self._generate_key("vector_search", {
            "query": query,
            "limit": limit,
            "embedding_hash": hashlib.md5(str(embedding).encode()).hexdigest()[:8]
        })
        return await self.get(key)
    
    async def set_vector_search(
        self,
        query: str,
        embedding: List[float],
        results: List[Dict[str, Any]],
        limit: int = 10,
        ttl: int = 1800  # 30 minutes
    ):
        """Set vector search results in cache."""
        key = self._generate_key("vector_search", {
            "query": query,
            "limit": limit,
            "embedding_hash": hashlib.md5(str(embedding).encode()).hexdigest()[:8]
        })
        await self.set(key, results, ttl)
    
    async def cache_graph_search(
        self,
        query: str,
        search_type: str = "similarity"
    ) -> Optional[Dict[str, Any]]:
        """Cache graph search results."""
        key = self._generate_key("graph_search", {
            "query": query,
            "search_type": search_type
        })
        return await self.get(key)
    
    async def set_graph_search(
        self,
        query: str,
        results: Dict[str, Any],
        search_type: str = "similarity",
        ttl: int = 1800
    ):
        """Set graph search results in cache."""
        key = self._generate_key("graph_search", {
            "query": query,
            "search_type": search_type
        })
        await self.set(key, results, ttl)
    
    async def cache_embedding(self, text: str) -> Optional[List[float]]:
        """Cache embedding for text."""
        key = self._generate_key("embedding", {"text": text[:100]})  # Limit key size
        return await self.get(key)
    
    async def set_embedding(self, text: str, embedding: List[float], ttl: int = 7200):
        """Set embedding in cache."""
        key = self._generate_key("embedding", {"text": text[:100]})
        await self.set(key, embedding, ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.total_hits + self.total_misses
        hit_rate = (self.total_hits / total * 100) if total > 0 else 0
        
        stats = {
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "memory_cache": self.memory_cache.get_stats(),
            "redis_enabled": self.enable_redis
        }
        
        return stats


# Global cache manager instance
cache_manager = CacheManager()


def cached(ttl: int = 3600, key_prefix: Optional[str] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Optional key prefix (defaults to function name)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or f"{func.__module__}.{func.__name__}"
            cache_key = cache_manager._generate_key(prefix, {
                "args": str(args),
                "kwargs": str(kwargs)
            })
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {prefix}")
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't use async cache
            # Just execute the function
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator