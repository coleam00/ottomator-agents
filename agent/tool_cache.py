"""
Simple in-memory cache for tool results to reduce redundant searches.
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    value: Any
    timestamp: float
    hits: int = 0


class ToolCache:
    """
    In-memory cache for tool results with TTL and size limits.
    
    This cache helps reduce redundant tool calls when the agent
    repeatedly searches for the same information within a session.
    """
    
    def __init__(
        self,
        ttl_seconds: int = 300,  # 5 minutes default
        max_entries: int = 100,
        enabled: bool = True
    ):
        """
        Initialize the tool cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            max_entries: Maximum number of entries to store
            enabled: Whether caching is enabled
        """
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.enabled = enabled
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # LRU tracking
        
    def _generate_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """
        Generate a cache key from tool name and arguments.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            
        Returns:
            Cache key string
        """
        # Create a deterministic string from args
        try:
            # Try JSON serialization first
            args_str = json.dumps(args, sort_keys=True)
        except (TypeError, ValueError) as e:
            # Fallback to repr for non-serializable objects
            logger.debug(f"Args not JSON serializable, using repr: {e}")
            args_str = repr(sorted(args.items()) if isinstance(args, dict) else args)
        
        combined = f"{tool_name}:{args_str}"
        
        # Use hash for shorter keys
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(
        self,
        tool_name: str,
        args: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Get cached result for a tool call.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            
        Returns:
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None
            
        key = self._generate_key(tool_name, args)
        
        if key not in self._cache:
            logger.debug(f"Cache miss for {tool_name}")
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if time.time() - entry.timestamp > self.ttl_seconds:
            logger.debug(f"Cache expired for {tool_name}")
            del self._cache[key]
            self._access_order.remove(key)
            return None
        
        # Update hits and access order
        entry.hits += 1
        self._update_access_order(key)
        
        logger.info(f"Cache hit for {tool_name} (hits: {entry.hits})")
        return entry.value
    
    def set(
        self,
        tool_name: str,
        args: Dict[str, Any],
        value: Any
    ) -> None:
        """
        Cache a tool result.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            value: Result to cache
        """
        if not self.enabled:
            return
        
        key = self._generate_key(tool_name, args)
        
        # Enforce size limit with LRU eviction
        if len(self._cache) >= self.max_entries and key not in self._cache:
            # Remove least recently used
            lru_key = self._access_order[0]
            del self._cache[lru_key]
            self._access_order.remove(lru_key)
            logger.debug(f"Evicted LRU cache entry")
        
        # Store new entry
        self._cache[key] = CacheEntry(
            value=value,
            timestamp=time.time()
        )
        self._update_access_order(key)
        
        logger.debug(f"Cached result for {tool_name}")
    
    def _update_access_order(self, key: str) -> None:
        """Update LRU access order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_hits = sum(entry.hits for entry in self._cache.values())
        
        return {
            "enabled": self.enabled,
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "total_hits": total_hits,
            "cache_keys": list(self._cache.keys())
        }


# Global cache instance for session-scoped caching
_session_caches: Dict[str, ToolCache] = {}


def get_session_cache(
    session_id: str,
    ttl_seconds: int = 300,
    max_entries: int = 100
) -> ToolCache:
    """
    Get or create a cache for a specific session.
    
    Args:
        session_id: Session identifier
        ttl_seconds: TTL for cache entries
        max_entries: Maximum cache size
        
    Returns:
        ToolCache instance for the session
    """
    if session_id not in _session_caches:
        _session_caches[session_id] = ToolCache(
            ttl_seconds=ttl_seconds,
            max_entries=max_entries
        )
        logger.debug(f"Created cache for session {session_id}")
    
    return _session_caches[session_id]


def clear_session_cache(session_id: str) -> None:
    """
    Clear cache for a specific session.
    
    Args:
        session_id: Session identifier
    """
    if session_id in _session_caches:
        _session_caches[session_id].clear()
        del _session_caches[session_id]
        logger.info(f"Cleared cache for session {session_id}")


def clear_all_caches() -> None:
    """Clear all session caches."""
    for cache in _session_caches.values():
        cache.clear()
    _session_caches.clear()
    logger.info("Cleared all session caches")