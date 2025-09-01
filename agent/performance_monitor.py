"""
Performance monitoring and metrics collection for the Medical RAG system.

This module provides comprehensive performance monitoring, request timing,
memory tracking, and slow query logging.
"""

import os
import time
import psutil
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    request_count: int = 0
    total_request_time: float = 0
    avg_request_time: float = 0
    min_request_time: float = float('inf')
    max_request_time: float = 0
    
    # Database metrics
    db_query_count: int = 0
    total_db_time: float = 0
    avg_db_time: float = 0
    slow_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0
    
    # Memory metrics
    memory_usage_mb: float = 0
    peak_memory_mb: float = 0
    
    # Error metrics
    error_count: int = 0
    timeout_count: int = 0
    
    def update_request_time(self, duration: float):
        """Update request timing metrics."""
        self.request_count += 1
        self.total_request_time += duration
        self.avg_request_time = self.total_request_time / self.request_count
        self.min_request_time = min(self.min_request_time, duration)
        self.max_request_time = max(self.max_request_time, duration)
    
    def update_db_time(self, duration: float, query: str = ""):
        """Update database timing metrics."""
        self.db_query_count += 1
        self.total_db_time += duration
        self.avg_db_time = self.total_db_time / self.db_query_count
        
        # Track slow queries (> 1 second)
        if duration > 1.0:
            self.slow_queries.append({
                "query": query[:200],  # Truncate long queries
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            })
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
    
    def update_cache_stats(self, hit: bool):
        """Update cache statistics."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        total = self.cache_hits + self.cache_misses
        self.cache_hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        process = psutil.Process()
        self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, self.memory_usage_mb)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "requests": {
                "count": self.request_count,
                "avg_time_ms": self.avg_request_time * 1000,
                "min_time_ms": self.min_request_time * 1000 if self.min_request_time != float('inf') else 0,
                "max_time_ms": self.max_request_time * 1000
            },
            "database": {
                "query_count": self.db_query_count,
                "avg_time_ms": self.avg_db_time * 1000,
                "total_time_s": self.total_db_time,
                "slow_queries_count": len(self.slow_queries)
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": f"{self.cache_hit_rate:.1f}%"
            },
            "memory": {
                "current_mb": f"{self.memory_usage_mb:.1f}",
                "peak_mb": f"{self.peak_memory_mb:.1f}"
            },
            "errors": {
                "count": self.error_count,
                "timeouts": self.timeout_count
            }
        }


class PerformanceMonitor:
    """Global performance monitor for the application."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.request_history = deque(maxlen=1000)  # Keep last 1000 requests
        self.endpoint_metrics = defaultdict(lambda: {"count": 0, "total_time": 0})
        self._start_time = time.time()
        self._monitoring_task = None
        
        # Configuration
        self.slow_query_threshold = float(os.getenv("SLOW_QUERY_THRESHOLD", "1.0"))
        self.enable_memory_monitoring = os.getenv("ENABLE_MEMORY_MONITORING", "true").lower() == "true"
        self.monitoring_interval = int(os.getenv("MONITORING_INTERVAL", "60"))  # seconds
    
    async def start(self):
        """Start background monitoring tasks."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._background_monitor())
            logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop background monitoring tasks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("Performance monitoring stopped")
    
    async def _background_monitor(self):
        """Background task for periodic monitoring."""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Update memory metrics
                if self.enable_memory_monitoring:
                    self.metrics.update_memory_usage()
                
                # Log current metrics
                metrics_summary = self.metrics.to_dict()
                logger.info(f"Performance metrics: {json.dumps(metrics_summary, indent=2)}")
                
                # Check for performance issues
                self._check_performance_issues()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    def _check_performance_issues(self):
        """Check for performance issues and log warnings."""
        # Check memory usage
        if self.metrics.memory_usage_mb > 500:
            logger.warning(f"High memory usage: {self.metrics.memory_usage_mb:.1f} MB")
        
        # Check cache hit rate
        if self.metrics.cache_hit_rate < 50 and self.metrics.cache_misses > 100:
            logger.warning(f"Low cache hit rate: {self.metrics.cache_hit_rate:.1f}%")
        
        # Check average response time
        if self.metrics.avg_request_time > 2.0 and self.metrics.request_count > 10:
            logger.warning(f"High average response time: {self.metrics.avg_request_time:.2f}s")
        
        # Check error rate
        if self.metrics.request_count > 0:
            error_rate = (self.metrics.error_count / self.metrics.request_count) * 100
            if error_rate > 5:
                logger.warning(f"High error rate: {error_rate:.1f}%")
    
    @asynccontextmanager
    async def track_request(self, endpoint: str):
        """Context manager to track request performance."""
        start_time = time.time()
        
        try:
            yield
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            self.metrics.update_request_time(duration)
            
            # Update endpoint-specific metrics
            self.endpoint_metrics[endpoint]["count"] += 1
            self.endpoint_metrics[endpoint]["total_time"] += duration
            
            # Add to history
            self.request_history.append({
                "endpoint": endpoint,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            # Log slow requests
            if duration > 2.0:
                logger.warning(f"Slow request: {endpoint} took {duration:.2f}s")
                
        except Exception as e:
            # Track errors
            self.metrics.error_count += 1
            
            # Add to history
            self.request_history.append({
                "endpoint": endpoint,
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)[:100]
            })
            
            raise
    
    @asynccontextmanager
    async def track_db_query(self, query: str):
        """Context manager to track database query performance."""
        start_time = time.time()
        
        try:
            yield
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Update metrics
            self.metrics.update_db_time(duration, query)
            
            # Log slow queries
            if duration > self.slow_query_threshold:
                logger.warning(f"Slow query ({duration:.2f}s): {query[:100]}...")
                
        except Exception as e:
            # Track query errors
            logger.error(f"Database query error: {e}")
            raise
    
    def track_cache_access(self, hit: bool):
        """Track cache access."""
        self.metrics.update_cache_stats(hit)
    
    def track_timeout(self):
        """Track timeout events."""
        self.metrics.timeout_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        uptime = time.time() - self._start_time
        
        return {
            **self.metrics.to_dict(),
            "uptime_seconds": uptime,
            "endpoints": {
                endpoint: {
                    "count": stats["count"],
                    "avg_time_ms": (stats["total_time"] / stats["count"] * 1000) if stats["count"] > 0 else 0
                }
                for endpoint, stats in self.endpoint_metrics.items()
            },
            "recent_slow_queries": self.metrics.slow_queries[-10:],  # Last 10 slow queries
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status based on performance metrics."""
        issues = []
        
        # Check various health indicators
        if self.metrics.avg_request_time > 3.0:
            issues.append("High average response time")
        
        if self.metrics.memory_usage_mb > 800:
            issues.append("High memory usage")
        
        if self.metrics.cache_hit_rate < 30 and self.metrics.cache_misses > 50:
            issues.append("Low cache performance")
        
        if self.metrics.error_count > 10:
            issues.append("High error count")
        
        return {
            "status": "healthy" if not issues else "degraded",
            "issues": issues,
            "metrics_summary": {
                "requests_per_minute": (self.metrics.request_count / max((time.time() - self._start_time) / 60, 1)),
                "avg_response_ms": self.metrics.avg_request_time * 1000,
                "memory_mb": self.metrics.memory_usage_mb,
                "cache_hit_rate": self.metrics.cache_hit_rate
            }
        }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.metrics = PerformanceMetrics()
        self.request_history.clear()
        self.endpoint_metrics.clear()
        self._start_time = time.time()
        logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for tracking function performance
def track_performance(name: Optional[str] = None):
    """Decorator to track function performance."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__name__}"
            async with performance_monitor.track_request(func_name):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            func_name = name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.metrics.update_request_time(duration)
                return result
            except Exception as e:
                performance_monitor.metrics.error_count += 1
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator