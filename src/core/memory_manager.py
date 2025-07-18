#!/usr/bin/env python3
"""
Memory Management System for Academic Agent v2.0
Task 16 Implementation - Comprehensive memory management and resource optimization

This module provides memory management capabilities including:
- Memory usage monitoring and alerts
- Automatic garbage collection and cleanup
- Resource pool management
- Memory-aware task scheduling
- Cache management with TTL and LRU policies
- Memory profiling and optimization
"""

import gc
import logging
import psutil
import time
import threading
import weakref
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from contextlib import contextmanager
from threading import Lock, RLock
import tracemalloc

from .exceptions import MemoryException, ResourceExhaustedException
from .monitoring import MetricsCollector


@dataclass
class MemoryStats:
    """Memory statistics snapshot."""
    timestamp: datetime
    total_memory_mb: float
    available_memory_mb: float
    used_memory_mb: float
    process_memory_mb: float
    memory_percent: float
    gc_collections: Dict[int, int]
    cache_size_mb: float
    active_objects: int


@dataclass
class MemoryThresholds:
    """Memory usage thresholds for alerts and actions."""
    warning_percent: float = 75.0
    critical_percent: float = 85.0
    emergency_percent: float = 95.0
    cache_cleanup_percent: float = 80.0
    gc_trigger_percent: float = 70.0


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class MemoryPool:
    """Memory pool for efficient resource allocation."""
    
    def __init__(self, name: str, max_size_mb: float = 100.0):
        self.name = name
        self.max_size_mb = max_size_mb
        self.current_size_mb = 0.0
        self.allocations: Dict[str, Any] = {}
        self.lock = RLock()
        self.logger = logging.getLogger(f"memory_pool.{name}")
    
    def allocate(self, key: str, size_mb: float, factory: Callable[[], Any]) -> Any:
        """Allocate resource with size tracking."""
        with self.lock:
            if key in self.allocations:
                return self.allocations[key]
            
            if self.current_size_mb + size_mb > self.max_size_mb:
                self._cleanup()
                if self.current_size_mb + size_mb > self.max_size_mb:
                    raise MemoryException(
                        f"Cannot allocate {size_mb}MB in pool {self.name}. "
                        f"Current: {self.current_size_mb}MB, Max: {self.max_size_mb}MB"
                    )
            
            resource = factory()
            self.allocations[key] = resource
            self.current_size_mb += size_mb
            
            self.logger.debug(f"Allocated {size_mb}MB for {key}. Pool usage: {self.current_size_mb}/{self.max_size_mb}MB")
            return resource
    
    def deallocate(self, key: str, size_mb: float):
        """Deallocate resource and update size tracking."""
        with self.lock:
            if key in self.allocations:
                del self.allocations[key]
                self.current_size_mb = max(0, self.current_size_mb - size_mb)
                self.logger.debug(f"Deallocated {size_mb}MB for {key}. Pool usage: {self.current_size_mb}MB")
    
    def _cleanup(self):
        """Internal cleanup method - override in subclasses."""
        gc.collect()


class LRUCache:
    """LRU Cache with TTL and size limits."""
    
    def __init__(self, max_size: int = 1000, max_size_mb: float = 100.0, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_mb = 0.0
        self.lock = RLock()
        self.logger = logging.getLogger("lru_cache")
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            entry.touch()
            self.cache.move_to_end(key)
            self.hits += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None, size_bytes: int = 0):
        """Put item in cache."""
        with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            # Check size limits
            entry_size_mb = size_bytes / (1024 * 1024)
            if self.current_size_mb + entry_size_mb > self.max_size_mb:
                self._evict_to_make_space(entry_size_mb)
            
            # Add to cache
            self.cache[key] = entry
            self.current_size_mb += entry_size_mb
            
            # Check count limit
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _remove_entry(self, key: str):
        """Remove entry and update size."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_size_mb -= entry.size_bytes / (1024 * 1024)
            del self.cache[key]
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            key = next(iter(self.cache))
            self._remove_entry(key)
    
    def _evict_to_make_space(self, needed_mb: float):
        """Evict items to make space."""
        while self.current_size_mb + needed_mb > self.max_size_mb and self.cache:
            self._evict_lru()
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_mb = 0.0
    
    def cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "size_mb": self.current_size_mb,
            "max_size_mb": self.max_size_mb,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": hit_rate
        }


class MemoryManager:
    """Comprehensive memory management system."""
    
    def __init__(self, 
                 thresholds: Optional[MemoryThresholds] = None,
                 enable_monitoring: bool = True,
                 monitoring_interval: float = 10.0):
        self.thresholds = thresholds or MemoryThresholds()
        self.enable_monitoring = enable_monitoring
        self.monitoring_interval = monitoring_interval
        
        # Monitoring and statistics
        self.logger = logging.getLogger("memory_manager")
        self.metrics = MetricsCollector()
        self.process = psutil.Process()
        
        # Memory pools
        self.pools: Dict[str, MemoryPool] = {}
        self.pools_lock = Lock()
        
        # Global cache
        self.cache = LRUCache(max_size=10000, max_size_mb=500.0, default_ttl=3600)
        
        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Weak references to tracked objects
        self._tracked_objects: Set[weakref.ref] = set()
        self._tracking_lock = Lock()
        
        # Memory profiling
        self.profiling_enabled = False
        self.snapshots: List[Any] = []
        
        if self.enable_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_memory, daemon=True)
            self._monitoring_thread.start()
            self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Memory monitoring stopped")
    
    def _monitor_memory(self):
        """Memory monitoring loop."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                stats = self.get_memory_stats()
                self._check_thresholds(stats)
                self._update_metrics(stats)
                
                # Cleanup expired cache entries
                self.cache.cleanup_expired()
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process_info = self.process.memory_info()
        
        # GC statistics
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]
        
        # Cache size
        cache_size_mb = self.cache.current_size_mb
        
        # Active objects count
        active_objects = len(gc.get_objects())
        
        return MemoryStats(
            timestamp=datetime.now(),
            total_memory_mb=memory.total / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            used_memory_mb=memory.used / (1024 * 1024),
            process_memory_mb=process_info.rss / (1024 * 1024),
            memory_percent=memory.percent,
            gc_collections=gc_stats,
            cache_size_mb=cache_size_mb,
            active_objects=active_objects
        )
    
    def _check_thresholds(self, stats: MemoryStats):
        """Check memory thresholds and take actions."""
        memory_percent = stats.memory_percent
        
        if memory_percent >= self.thresholds.emergency_percent:
            self.logger.critical(f"Emergency memory usage: {memory_percent:.1f}%")
            self._emergency_cleanup()
            
        elif memory_percent >= self.thresholds.critical_percent:
            self.logger.error(f"Critical memory usage: {memory_percent:.1f}%")
            self._critical_cleanup()
            
        elif memory_percent >= self.thresholds.warning_percent:
            self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
            
        elif memory_percent >= self.thresholds.cache_cleanup_percent:
            self._cache_cleanup()
            
        elif memory_percent >= self.thresholds.gc_trigger_percent:
            self._trigger_gc()
    
    def _update_metrics(self, stats: MemoryStats):
        """Update monitoring metrics."""
        self.metrics.gauge("memory.total_mb", stats.total_memory_mb)
        self.metrics.gauge("memory.available_mb", stats.available_memory_mb)
        self.metrics.gauge("memory.used_mb", stats.used_memory_mb)
        self.metrics.gauge("memory.process_mb", stats.process_memory_mb)
        self.metrics.gauge("memory.percent", stats.memory_percent)
        self.metrics.gauge("memory.cache_mb", stats.cache_size_mb)
        self.metrics.gauge("memory.active_objects", stats.active_objects)
    
    def _trigger_gc(self):
        """Trigger garbage collection."""
        self.logger.debug("Triggering garbage collection")
        collected = gc.collect()
        self.logger.debug(f"Garbage collection freed {collected} objects")
        self.metrics.counter("memory.gc_triggered", 1)
    
    def _cache_cleanup(self):
        """Clean up cache to free memory."""
        self.logger.debug("Performing cache cleanup")
        initial_size = self.cache.current_size_mb
        
        # Remove expired entries
        self.cache.cleanup_expired()
        
        # If still too large, remove LRU entries
        target_size = self.cache.max_size_mb * 0.7  # Target 70% of max
        while self.cache.current_size_mb > target_size and self.cache.cache:
            self.cache._evict_lru()
        
        freed_mb = initial_size - self.cache.current_size_mb
        self.logger.debug(f"Cache cleanup freed {freed_mb:.2f}MB")
        self.metrics.counter("memory.cache_cleanup", 1)
    
    def _critical_cleanup(self):
        """Perform critical memory cleanup."""
        self.logger.warning("Performing critical memory cleanup")
        
        # Aggressive cache cleanup
        self.cache.clear()
        
        # Trigger garbage collection
        self._trigger_gc()
        
        # Clean up memory pools
        with self.pools_lock:
            for pool in self.pools.values():
                pool._cleanup()
        
        self.metrics.counter("memory.critical_cleanup", 1)
    
    def _emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        self.logger.critical("Performing emergency memory cleanup")
        
        # Clear everything possible
        self.cache.clear()
        
        # Multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # Clear weak references
        with self._tracking_lock:
            self._tracked_objects.clear()
        
        self.metrics.counter("memory.emergency_cleanup", 1)
        
        # Check if we're still in critical state
        stats = self.get_memory_stats()
        if stats.memory_percent >= self.thresholds.emergency_percent:
            raise ResourceExhaustedException(
                f"Cannot free sufficient memory. Current usage: {stats.memory_percent:.1f}%"
            )
    
    def create_pool(self, name: str, max_size_mb: float = 100.0) -> MemoryPool:
        """Create a new memory pool."""
        with self.pools_lock:
            if name in self.pools:
                return self.pools[name]
            
            pool = MemoryPool(name, max_size_mb)
            self.pools[name] = pool
            self.logger.info(f"Created memory pool '{name}' with max size {max_size_mb}MB")
            return pool
    
    def get_pool(self, name: str) -> Optional[MemoryPool]:
        """Get existing memory pool."""
        with self.pools_lock:
            return self.pools.get(name)
    
    def remove_pool(self, name: str):
        """Remove memory pool."""
        with self.pools_lock:
            if name in self.pools:
                del self.pools[name]
                self.logger.info(f"Removed memory pool '{name}'")
    
    @contextmanager
    def memory_limit(self, limit_mb: float):
        """Context manager for memory-limited operations."""
        initial_stats = self.get_memory_stats()
        initial_memory = initial_stats.process_memory_mb
        
        try:
            yield
        finally:
            final_stats = self.get_memory_stats()
            final_memory = final_stats.process_memory_mb
            memory_used = final_memory - initial_memory
            
            if memory_used > limit_mb:
                self.logger.warning(
                    f"Memory limit exceeded: used {memory_used:.2f}MB, limit {limit_mb}MB"
                )
                self._trigger_gc()
    
    def track_object(self, obj: Any) -> weakref.ref:
        """Track object for memory monitoring."""
        def cleanup_callback(ref):
            with self._tracking_lock:
                self._tracked_objects.discard(ref)
        
        ref = weakref.ref(obj, cleanup_callback)
        
        with self._tracking_lock:
            self._tracked_objects.add(ref)
        
        return ref
    
    def get_tracked_objects_count(self) -> int:
        """Get count of currently tracked objects."""
        with self._tracking_lock:
            # Clean up dead references
            dead_refs = {ref for ref in self._tracked_objects if ref() is None}
            self._tracked_objects -= dead_refs
            return len(self._tracked_objects)
    
    def start_profiling(self):
        """Start memory profiling."""
        if not self.profiling_enabled:
            tracemalloc.start()
            self.profiling_enabled = True
            self.logger.info("Memory profiling started")
    
    def stop_profiling(self):
        """Stop memory profiling."""
        if self.profiling_enabled:
            tracemalloc.stop()
            self.profiling_enabled = False
            self.logger.info("Memory profiling stopped")
    
    def take_snapshot(self) -> Optional[Any]:
        """Take memory snapshot for profiling."""
        if self.profiling_enabled:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append(snapshot)
            return snapshot
        return None
    
    def compare_snapshots(self, snapshot1: Any, snapshot2: Any) -> List[Any]:
        """Compare memory snapshots."""
        if snapshot1 and snapshot2:
            return snapshot2.compare_to(snapshot1, 'lineno')
        return []
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        stats = self.get_memory_stats()
        cache_stats = self.cache.get_stats()
        
        pool_stats = {}
        with self.pools_lock:
            for name, pool in self.pools.items():
                pool_stats[name] = {
                    "current_size_mb": pool.current_size_mb,
                    "max_size_mb": pool.max_size_mb,
                    "allocations": len(pool.allocations)
                }
        
        return {
            "timestamp": stats.timestamp.isoformat(),
            "system_memory": {
                "total_mb": stats.total_memory_mb,
                "available_mb": stats.available_memory_mb,
                "used_mb": stats.used_memory_mb,
                "percent": stats.memory_percent
            },
            "process_memory": {
                "memory_mb": stats.process_memory_mb,
                "active_objects": stats.active_objects,
                "tracked_objects": self.get_tracked_objects_count()
            },
            "cache": cache_stats,
            "pools": pool_stats,
            "gc_collections": stats.gc_collections,
            "profiling_enabled": self.profiling_enabled,
            "snapshots_count": len(self.snapshots)
        }
    
    def cleanup(self):
        """Clean up memory manager resources."""
        self.stop_monitoring()
        self.cache.clear()
        
        with self.pools_lock:
            self.pools.clear()
        
        if self.profiling_enabled:
            self.stop_profiling()
        
        self.logger.info("Memory manager cleanup completed")


# Singleton instance for global memory management
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def shutdown_memory_manager():
    """Shutdown global memory manager."""
    global _memory_manager
    if _memory_manager:
        _memory_manager.cleanup()
        _memory_manager = None


# Convenience decorators and context managers
def memory_monitor(limit_mb: Optional[float] = None):
    """Decorator for memory monitoring."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_memory_manager()
            
            if limit_mb:
                with manager.memory_limit(limit_mb):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


@contextmanager
def memory_profile():
    """Context manager for memory profiling."""
    manager = get_memory_manager()
    manager.start_profiling()
    snapshot1 = manager.take_snapshot()
    
    try:
        yield manager
    finally:
        snapshot2 = manager.take_snapshot()
        if snapshot1 and snapshot2:
            differences = manager.compare_snapshots(snapshot1, snapshot2)
            for diff in differences[:10]:  # Top 10 differences
                manager.logger.info(f"Memory diff: {diff}")