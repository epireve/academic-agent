#!/usr/bin/env python3
"""
Asynchronous Monitoring and Metrics System
Task 15 Implementation - Real-time monitoring for async operations

This module provides comprehensive monitoring capabilities for async operations:
- Real-time performance metrics collection
- Task execution monitoring and analysis
- Resource usage tracking and alerting
- Async-aware health checks and diagnostics
- Integration with existing monitoring systems
"""

import asyncio
import time
import json
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import psutil
import gc
from collections import defaultdict, deque
from contextlib import asynccontextmanager

from .async_framework import AsyncTask, TaskStatus, WorkerPool, AsyncProgressTracker

try:
    from ..monitoring.integration import get_monitoring_integration
    MONITORING_INTEGRATION_AVAILABLE = True
except ImportError:
    MONITORING_INTEGRATION_AVAILABLE = False


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    level: AlertLevel
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """System performance snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_usage_percent: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    queue_size: int
    worker_utilization: float


class AsyncMetricsCollector:
    """Async-aware metrics collector."""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: float = 30.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Metrics storage
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Collection state
        self.collection_active = False
        self.collection_task: Optional[asyncio.Task] = None
        self.last_flush = time.time()
        
        # Callbacks
        self.flush_callbacks: List[Callable[[List[MetricPoint]], None]] = []
        
        self.logger = logging.getLogger("async_metrics_collector")
    
    async def start_collection(self):
        """Start metrics collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Async metrics collection started")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        self.collection_active = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_metrics()
        self.logger.info("Async metrics collection stopped")
    
    async def collect_metric(self, metric: MetricPoint):
        """Collect a single metric."""
        self.metrics_buffer.append(metric)
        
        # Update aggregated metrics
        await self._update_aggregated_metric(metric)
        
        # Add to history
        self.metric_history[metric.name].append({
            "value": metric.value,
            "timestamp": metric.timestamp,
            "tags": metric.tags
        })
    
    async def collect_metrics_batch(self, metrics: List[MetricPoint]):
        """Collect multiple metrics efficiently."""
        for metric in metrics:
            await self.collect_metric(metric)
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                # Check if flush is needed
                if time.time() - self.last_flush >= self.flush_interval:
                    await self._flush_metrics()
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5.0)  # Error recovery delay
    
    async def _flush_metrics(self):
        """Flush collected metrics."""
        if not self.metrics_buffer:
            return
        
        # Get current buffer contents
        metrics_to_flush = list(self.metrics_buffer)
        self.metrics_buffer.clear()
        
        # Call flush callbacks
        for callback in self.flush_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics_to_flush)
                else:
                    callback(metrics_to_flush)
            except Exception as e:
                self.logger.error(f"Error in flush callback: {e}")
        
        self.last_flush = time.time()
        self.logger.debug(f"Flushed {len(metrics_to_flush)} metrics")
    
    async def _update_aggregated_metric(self, metric: MetricPoint):
        """Update aggregated metric statistics."""
        name = metric.name
        
        if name not in self.aggregated_metrics:
            self.aggregated_metrics[name] = {
                "count": 0,
                "sum": 0.0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0,
                "last_value": 0.0,
                "last_updated": metric.timestamp
            }
        
        stats = self.aggregated_metrics[name]
        stats["count"] += 1
        stats["sum"] += metric.value
        stats["min"] = min(stats["min"], metric.value)
        stats["max"] = max(stats["max"], metric.value)
        stats["avg"] = stats["sum"] / stats["count"]
        stats["last_value"] = metric.value
        stats["last_updated"] = metric.timestamp
    
    def add_flush_callback(self, callback: Callable[[List[MetricPoint]], None]):
        """Add callback for metric flushing."""
        self.flush_callbacks.append(callback)
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        return self.aggregated_metrics.get(metric_name)
    
    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent history for a metric."""
        history = self.metric_history.get(metric_name, deque())
        return list(history)[-limit:]
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all collected metrics."""
        return dict(self.aggregated_metrics)


class AsyncTaskMonitor:
    """Monitor async task execution and performance."""
    
    def __init__(self, metrics_collector: AsyncMetricsCollector):
        self.metrics_collector = metrics_collector
        
        # Task tracking
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.task_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance tracking
        self.task_durations: deque = deque(maxlen=1000)
        self.task_success_rate: deque = deque(maxlen=100)
        self.worker_utilization: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        self.logger = logging.getLogger("async_task_monitor")
    
    async def register_task(self, task: AsyncTask):
        """Register a task for monitoring."""
        self.active_tasks[task.task_id] = task
        
        # Collect initial metric
        await self.metrics_collector.collect_metric(MetricPoint(
            name="task_registered",
            value=1,
            metric_type=MetricType.COUNTER,
            tags={
                "task_id": task.task_id,
                "priority": task.priority.name,
                "status": task.status.name
            }
        ))
        
        self.logger.debug(f"Registered task {task.task_id} for monitoring")
    
    async def update_task_status(self, task_id: str, status: TaskStatus, worker_id: Optional[str] = None):
        """Update task status and collect metrics."""
        if task_id not in self.active_tasks:
            return
        
        task = self.active_tasks[task_id]
        previous_status = task.status
        task.status = status
        
        # Collect status change metric
        await self.metrics_collector.collect_metric(MetricPoint(
            name="task_status_change",
            value=1,
            metric_type=MetricType.COUNTER,
            tags={
                "task_id": task_id,
                "from_status": previous_status.name,
                "to_status": status.name,
                "worker_id": worker_id or "unknown"
            }
        ))
        
        # Handle completion
        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            await self._handle_task_completion(task_id, status, worker_id)
    
    async def _handle_task_completion(self, task_id: str, status: TaskStatus, worker_id: Optional[str]):
        """Handle task completion and collect final metrics."""
        task = self.active_tasks.pop(task_id, None)
        if not task:
            return
        
        # Calculate duration
        duration = 0.0
        if task.metrics.started_at and task.metrics.completed_at:
            duration = (task.metrics.completed_at - task.metrics.started_at).total_seconds()
        
        # Record completion
        completion_data = {
            "task_id": task_id,
            "status": status.name,
            "duration": duration,
            "worker_id": worker_id,
            "priority": task.priority.name,
            "completed_at": datetime.now(),
            "retry_count": task.metrics.retry_count,
            "memory_usage": task.metrics.memory_usage_mb
        }
        
        self.completed_tasks.append(completion_data)
        
        # Update performance tracking
        self.task_durations.append(duration)
        self.task_success_rate.append(1 if status == TaskStatus.COMPLETED else 0)
        
        if worker_id:
            self.worker_utilization[worker_id].append(duration)
        
        # Collect completion metrics
        await self.metrics_collector.collect_metric(MetricPoint(
            name="task_duration",
            value=duration,
            metric_type=MetricType.TIMING,
            tags={
                "task_id": task_id,
                "status": status.name,
                "worker_id": worker_id or "unknown",
                "priority": task.priority.name
            }
        ))
        
        await self.metrics_collector.collect_metric(MetricPoint(
            name="task_completed",
            value=1,
            metric_type=MetricType.COUNTER,
            tags={
                "status": status.name,
                "worker_id": worker_id or "unknown"
            }
        ))
        
        self.logger.debug(f"Task {task_id} completed with status {status.name} in {duration:.2f}s")
    
    def get_active_task_count(self) -> int:
        """Get count of active tasks."""
        return len(self.active_tasks)
    
    def get_completion_stats(self) -> Dict[str, Any]:
        """Get task completion statistics."""
        if not self.completed_tasks:
            return {}
        
        recent_tasks = self.completed_tasks[-100:]  # Last 100 tasks
        
        completed_count = sum(1 for t in recent_tasks if t["status"] == "COMPLETED")
        failed_count = sum(1 for t in recent_tasks if t["status"] == "FAILED")
        cancelled_count = sum(1 for t in recent_tasks if t["status"] == "CANCELLED")
        
        durations = [t["duration"] for t in recent_tasks if t["duration"] > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_completed": len(self.completed_tasks),
            "recent_completed": completed_count,
            "recent_failed": failed_count,
            "recent_cancelled": cancelled_count,
            "success_rate": completed_count / len(recent_tasks) if recent_tasks else 0,
            "average_duration": avg_duration,
            "active_tasks": len(self.active_tasks)
        }
    
    def get_worker_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get worker performance statistics."""
        worker_stats = {}
        
        for worker_id, durations in self.worker_utilization.items():
            if not durations:
                continue
            
            worker_stats[worker_id] = {
                "tasks_processed": len(durations),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_processing_time": sum(durations)
            }
        
        return worker_stats


class AsyncSystemMonitor:
    """Monitor system resources and performance for async operations."""
    
    def __init__(self, metrics_collector: AsyncMetricsCollector, check_interval: float = 5.0):
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Performance history
        self.performance_history: deque = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_critical": 90.0,
            "cpu_warning": 75.0,
            "memory_critical": 90.0,
            "memory_warning": 75.0,
            "disk_critical": 95.0,
            "disk_warning": 85.0
        }
        
        self.logger = logging.getLogger("async_system_monitor")
    
    async def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Async system monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Async system monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.used / disk.total * 100
            
            # Create performance snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_percent,
                active_tasks=0,  # Would be set by task monitor
                completed_tasks=0,  # Would be set by task monitor
                failed_tasks=0,  # Would be set by task monitor
                queue_size=0,  # Would be set by queue monitor
                worker_utilization=0.0  # Would be calculated
            )
            
            self.performance_history.append(snapshot)
            
            # Collect individual metrics
            metrics = [
                MetricPoint("system_cpu_percent", cpu_percent, MetricType.GAUGE),
                MetricPoint("system_memory_percent", memory_percent, MetricType.GAUGE),
                MetricPoint("system_memory_used_gb", memory_used_gb, MetricType.GAUGE),
                MetricPoint("system_memory_available_gb", memory_available_gb, MetricType.GAUGE),
                MetricPoint("system_disk_percent", disk_percent, MetricType.GAUGE),
            ]
            
            await self.metrics_collector.collect_metrics_batch(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _check_alert_conditions(self):
        """Check for alert conditions."""
        if not self.performance_history:
            return
        
        latest = self.performance_history[-1]
        
        # CPU alerts
        if latest.cpu_percent >= self.alert_thresholds["cpu_critical"]:
            await self._emit_alert(
                AlertLevel.CRITICAL,
                "High CPU Usage",
                f"CPU usage is {latest.cpu_percent:.1f}%",
                {"cpu_percent": latest.cpu_percent}
            )
        elif latest.cpu_percent >= self.alert_thresholds["cpu_warning"]:
            await self._emit_alert(
                AlertLevel.WARNING,
                "Elevated CPU Usage",
                f"CPU usage is {latest.cpu_percent:.1f}%",
                {"cpu_percent": latest.cpu_percent}
            )
        
        # Memory alerts
        if latest.memory_percent >= self.alert_thresholds["memory_critical"]:
            await self._emit_alert(
                AlertLevel.CRITICAL,
                "High Memory Usage",
                f"Memory usage is {latest.memory_percent:.1f}%",
                {"memory_percent": latest.memory_percent}
            )
        elif latest.memory_percent >= self.alert_thresholds["memory_warning"]:
            await self._emit_alert(
                AlertLevel.WARNING,
                "Elevated Memory Usage",
                f"Memory usage is {latest.memory_percent:.1f}%",
                {"memory_percent": latest.memory_percent}
            )
        
        # Disk alerts
        if latest.disk_usage_percent >= self.alert_thresholds["disk_critical"]:
            await self._emit_alert(
                AlertLevel.CRITICAL,
                "High Disk Usage",
                f"Disk usage is {latest.disk_usage_percent:.1f}%",
                {"disk_percent": latest.disk_usage_percent}
            )
    
    async def _emit_alert(self, level: AlertLevel, title: str, message: str, context: Dict[str, Any]):
        """Emit a system alert."""
        alert = Alert(
            id=f"system_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            source="async_system_monitor",
            context=context
        )
        
        # Collect alert metric
        await self.metrics_collector.collect_metric(MetricPoint(
            name="system_alert",
            value=1,
            metric_type=MetricType.COUNTER,
            tags={
                "level": level.value,
                "title": title,
                "source": "system_monitor"
            }
        ))
        
        self.logger.warning(f"Alert [{level.value.upper()}]: {title} - {message}")
    
    def get_current_performance(self) -> Optional[PerformanceSnapshot]:
        """Get current performance snapshot."""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_performance_trend(self, duration_minutes: int = 30) -> List[PerformanceSnapshot]:
        """Get performance trend over specified duration."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        return [
            snapshot for snapshot in self.performance_history 
            if snapshot.timestamp >= cutoff_time
        ]


class AsyncMonitoringSystem:
    """Comprehensive async monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.metrics_collector = AsyncMetricsCollector(
            buffer_size=self.config.get("metrics_buffer_size", 10000),
            flush_interval=self.config.get("metrics_flush_interval", 30.0)
        )
        
        self.task_monitor = AsyncTaskMonitor(self.metrics_collector)
        
        self.system_monitor = AsyncSystemMonitor(
            self.metrics_collector,
            check_interval=self.config.get("system_check_interval", 5.0)
        )
        
        # Integration with existing monitoring
        self.external_integration = None
        if MONITORING_INTEGRATION_AVAILABLE:
            try:
                self.external_integration = get_monitoring_integration()
            except Exception as e:
                logging.warning(f"Could not initialize external monitoring: {e}")
        
        # Setup callbacks
        self._setup_integration_callbacks()
        
        self.logger = logging.getLogger("async_monitoring_system")
    
    def _setup_integration_callbacks(self):
        """Setup callbacks for external monitoring integration."""
        if not self.external_integration:
            return
        
        async def external_flush_callback(metrics: List[MetricPoint]):
            """Forward metrics to external monitoring system."""
            try:
                # Convert to external format and send
                for metric in metrics:
                    # This would integrate with Prometheus or other systems
                    pass
            except Exception as e:
                self.logger.error(f"Error forwarding metrics to external system: {e}")
        
        self.metrics_collector.add_flush_callback(external_flush_callback)
    
    async def start(self):
        """Start the monitoring system."""
        await self.metrics_collector.start_collection()
        await self.system_monitor.start_monitoring()
        
        if self.external_integration:
            try:
                self.external_integration.start_monitoring()
            except Exception as e:
                self.logger.warning(f"Could not start external monitoring: {e}")
        
        self.logger.info("Async monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system."""
        await self.system_monitor.stop_monitoring()
        await self.metrics_collector.stop_collection()
        
        if self.external_integration:
            try:
                self.external_integration.stop_monitoring()
            except Exception as e:
                self.logger.warning(f"Error stopping external monitoring: {e}")
        
        self.logger.info("Async monitoring system stopped")
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, **tags):
        """Context manager for monitoring operations."""
        start_time = time.time()
        
        # Start operation metric
        await self.metrics_collector.collect_metric(MetricPoint(
            name="operation_started",
            value=1,
            metric_type=MetricType.COUNTER,
            tags={"operation": operation_name, **tags}
        ))
        
        try:
            yield
            
            # Success metric
            duration = time.time() - start_time
            await self.metrics_collector.collect_metric(MetricPoint(
                name="operation_duration",
                value=duration,
                metric_type=MetricType.TIMING,
                tags={"operation": operation_name, "status": "success", **tags}
            ))
            
        except Exception as e:
            # Error metric
            duration = time.time() - start_time
            await self.metrics_collector.collect_metric(MetricPoint(
                name="operation_duration",
                value=duration,
                metric_type=MetricType.TIMING,
                tags={"operation": operation_name, "status": "error", **tags}
            ))
            
            await self.metrics_collector.collect_metric(MetricPoint(
                name="operation_error",
                value=1,
                metric_type=MetricType.COUNTER,
                tags={
                    "operation": operation_name,
                    "error_type": type(e).__name__,
                    **tags
                }
            ))
            
            raise
    
    async def monitor_worker_pool(self, worker_pool: WorkerPool):
        """Monitor a worker pool continuously."""
        while worker_pool.is_running:
            try:
                stats = worker_pool.get_pool_stats()
                
                # Collect worker pool metrics
                await self.metrics_collector.collect_metrics_batch([
                    MetricPoint("worker_pool_size", stats["pool_size"], MetricType.GAUGE),
                    MetricPoint("worker_pool_active", len([w for w in stats["workers"] if w["is_running"]]), MetricType.GAUGE),
                    MetricPoint("worker_pool_tasks_completed", stats["totals"]["tasks_completed"], MetricType.COUNTER),
                    MetricPoint("worker_pool_tasks_failed", stats["totals"]["tasks_failed"], MetricType.COUNTER),
                    MetricPoint("worker_pool_queue_pending", stats["queue_stats"]["pending_tasks"], MetricType.GAUGE)
                ])
                
                # Monitor individual workers
                for worker_stats in stats["workers"]:
                    await self.metrics_collector.collect_metrics_batch([
                        MetricPoint(
                            "worker_utilization",
                            worker_stats["total_processing_time"],
                            MetricType.GAUGE,
                            tags={"worker_id": worker_stats["worker_id"]}
                        ),
                        MetricPoint(
                            "worker_tasks_completed",
                            worker_stats["tasks_completed"],
                            MetricType.COUNTER,
                            tags={"worker_id": worker_stats["worker_id"]}
                        )
                    ])
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring worker pool: {e}")
                await asyncio.sleep(10)
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "monitoring_system": {
                "status": "active",
                "metrics_collected": len(self.metrics_collector.metrics_buffer),
                "external_integration": self.external_integration is not None
            },
            "task_monitoring": self.task_monitor.get_completion_stats(),
            "worker_performance": self.task_monitor.get_worker_performance(),
            "system_performance": (
                self.system_monitor.get_current_performance().__dict__ 
                if self.system_monitor.get_current_performance() else {}
            ),
            "metrics_summary": self.metrics_collector.get_all_metrics_summary(),
            "timestamp": datetime.now().isoformat()
        }
    
    async def generate_performance_report(self, duration_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Get performance trend
        trend = self.system_monitor.get_performance_trend(duration_hours * 60)
        
        if not trend:
            return {"error": "No performance data available"}
        
        # Calculate statistics
        cpu_values = [s.cpu_percent for s in trend]
        memory_values = [s.memory_percent for s in trend]
        
        report = {
            "report_period": {
                "duration_hours": duration_hours,
                "start_time": trend[0].timestamp.isoformat() if trend else None,
                "end_time": trend[-1].timestamp.isoformat() if trend else None,
                "data_points": len(trend)
            },
            "cpu_statistics": {
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "current": trend[-1].cpu_percent if trend else 0
            },
            "memory_statistics": {
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "current": trend[-1].memory_percent if trend else 0
            },
            "task_statistics": self.task_monitor.get_completion_stats(),
            "worker_performance": self.task_monitor.get_worker_performance(),
            "generated_at": datetime.now().isoformat()
        }
        
        return report


# Utility functions and decorators
def monitor_async_function(monitoring_system: AsyncMonitoringSystem, operation_name: str = None):
    """Decorator to monitor async functions."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            async with monitoring_system.monitor_operation(op_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


async def create_monitoring_system(config: Optional[Dict[str, Any]] = None) -> AsyncMonitoringSystem:
    """Create and start monitoring system."""
    monitoring = AsyncMonitoringSystem(config)
    await monitoring.start()
    return monitoring


# Example usage
async def main():
    """Example usage of async monitoring system."""
    
    config = {
        "metrics_buffer_size": 5000,
        "metrics_flush_interval": 10.0,
        "system_check_interval": 3.0
    }
    
    monitoring = AsyncMonitoringSystem(config)
    
    try:
        await monitoring.start()
        
        # Example monitored operation
        async with monitoring.monitor_operation("example_operation", component="test"):
            await asyncio.sleep(1)
            print("Operation completed")
        
        # Get status
        status = await monitoring.get_comprehensive_status()
        print(f"System status: {json.dumps(status, indent=2, default=str)}")
        
        # Generate report
        report = await monitoring.generate_performance_report(1)  # 1 hour
        print(f"Performance report: {json.dumps(report, indent=2, default=str)}")
        
        await asyncio.sleep(5)  # Let monitoring collect some data
        
    finally:
        await monitoring.stop()


if __name__ == "__main__":
    asyncio.run(main())