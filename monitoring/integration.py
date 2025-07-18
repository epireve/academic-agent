"""Integration layer for Prometheus monitoring with Academic Agent system."""

import asyncio
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from academic_agent_v2.src.core.logging import get_logger
from academic_agent_v2.src.core.monitoring import (
    Alert,
    Metric,
    get_system_monitor
)
from .prometheus_config import (
    get_prometheus_monitor,
    record_error,
    record_operation,
    record_pdf_processing,
    record_quality_score,
    start_prometheus_monitoring
)


class MonitoringIntegration:
    """Integrates Prometheus monitoring with the existing monitoring system."""
    
    def __init__(self):
        self.logger = get_logger("monitoring_integration")
        self.system_monitor = get_system_monitor()
        self.prometheus_monitor = get_prometheus_monitor()
        
        # Monitoring state
        self.monitoring_active = False
        self.collection_thread = None
        self.collection_interval = 30  # seconds
        
        # Operation tracking
        self.active_operations = {}
        
        # Setup integration hooks
        self._setup_hooks()
        
        self.logger.info("Monitoring integration initialized")
    
    def _setup_hooks(self):
        """Set up hooks to capture metrics from the existing system."""
        # Register custom metric collection callbacks
        if hasattr(self.system_monitor.metrics_collector, 'register_callback'):
            self.system_monitor.metrics_collector.register_callback(
                self._sync_metrics_to_prometheus
            )
        
        # Register alert notification handler
        if hasattr(self.system_monitor.alert_manager, 'register_notification_handler'):
            self.system_monitor.alert_manager.register_notification_handler(
                self._handle_alert_notification
            )
    
    def start_monitoring(self):
        """Start the integrated monitoring system."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        try:
            # Start Prometheus monitoring
            start_prometheus_monitoring()
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            # Start metric collection thread
            self.monitoring_active = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                daemon=True
            )
            self.collection_thread.start()
            
            self.logger.info("Integrated monitoring system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
            raise
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        
        # Stop Prometheus monitoring
        if self.prometheus_monitor:
            self.prometheus_monitor.stop_server()
        
        self.logger.info("Monitoring system stopped")
    
    def _collection_loop(self):
        """Main collection loop for metrics synchronization."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                if self.prometheus_monitor.enabled:
                    self.prometheus_monitor.collect_system_metrics()
                
                # Sync metrics between systems
                self._sync_metrics_to_prometheus()
                
                # Update agent status
                self._update_agent_status()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in collection loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _sync_metrics_to_prometheus(self):
        """Synchronize metrics from the core system to Prometheus."""
        if not self.prometheus_monitor.enabled:
            return
        
        try:
            # Get metrics from the core system
            metrics = self.system_monitor.metrics_collector.get_metrics()
            
            for metric in metrics:
                self._convert_metric_to_prometheus(metric)
            
            # Update aggregated metrics
            aggregated = self.system_monitor.metrics_collector.get_aggregated_metrics()
            self._update_prometheus_aggregations(aggregated)
            
        except Exception as e:
            self.logger.error(f"Error syncing metrics to Prometheus: {e}")
    
    def _convert_metric_to_prometheus(self, metric: Metric):
        """Convert a core system metric to Prometheus format."""
        if not self.prometheus_monitor.enabled:
            return
        
        prometheus_metrics = self.prometheus_monitor.metrics
        
        # Map metric names to Prometheus metrics
        metric_mappings = {
            'memory_usage': self._update_memory_metric,
            'cpu_usage': self._update_cpu_metric,
            'operation_duration': self._update_operation_duration,
            'quality_score': self._update_quality_metric,
            'error_count': self._update_error_metric,
            'pdf_processing_time': self._update_pdf_metric,
            'agent_message_latency': self._update_communication_metric
        }
        
        # Find and execute the appropriate mapping
        for pattern, handler in metric_mappings.items():
            if pattern in metric.name:
                try:
                    handler(metric)
                except Exception as e:
                    self.logger.warning(f"Error updating Prometheus metric {metric.name}: {e}")
                break
    
    def _update_memory_metric(self, metric: Metric):
        """Update memory usage in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'memory_usage'):
            memory_type = metric.tags.get('type', 'unknown')
            self.prometheus_monitor.metrics.memory_usage.labels(type=memory_type).set(metric.value)
    
    def _update_cpu_metric(self, metric: Metric):
        """Update CPU usage in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'cpu_usage'):
            self.prometheus_monitor.metrics.cpu_usage.set(metric.value)
    
    def _update_operation_duration(self, metric: Metric):
        """Update operation duration in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'operation_duration'):
            operation_type = metric.tags.get('operation_type', 'unknown')
            status = metric.tags.get('status', 'success')
            self.prometheus_monitor.metrics.operation_duration.labels(
                operation_type=operation_type,
                status=status
            ).observe(metric.value)
    
    def _update_quality_metric(self, metric: Metric):
        """Update quality score in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'quality_scores'):
            content_type = metric.tags.get('content_type', 'unknown')
            agent = metric.tags.get('agent', 'unknown')
            self.prometheus_monitor.metrics.quality_scores.labels(
                content_type=content_type,
                agent=agent
            ).observe(metric.value)
    
    def _update_error_metric(self, metric: Metric):
        """Update error count in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'error_count'):
            error_type = metric.tags.get('error_type', 'unknown')
            component = metric.tags.get('component', 'unknown')
            severity = metric.tags.get('severity', 'error')
            self.prometheus_monitor.metrics.error_count.labels(
                error_type=error_type,
                component=component,
                severity=severity
            ).inc()
    
    def _update_pdf_metric(self, metric: Metric):
        """Update PDF processing metrics in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'pdf_processing_duration'):
            processor = metric.tags.get('processor', 'unknown')
            status = metric.tags.get('status', 'success')
            self.prometheus_monitor.metrics.pdf_processing_duration.labels(
                processor=processor,
                status=status
            ).observe(metric.value)
    
    def _update_communication_metric(self, metric: Metric):
        """Update communication metrics in Prometheus."""
        if hasattr(self.prometheus_monitor.metrics, 'message_latency'):
            sender = metric.tags.get('sender', 'unknown')
            receiver = metric.tags.get('receiver', 'unknown')
            self.prometheus_monitor.metrics.message_latency.labels(
                sender=sender,
                receiver=receiver
            ).observe(metric.value)
    
    def _update_prometheus_aggregations(self, aggregated_metrics: Dict[str, Dict[str, Any]]):
        """Update Prometheus with aggregated metrics."""
        if not self.prometheus_monitor.enabled:
            return
        
        for metric_name, stats in aggregated_metrics.items():
            try:
                # Update throughput metrics
                if 'throughput' in metric_name:
                    operation_type = metric_name.split('_')[0]
                    if hasattr(self.prometheus_monitor.metrics, 'processing_throughput'):
                        # Calculate items per second from recent values
                        recent_values = stats.get('recent_values', [])
                        if recent_values:
                            avg_throughput = sum(recent_values) / len(recent_values)
                            self.prometheus_monitor.metrics.processing_throughput.labels(
                                operation_type=operation_type
                            ).set(avg_throughput)
                
            except Exception as e:
                self.logger.warning(f"Error updating aggregated metric {metric_name}: {e}")
    
    def _update_agent_status(self):
        """Update agent status in Prometheus."""
        if not self.prometheus_monitor.enabled:
            return
        
        # This would be implemented based on the actual agent status tracking
        # For now, we'll assume all agents are active
        try:
            agents = [
                ('ingestion_agent', 'ingestion'),
                ('analysis_agent', 'analysis'),
                ('outline_agent', 'outline'),
                ('notes_agent', 'notes'),
                ('quality_agent', 'quality'),
                ('update_agent', 'update')
            ]
            
            for agent_name, agent_type in agents:
                if hasattr(self.prometheus_monitor.metrics, 'agent_status'):
                    # In a real implementation, this would check actual agent status
                    self.prometheus_monitor.metrics.agent_status.labels(
                        agent_name=agent_name,
                        agent_type=agent_type
                    ).set(1.0)  # Assume active for now
                    
        except Exception as e:
            self.logger.error(f"Error updating agent status: {e}")
    
    def _handle_alert_notification(self, alert: Alert):
        """Handle alert notifications from the core system."""
        try:
            # Log the alert
            self.logger.warning(f"Alert received: {alert.name} - {alert.message}")
            
            # Record the alert as an error metric in Prometheus
            if self.prometheus_monitor.enabled:
                record_error(
                    error_type="alert",
                    component=alert.source or "unknown",
                    severity=alert.level.lower()
                )
            
            # Additional alert handling could be implemented here
            # (e.g., sending notifications, triggering automated responses)
            
        except Exception as e:
            self.logger.error(f"Error handling alert notification: {e}")
    
    @contextmanager
    def track_operation(self, operation_type: str, operation_id: str = None):
        """Context manager to track operation metrics."""
        operation_id = operation_id or f"{operation_type}_{int(time.time())}"
        start_time = time.time()
        
        # Track concurrent operations
        if self.prometheus_monitor.enabled:
            current_count = self.active_operations.get(operation_type, 0)
            self.active_operations[operation_type] = current_count + 1
            
            if hasattr(self.prometheus_monitor.metrics, 'concurrent_operations'):
                self.prometheus_monitor.metrics.concurrent_operations.labels(
                    operation_type=operation_type
                ).set(self.active_operations[operation_type])
        
        try:
            yield operation_id
            
            # Record successful operation
            duration = time.time() - start_time
            record_operation(operation_type, duration, "success")
            
        except Exception as e:
            # Record failed operation
            duration = time.time() - start_time
            record_operation(operation_type, duration, "error")
            
            # Record the specific error
            record_error(
                error_type=type(e).__name__,
                component=operation_type,
                severity="error"
            )
            
            raise
            
        finally:
            # Update concurrent operations count
            if self.prometheus_monitor.enabled:
                self.active_operations[operation_type] = max(0, self.active_operations[operation_type] - 1)
                
                if hasattr(self.prometheus_monitor.metrics, 'concurrent_operations'):
                    self.prometheus_monitor.metrics.concurrent_operations.labels(
                        operation_type=operation_type
                    ).set(self.active_operations[operation_type])
    
    def track_pdf_processing(self, processor: str, file_path: Path, 
                           pages: int, status: str = "success"):
        """Track PDF processing metrics."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    
                    # Record PDF processing metrics
                    record_pdf_processing(
                        processor=processor,
                        duration=duration,
                        status=status,
                        pages=pages,
                        file_size=file_size
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    file_size = file_path.stat().st_size if file_path.exists() else 0
                    
                    # Record failed PDF processing
                    record_pdf_processing(
                        processor=processor,
                        duration=duration,
                        status="error",
                        pages=pages,
                        file_size=file_size
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def track_quality_score(self, content_type: str, agent: str):
        """Track quality score metrics."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                # Extract quality score from result
                if isinstance(result, dict) and 'quality_score' in result:
                    score = result['quality_score']
                    record_quality_score(content_type, agent, score)
                elif isinstance(result, (int, float)):
                    record_quality_score(content_type, agent, result)
                
                return result
            
            return wrapper
        return decorator
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "integration_active": self.monitoring_active,
            "prometheus_enabled": self.prometheus_monitor.enabled if self.prometheus_monitor else False,
            "system_monitor_active": bool(self.system_monitor),
            "active_operations": dict(self.active_operations),
            "collection_interval": self.collection_interval,
            "last_sync": datetime.now().isoformat()
        }


# Global integration instance
_monitoring_integration: Optional[MonitoringIntegration] = None


def get_monitoring_integration() -> MonitoringIntegration:
    """Get the global monitoring integration instance."""
    global _monitoring_integration
    
    if _monitoring_integration is None:
        _monitoring_integration = MonitoringIntegration()
    
    return _monitoring_integration


def start_integrated_monitoring():
    """Start the integrated monitoring system."""
    integration = get_monitoring_integration()
    integration.start_monitoring()
    return integration


def stop_integrated_monitoring():
    """Stop the integrated monitoring system."""
    global _monitoring_integration
    if _monitoring_integration:
        _monitoring_integration.stop_monitoring()


# Convenience decorators and context managers
def track_operation(operation_type: str, operation_id: str = None):
    """Decorator/context manager to track operations."""
    integration = get_monitoring_integration()
    return integration.track_operation(operation_type, operation_id)


def track_pdf_processing(processor: str, file_path: Path, pages: int, status: str = "success"):
    """Decorator to track PDF processing."""
    integration = get_monitoring_integration()
    return integration.track_pdf_processing(processor, file_path, pages, status)


def track_quality_score(content_type: str, agent: str):
    """Decorator to track quality scores."""
    integration = get_monitoring_integration()
    return integration.track_quality_score(content_type, agent)