"""Prometheus configuration and integration for Academic Agent monitoring."""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        start_http_server,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from academic_agent_v2.src.core.logging import get_logger
from academic_agent_v2.src.core.config_manager import get_config
from academic_agent_v2.src.core.exceptions import AcademicAgentError


@dataclass
class PrometheusConfig:
    """Configuration for Prometheus monitoring."""
    
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 9090
    metrics_path: str = "/metrics"
    registry_name: str = "academic_agent"
    
    # Metric collection settings
    collection_interval: int = 60  # seconds
    retention_period: int = 3600  # seconds (1 hour)
    
    # Custom metric settings
    enable_custom_metrics: bool = True
    enable_system_metrics: bool = True
    enable_performance_metrics: bool = True
    enable_quality_metrics: bool = True
    enable_pdf_metrics: bool = True
    
    # Alert thresholds
    high_memory_threshold: float = 1000.0  # MB
    high_cpu_threshold: float = 80.0  # percent
    error_rate_threshold: float = 0.1  # 10%
    processing_time_threshold: float = 300.0  # seconds
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not PROMETHEUS_AVAILABLE and self.enabled:
            raise AcademicAgentError(
                "Prometheus client not available. Install with: pip install prometheus-client"
            )


class PrometheusMetrics:
    """Prometheus metrics collector for Academic Agent."""
    
    def __init__(self, config: PrometheusConfig):
        self.config = config
        self.logger = get_logger("prometheus_metrics")
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available. Metrics collection disabled.")
            return
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_system_metrics()
        self._init_performance_metrics()
        self._init_quality_metrics()
        self._init_pdf_metrics()
        self._init_agent_metrics()
        self._init_error_metrics()
        
        self.logger.info("Prometheus metrics initialized")
    
    def _init_system_metrics(self):
        """Initialize system-level metrics."""
        if not self.config.enable_system_metrics:
            return
        
        # System information
        self.system_info = Info(
            'academic_agent_system_info',
            'System information',
            registry=self.registry
        )
        
        # Memory usage
        self.memory_usage = Gauge(
            'academic_agent_memory_usage_bytes',
            'Current memory usage in bytes',
            ['type'],  # RSS, VMS, etc.
            registry=self.registry
        )
        
        # CPU usage
        self.cpu_usage = Gauge(
            'academic_agent_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        # Disk usage
        self.disk_usage = Gauge(
            'academic_agent_disk_usage_bytes',
            'Current disk usage in bytes',
            ['path', 'type'],  # type: total, used, free
            registry=self.registry
        )
        
        # Uptime
        self.uptime = Gauge(
            'academic_agent_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        self.start_time = time.time()
    
    def _init_performance_metrics(self):
        """Initialize performance metrics."""
        if not self.config.enable_performance_metrics:
            return
        
        # Operation duration
        self.operation_duration = Histogram(
            'academic_agent_operation_duration_seconds',
            'Time spent on operations',
            ['operation_type', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0, float('inf')],
            registry=self.registry
        )
        
        # Operation count
        self.operation_count = Counter(
            'academic_agent_operations_total',
            'Total number of operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        # Concurrent operations
        self.concurrent_operations = Gauge(
            'academic_agent_concurrent_operations',
            'Number of concurrent operations',
            ['operation_type'],
            registry=self.registry
        )
        
        # Processing throughput
        self.processing_throughput = Gauge(
            'academic_agent_throughput_items_per_second',
            'Processing throughput in items per second',
            ['operation_type'],
            registry=self.registry
        )
    
    def _init_quality_metrics(self):
        """Initialize quality metrics."""
        if not self.config.enable_quality_metrics:
            return
        
        # Quality scores
        self.quality_scores = Histogram(
            'academic_agent_quality_score',
            'Quality scores for generated content',
            ['content_type', 'agent'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Improvement cycles
        self.improvement_cycles = Counter(
            'academic_agent_improvement_cycles_total',
            'Total number of improvement cycles',
            ['content_type', 'agent', 'success'],
            registry=self.registry
        )
        
        # Quality threshold violations
        self.quality_violations = Counter(
            'academic_agent_quality_violations_total',
            'Number of quality threshold violations',
            ['content_type', 'threshold_type'],
            registry=self.registry
        )
    
    def _init_pdf_metrics(self):
        """Initialize PDF processing metrics."""
        if not self.config.enable_pdf_metrics:
            return
        
        # PDF processing time
        self.pdf_processing_duration = Histogram(
            'academic_agent_pdf_processing_duration_seconds',
            'Time spent processing PDFs',
            ['processor', 'status'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, float('inf')],
            registry=self.registry
        )
        
        # PDF page count
        self.pdf_pages_processed = Counter(
            'academic_agent_pdf_pages_processed_total',
            'Total number of PDF pages processed',
            ['processor'],
            registry=self.registry
        )
        
        # PDF file size
        self.pdf_file_size = Histogram(
            'academic_agent_pdf_file_size_bytes',
            'Size of processed PDF files',
            ['processor'],
            buckets=[1e6, 5e6, 10e6, 50e6, 100e6, 500e6, float('inf')],  # 1MB to 500MB+
            registry=self.registry
        )
        
        # Image extraction
        self.images_extracted = Counter(
            'academic_agent_images_extracted_total',
            'Total number of images extracted from PDFs',
            ['processor', 'image_type'],
            registry=self.registry
        )
    
    def _init_agent_metrics(self):
        """Initialize agent-specific metrics."""
        # Agent communication
        self.agent_messages = Counter(
            'academic_agent_messages_total',
            'Total number of agent messages',
            ['sender', 'receiver', 'message_type'],
            registry=self.registry
        )
        
        # Message latency
        self.message_latency = Histogram(
            'academic_agent_message_latency_seconds',
            'Agent message latency',
            ['sender', 'receiver'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, float('inf')],
            registry=self.registry
        )
        
        # Agent status
        self.agent_status = Gauge(
            'academic_agent_status',
            'Agent status (1=active, 0=inactive)',
            ['agent_name', 'agent_type'],
            registry=self.registry
        )
    
    def _init_error_metrics(self):
        """Initialize error tracking metrics."""
        # Error count
        self.error_count = Counter(
            'academic_agent_errors_total',
            'Total number of errors',
            ['error_type', 'component', 'severity'],
            registry=self.registry
        )
        
        # Error rate
        self.error_rate = Gauge(
            'academic_agent_error_rate',
            'Current error rate (errors per minute)',
            ['component'],
            registry=self.registry
        )
        
        # Recovery count
        self.recovery_count = Counter(
            'academic_agent_recoveries_total',
            'Total number of error recoveries',
            ['recovery_type', 'component'],
            registry=self.registry
        )
    
    def update_system_info(self, info: Dict[str, str]):
        """Update system information metric."""
        if hasattr(self, 'system_info'):
            self.system_info.info(info)
    
    def update_memory_usage(self, memory_info: Dict[str, float]):
        """Update memory usage metrics."""
        if hasattr(self, 'memory_usage'):
            for memory_type, value in memory_info.items():
                self.memory_usage.labels(type=memory_type).set(value)
    
    def update_cpu_usage(self, cpu_percent: float):
        """Update CPU usage metric."""
        if hasattr(self, 'cpu_usage'):
            self.cpu_usage.set(cpu_percent)
    
    def update_disk_usage(self, path: str, usage_info: Dict[str, int]):
        """Update disk usage metrics."""
        if hasattr(self, 'disk_usage'):
            for usage_type, value in usage_info.items():
                self.disk_usage.labels(path=path, type=usage_type).set(value)
    
    def update_uptime(self):
        """Update uptime metric."""
        if hasattr(self, 'uptime'):
            current_uptime = time.time() - self.start_time
            self.uptime.set(current_uptime)
    
    def record_operation(self, operation_type: str, duration: float, status: str = "success"):
        """Record an operation's completion."""
        if hasattr(self, 'operation_duration'):
            self.operation_duration.labels(
                operation_type=operation_type,
                status=status
            ).observe(duration)
        
        if hasattr(self, 'operation_count'):
            self.operation_count.labels(
                operation_type=operation_type,
                status=status
            ).inc()
    
    def set_concurrent_operations(self, operation_type: str, count: int):
        """Set the number of concurrent operations."""
        if hasattr(self, 'concurrent_operations'):
            self.concurrent_operations.labels(operation_type=operation_type).set(count)
    
    def update_throughput(self, operation_type: str, items_per_second: float):
        """Update processing throughput."""
        if hasattr(self, 'processing_throughput'):
            self.processing_throughput.labels(operation_type=operation_type).set(items_per_second)
    
    def record_quality_score(self, content_type: str, agent: str, score: float):
        """Record a quality score."""
        if hasattr(self, 'quality_scores'):
            self.quality_scores.labels(content_type=content_type, agent=agent).observe(score)
    
    def record_improvement_cycle(self, content_type: str, agent: str, success: bool):
        """Record an improvement cycle."""
        if hasattr(self, 'improvement_cycles'):
            self.improvement_cycles.labels(
                content_type=content_type,
                agent=agent,
                success=str(success).lower()
            ).inc()
    
    def record_quality_violation(self, content_type: str, threshold_type: str):
        """Record a quality threshold violation."""
        if hasattr(self, 'quality_violations'):
            self.quality_violations.labels(
                content_type=content_type,
                threshold_type=threshold_type
            ).inc()
    
    def record_pdf_processing(self, processor: str, duration: float, 
                            status: str, pages: int, file_size: int):
        """Record PDF processing metrics."""
        if hasattr(self, 'pdf_processing_duration'):
            self.pdf_processing_duration.labels(
                processor=processor,
                status=status
            ).observe(duration)
        
        if hasattr(self, 'pdf_pages_processed'):
            self.pdf_pages_processed.labels(processor=processor).inc(pages)
        
        if hasattr(self, 'pdf_file_size'):
            self.pdf_file_size.labels(processor=processor).observe(file_size)
    
    def record_image_extraction(self, processor: str, image_type: str, count: int = 1):
        """Record image extraction."""
        if hasattr(self, 'images_extracted'):
            self.images_extracted.labels(
                processor=processor,
                image_type=image_type
            ).inc(count)
    
    def record_agent_message(self, sender: str, receiver: str, message_type: str, latency: float):
        """Record agent message metrics."""
        if hasattr(self, 'agent_messages'):
            self.agent_messages.labels(
                sender=sender,
                receiver=receiver,
                message_type=message_type
            ).inc()
        
        if hasattr(self, 'message_latency'):
            self.message_latency.labels(sender=sender, receiver=receiver).observe(latency)
    
    def set_agent_status(self, agent_name: str, agent_type: str, active: bool):
        """Set agent status."""
        if hasattr(self, 'agent_status'):
            status_value = 1.0 if active else 0.0
            self.agent_status.labels(agent_name=agent_name, agent_type=agent_type).set(status_value)
    
    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record an error."""
        if hasattr(self, 'error_count'):
            self.error_count.labels(
                error_type=error_type,
                component=component,
                severity=severity
            ).inc()
    
    def update_error_rate(self, component: str, rate: float):
        """Update error rate."""
        if hasattr(self, 'error_rate'):
            self.error_rate.labels(component=component).set(rate)
    
    def record_recovery(self, recovery_type: str, component: str):
        """Record an error recovery."""
        if hasattr(self, 'recovery_count'):
            self.recovery_count.labels(
                recovery_type=recovery_type,
                component=component
            ).inc()
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE:
            return b""
        
        return generate_latest(self.registry)


class PrometheusMonitor:
    """Main Prometheus monitoring system."""
    
    def __init__(self, config: Optional[PrometheusConfig] = None):
        self.config = config or PrometheusConfig()
        self.logger = get_logger("prometheus_monitor")
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available. Monitoring disabled.")
            self.enabled = False
            return
        
        self.enabled = self.config.enabled
        if not self.enabled:
            self.logger.info("Prometheus monitoring disabled in configuration")
            return
        
        # Initialize metrics
        self.metrics = PrometheusMetrics(self.config)
        
        # HTTP server for metrics endpoint
        self.http_server = None
        
        # System monitoring
        self._setup_system_monitoring()
        
        self.logger.info(f"Prometheus monitor initialized on {self.config.host}:{self.config.port}")
    
    def _setup_system_monitoring(self):
        """Set up system-level monitoring."""
        # Update system info
        import platform
        import sys
        
        system_info = {
            'version': '2.0.0',  # Academic Agent version
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node(),
            'architecture': platform.architecture()[0]
        }
        
        self.metrics.update_system_info(system_info)
    
    def start_server(self):
        """Start the Prometheus metrics HTTP server."""
        if not self.enabled:
            return
        
        try:
            # Start HTTP server
            self.http_server = start_http_server(
                self.config.port,
                addr=self.config.host,
                registry=self.metrics.registry
            )
            
            self.logger.info(
                f"Prometheus metrics server started on "
                f"http://{self.config.host}:{self.config.port}{self.config.metrics_path}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
            self.enabled = False
    
    def stop_server(self):
        """Stop the Prometheus metrics HTTP server."""
        if self.http_server:
            self.http_server.shutdown()
            self.http_server = None
            self.logger.info("Prometheus metrics server stopped")
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        if not self.enabled:
            return
        
        try:
            import psutil
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.metrics.update_memory_usage({
                'rss': memory_info.rss,
                'vms': memory_info.vms
            })
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.update_cpu_usage(cpu_percent)
            
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            self.metrics.update_disk_usage('/', {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free
            })
            
            # Uptime
            self.metrics.update_uptime()
            
        except ImportError:
            self.logger.warning("psutil not available for system metrics")
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not self.enabled:
            return ""
        
        try:
            metrics_bytes = self.metrics.get_metrics()
            return metrics_bytes.decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return ""
    
    def check_thresholds(self):
        """Check metrics against configured thresholds and create alerts."""
        if not self.enabled:
            return
        
        # This would integrate with the existing AlertManager
        # Implementation depends on current metrics values
        pass


# Global Prometheus monitor instance
_prometheus_monitor: Optional[PrometheusMonitor] = None


def get_prometheus_monitor() -> PrometheusMonitor:
    """Get the global Prometheus monitor instance."""
    global _prometheus_monitor
    
    if _prometheus_monitor is None:
        # Load configuration from the global config
        try:
            config = get_config()
            prometheus_config = PrometheusConfig(
                enabled=config.metrics.export.prometheus.enabled,
                port=config.metrics.export.prometheus.port,
                metrics_path=config.metrics.export.prometheus.path
            )
        except:
            # Use default configuration if global config not available
            prometheus_config = PrometheusConfig()
        
        _prometheus_monitor = PrometheusMonitor(prometheus_config)
    
    return _prometheus_monitor


def start_prometheus_monitoring():
    """Start the global Prometheus monitoring system."""
    monitor = get_prometheus_monitor()
    monitor.start_server()
    return monitor


def stop_prometheus_monitoring():
    """Stop the global Prometheus monitoring system."""
    global _prometheus_monitor
    if _prometheus_monitor:
        _prometheus_monitor.stop_server()


def record_operation(operation_type: str, duration: float, status: str = "success"):
    """Record an operation to Prometheus metrics."""
    monitor = get_prometheus_monitor()
    if monitor.enabled:
        monitor.metrics.record_operation(operation_type, duration, status)


def record_pdf_processing(processor: str, duration: float, status: str, 
                         pages: int, file_size: int):
    """Record PDF processing metrics to Prometheus."""
    monitor = get_prometheus_monitor()
    if monitor.enabled:
        monitor.metrics.record_pdf_processing(processor, duration, status, pages, file_size)


def record_quality_score(content_type: str, agent: str, score: float):
    """Record a quality score to Prometheus metrics."""
    monitor = get_prometheus_monitor()
    if monitor.enabled:
        monitor.metrics.record_quality_score(content_type, agent, score)


def record_error(error_type: str, component: str, severity: str = "error"):
    """Record an error to Prometheus metrics."""
    monitor = get_prometheus_monitor()
    if monitor.enabled:
        monitor.metrics.record_error(error_type, component, severity)