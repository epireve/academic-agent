#!/usr/bin/env python3
"""
Monitoring and Logging System for High-Performance PDF Processor
Academic Agent v2 - Task 11 Implementation

This module provides comprehensive monitoring, logging, and performance tracking
for the PDF processing system with real-time metrics and alerting.
"""

import asyncio
import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Deque
import psutil
import traceback
from enum import Enum

# GPU monitoring
try:
    import torch
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert information."""
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class MetricData:
    """Metric data point."""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'type': self.metric_type.value,
            'tags': self.tags
        }


@dataclass
class ProcessingEvent:
    """Processing event for logging."""
    event_type: str
    timestamp: datetime
    file_path: str
    status: str
    processing_time: float
    memory_usage: float
    pages_processed: int
    file_size: float
    error_message: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class MetricsCollector:
    """Collects and manages metrics data."""
    
    def __init__(self, max_history: int = 10000):
        self.metrics: Dict[str, Deque[MetricData]] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str] = None):
        """Record a metric value."""
        with self.lock:
            metric = MetricData(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metric_type=metric_type,
                tags=tags or {}
            )
            
            self.metrics[name].append(metric)
            
            # Update type-specific storage
            if metric_type == MetricType.COUNTER:
                self.counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self.gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self.histograms[name].append(value)
            elif metric_type == MetricType.TIMER:
                self.timers[name].append(value)
                
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return {}
                
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [m for m in self.metrics[name] if m.timestamp > cutoff_time]
            
            if not recent_metrics:
                return {}
                
            values = [m.value for m in recent_metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'sum': sum(values),
                'latest': values[-1] if values else None,
                'window_minutes': window_minutes
            }
            
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all current metrics."""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {k: {
                    'count': len(v),
                    'avg': sum(v) / len(v) if v else 0,
                    'min': min(v) if v else 0,
                    'max': max(v) if v else 0
                } for k, v in self.histograms.items()},
                'timers': {k: {
                    'count': len(v),
                    'avg': sum(v) / len(v) if v else 0,
                    'min': min(v) if v else 0,
                    'max': max(v) if v else 0
                } for k, v in self.timers.items()}
            }


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring = False
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info(f"System monitoring started with {interval}s interval")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.metrics_collector.record_metric(
                    'system.cpu.usage_percent',
                    cpu_percent,
                    MetricType.GAUGE
                )
                
                # Memory monitoring
                memory = psutil.virtual_memory()
                self.metrics_collector.record_metric(
                    'system.memory.usage_percent',
                    memory.percent,
                    MetricType.GAUGE
                )
                self.metrics_collector.record_metric(
                    'system.memory.available_gb',
                    memory.available / (1024**3),
                    MetricType.GAUGE
                )
                
                # Disk monitoring
                disk = psutil.disk_usage('/')
                self.metrics_collector.record_metric(
                    'system.disk.usage_percent',
                    disk.percent,
                    MetricType.GAUGE
                )
                
                # Process-specific monitoring
                process = psutil.Process()
                process_memory = process.memory_info()
                self.metrics_collector.record_metric(
                    'process.memory.rss_mb',
                    process_memory.rss / (1024**2),
                    MetricType.GAUGE
                )
                self.metrics_collector.record_metric(
                    'process.memory.vms_mb',
                    process_memory.vms / (1024**2),
                    MetricType.GAUGE
                )
                
                # GPU monitoring (if available)
                if GPU_MONITORING_AVAILABLE:
                    self._monitor_gpu()
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(interval)
            
    def _monitor_gpu(self):
        """Monitor GPU resources."""
        try:
            # PyTorch GPU monitoring
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    self.metrics_collector.record_metric(
                        f'gpu.{i}.memory.allocated_gb',
                        memory_allocated,
                        MetricType.GAUGE
                    )
                    self.metrics_collector.record_metric(
                        f'gpu.{i}.memory.reserved_gb',
                        memory_reserved,
                        MetricType.GAUGE
                    )
                    
            # GPUtil monitoring
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    self.metrics_collector.record_metric(
                        f'gpu.{gpu.id}.utilization_percent',
                        gpu.load * 100,
                        MetricType.GAUGE
                    )
                    self.metrics_collector.record_metric(
                        f'gpu.{gpu.id}.temperature_c',
                        gpu.temperature,
                        MetricType.GAUGE
                    )
                    self.metrics_collector.record_metric(
                        f'gpu.{gpu.id}.memory.usage_percent',
                        (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        MetricType.GAUGE
                    )
            except Exception:
                pass  # GPUtil not available or no GPUs
                
        except Exception as e:
            self.logger.error(f"GPU monitoring error: {e}")


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.logger = logging.getLogger(__name__)
        
    def add_alert_rule(self, 
                      metric_name: str, 
                      threshold: float, 
                      level: AlertLevel,
                      comparison: str = 'greater',
                      message_template: str = None):
        """Add an alert rule."""
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'level': level,
            'comparison': comparison,
            'message_template': message_template or f"{metric_name} {comparison} {threshold}"
        }
        
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)
        
    def check_alerts(self):
        """Check all alert rules against current metrics."""
        for metric_name, rule in self.alert_rules.items():
            try:
                summary = self.metrics_collector.get_metric_summary(metric_name, window_minutes=5)
                if not summary:
                    continue
                    
                current_value = summary['latest']
                threshold = rule['threshold']
                comparison = rule['comparison']
                
                should_alert = False
                if comparison == 'greater' and current_value > threshold:
                    should_alert = True
                elif comparison == 'less' and current_value < threshold:
                    should_alert = True
                elif comparison == 'equal' and abs(current_value - threshold) < 0.001:
                    should_alert = True
                    
                if should_alert:
                    alert = Alert(
                        level=rule['level'],
                        message=rule['message_template'].format(
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold=threshold
                        ),
                        timestamp=datetime.now(),
                        metric_name=metric_name,
                        metric_value=current_value,
                        threshold=threshold
                    )
                    
                    self._trigger_alert(alert)
                    
            except Exception as e:
                self.logger.error(f"Error checking alert for {metric_name}: {e}")
                
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert."""
        self.alerts.append(alert)
        self.logger.log(
            logging.WARNING if alert.level == AlertLevel.WARNING else logging.ERROR,
            f"ALERT [{alert.level.value.upper()}]: {alert.message}"
        )
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
                
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [a for a in self.alerts if not a.resolved]
        
    def resolve_alert(self, alert: Alert):
        """Resolve an alert."""
        alert.resolved = True
        alert.resolution_time = datetime.now()


class ProcessingLogger:
    """Structured logging for processing events."""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.events: List[ProcessingEvent] = []
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # Setup structured logging
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def log_event(self, event: ProcessingEvent):
        """Log a processing event."""
        self.events.append(event)
        
        # Log to standard logger
        log_level = logging.INFO
        if event.status == 'failed':
            log_level = logging.ERROR
        elif event.status == 'warning':
            log_level = logging.WARNING
            
        self.logger.log(
            log_level,
            f"Processing event: {event.event_type} - {event.file_path} - {event.status}"
        )
        
        # Save to file if configured
        if self.log_file:
            self._save_event_to_file(event)
            
    def _save_event_to_file(self, event: ProcessingEvent):
        """Save event to structured log file."""
        try:
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event': event.to_dict()
            }
            
            # Append to JSONL file
            structured_log_file = self.log_file.with_suffix('.jsonl')
            with open(structured_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error saving event to file: {e}")
            
    def get_events(self, 
                  event_type: Optional[str] = None,
                  status: Optional[str] = None,
                  hours: int = 24) -> List[ProcessingEvent]:
        """Get processing events with optional filtering."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_events = [
            e for e in self.events
            if e.timestamp > cutoff_time
        ]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
            
        if status:
            filtered_events = [e for e in filtered_events if e.status == status]
            
        return filtered_events
        
    def get_processing_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get processing summary statistics."""
        events = self.get_events(hours=hours)
        
        if not events:
            return {}
            
        total_events = len(events)
        successful_events = len([e for e in events if e.status == 'success'])
        failed_events = len([e for e in events if e.status == 'failed'])
        
        processing_times = [e.processing_time for e in events if e.processing_time > 0]
        memory_usages = [e.memory_usage for e in events if e.memory_usage > 0]
        pages_processed = [e.pages_processed for e in events if e.pages_processed > 0]
        
        return {
            'total_events': total_events,
            'successful_events': successful_events,
            'failed_events': failed_events,
            'success_rate': successful_events / total_events if total_events > 0 else 0,
            'processing_time': {
                'avg': sum(processing_times) / len(processing_times) if processing_times else 0,
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0,
                'total': sum(processing_times)
            },
            'memory_usage': {
                'avg': sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                'min': min(memory_usages) if memory_usages else 0,
                'max': max(memory_usages) if memory_usages else 0
            },
            'pages_processed': {
                'total': sum(pages_processed),
                'avg': sum(pages_processed) / len(pages_processed) if pages_processed else 0
            }
        }


class PerformanceProfiler:
    """Performance profiling and optimization recommendations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.profiling_enabled = False
        self.profile_data = {}
        self.logger = logging.getLogger(__name__)
        
    def start_profiling(self):
        """Start performance profiling."""
        self.profiling_enabled = True
        self.profile_data = {}
        self.logger.info("Performance profiling started")
        
    def stop_profiling(self):
        """Stop performance profiling."""
        self.profiling_enabled = False
        self.logger.info("Performance profiling stopped")
        
    def profile_function(self, func_name: str):
        """Decorator for profiling functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.profiling_enabled:
                    return func(*args, **kwargs)
                    
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                    
                    profile_info = {
                        'execution_time': end_time - start_time,
                        'memory_delta': end_memory - start_memory,
                        'success': success,
                        'error': error,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    if func_name not in self.profile_data:
                        self.profile_data[func_name] = []
                    self.profile_data[func_name].append(profile_info)
                    
                return result
            return wrapper
        return decorator
        
    def get_profile_report(self) -> Dict[str, Any]:
        """Get comprehensive profiling report."""
        report = {}
        
        for func_name, profiles in self.profile_data.items():
            if not profiles:
                continue
                
            execution_times = [p['execution_time'] for p in profiles]
            memory_deltas = [p['memory_delta'] for p in profiles]
            success_count = sum(1 for p in profiles if p['success'])
            
            report[func_name] = {
                'call_count': len(profiles),
                'success_rate': success_count / len(profiles),
                'execution_time': {
                    'avg': sum(execution_times) / len(execution_times),
                    'min': min(execution_times),
                    'max': max(execution_times),
                    'total': sum(execution_times)
                },
                'memory_usage': {
                    'avg': sum(memory_deltas) / len(memory_deltas),
                    'min': min(memory_deltas),
                    'max': max(memory_deltas)
                },
                'errors': [p['error'] for p in profiles if p['error']]
            }
            
        return report
        
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on profiling data."""
        recommendations = []
        profile_report = self.get_profile_report()
        
        for func_name, stats in profile_report.items():
            # High execution time
            if stats['execution_time']['avg'] > 10.0:
                recommendations.append({
                    'type': 'performance',
                    'severity': 'high',
                    'function': func_name,
                    'issue': 'High average execution time',
                    'recommendation': 'Consider optimizing algorithm or adding parallelization',
                    'metric': f"Average time: {stats['execution_time']['avg']:.2f}s"
                })
                
            # High memory usage
            if stats['memory_usage']['avg'] > 500 * 1024 * 1024:  # 500MB
                recommendations.append({
                    'type': 'memory',
                    'severity': 'medium',
                    'function': func_name,
                    'issue': 'High memory usage',
                    'recommendation': 'Consider processing in smaller batches or optimizing memory usage',
                    'metric': f"Average memory: {stats['memory_usage']['avg'] / (1024**2):.1f}MB"
                })
                
            # Low success rate
            if stats['success_rate'] < 0.95:
                recommendations.append({
                    'type': 'reliability',
                    'severity': 'high',
                    'function': func_name,
                    'issue': 'Low success rate',
                    'recommendation': 'Improve error handling and add retry logic',
                    'metric': f"Success rate: {stats['success_rate']*100:.1f}%"
                })
                
        return recommendations


class MonitoringSystem:
    """Comprehensive monitoring system for PDF processor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.processing_logger = ProcessingLogger()
        self.performance_profiler = PerformanceProfiler(self.metrics_collector)
        self.logger = logging.getLogger(__name__)
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Start monitoring
        self.start_monitoring()
        
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # Memory usage alerts
        self.alert_manager.add_alert_rule(
            'system.memory.usage_percent',
            80.0,
            AlertLevel.WARNING,
            'greater',
            'High memory usage: {current_value:.1f}%'
        )
        
        self.alert_manager.add_alert_rule(
            'system.memory.usage_percent',
            95.0,
            AlertLevel.CRITICAL,
            'greater',
            'Critical memory usage: {current_value:.1f}%'
        )
        
        # Processing time alerts
        self.alert_manager.add_alert_rule(
            'processing.time.seconds',
            300.0,
            AlertLevel.WARNING,
            'greater',
            'Long processing time: {current_value:.1f}s'
        )
        
        # Error rate alerts
        self.alert_manager.add_alert_rule(
            'processing.error_rate',
            0.1,
            AlertLevel.ERROR,
            'greater',
            'High error rate: {current_value:.1%}'
        )
        
    def start_monitoring(self):
        """Start all monitoring components."""
        # Start system monitoring
        monitor_interval = self.config.get('monitor_interval', 1.0)
        self.system_monitor.start_monitoring(monitor_interval)
        
        # Start alert checking
        self._start_alert_checker()
        
        self.logger.info("Monitoring system started")
        
    def _start_alert_checker(self):
        """Start alert checking loop."""
        async def alert_checker():
            while True:
                try:
                    self.alert_manager.check_alerts()
                    await asyncio.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Error in alert checker: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        # Start alert checker as background task
        asyncio.create_task(alert_checker())
        
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.system_monitor.stop_monitoring()
        self.logger.info("Monitoring system stopped")
        
    def record_processing_event(self, event: ProcessingEvent):
        """Record a processing event."""
        self.processing_logger.log_event(event)
        
        # Record metrics
        self.metrics_collector.record_metric(
            'processing.time.seconds',
            event.processing_time,
            MetricType.TIMER
        )
        
        self.metrics_collector.record_metric(
            'processing.memory.mb',
            event.memory_usage,
            MetricType.HISTOGRAM
        )
        
        self.metrics_collector.record_metric(
            'processing.pages.count',
            event.pages_processed,
            MetricType.COUNTER
        )
        
        self.metrics_collector.record_metric(
            'processing.file_size.mb',
            event.file_size,
            MetricType.HISTOGRAM
        )
        
        # Success/failure metrics
        if event.status == 'success':
            self.metrics_collector.record_metric(
                'processing.success.count',
                1,
                MetricType.COUNTER
            )
        else:
            self.metrics_collector.record_metric(
                'processing.error.count',
                1,
                MetricType.COUNTER
            )
            
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            'system_metrics': self.metrics_collector.get_all_metrics(),
            'processing_summary': self.processing_logger.get_processing_summary(),
            'active_alerts': [asdict(a) for a in self.alert_manager.get_active_alerts()],
            'performance_profile': self.performance_profiler.get_profile_report(),
            'optimization_recommendations': self.performance_profiler.get_optimization_recommendations(),
            'timestamp': datetime.now().isoformat()
        }
        
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in various formats."""
        data = self.get_dashboard_data()
        
        if format == 'json':
            return json.dumps(data, indent=2)
        elif format == 'prometheus':
            # Convert to Prometheus format
            metrics = []
            for metric_name, metric_data in data['system_metrics'].items():
                if isinstance(metric_data, dict):
                    for key, value in metric_data.items():
                        metrics.append(f"{metric_name}_{key} {value}")
                else:
                    metrics.append(f"{metric_name} {metric_data}")
            return '\n'.join(metrics)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def __del__(self):
        """Cleanup resources."""
        self.stop_monitoring()


# Context manager for monitoring
class MonitoringContext:
    """Context manager for monitoring processing operations."""
    
    def __init__(self, monitoring_system: MonitoringSystem, operation_name: str):
        self.monitoring_system = monitoring_system
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        processing_time = end_time - self.start_time
        memory_usage = (end_memory - self.start_memory) / (1024**2)  # MB
        
        # Record metrics
        self.monitoring_system.metrics_collector.record_metric(
            f'operation.{self.operation_name}.time',
            processing_time,
            MetricType.TIMER
        )
        
        self.monitoring_system.metrics_collector.record_metric(
            f'operation.{self.operation_name}.memory',
            memory_usage,
            MetricType.HISTOGRAM
        )
        
        # Success/failure
        success = exc_type is None
        self.monitoring_system.metrics_collector.record_metric(
            f'operation.{self.operation_name}.success',
            1 if success else 0,
            MetricType.COUNTER
        )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create monitoring system
        monitoring = MonitoringSystem()
        
        # Example processing event
        event = ProcessingEvent(
            event_type='pdf_processing',
            timestamp=datetime.now(),
            file_path='test.pdf',
            status='success',
            processing_time=5.2,
            memory_usage=150.0,
            pages_processed=25,
            file_size=2.5
        )
        
        monitoring.record_processing_event(event)
        
        # Wait a bit for metrics to be collected
        await asyncio.sleep(2)
        
        # Get dashboard data
        dashboard = monitoring.get_dashboard_data()
        print(json.dumps(dashboard, indent=2))
        
        # Example of using monitoring context
        with MonitoringContext(monitoring, 'test_operation'):
            await asyncio.sleep(1)  # Simulate work
            
        # Export metrics
        metrics_json = monitoring.export_metrics('json')
        print(metrics_json)
        
    asyncio.run(main())