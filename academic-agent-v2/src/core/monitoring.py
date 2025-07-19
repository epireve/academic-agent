"""Monitoring and alerting system for Academic Agent v2."""

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .config_manager import get_config
from .exceptions import AcademicAgentError
from .logging import get_logger


@dataclass
class Metric:
    """Represents a single metric measurement."""
    
    name: str
    value: Union[int, float]
    unit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class Alert:
    """Represents an alert condition."""
    
    name: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'name': self.name,
            'level': self.level,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'context': self.context,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class MetricsCollector:
    """Collects and manages metrics from various sources."""
    
    def __init__(self, logger_name: str = "metrics_collector"):
        self.logger = get_logger(logger_name)
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.collection_callbacks: List[Callable] = []
        self.config = get_config().metrics
        
        # Initialize metric collectors
        self._init_performance_collectors()
        self._init_quality_collectors()
        self._init_communication_collectors()
    
    def _init_performance_collectors(self):
        """Initialize performance metric collectors."""
        if not self.config.enabled:
            return
        
        performance_metrics = self.config.performance
        
        if 'operation_duration' in performance_metrics:
            self.register_callback(self._collect_operation_duration)
        
        if 'memory_usage' in performance_metrics:
            self.register_callback(self._collect_memory_usage)
        
        if 'cpu_usage' in performance_metrics:
            self.register_callback(self._collect_cpu_usage)
        
        if 'processing_throughput' in performance_metrics:
            self.register_callback(self._collect_processing_throughput)
    
    def _init_quality_collectors(self):
        """Initialize quality metric collectors."""
        if not self.config.enabled:
            return
        
        quality_metrics = self.config.quality
        
        if 'quality_scores' in quality_metrics:
            self.register_callback(self._collect_quality_scores)
        
        if 'improvement_cycles' in quality_metrics:
            self.register_callback(self._collect_improvement_cycles)
        
        if 'success_rates' in quality_metrics:
            self.register_callback(self._collect_success_rates)
        
        if 'error_rates' in quality_metrics:
            self.register_callback(self._collect_error_rates)
    
    def _init_communication_collectors(self):
        """Initialize communication metric collectors."""
        if not self.config.enabled:
            return

        communication_metrics = self.config.communication

        if 'message_count' in communication_metrics:
            self.register_callback(self._collect_message_count)

        if 'message_latency' in communication_metrics:
            self.register_callback(self._collect_message_latency)

        if 'failed_messages' in communication_metrics:
            self.register_callback(self._collect_failed_messages)

        if 'retry_counts' in communication_metrics:
            self.register_callback(self._collect_retry_counts)

    def register_callback(self, callback: Callable):
        """Register a callback for metric collection."""
        self.collection_callbacks.append(callback)

    def collect_metric(self, metric: Metric):
        """Collect a single metric."""
        self.metrics.append(metric)
        self.logger.metric(metric.name, metric.value, metric.unit, **metric.tags)

        # Update aggregated metrics
        self._update_aggregated_metrics(metric)

    def collect_metrics(self, metrics: List[Metric]):
        """Collect multiple metrics."""
        for metric in metrics:
            self.collect_metric(metric)

    def _update_aggregated_metrics(self, metric: Metric):
        """Update aggregated metrics for trend analysis."""
        key = f"{metric.name}_{metric.unit or 'value'}"

        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0,
                'recent_values': deque(maxlen=100)
            }

        agg = self.aggregated_metrics[key]
        agg['count'] += 1
        agg['sum'] += metric.value
        agg['min'] = min(agg['min'], metric.value)
        agg['max'] = max(agg['max'], metric.value)
        agg['avg'] = agg['sum'] / agg['count']
        agg['recent_values'].append(metric.value)

    def get_metrics(self, 
                   name_filter: Optional[str] = None,
                   time_range: Optional[timedelta] = None,
                   tags: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Get metrics with optional filtering."""
        filtered_metrics = list(self.metrics)

        if name_filter:
            filtered_metrics = [m for m in filtered_metrics if name_filter in m.name]

        if time_range:
            cutoff_time = datetime.now() - time_range
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= cutoff_time]

        if tags:
            filtered_metrics = [
                m for m in filtered_metrics
                if all(m.tags.get(k) == v for k, v in tags.items())
            ]

        return filtered_metrics

    def get_aggregated_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated metrics for analysis."""
        return dict(self.aggregated_metrics)

    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a specific metric."""
        key = next((k for k in self.aggregated_metrics.keys() if metric_name in k), None)
        if key:
            return self.aggregated_metrics[key].copy()
        return None

    def collect_all_metrics(self):
        """Trigger collection from all registered callbacks."""
        for callback in self.collection_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Error in metric collection callback: {e}")

    def _collect_operation_duration(self):
        """Collect operation duration metrics."""
        # This would be implemented based on actual operation tracking
        pass

    def _collect_memory_usage(self):
        """Collect memory usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            self.collect_metric(Metric(
                name="memory_usage",
                value=memory_info.rss / 1024 / 1024,  # MB
                unit="MB",
                tags={"type": "rss"}
            ))

            self.collect_metric(Metric(
                name="memory_usage",
                value=memory_info.vms / 1024 / 1024,  # MB
                unit="MB",
                tags={"type": "vms"}
            ))

        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
        except Exception as e:
            self.logger.error(f"Error collecting memory metrics: {e}")

    def _collect_cpu_usage(self):
        """Collect CPU usage metrics."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)

            self.collect_metric(Metric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent"
            ))

        except ImportError:
            self.logger.warning("psutil not available for CPU monitoring")
        except Exception as e:
            self.logger.error(f"Error collecting CPU metrics: {e}")

    def _collect_processing_throughput(self):
        """Collect processing throughput metrics."""
        # This would be implemented based on actual processing tracking
        pass

    def _collect_quality_scores(self):
        """Collect quality score metrics."""
        # This would be implemented based on actual quality tracking
        pass

    def _collect_improvement_cycles(self):
        """Collect improvement cycle metrics."""
        # This would be implemented based on actual improvement tracking
        pass

    def _collect_success_rates(self):
        """Collect success rate metrics."""
        # This would be implemented based on actual success tracking
        pass

    def _collect_error_rates(self):
        """Collect error rate metrics."""
        # This would be implemented based on actual error tracking
        pass

    def _collect_message_count(self):
        """Collect message count metrics."""
        # This would be implemented based on actual message tracking
        pass

    def _collect_message_latency(self):
        """Collect message latency metrics."""
        # This would be implemented based on actual message tracking
        pass

    def _collect_failed_messages(self):
        """Collect failed message metrics."""
        # This would be implemented based on actual message tracking
        pass

    def _collect_retry_counts(self):
        """Collect retry count metrics."""
        # This would be implemented based on actual retry tracking
        pass

    def export_metrics(self, output_path: Path, format: str = "json"):
        """Export metrics to file."""
        metrics_data = [metric.to_dict() for metric in self.metrics]

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                if metrics_data:
                    writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                    writer.writeheader()
                    writer.writerows(metrics_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self.logger.info(f"Metrics exported to {output_path}")


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self, logger_name: str = "alert_manager"):
        self.logger = get_logger(logger_name)
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Callable] = {}
        self.notification_handlers: List[Callable] = []
        self.config = get_config().error_handling.monitoring\n    \n    def register_alert_rule(self, name: str, rule_func: Callable):\n        \"\"\"Register an alert rule.\"\"\"\n        self.alert_rules[name] = rule_func\n    \n    def register_notification_handler(self, handler: Callable):\n        \"\"\"Register a notification handler.\"\"\"\n        self.notification_handlers.append(handler)\n    \n    def create_alert(self, alert: Alert):\n        \"\"\"Create a new alert.\"\"\"\n        self.alerts.append(alert)\n        self.logger.warning(f\"Alert created: {alert.name} - {alert.message}\")\n        \n        # Trigger notifications\n        for handler in self.notification_handlers:\n            try:\n                handler(alert)\n            except Exception as e:\n                self.logger.error(f\"Error in notification handler: {e}\")\n    \n    def resolve_alert(self, alert_name: str):\n        \"\"\"Resolve an alert.\"\"\"\n        for alert in self.alerts:\n            if alert.name == alert_name and not alert.resolved:\n                alert.resolved = True\n                alert.resolved_at = datetime.now()\n                self.logger.info(f\"Alert resolved: {alert_name}\")\n                return True\n        return False\n    \n    def get_active_alerts(self) -> List[Alert]:\n        \"\"\"Get all active (unresolved) alerts.\"\"\"\n        return [alert for alert in self.alerts if not alert.resolved]\n    \n    def get_alerts(self, \n                   level: Optional[str] = None,\n                   resolved: Optional[bool] = None,\n                   time_range: Optional[timedelta] = None) -> List[Alert]:\n        \"\"\"Get alerts with optional filtering.\"\"\"\n        filtered_alerts = self.alerts\n        \n        if level:\n            filtered_alerts = [a for a in filtered_alerts if a.level == level]\n        \n        if resolved is not None:\n            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]\n        \n        if time_range:\n            cutoff_time = datetime.now() - time_range\n            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= cutoff_time]\n        \n        return filtered_alerts\n    \n    def check_alert_rules(self, metrics: List[Metric]):\n        \"\"\"Check all alert rules against current metrics.\"\"\"\n        for rule_name, rule_func in self.alert_rules.items():\n            try:\n                rule_func(metrics)\n            except Exception as e:\n                self.logger.error(f\"Error in alert rule {rule_name}: {e}\")\n    \n    def export_alerts(self, output_path: Path, format: str = \"json\"):\n        \"\"\"Export alerts to file.\"\"\"\n        alerts_data = [alert.to_dict() for alert in self.alerts]\n        \n        if format == \"json\":\n            with open(output_path, 'w') as f:\n                json.dump(alerts_data, f, indent=2)\n        else:\n            raise ValueError(f\"Unsupported format: {format}\")\n        \n        self.logger.info(f\"Alerts exported to {output_path}\")\n\n\nclass HealthChecker:\n    \"\"\"Monitors system health and component status.\"\"\"\n    \n    def __init__(self, logger_name: str = \"health_checker\"):\n        self.logger = get_logger(logger_name)\n        self.health_checks: Dict[str, Callable] = {}\n        self.component_status: Dict[str, Dict[str, Any]] = {}\n        self.last_check_time = datetime.now()\n    \n    def register_health_check(self, component_name: str, check_func: Callable):\n        \"\"\"Register a health check for a component.\"\"\"\n        self.health_checks[component_name] = check_func\n    \n    def check_component_health(self, component_name: str) -> Dict[str, Any]:\n        \"\"\"Check health of a specific component.\"\"\"\n        if component_name not in self.health_checks:\n            return {\"status\": \"unknown\", \"message\": \"No health check registered\"}\n        \n        try:\n            check_func = self.health_checks[component_name]\n            result = check_func()\n            \n            # Ensure result has required fields\n            if not isinstance(result, dict):\n                result = {\"status\": \"unknown\", \"message\": str(result)}\n            \n            if \"status\" not in result:\n                result[\"status\"] = \"unknown\"\n            \n            if \"timestamp\" not in result:\n                result[\"timestamp\"] = datetime.now().isoformat()\n            \n            self.component_status[component_name] = result\n            return result\n            \n        except Exception as e:\n            self.logger.error(f\"Health check failed for {component_name}: {e}\")\n            result = {\n                \"status\": \"error\",\n                \"message\": str(e),\n                \"timestamp\": datetime.now().isoformat()\n            }\n            self.component_status[component_name] = result\n            return result\n    \n    def check_all_health(self) -> Dict[str, Dict[str, Any]]:\n        \"\"\"Check health of all registered components.\"\"\"\n        results = {}\n        \n        for component_name in self.health_checks.keys():\n            results[component_name] = self.check_component_health(component_name)\n        \n        self.last_check_time = datetime.now()\n        return results\n    \n    def get_overall_health(self) -> Dict[str, Any]:\n        \"\"\"Get overall system health status.\"\"\"\n        all_health = self.check_all_health()\n        \n        if not all_health:\n            return {\"status\": \"unknown\", \"message\": \"No health checks registered\"}\n        \n        # Determine overall status\n        statuses = [status[\"status\"] for status in all_health.values()]\n        \n        if \"error\" in statuses:\n            overall_status = \"error\"\n        elif \"warning\" in statuses:\n            overall_status = \"warning\"\n        elif all(status == \"healthy\" for status in statuses):\n            overall_status = \"healthy\"\n        else:\n            overall_status = \"unknown\"\n        \n        return {\n            \"status\": overall_status,\n            \"components\": all_health,\n            \"timestamp\": datetime.now().isoformat(),\n            \"last_check\": self.last_check_time.isoformat()\n        }\n\n\nclass SystemMonitor:\n    \"\"\"Main monitoring system that coordinates all monitoring components.\"\"\"\n    \n    def __init__(self):\n        self.logger = get_logger(\"system_monitor\")\n        self.metrics_collector = MetricsCollector()\n        self.alert_manager = AlertManager()\n        self.health_checker = HealthChecker()\n        self.config = get_config().metrics\n        \n        # Set up default health checks\n        self._setup_default_health_checks()\n        \n        # Set up default alert rules\n        self._setup_default_alert_rules()\n        \n        # Set up default notification handlers\n        self._setup_default_notification_handlers()\n    \n    def _setup_default_health_checks(self):\n        \"\"\"Set up default health checks.\"\"\"\n        self.health_checker.register_health_check(\"logging\", self._check_logging_health)\n        self.health_checker.register_health_check(\"configuration\", self._check_config_health)\n        self.health_checker.register_health_check(\"storage\", self._check_storage_health)\n    \n    def _setup_default_alert_rules(self):\n        \"\"\"Set up default alert rules.\"\"\"\n        self.alert_manager.register_alert_rule(\"high_error_rate\", self._check_error_rate_rule)\n        self.alert_manager.register_alert_rule(\"low_quality_scores\", self._check_quality_rule)\n        self.alert_manager.register_alert_rule(\"high_memory_usage\", self._check_memory_rule)\n    \n    def _setup_default_notification_handlers(self):\n        \"\"\"Set up default notification handlers.\"\"\"\n        self.alert_manager.register_notification_handler(self._log_alert_handler)\n        \n        # Additional handlers can be added here\n        # self.alert_manager.register_notification_handler(self._email_alert_handler)\n        # self.alert_manager.register_notification_handler(self._slack_alert_handler)\n    \n    def _check_logging_health(self) -> Dict[str, Any]:\n        \"\"\"Check logging system health.\"\"\"\n        try:\n            # Check if log directory exists and is writable\n            log_dir = Path(self.config.export.get(\"json_file\", {}).get(\"path\", \"logs\")).parent\n            \n            if not log_dir.exists():\n                return {\"status\": \"error\", \"message\": f\"Log directory does not exist: {log_dir}\"}\n            \n            if not log_dir.is_dir():\n                return {\"status\": \"error\", \"message\": f\"Log path is not a directory: {log_dir}\"}\n            \n            # Try to write a test file\n            test_file = log_dir / \"health_check_test.tmp\"\n            try:\n                test_file.write_text(\"test\")\n                test_file.unlink()\n            except Exception as e:\n                return {\"status\": \"error\", \"message\": f\"Cannot write to log directory: {e}\"}\n            \n            return {\"status\": \"healthy\", \"message\": \"Logging system is operational\"}\n            \n        except Exception as e:\n            return {\"status\": \"error\", \"message\": f\"Logging health check failed: {e}\"}\n    \n    def _check_config_health(self) -> Dict[str, Any]:\n        \"\"\"Check configuration health.\"\"\"\n        try:\n            config = get_config()\n            \n            if config is None:\n                return {\"status\": \"error\", \"message\": \"Configuration not loaded\"}\n            \n            # Check critical configuration values\n            if not (0 <= config.quality_threshold <= 1):\n                return {\"status\": \"warning\", \"message\": \"Quality threshold out of range\"}\n            \n            if config.max_improvement_cycles < 1:\n                return {\"status\": \"warning\", \"message\": \"Max improvement cycles too low\"}\n            \n            return {\"status\": \"healthy\", \"message\": \"Configuration is valid\"}\n            \n        except Exception as e:\n            return {\"status\": \"error\", \"message\": f\"Configuration health check failed: {e}\"}\n    \n    def _check_storage_health(self) -> Dict[str, Any]:\n        \"\"\"Check storage health.\"\"\"\n        try:\n            import shutil\n            \n            # Check disk space\n            total, used, free = shutil.disk_usage(\"/\")\n            free_percent = (free / total) * 100\n            \n            if free_percent < 5:\n                return {\"status\": \"error\", \"message\": f\"Disk space critically low: {free_percent:.1f}% free\"}\n            elif free_percent < 20:\n                return {\"status\": \"warning\", \"message\": f\"Disk space low: {free_percent:.1f}% free\"}\n            else:\n                return {\"status\": \"healthy\", \"message\": f\"Disk space adequate: {free_percent:.1f}% free\"}\n            \n        except Exception as e:\n            return {\"status\": \"error\", \"message\": f\"Storage health check failed: {e}\"}\n    \n    def _check_error_rate_rule(self, metrics: List[Metric]):\n        \"\"\"Check if error rate is too high.\"\"\"\n        # This would be implemented based on actual error tracking\n        pass\n    \n    def _check_quality_rule(self, metrics: List[Metric]):\n        \"\"\"Check if quality scores are too low.\"\"\"\n        # This would be implemented based on actual quality tracking\n        pass\n    \n    def _check_memory_rule(self, metrics: List[Metric]):\n        \"\"\"Check if memory usage is too high.\"\"\"\n        memory_metrics = [m for m in metrics if m.name == \"memory_usage\"]\n        \n        if not memory_metrics:\n            return\n        \n        # Get recent memory usage\n        recent_memory = [m for m in memory_metrics if (datetime.now() - m.timestamp).seconds < 300]\n        \n        if recent_memory:\n            avg_memory = sum(m.value for m in recent_memory) / len(recent_memory)\n            \n            if avg_memory > 1000:  # 1GB\n                alert = Alert(\n                    name=\"high_memory_usage\",\n                    level=\"WARNING\",\n                    message=f\"High memory usage detected: {avg_memory:.1f}MB\",\n                    source=\"system_monitor\",\n                    context={\"average_memory_mb\": avg_memory}\n                )\n                self.alert_manager.create_alert(alert)\n    \n    def _log_alert_handler(self, alert: Alert):\n        \"\"\"Handle alerts by logging them.\"\"\"\n        log_level = getattr(self.logger, alert.level.lower(), self.logger.info)\n        log_level(f\"ALERT: {alert.name} - {alert.message}\", extra_context=alert.context)\n    \n    def start_monitoring(self, interval: int = None):\n        \"\"\"Start the monitoring system.\"\"\"\n        if interval is None:\n            interval = self.config.collection_interval\n        \n        self.logger.info(f\"Starting monitoring system with {interval}s interval\")\n        \n        # This would typically run in a separate thread or process\n        # For now, we'll just set up the monitoring infrastructure\n        \n        # Collect initial metrics\n        self.metrics_collector.collect_all_metrics()\n        \n        # Check initial health\n        health_status = self.health_checker.get_overall_health()\n        self.logger.info(f\"Initial health status: {health_status['status']}\")\n        \n        # Check for any initial alerts\n        current_metrics = list(self.metrics_collector.metrics)\n        self.alert_manager.check_alert_rules(current_metrics)\n    \n    def get_monitoring_summary(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive monitoring summary.\"\"\"\n        return {\n            \"health\": self.health_checker.get_overall_health(),\n            \"metrics_summary\": self.metrics_collector.get_aggregated_metrics(),\n            \"active_alerts\": [alert.to_dict() for alert in self.alert_manager.get_active_alerts()],\n            \"total_metrics\": len(self.metrics_collector.metrics),\n            \"total_alerts\": len(self.alert_manager.alerts),\n            \"timestamp\": datetime.now().isoformat()\n        }\n    \n    def export_monitoring_data(self, output_dir: Path):\n        \"\"\"Export all monitoring data.\"\"\"\n        output_dir.mkdir(parents=True, exist_ok=True)\n        \n        # Export metrics\n        self.metrics_collector.export_metrics(output_dir / \"metrics.json\")\n        \n        # Export alerts\n        self.alert_manager.export_alerts(output_dir / \"alerts.json\")\n        \n        # Export health status\n        health_status = self.health_checker.get_overall_health()\n        with open(output_dir / \"health.json\", 'w') as f:\n            json.dump(health_status, f, indent=2)\n        \n        # Export summary\n        summary = self.get_monitoring_summary()\n        with open(output_dir / \"monitoring_summary.json\", 'w') as f:\n            json.dump(summary, f, indent=2)\n        \n        self.logger.info(f\"Monitoring data exported to {output_dir}\")\n\n\n# Global monitoring instance\n_system_monitor: Optional[SystemMonitor] = None\n\n\ndef get_system_monitor() -> SystemMonitor:\n    \"\"\"Get the global system monitor instance.\"\"\"\n    global _system_monitor\n    \n    if _system_monitor is None:\n        _system_monitor = SystemMonitor()\n    \n    return _system_monitor\n\n\ndef start_monitoring(interval: int = None):\n    \"\"\"Start the global monitoring system.\"\"\"\n    get_system_monitor().start_monitoring(interval)\n\n\ndef get_monitoring_summary() -> Dict[str, Any]:\n    \"\"\"Get comprehensive monitoring summary.\"\"\"\n    return get_system_monitor().get_monitoring_summary()"