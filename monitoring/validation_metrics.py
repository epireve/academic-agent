#!/usr/bin/env python3
"""
Validation Metrics Integration for Academic Agent Monitoring

This module integrates the Quality Validation System with the existing
Prometheus-based monitoring infrastructure to provide real-time metrics
on PDF-to-markdown conversion accuracy and quality validation performance.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
from collections import defaultdict

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass


logger = logging.getLogger(__name__)


class ValidationMetricsCollector:
    """
    Prometheus metrics collector for Quality Validation System
    
    Collects and exposes metrics related to PDF-to-markdown conversion
    accuracy, validation performance, and quality trends.
    """

    def __init__(self, registry=None):
        """Initialize metrics collector with Prometheus metrics."""
        
        # Validation test metrics
        self.validation_tests_total = Counter(
            'academic_agent_validation_tests_total',
            'Total number of validation tests run',
            ['benchmark_id', 'content_type', 'difficulty_level', 'status'],
            registry=registry
        )
        
        self.validation_accuracy_score = Histogram(
            'academic_agent_validation_accuracy_score',
            'Accuracy scores from validation tests',
            ['benchmark_id', 'content_type', 'difficulty_level'],
            buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=registry
        )
        
        self.validation_processing_time = Histogram(
            'academic_agent_validation_processing_time_seconds',
            'Time taken for PDF processing during validation',
            ['benchmark_id', 'content_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=registry
        )
        
        # Component-specific accuracy metrics
        self.content_accuracy = Histogram(
            'academic_agent_content_accuracy_score',
            'Content extraction accuracy scores',
            ['benchmark_id', 'content_type'],
            buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=registry
        )
        
        self.structure_accuracy = Histogram(
            'academic_agent_structure_accuracy_score',
            'Document structure preservation accuracy',
            ['benchmark_id', 'content_type'],
            buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=registry
        )
        
        self.formatting_accuracy = Histogram(
            'academic_agent_formatting_accuracy_score',
            'Markdown formatting accuracy scores',
            ['benchmark_id', 'content_type'],
            buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=registry
        )
        
        self.metadata_accuracy = Histogram(
            'academic_agent_metadata_accuracy_score',
            'Metadata extraction accuracy scores',
            ['benchmark_id', 'content_type'],
            buckets=[0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            registry=registry
        )
        
        # Quality trend metrics
        self.overall_quality_gauge = Gauge(
            'academic_agent_overall_quality_score',
            'Current overall quality score across all benchmarks',
            ['time_window'],
            registry=registry
        )
        
        self.pass_rate_gauge = Gauge(
            'academic_agent_validation_pass_rate',
            'Current validation pass rate percentage',
            ['time_window', 'content_type'],
            registry=registry
        )
        
        self.regression_count = Gauge(
            'academic_agent_quality_regressions_detected',
            'Number of quality regressions detected',
            ['time_window'],
            registry=registry
        )
        
        # Benchmark metrics
        self.benchmark_count = Gauge(
            'academic_agent_validation_benchmarks_total',
            'Total number of validation benchmarks',
            ['content_type', 'difficulty_level'],
            registry=registry
        )
        
        self.benchmark_average_accuracy = Gauge(
            'academic_agent_benchmark_average_accuracy',
            'Average accuracy score for each benchmark',
            ['benchmark_id', 'benchmark_name', 'content_type'],
            registry=registry
        )
        
        # Validation suite metrics
        self.validation_suite_duration = Summary(
            'academic_agent_validation_suite_duration_seconds',
            'Time taken to run full validation suite',
            ['benchmark_count'],
            registry=registry
        )
        
        self.validation_suite_tests = Gauge(
            'academic_agent_validation_suite_tests',
            'Number of tests in last validation suite run',
            ['status'],
            registry=registry
        )
        
        # Error and warning metrics
        self.validation_errors = Counter(
            'academic_agent_validation_errors_total',
            'Total validation errors encountered',
            ['error_type', 'benchmark_id'],
            registry=registry
        )
        
        self.validation_warnings = Counter(
            'academic_agent_validation_warnings_total',
            'Total validation warnings encountered',
            ['warning_type', 'benchmark_id'],
            registry=registry
        )
        
        # Performance degradation metrics
        self.accuracy_degradation = Counter(
            'academic_agent_accuracy_degradation_total',
            'Number of times accuracy dropped below threshold',
            ['benchmark_id', 'severity'],
            registry=registry
        )
        
        # System information
        self.validation_system_info = Info(
            'academic_agent_validation_system_info',
            'Information about the validation system',
            registry=registry
        )
        
        logger.info("Validation metrics collector initialized")

    def record_validation_test(self, validation_result, benchmark_info: Optional[Dict] = None):
        """
        Record metrics from a validation test result
        
        Args:
            validation_result: ValidationResult object
            benchmark_info: Optional benchmark information for labeling
        """
        try:
            # Extract labels
            benchmark_id = validation_result.benchmark_id
            content_type = benchmark_info.get('content_type', 'unknown') if benchmark_info else 'unknown'
            difficulty_level = benchmark_info.get('difficulty_level', 'unknown') if benchmark_info else 'unknown'
            status = 'passed' if validation_result.passed else 'failed'
            
            # Record test count
            self.validation_tests_total.labels(
                benchmark_id=benchmark_id,
                content_type=content_type,
                difficulty_level=difficulty_level,
                status=status
            ).inc()
            
            # Record accuracy scores
            self.validation_accuracy_score.labels(
                benchmark_id=benchmark_id,
                content_type=content_type,
                difficulty_level=difficulty_level
            ).observe(validation_result.accuracy_score)
            
            # Record component accuracies
            self.content_accuracy.labels(
                benchmark_id=benchmark_id,
                content_type=content_type
            ).observe(validation_result.content_accuracy)
            
            self.structure_accuracy.labels(
                benchmark_id=benchmark_id,
                content_type=content_type
            ).observe(validation_result.structure_accuracy)
            
            self.formatting_accuracy.labels(
                benchmark_id=benchmark_id,
                content_type=content_type
            ).observe(validation_result.formatting_accuracy)
            
            self.metadata_accuracy.labels(
                benchmark_id=benchmark_id,
                content_type=content_type
            ).observe(validation_result.metadata_accuracy)
            
            # Record processing time
            self.validation_processing_time.labels(
                benchmark_id=benchmark_id,
                content_type=content_type
            ).observe(validation_result.processing_time)
            
            # Record errors and warnings
            for error in validation_result.errors:
                error_type = self._categorize_error(error)
                self.validation_errors.labels(
                    error_type=error_type,
                    benchmark_id=benchmark_id
                ).inc()
            
            for warning in validation_result.warnings:
                warning_type = self._categorize_warning(warning)
                self.validation_warnings.labels(
                    warning_type=warning_type,
                    benchmark_id=benchmark_id
                ).inc()
            
            # Check for accuracy degradation
            if validation_result.accuracy_score < 0.9:
                severity = 'critical' if validation_result.accuracy_score < 0.8 else 'warning'
                self.accuracy_degradation.labels(
                    benchmark_id=benchmark_id,
                    severity=severity
                ).inc()
            
            logger.debug(f"Recorded validation metrics for test {validation_result.test_id}")
            
        except Exception as e:
            logger.error(f"Error recording validation test metrics: {str(e)}")

    def record_validation_suite(self, suite_results: List, duration: float, benchmark_info_map: Dict = None):
        """
        Record metrics from a full validation suite run
        
        Args:
            suite_results: List of ValidationResult objects
            duration: Total time taken for the suite
            benchmark_info_map: Map of benchmark_id to benchmark info
        """
        try:
            benchmark_count = len(set(result.benchmark_id for result in suite_results))
            
            # Record suite duration
            self.validation_suite_duration.labels(
                benchmark_count=str(benchmark_count)
            ).observe(duration)
            
            # Count test statuses
            status_counts = defaultdict(int)
            for result in suite_results:
                status = 'passed' if result.passed else 'failed'
                status_counts[status] += 1
            
            # Record test counts by status
            for status, count in status_counts.items():
                self.validation_suite_tests.labels(status=status).set(count)
            
            # Record individual test metrics
            for result in suite_results:
                benchmark_info = benchmark_info_map.get(result.benchmark_id, {}) if benchmark_info_map else {}
                self.record_validation_test(result, benchmark_info)
            
            logger.info(f"Recorded validation suite metrics: {len(suite_results)} tests, {duration:.2f}s duration")
            
        except Exception as e:
            logger.error(f"Error recording validation suite metrics: {str(e)}")

    def update_quality_trends(self, validation_results: List, time_window: str = "24h"):
        """
        Update quality trend metrics
        
        Args:
            validation_results: Recent validation results
            time_window: Time window label (e.g., "24h", "7d", "30d")
        """
        try:
            if not validation_results:
                return
            
            # Calculate overall quality
            accuracy_scores = [result.accuracy_score for result in validation_results]
            overall_quality = sum(accuracy_scores) / len(accuracy_scores)
            
            self.overall_quality_gauge.labels(time_window=time_window).set(overall_quality)
            
            # Calculate pass rates by content type
            results_by_content_type = defaultdict(list)
            for result in validation_results:
                # We would need benchmark info to get content type
                # For now, use a default
                content_type = 'unknown'  # This would be filled from benchmark info
                results_by_content_type[content_type].append(result)
            
            for content_type, type_results in results_by_content_type.items():
                passed = sum(1 for result in type_results if result.passed)
                total = len(type_results)
                pass_rate = (passed / total) * 100 if total > 0 else 0
                
                self.pass_rate_gauge.labels(
                    time_window=time_window,
                    content_type=content_type
                ).set(pass_rate)
            
            logger.debug(f"Updated quality trend metrics for {time_window}: {overall_quality:.3f} overall quality")
            
        except Exception as e:
            logger.error(f"Error updating quality trend metrics: {str(e)}")

    def update_benchmark_metrics(self, benchmarks: Dict):
        """
        Update benchmark-related metrics
        
        Args:
            benchmarks: Dictionary of benchmark_id -> ValidationBenchmark
        """
        try:
            # Count benchmarks by category
            content_type_counts = defaultdict(int)
            difficulty_counts = defaultdict(int)
            
            for benchmark in benchmarks.values():
                content_type_counts[benchmark.content_type] += 1
                difficulty_counts[benchmark.difficulty_level] += 1
                
                # Update individual benchmark metrics
                self.benchmark_average_accuracy.labels(
                    benchmark_id=benchmark.benchmark_id,
                    benchmark_name=benchmark.name,
                    content_type=benchmark.content_type
                ).set(benchmark.average_accuracy)
            
            # Update category counts
            for content_type, count in content_type_counts.items():
                for difficulty_level, diff_count in difficulty_counts.items():
                    # This gives us a cross-product, but we'll set 0 for non-matching combinations
                    actual_count = sum(1 for b in benchmarks.values() 
                                     if b.content_type == content_type and b.difficulty_level == difficulty_level)
                    self.benchmark_count.labels(
                        content_type=content_type,
                        difficulty_level=difficulty_level
                    ).set(actual_count)
            
            logger.debug(f"Updated metrics for {len(benchmarks)} benchmarks")
            
        except Exception as e:
            logger.error(f"Error updating benchmark metrics: {str(e)}")

    def record_regression_detection(self, regression_analysis: Dict, time_window: str = "7d"):
        """
        Record regression detection metrics
        
        Args:
            regression_analysis: Regression analysis results
            time_window: Time window for regression detection
        """
        try:
            regression_count = regression_analysis.get('regression_count', 0)
            
            self.regression_count.labels(time_window=time_window).set(regression_count)
            
            # Record individual regressions
            for regression in regression_analysis.get('regressions', []):
                benchmark_id = regression.get('benchmark_id', 'unknown')
                severity = 'critical' if regression.get('regression_magnitude', 0) > 0.1 else 'warning'
                
                self.accuracy_degradation.labels(
                    benchmark_id=benchmark_id,
                    severity=severity
                ).inc()
            
            logger.info(f"Recorded regression detection: {regression_count} regressions found")
            
        except Exception as e:
            logger.error(f"Error recording regression detection metrics: {str(e)}")

    def update_system_info(self, system_info: Dict):
        """
        Update validation system information metric
        
        Args:
            system_info: System information dictionary
        """
        try:
            # Filter out sensitive or large data
            safe_info = {
                'python_version': system_info.get('python_version', 'unknown'),
                'platform': system_info.get('platform', 'unknown'),
                'processor_version': system_info.get('processor_version', 'unknown'),
                'cpu_count': str(system_info.get('cpu_count', 'unknown')),
                'last_updated': datetime.now().isoformat()
            }
            
            self.validation_system_info.info(safe_info)
            
        except Exception as e:
            logger.error(f"Error updating system info metrics: {str(e)}")

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message for metrics labeling."""
        error_lower = error_message.lower()
        
        if 'timeout' in error_lower:
            return 'timeout'
        elif 'memory' in error_lower:
            return 'memory'
        elif 'file' in error_lower or 'path' in error_lower:
            return 'file_system'
        elif 'permission' in error_lower:
            return 'permission'
        elif 'network' in error_lower or 'connection' in error_lower:
            return 'network'
        elif 'pdf' in error_lower:
            return 'pdf_processing'
        elif 'format' in error_lower:
            return 'format'
        else:
            return 'other'

    def _categorize_warning(self, warning_message: str) -> str:
        """Categorize warning message for metrics labeling."""
        warning_lower = warning_message.lower()
        
        if 'performance' in warning_lower or 'slow' in warning_lower:
            return 'performance'
        elif 'quality' in warning_lower:
            return 'quality'
        elif 'format' in warning_lower:
            return 'formatting'
        elif 'deprecated' in warning_lower:
            return 'deprecated'
        else:
            return 'other'

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current metrics (for debugging/monitoring)
        
        Returns:
            Dictionary with metrics summary
        """
        try:
            if not PROMETHEUS_AVAILABLE:
                return {"error": "Prometheus client not available"}
            
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            # This would typically be called by the metrics endpoint
            metrics_output = generate_latest().decode('utf-8')
            
            # Parse some basic stats
            lines = metrics_output.split('\n')
            validation_metrics = [line for line in lines if 'academic_agent_validation' in line]
            
            return {
                "total_metrics": len(lines),
                "validation_metrics": len(validation_metrics),
                "timestamp": datetime.now().isoformat(),
                "prometheus_available": True
            }
            
        except Exception as e:
            logger.error(f"Error generating metrics summary: {str(e)}")
            return {"error": str(e)}


class ValidationMetricsIntegration:
    """
    Integration layer between Quality Validation System and Monitoring
    
    Provides automatic metrics collection and reporting for validation activities.
    """

    def __init__(self, metrics_collector: Optional[ValidationMetricsCollector] = None):
        """
        Initialize metrics integration
        
        Args:
            metrics_collector: Optional custom metrics collector
        """
        self.metrics_collector = metrics_collector or ValidationMetricsCollector()
        self.last_update = datetime.now()
        self.update_interval = timedelta(minutes=5)  # Update trends every 5 minutes
        
        logger.info("Validation metrics integration initialized")

    def hook_validation_system(self, validation_system):
        """
        Hook into validation system to automatically collect metrics
        
        Args:
            validation_system: QualityValidationSystem instance
        """
        try:
            # Store reference to validation system
            self.validation_system = validation_system
            
            # Override key methods to collect metrics
            original_validate_benchmark = validation_system.validate_benchmark
            original_run_full_suite = validation_system.run_full_validation_suite
            original_detect_regressions = validation_system.detect_quality_regressions
            
            def wrapped_validate_benchmark(benchmark_id, pdf_processor_func, processor_config=None):
                start_time = time.time()
                result = original_validate_benchmark(benchmark_id, pdf_processor_func, processor_config)
                
                # Collect metrics
                benchmark_info = validation_system.benchmarks.get(benchmark_id, {})
                if hasattr(benchmark_info, 'to_dict'):
                    benchmark_info = benchmark_info.to_dict()
                
                self.metrics_collector.record_validation_test(result, benchmark_info)
                
                return result
            
            def wrapped_run_full_suite(pdf_processor_func, processor_config=None, benchmark_filter=None):
                start_time = time.time()
                results = original_run_full_suite(pdf_processor_func, processor_config, benchmark_filter)
                duration = time.time() - start_time
                
                # Collect metrics
                benchmark_info_map = {}
                for benchmark_id, benchmark in validation_system.benchmarks.items():
                    if hasattr(benchmark, 'to_dict'):
                        benchmark_info_map[benchmark_id] = benchmark.to_dict()
                
                self.metrics_collector.record_validation_suite(results, duration, benchmark_info_map)
                
                return results
            
            def wrapped_detect_regressions(window_days=7):
                result = original_detect_regressions(window_days)
                
                # Collect metrics
                self.metrics_collector.record_regression_detection(result, f"{window_days}d")
                
                return result
            
            # Replace methods
            validation_system.validate_benchmark = wrapped_validate_benchmark
            validation_system.run_full_validation_suite = wrapped_run_full_suite
            validation_system.detect_quality_regressions = wrapped_detect_regressions
            
            # Initial metrics update
            self._update_static_metrics()
            
            logger.info("Successfully hooked validation system for metrics collection")
            
        except Exception as e:
            logger.error(f"Error hooking validation system: {str(e)}")

    def update_metrics(self, force: bool = False):
        """
        Update metrics (called periodically)
        
        Args:
            force: Force update regardless of interval
        """
        try:
            now = datetime.now()
            
            if not force and (now - self.last_update) < self.update_interval:
                return
            
            if hasattr(self, 'validation_system'):
                self._update_static_metrics()
                self._update_trend_metrics()
            
            self.last_update = now
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def _update_static_metrics(self):
        """Update static metrics (benchmarks, system info)."""
        try:
            if hasattr(self, 'validation_system'):
                # Update benchmark metrics
                self.metrics_collector.update_benchmark_metrics(self.validation_system.benchmarks)
                
                # Update system info
                system_info = self.validation_system._collect_system_info()
                self.metrics_collector.update_system_info(system_info)
            
        except Exception as e:
            logger.error(f"Error updating static metrics: {str(e)}")

    def _update_trend_metrics(self):
        """Update trend metrics based on recent validation results."""
        try:
            if hasattr(self, 'validation_system'):
                # Get recent results for different time windows
                now = datetime.now()
                
                for window_hours, label in [(24, "24h"), (168, "7d"), (720, "30d")]:
                    cutoff = now - timedelta(hours=window_hours)
                    recent_results = [
                        result for result in self.validation_system.validation_results
                        if result.timestamp >= cutoff
                    ]
                    
                    if recent_results:
                        self.metrics_collector.update_quality_trends(recent_results, label)
            
        except Exception as e:
            logger.error(f"Error updating trend metrics: {str(e)}")

    def get_metrics_endpoint(self):
        """
        Get metrics in Prometheus format for scraping
        
        Returns:
            Tuple of (content, content_type) for HTTP response
        """
        try:
            if not PROMETHEUS_AVAILABLE:
                return "# Prometheus client not available\n", "text/plain"
            
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            # Update metrics before serving
            self.update_metrics(force=True)
            
            content = generate_latest()
            return content, CONTENT_TYPE_LATEST
            
        except Exception as e:
            logger.error(f"Error generating metrics endpoint: {str(e)}")
            return f"# Error: {str(e)}\n", "text/plain"

    def export_metrics_json(self) -> Dict[str, Any]:
        """
        Export metrics in JSON format for API consumption
        
        Returns:
            Dictionary with current metrics data
        """
        try:
            metrics_summary = self.metrics_collector.get_metrics_summary()
            
            if hasattr(self, 'validation_system'):
                # Add validation system statistics
                metrics_summary.update({
                    "validation_system": {
                        "total_benchmarks": len(self.validation_system.benchmarks),
                        "total_validation_results": len(self.validation_system.validation_results),
                        "accuracy_threshold": self.validation_system.accuracy_threshold,
                        "regression_threshold": self.validation_system.regression_threshold
                    }
                })
            
            return metrics_summary
            
        except Exception as e:
            logger.error(f"Error exporting metrics JSON: {str(e)}")
            return {"error": str(e)}


def create_validation_metrics_integration(validation_system=None) -> ValidationMetricsIntegration:
    """
    Factory function to create and configure validation metrics integration
    
    Args:
        validation_system: Optional QualityValidationSystem to hook into
        
    Returns:
        Configured ValidationMetricsIntegration instance
    """
    try:
        metrics_collector = ValidationMetricsCollector()
        integration = ValidationMetricsIntegration(metrics_collector)
        
        if validation_system:
            integration.hook_validation_system(validation_system)
        
        logger.info("Created validation metrics integration")
        return integration
        
    except Exception as e:
        logger.error(f"Error creating validation metrics integration: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage and testing
    print("Validation Metrics Integration")
    print(f"Prometheus available: {PROMETHEUS_AVAILABLE}")
    
    # Create metrics collector
    collector = ValidationMetricsCollector()
    
    # Get metrics summary
    summary = collector.get_metrics_summary()
    print("Metrics summary:", summary)