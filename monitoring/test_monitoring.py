#!/usr/bin/env python3
"""Test script for Academic Agent monitoring system."""

import time
import random
from pathlib import Path

from academic_agent_v2.src.core.logging import get_logger
from .integration import (
    get_monitoring_integration,
    track_operation,
    track_pdf_processing,
    track_quality_score,
    record_error
)


class MonitoringTester:
    """Test suite for the monitoring system."""
    
    def __init__(self):
        self.logger = get_logger("monitoring_tester")
        self.integration = get_monitoring_integration()
    
    def test_basic_metrics(self):
        """Test basic metric collection."""
        self.logger.info("Testing basic metrics collection...")
        
        # Test operation tracking
        with track_operation("test_operation", "test_001"):
            time.sleep(random.uniform(0.1, 2.0))
            self.logger.info("Completed test operation")
        
        # Test error recording
        record_error("TestError", "test_component", "warning")
        
        self.logger.info("Basic metrics test completed")
    
    def test_pdf_processing_metrics(self):
        """Test PDF processing metrics."""
        self.logger.info("Testing PDF processing metrics...")
        
        # Simulate PDF processing
        test_file = Path("test.pdf")
        
        @track_pdf_processing("test_processor", test_file, 10, "success")
        def simulate_pdf_processing():
            time.sleep(random.uniform(1.0, 5.0))
            return {"pages": 10, "images": 3}
        
        try:
            result = simulate_pdf_processing()
            self.logger.info(f"PDF processing result: {result}")
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
        
        self.logger.info("PDF processing metrics test completed")
    
    def test_quality_metrics(self):
        """Test quality score metrics."""
        self.logger.info("Testing quality metrics...")
        
        @track_quality_score("test_content", "test_agent")
        def simulate_quality_assessment():
            # Simulate variable quality scores
            score = random.uniform(0.5, 1.0)
            return {"quality_score": score}
        
        # Generate multiple quality assessments
        for i in range(5):
            result = simulate_quality_assessment()
            self.logger.info(f"Quality assessment {i+1}: {result['quality_score']:.2f}")
            time.sleep(0.5)
        
        self.logger.info("Quality metrics test completed")
    
    def test_concurrent_operations(self):
        """Test concurrent operation tracking."""
        self.logger.info("Testing concurrent operations...")
        
        import threading
        
        def worker(worker_id):
            with track_operation("concurrent_test", f"worker_{worker_id}"):
                sleep_time = random.uniform(0.5, 3.0)
                self.logger.info(f"Worker {worker_id} working for {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Start multiple concurrent operations
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all operations to complete
        for thread in threads:
            thread.join()
        
        self.logger.info("Concurrent operations test completed")
    
    def test_error_scenarios(self):
        """Test error metric recording."""
        self.logger.info("Testing error scenarios...")
        
        # Test different error types
        error_types = [
            ("ValidationError", "input_validation", "error"),
            ("ConnectionError", "external_api", "warning"),
            ("ProcessingError", "pdf_processor", "error"),
            ("QualityError", "quality_assessment", "warning")
        ]
        
        for error_type, component, severity in error_types:
            record_error(error_type, component, severity)
            self.logger.info(f"Recorded {severity} error: {error_type} in {component}")
            time.sleep(0.2)
        
        self.logger.info("Error scenarios test completed")
    
    def test_threshold_violations(self):
        """Test metrics that should trigger alerts."""
        self.logger.info("Testing threshold violations...")
        
        # Simulate slow operations that should trigger alerts
        with track_operation("slow_operation", "threshold_test"):
            self.logger.info("Simulating slow operation (should trigger alert)...")
            time.sleep(6.0)  # Longer than typical threshold
        
        # Simulate multiple errors
        for i in range(5):
            record_error("ThresholdTestError", "test_component", "error")
            time.sleep(0.1)
        
        self.logger.info("Threshold violations test completed")
    
    def test_metric_aggregation(self):
        """Test metric aggregation and calculations."""
        self.logger.info("Testing metric aggregation...")
        
        # Generate multiple operations for aggregation
        operation_types = ["ingestion", "analysis", "generation", "quality_check"]
        
        for _ in range(20):
            operation_type = random.choice(operation_types)
            operation_duration = random.uniform(0.1, 5.0)
            
            with track_operation(operation_type, f"agg_test_{int(time.time())}"):
                time.sleep(operation_duration)
        
        self.logger.info("Metric aggregation test completed")
    
    def run_all_tests(self):
        """Run all monitoring tests."""
        self.logger.info("Starting comprehensive monitoring test suite...")
        
        test_methods = [
            self.test_basic_metrics,
            self.test_pdf_processing_metrics,
            self.test_quality_metrics,
            self.test_concurrent_operations,
            self.test_error_scenarios,
            self.test_threshold_violations,
            self.test_metric_aggregation
        ]
        
        failed_tests = []
        
        for test_method in test_methods:
            try:
                test_method()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.logger.error(f"Test failed {test_method.__name__}: {e}")
                failed_tests.append(test_method.__name__)
        
        # Summary
        if failed_tests:
            self.logger.error(f"Test suite completed with failures: {', '.join(failed_tests)}")
        else:
            self.logger.info("All monitoring tests completed successfully!")
        
        # Display monitoring status
        status = self.integration.get_monitoring_status()
        self.logger.info(f"Monitoring status: {status}")
        
        return len(failed_tests) == 0


def run_load_test(duration_minutes: int = 5):
    """Run a load test to generate metrics."""
    logger = get_logger("load_tester")
    logger.info(f"Starting {duration_minutes}-minute load test...")
    
    end_time = time.time() + (duration_minutes * 60)
    operation_count = 0
    
    while time.time() < end_time:
        # Random operation
        operation_type = random.choice([
            "pdf_processing", "content_analysis", "outline_generation",
            "notes_generation", "quality_assessment", "agent_communication"
        ])
        
        with track_operation(operation_type, f"load_test_{operation_count}"):
            # Simulate work with variable duration
            work_duration = random.exponential(1.0)  # Exponential distribution
            time.sleep(min(work_duration, 5.0))  # Cap at 5 seconds
        
        # Occasionally record errors
        if random.random() < 0.1:  # 10% error rate
            error_type = random.choice(["TimeoutError", "ValidationError", "ProcessingError"])
            component = random.choice(["processor", "analyzer", "generator"])
            record_error(error_type, component, "error")
        
        # Occasionally record quality scores
        if random.random() < 0.3:  # 30% chance
            score = random.betavariate(8, 2)  # Beta distribution favoring higher scores
            from monitoring.prometheus_config import record_quality_score
            record_quality_score("test_content", "test_agent", score)
        
        operation_count += 1
        
        # Brief pause between operations
        time.sleep(random.uniform(0.1, 1.0))
    
    logger.info(f"Load test completed. Processed {operation_count} operations.")


def main():
    """Main function for the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Academic Agent Monitoring Test")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--load-test", type=int, metavar="MINUTES", 
                       help="Run load test for specified minutes")
    parser.add_argument("--start-monitoring", action="store_true", 
                       help="Start monitoring integration")
    
    args = parser.parse_args()
    
    if args.start_monitoring:
        from monitoring.integration import start_integrated_monitoring
        monitoring = start_integrated_monitoring()
        print("Monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(60)
                status = monitoring.get_monitoring_status()
                print(f"Monitoring status: {status['integration_active']}")
        except KeyboardInterrupt:
            monitoring.stop_monitoring()
            print("Monitoring stopped.")
    
    elif args.test:
        tester = MonitoringTester()
        success = tester.run_all_tests()
        
        if success:
            print("All tests passed!")
            exit(0)
        else:
            print("Some tests failed!")
            exit(1)
    
    elif args.load_test:
        run_load_test(args.load_test)
    
    else:
        # Default: run basic test
        tester = MonitoringTester()
        tester.test_basic_metrics()
        print("Basic monitoring test completed. Use --help for more options.")


if __name__ == "__main__":
    main()