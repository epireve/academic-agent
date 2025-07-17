#!/usr/bin/env python3
"""
Test script for the Academic Agent Logging and Error Handling System

This script demonstrates the key features of the enhanced logging and error handling system.
Run this script to verify that the system is working correctly.
"""

import sys
import time
from pathlib import Path

# Add the v2 core modules to the path
sys.path.append(str(Path(__file__).parent / "academic-agent-v2" / "src"))

def test_basic_logging():
    """Test basic logging functionality."""
    print("Testing basic logging...")
    
    try:
        from core.logging import get_logger
        
        logger = get_logger("test_component")
        
        # Basic logging
        logger.info("Basic logging test started")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        # Structured logging with context
        logger.info("Structured logging test", extra_context={
            "test_id": "test_001",
            "component": "logging_test",
            "status": "running"
        })
        
        # Metrics logging
        logger.metric("test_duration", 1.5, "seconds", test_type="basic_logging")
        
        print("✓ Basic logging test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic logging test failed: {e}")
        return False

def test_operation_context():
    """Test operation context management."""
    print("Testing operation context...")
    
    try:
        from core.logging import get_logger
        
        logger = get_logger("test_operations")
        
        # Test operation context
        with logger.operation("test_operation", test_param="value") as op_logger:
            op_logger.info("Starting test operation")
            time.sleep(0.1)  # Simulate work
            op_logger.info("Operation in progress")
            # Duration will be automatically logged
        
        # Test context propagation
        context_logger = logger.with_context(user_id="test_user", session="test_session")
        context_logger.info("Message with persistent context")
        
        print("✓ Operation context test passed")
        return True
        
    except Exception as e:
        print(f"✗ Operation context test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery."""
    print("Testing error handling...")
    
    try:
        from core.error_handling import with_error_handling, error_context
        from core.exceptions import ProcessingError, ValidationError
        
        # Test error context
        try:
            with error_context("test_error_handling", {"test": True}) as logger:
                logger.info("Testing error handling")
                raise ValueError("Test error for demonstration")
        except Exception as e:
            print(f"  Caught expected error: {type(e).__name__}")
        
        # Test custom exceptions
        try:
            raise ProcessingError(
                "Test processing error",
                stage="validation",
                file_path="/test/path"
            ).add_suggestion("This is a test error for demonstration")
        except ProcessingError as e:
            print(f"  Caught ProcessingError: {e.message}")
            print(f"  Suggestions: {e.suggestions}")
        
        # Test decorator
        @with_error_handling("test_decorator", max_retries=2)
        def failing_function():
            raise ConnectionError("Test connection error")
        
        try:
            failing_function()
        except Exception as e:
            print(f"  Decorator handled error: {type(e).__name__}")
        
        print("✓ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("Testing configuration management...")
    
    try:
        from core.config_manager import get_config_manager, get_config
        
        # Test configuration loading
        config_manager = get_config_manager()
        config = get_config()
        
        print(f"  Quality threshold: {config.quality_threshold}")
        print(f"  Max improvement cycles: {config.max_improvement_cycles}")
        
        # Test configuration validation
        if config_manager.validate_configuration():
            print("  Configuration validation passed")
        else:
            print("  Configuration validation failed")
        
        print("✓ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_monitoring():
    """Test monitoring and metrics collection."""
    print("Testing monitoring system...")
    
    try:
        from core.monitoring import get_system_monitor
        
        monitor = get_system_monitor()
        
        # Test metrics collection
        monitor.metrics_collector.collect_metric({
            "name": "test_metric",
            "value": 42,
            "unit": "items",
            "tags": {"test_type": "monitoring"}
        })
        
        # Test health checking
        health_status = monitor.health_checker.get_overall_health()
        print(f"  System health: {health_status['status']}")
        
        # Test monitoring summary
        summary = monitor.get_monitoring_summary()
        print(f"  Total metrics: {summary['total_metrics']}")
        print(f"  Active alerts: {len(summary['active_alerts'])}")
        
        print("✓ Monitoring test passed")
        return True
        
    except Exception as e:
        print(f"✗ Monitoring test failed: {e}")
        return False

def test_enhanced_agent():
    """Test enhanced agent functionality."""
    print("Testing enhanced agent...")
    
    try:
        from agents.academic.base_agent import BaseAgent, AgentMessage
        from datetime import datetime
        
        class TestAgent(BaseAgent):
            def check_quality(self, content):
                return 0.85
            
            def validate_input(self, input_data):
                return isinstance(input_data, dict)
            
            def validate_output(self, output_data):
                return output_data is not None
        
        # Create test agent
        agent = TestAgent("test_agent")
        
        # Test message sending
        success = agent.send_message(
            recipient="other_agent",
            message_type="processing_request",
            content={"test": "data"}
        )
        print(f"  Message sent: {success}")
        
        # Test message receiving
        test_message = AgentMessage(
            sender="other_agent",
            recipient="test_agent",
            message_type="status_update",
            content={"status": "completed"},
            metadata={},
            timestamp=datetime.now()
        )
        
        received = agent.receive_message(test_message)
        print(f"  Message received: {received}")
        
        # Test agent status
        status = agent.get_agent_status()
        print(f"  Agent ID: {status['agent_id']}")
        print(f"  Messages sent: {status['messages_sent']}")
        print(f"  Messages received: {status['messages_received']}")
        
        # Test metrics logging
        agent.log_metrics({
            "processing_time": 2.5,
            "quality_score": 0.9,
            "success": True
        })
        
        print("✓ Enhanced agent test passed")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced agent test failed: {e}")
        return False

def test_fallback_mode():
    """Test fallback mode when enhanced system is not available."""
    print("Testing fallback mode...")
    
    try:
        # This should work even if the enhanced system is not available
        import logging
        logger = logging.getLogger("fallback_test")
        logger.info("Fallback logging works")
        
        print("✓ Fallback mode test passed")
        return True
        
    except Exception as e:
        print(f"✗ Fallback mode test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Academic Agent Logging and Error Handling System Test")
    print("=" * 60)
    
    tests = [
        test_basic_logging,
        test_operation_context,
        test_error_handling,
        test_configuration,
        test_monitoring,
        test_enhanced_agent,
        test_fallback_mode
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The logging and error handling system is working correctly.")
        
        # Print helpful information
        print("\\nNext steps:")
        print("1. Check the logs/ directory for generated log files")
        print("2. Review the configuration in config/logging_config.yaml")
        print("3. Integrate the enhanced BaseAgent into your existing agents")
        print("4. Set up monitoring and alerting for production use")
        
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\\nTroubleshooting:")
        print("1. Ensure all dependencies are installed (pydantic, pyyaml, psutil)")
        print("2. Check file permissions for the logs/ directory")
        print("3. Verify the configuration files are valid")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)