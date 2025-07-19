#!/usr/bin/env python3
"""
Minimal validation tests for core unified architecture components.
Tests only the essential components that are confirmed working.
"""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_feature_flags_system():
    """Test feature flags system functionality."""
    print("Testing feature flags system...")
    
    try:
        from src.core.feature_flags import FeatureFlagManager, FlagState, get_flag_manager
        
        # Create manager
        manager = FeatureFlagManager()
        
        # Test default flags exist
        flags = manager.list_flags()
        expected_flags = [
            "USE_UNIFIED_AGENTS",
            "USE_LEGACY_ADAPTER", 
            "ENABLE_ASYNC_PROCESSING",
            "USE_UNIFIED_CONFIG",
            "ENABLE_PERFORMANCE_MONITORING"
        ]
        
        missing_flags = []
        for flag in expected_flags:
            if flag not in flags:
                missing_flags.append(flag)
        
        if missing_flags:
            print(f"✗ Missing expected flags: {missing_flags}")
            return False
        
        # Test flag evaluation
        legacy_enabled = manager.is_enabled("USE_LEGACY_ADAPTER")
        unified_enabled = manager.is_enabled("USE_UNIFIED_AGENTS")
        
        print(f"  - Legacy adapter enabled: {legacy_enabled}")
        print(f"  - Unified agents enabled: {unified_enabled}")
        
        # Test global instance
        global_manager = get_flag_manager()
        if global_manager is None:
            print("✗ Global flag manager is None")
            return False
            
        print("✓ Feature flags system working")
        return True
        
    except Exception as e:
        print(f"✗ Feature flags test failed: {e}")
        return False


def test_base_agent_components():
    """Test base agent message and result structures."""
    print("\nTesting base agent components...")
    
    try:
        from src.agents.base_agent import AgentMessage, TaskResult
        
        # Test message creation
        message = AgentMessage(
            sender="test_sender",
            recipient="test_recipient", 
            message_type="test",
            content={"data": "test"}
        )
        
        if message.sender != "test_sender":
            print("✗ Message sender not set correctly")
            return False
            
        # Test message serialization
        msg_dict = message.to_dict()
        required_fields = ["message_id", "sender", "recipient", "message_type", "content"]
        for field in required_fields:
            if field not in msg_dict:
                print(f"✗ Message dict missing {field}")
                return False
            
        # Test message deserialization
        restored = AgentMessage.from_dict(msg_dict)
        if restored.sender != message.sender:
            print("✗ Message deserialization failed")
            return False
            
        # Test task result
        result = TaskResult(
            task_id="test_task",
            task_type="test",
            success=True,
            result={"output": "test"}
        )
        
        result_dict = result.to_dict()
        if result_dict["success"] != True:
            print("✗ Task result serialization failed")
            return False
            
        print("✓ Base agent components working")
        return True
        
    except Exception as e:
        print(f"✗ Base agent components test failed: {e}")
        return False


def test_unified_config():
    """Test unified configuration system."""
    print("\nTesting unified configuration...")
    
    try:
        from src.config.unified_config import UnifiedConfig, ConfigMigrator
        
        # Test v2 config migration
        v2_config = {
            "version": "2.0",
            "agents": [
                {
                    "agent_id": "test_agent",
                    "enabled": True,
                    "config": {"key": "value"}
                }
            ]
        }
        
        unified = UnifiedConfig.from_v2_config(v2_config)
        
        if unified.version != "3.0.0":
            print(f"✗ Expected version 3.0.0, got {unified.version}")
            return False
            
        if len(unified.agents) != 1:
            print(f"✗ Expected 1 agent, got {len(unified.agents)}")
            return False
            
        # Test format detection
        formats = [
            ({"version": "3.0.0"}, "unified"),
            ({"version": "2.0", "agents": []}, "v2"),
            ({"agents": {"agent1": {}}}, "legacy")
        ]
        
        for config, expected_format in formats:
            detected = ConfigMigrator.detect_format(config)
            if detected != expected_format:
                print(f"✗ Expected '{expected_format}', got '{detected}' for {config}")
                return False
            
        print("✓ Unified configuration system working")
        return True
        
    except Exception as e:
        print(f"✗ Unified config test failed: {e}")
        return False


def test_core_exceptions():
    """Test core exception classes."""
    print("\nTesting core exceptions...")
    
    try:
        from src.core.exceptions import (
            AcademicAgentError,
            CommunicationError,
            ValidationError,
            ProcessingError
        )
        
        # Test basic exception creation
        base_error = AcademicAgentError("Base error")
        if str(base_error) != "Base error":
            print("✗ Base exception string representation failed")
            return False
            
        # Test inheritance
        if not issubclass(CommunicationError, AcademicAgentError):
            print("✗ CommunicationError should inherit from AcademicAgentError")
            return False
            
        if not issubclass(ValidationError, AcademicAgentError):
            print("✗ ValidationError should inherit from AcademicAgentError")
            return False
            
        if not issubclass(ProcessingError, AcademicAgentError):
            print("✗ ProcessingError should inherit from AcademicAgentError")
            return False
            
        print("✓ Core exceptions working")
        return True
        
    except Exception as e:
        print(f"✗ Core exceptions test failed: {e}")
        return False


def test_core_logging():
    """Test core logging functionality."""
    print("\nTesting core logging...")
    
    try:
        from src.core.logging import get_logger
        
        # Test logger creation
        logger = get_logger("test_logger")
        if logger is None:
            print("✗ Logger creation failed")
            return False
            
        # The logger might not have the full name depending on implementation
        if "test_logger" not in logger.logger.name:
            print(f"✗ Expected logger name to contain 'test_logger', got {logger.logger.name}")
            return False
            
        # Test that we can call logging methods without error
        logger.info("Test info message")
        logger.debug("Test debug message")
        logger.warning("Test warning message")
        
        print("✓ Core logging working")
        return True
        
    except Exception as e:
        print(f"✗ Core logging test failed: {e}")
        return False


def run_all_tests():
    """Run all minimal unified architecture tests."""
    print("Running minimal unified architecture validation tests...\n")
    
    tests = [
        ("Feature Flags System", test_feature_flags_system),
        ("Base Agent Components", test_base_agent_components),
        ("Unified Config", test_unified_config),
        ("Core Exceptions", test_core_exceptions),
        ("Core Logging", test_core_logging),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All minimal unified architecture tests passed!")
        print("\nCore unified architecture components validated:")
        print("- ✓ Feature flags system")
        print("- ✓ Base agent messaging") 
        print("- ✓ Unified configuration")
        print("- ✓ Core exceptions")
        print("- ✓ Core logging")
        print("\nNext phase: Complete agent implementations and integration")
        return True
    else:
        print(f"\n❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)