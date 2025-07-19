#!/usr/bin/env python3
"""
Basic validation tests for unified architecture components.
Tests only the core components that are properly implemented.
"""

import asyncio
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_core_imports():
    """Test core unified components can be imported."""
    print("Testing core unified architecture imports...")
    
    try:
        from src.core.feature_flags import FeatureFlagManager, FlagState, get_flag_manager
        print("✓ Feature flags imported")
        
        from src.processors.pdf_processor import UnifiedPDFProcessor, create_pdf_processor
        print("✓ PDF processor imported")
        
        from src.config.unified_config import UnifiedConfig
        print("✓ Unified config imported")
        
        from src.agents.base_agent import BaseAgent, AgentMessage, TaskResult
        print("✓ Base agent imported")
        
        print("✓ All core imports successful")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_feature_flags_system():
    """Test feature flags system functionality."""
    print("\nTesting feature flags system...")
    
    try:
        from src.core.feature_flags import FeatureFlagManager, FlagState
        
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
        
        for flag in expected_flags:
            if flag not in flags:
                print(f"✗ Missing expected flag: {flag}")
                return False
        
        # Test flag evaluation
        legacy_enabled = manager.is_enabled("USE_LEGACY_ADAPTER")
        unified_enabled = manager.is_enabled("USE_UNIFIED_AGENTS")
        
        print(f"  - Legacy adapter enabled: {legacy_enabled}")
        print(f"  - Unified agents enabled: {unified_enabled}")
        
        # Test setting flag state
        old_state = manager.get_flag("USE_UNIFIED_AGENTS").state
        manager.set_flag_state("USE_UNIFIED_AGENTS", FlagState.ENABLED, "test")
        new_enabled = manager.is_enabled("USE_UNIFIED_AGENTS")
        
        if not new_enabled:
            print("✗ Flag state change failed")
            return False
            
        print("✓ Feature flags system working")
        return True
        
    except Exception as e:
        print(f"✗ Feature flags test failed: {e}")
        return False


def test_pdf_processor():
    """Test unified PDF processor."""
    print("\nTesting PDF processor...")
    
    try:
        from src.processors.pdf_processor import UnifiedPDFProcessor, create_pdf_processor
        
        # Test factory function
        processor = create_pdf_processor()
        
        # Test available backends
        backends = processor.get_available_backends()
        print(f"  - Available backends: {backends}")
        
        if "fallback" not in backends:
            print("✗ Fallback backend not available")
            return False
        
        # Test preferred backend selection
        preferred = processor.preferred_backend
        print(f"  - Preferred backend: {preferred}")
        
        processor.shutdown()
        print("✓ PDF processor working")
        return True
        
    except Exception as e:
        print(f"✗ PDF processor test failed: {e}")
        return False


async def test_pdf_processor_async():
    """Test PDF processor async functionality."""
    print("\nTesting PDF processor async operations...")
    
    try:
        from src.processors.pdf_processor import UnifiedPDFProcessor
        
        processor = UnifiedPDFProcessor()
        
        # Test with non-existent file (should fail gracefully)
        result = await processor.process_pdf("nonexistent.pdf")
        
        if result.success:
            print("✗ Should have failed for non-existent file")
            return False
            
        if result.error_message is None:
            print("✗ Should have error message")
            return False
            
        print(f"  - Correctly failed with: {result.error_message}")
        
        processor.shutdown()
        print("✓ PDF processor async operations working")
        return True
        
    except Exception as e:
        print(f"✗ PDF processor async test failed: {e}")
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
            
        # Test legacy config migration
        legacy_config = {
            "agents": {
                "legacy_agent": {
                    "enabled": True,
                    "setting": "value"
                }
            }
        }
        
        unified_legacy = UnifiedConfig.from_legacy_config(legacy_config)
        
        if not unified_legacy.legacy_mode:
            print("✗ Legacy mode should be enabled")
            return False
            
        # Test format detection
        unified_format = ConfigMigrator.detect_format({"version": "3.0.0"})
        v2_format = ConfigMigrator.detect_format({"version": "2.0", "agents": []})
        legacy_format = ConfigMigrator.detect_format({"agents": {"agent1": {}}})
        
        if unified_format != "unified":
            print(f"✗ Expected 'unified', got {unified_format}")
            return False
            
        if v2_format != "v2":
            print(f"✗ Expected 'v2', got {v2_format}")
            return False
            
        if legacy_format != "legacy":
            print(f"✗ Expected 'legacy', got {legacy_format}")
            return False
            
        print("✓ Unified configuration system working")
        return True
        
    except Exception as e:
        print(f"✗ Unified config test failed: {e}")
        return False


def test_base_agent_structure():
    """Test base agent class structure without instantiation."""
    print("\nTesting base agent structure...")
    
    try:
        from src.agents.base_agent import BaseAgent, AgentMessage, TaskResult
        
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
        if "message_id" not in msg_dict:
            print("✗ Message dict missing message_id")
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
            
        print("✓ Base agent structure working")
        return True
        
    except Exception as e:
        print(f"✗ Base agent structure test failed: {e}")
        return False


async def run_all_tests():
    """Run all basic unified architecture tests."""
    print("Running basic unified architecture validation tests...\n")
    
    tests = [
        ("Core Imports", test_core_imports()),
        ("Feature Flags System", test_feature_flags_system()),
        ("PDF Processor", test_pdf_processor()),
        ("PDF Processor Async", await test_pdf_processor_async()),
        ("Unified Config", test_unified_config()),
        ("Base Agent Structure", test_base_agent_structure()),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All basic unified architecture tests passed!")
        print("\nThe core unified architecture components are working correctly.")
        print("Next steps:")
        print("- Complete agent implementations with initialize/cleanup methods")
        print("- Add remaining missing dependencies")
        print("- Implement comprehensive integration tests")
        return True
    else:
        print(f"\n❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)