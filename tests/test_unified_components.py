#!/usr/bin/env python3
"""
Simple integration tests for unified architecture components.
Tests the basic functionality of migrated agents without external dependencies.
"""

import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all unified components can be imported successfully."""
    try:
        from src.agents.base_agent import BaseAgent
        from src.agents.notes_agent import NotesAgent
        from src.agents.consolidation_agent import ConsolidationAgent  
        from src.agents.ingestion_agent import IngestionAgent
        from src.agents.content_quality_agent import ContentQualityAgent
        from src.core.feature_flags import FeatureFlagManager
        from src.core.memory_manager import MemoryManager
        from src.processors.pdf_processor import UnifiedPDFProcessor
        print("✓ All unified components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_base_agent_functionality():
    """Test basic BaseAgent functionality."""
    try:
        from src.agents.base_agent import BaseAgent
        
        # Test that we can create a base agent instance
        agent = BaseAgent("test_agent")
        assert agent.agent_name == "test_agent"
        assert hasattr(agent, 'logger')
        assert hasattr(agent, 'base_dir')
        
        print("✓ BaseAgent basic functionality works")
        return True
    except Exception as e:
        print(f"✗ BaseAgent test failed: {e}")
        return False

async def test_notes_agent_async():
    """Test NotesAgent async functionality."""
    try:
        from src.agents.notes_agent import NotesAgent
        
        agent = NotesAgent()
        assert agent.agent_name == "notes_agent"
        
        # Test input validation
        result = await agent.validate_input("test")
        assert isinstance(result, bool)
        
        print("✓ NotesAgent async functionality works")
        return True
    except Exception as e:
        print(f"✗ NotesAgent test failed: {e}")
        return False

async def test_consolidation_agent_async():
    """Test ConsolidationAgent async functionality."""
    try:
        from src.agents.consolidation_agent import ConsolidationAgent
        
        agent = ConsolidationAgent()
        assert agent.agent_name == "consolidation_agent"
        
        # Test input validation with empty dict
        result = await agent.validate_input({})
        assert result is False
        
        print("✓ ConsolidationAgent async functionality works")
        return True
    except Exception as e:
        print(f"✗ ConsolidationAgent test failed: {e}")
        return False

async def test_ingestion_agent_async():
    """Test IngestionAgent async functionality."""
    try:
        from src.agents.ingestion_agent import IngestionAgent
        
        agent = IngestionAgent()
        assert agent.agent_name == "ingestion_agent"
        
        # Test input validation with non-existent file
        result = await agent.validate_input("nonexistent.pdf")
        assert result is False
        
        print("✓ IngestionAgent async functionality works")
        return True
    except Exception as e:
        print(f"✗ IngestionAgent test failed: {e}")
        return False

async def test_content_quality_agent_async():
    """Test ContentQualityAgent async functionality."""
    try:
        from src.agents.content_quality_agent import ContentQualityAgent
        
        agent = ContentQualityAgent()
        assert agent.agent_name == "content_quality_agent"
        
        # Test input validation with empty content
        result = await agent.validate_input("")
        assert result is False
        
        print("✓ ContentQualityAgent async functionality works")
        return True
    except Exception as e:
        print(f"✗ ContentQualityAgent test failed: {e}")
        return False

def test_feature_flags():
    """Test feature flags system."""
    try:
        from src.core.feature_flags import FeatureFlagManager, FlagState
        
        manager = FeatureFlagManager()
        
        # Test default flags exist
        flags = manager.list_flags()
        assert "USE_UNIFIED_AGENTS" in flags
        assert "USE_LEGACY_ADAPTER" in flags
        
        # Test flag evaluation
        result = manager.is_enabled("USE_LEGACY_ADAPTER")
        assert isinstance(result, bool)
        
        print("✓ Feature flags system works")
        return True
    except Exception as e:
        print(f"✗ Feature flags test failed: {e}")
        return False

def test_memory_manager():
    """Test memory management system."""
    try:
        from src.core.memory_manager import MemoryManager
        
        manager = MemoryManager(enable_monitoring=False)  # Disable monitoring for test
        
        # Test memory stats
        stats = manager.get_memory_stats()
        assert hasattr(stats, 'total_memory_mb')
        assert hasattr(stats, 'process_memory_mb')
        
        # Test cache
        assert hasattr(manager, 'cache')
        
        manager.cleanup()
        print("✓ Memory manager works")
        return True
    except Exception as e:
        print(f"✗ Memory manager test failed: {e}")
        return False

def test_pdf_processor():
    """Test unified PDF processor."""
    try:
        from src.processors.pdf_processor import UnifiedPDFProcessor
        
        processor = UnifiedPDFProcessor()
        
        # Test available backends
        backends = processor.get_available_backends()
        assert "fallback" in backends  # Fallback should always be available
        
        processor.shutdown()
        print("✓ PDF processor works")
        return True
    except Exception as e:
        print(f"✗ PDF processor test failed: {e}")
        return False

async def test_unified_architecture_integration():
    """Test integration between unified components."""
    try:
        from src.core.feature_flags import get_flag_manager
        from src.core.memory_manager import get_memory_manager
        
        # Test that global instances work
        flag_manager = get_flag_manager()
        memory_manager = get_memory_manager()
        
        assert flag_manager is not None
        assert memory_manager is not None
        
        # Test feature flag for unified agents
        unified_enabled = flag_manager.is_enabled("USE_UNIFIED_AGENTS")
        assert isinstance(unified_enabled, bool)
        
        print("✓ Unified architecture integration works")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all unified architecture tests."""
    print("Running unified architecture tests...\n")
    
    tests = [
        ("Import Tests", test_imports()),
        ("BaseAgent Tests", test_base_agent_functionality()),
        ("NotesAgent Tests", await test_notes_agent_async()),
        ("ConsolidationAgent Tests", await test_consolidation_agent_async()),
        ("IngestionAgent Tests", await test_ingestion_agent_async()),
        ("ContentQualityAgent Tests", await test_content_quality_agent_async()),
        ("Feature Flags Tests", test_feature_flags()),
        ("Memory Manager Tests", test_memory_manager()),
        ("PDF Processor Tests", test_pdf_processor()),
        ("Integration Tests", await test_unified_architecture_integration()),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, result in tests:
        if result:
            passed += 1
        else:
            failed += 1
        print()
    
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All unified architecture tests passed!")
        return True
    else:
        print(f"\n❌ {failed} tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)