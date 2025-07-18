"""
Tests for the unified architecture components.
"""

import asyncio
from pathlib import Path
import sys

# Make pytest optional
try:
    import pytest
except ImportError:
    pytest = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base_agent import BaseAgent, AgentCapability, TaskResult
from src.agents.legacy_adapter import LegacyAgentAdapter, create_legacy_adapter
from src.processors.pdf_processor import UnifiedPDFProcessor, create_pdf_processor
from src.config.unified_config import UnifiedConfig, ConfigMigrator


class TestCapability(AgentCapability):
    """Test capability for unit tests."""
    
    async def execute(self, task_data):
        await asyncio.sleep(0.1)  # Simulate work
        return TaskResult(
            task_id=task_data.get("task_id", "test"),
            task_type=self.name,
            success=True,
            result={"processed": True, "input": task_data}
        )


class TestAgent(BaseAgent):
    """Test agent implementation."""
    
    async def initialize(self):
        self.register_capability(TestCapability("test_capability"))
    
    async def cleanup(self):
        pass


class LegacyTestAgent:
    """Legacy agent for adapter testing."""
    
    def __init__(self):
        self.agent_id = "legacy_test"
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
    
    def process_data(self, data):
        return {"legacy_processed": True, "data": data}
    
    async def async_process(self, data):
        await asyncio.sleep(0.1)
        return {"async_processed": True, "data": data}


async def test_base_agent_lifecycle():
    """Test base agent lifecycle."""
    agent = TestAgent("test_agent")
    
    # Start agent
    await agent.start()
    assert agent.status == "running"
    assert "test_capability" in agent.capabilities
    
    # Get status
    status = agent.get_status()
    assert status["agent_id"] == "test_agent"
    assert status["status"] == "running"
    
    # Stop agent
    await agent.stop()
    assert agent.status == "stopped"


async def test_capability_execution():
    """Test capability execution."""
    agent = TestAgent("test_agent")
    await agent.start()
    
    # Execute capability
    result = await agent.execute_capability(
        "test_capability",
        {"task_id": "test_task", "data": "test"}
    )
    
    assert result.success
    assert result.task_type == "test_capability"
    assert result.result["processed"] is True
    
    await agent.stop()


async def test_message_handling():
    """Test message passing."""
    agent = TestAgent("test_agent")
    await agent.start()
    
    # Register handler
    messages_received = []
    
    def handler(message):
        messages_received.append(message)
    
    agent.register_message_handler("test_message", handler)
    
    # Send message
    message = await agent.send_message(
        "other_agent",
        "test_message",
        {"content": "test"}
    )
    
    # Simulate receiving
    await agent.receive_message(message)
    
    # Wait for processing
    await asyncio.sleep(0.2)
    
    assert len(messages_received) == 1
    assert messages_received[0].content["content"] == "test"
    
    await agent.stop()


async def test_legacy_adapter():
    """Test legacy agent adapter."""
    legacy_agent = LegacyTestAgent()
    adapter = create_legacy_adapter("legacy_adapter", legacy_agent)
    
    # Start adapter
    await adapter.start()
    assert legacy_agent.initialized
    
    # Expose legacy method
    adapter.expose_legacy_method("process_data", "data_processing")
    
    # Execute via capability
    result = await adapter.execute_capability(
        "data_processing",
        {"data": "test_data"}
    )
    
    assert result.success
    assert result.result["legacy_processed"] is True
    
    await adapter.stop()


async def test_pdf_processor():
    """Test unified PDF processor."""
    processor = create_pdf_processor()
    
    # Check available backends
    backends = processor.get_available_backends()
    assert "fallback" in backends  # Fallback is always available
    
    # Test with non-existent file (should fail gracefully)
    result = await processor.process_pdf("nonexistent.pdf")
    assert not result.success
    assert result.error_message is not None
    
    processor.shutdown()


def test_config_migration():
    """Test configuration migration."""
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
    assert unified.version == "3.0.0"
    assert len(unified.agents) == 1
    assert unified.agents[0].agent_id == "test_agent"
    
    # Test legacy config migration
    legacy_config = {
        "agents": {
            "legacy_agent": {
                "enabled": True,
                "setting": "value"
            }
        }
    }
    
    unified = UnifiedConfig.from_legacy_config(legacy_config)
    assert unified.legacy_mode is True
    assert len(unified.agents) == 1
    assert unified.agents[0].agent_type == "legacy"


def test_config_format_detection():
    """Test configuration format detection."""
    # Unified format
    assert ConfigMigrator.detect_format({"version": "3.0.0"}) == "unified"
    
    # V2 format
    assert ConfigMigrator.detect_format({
        "version": "2.0",
        "agents": []
    }) == "v2"
    
    # Legacy format
    assert ConfigMigrator.detect_format({
        "agents": {"agent1": {}}
    }) == "legacy"


async def test_health_check():
    """Test agent health check."""
    agent = TestAgent("test_agent")
    await agent.start()
    
    health = await agent.health_check()
    assert health["healthy"] is True
    assert health["status"] == "running"
    assert health["checks"]["capabilities"]["healthy"] is True
    
    await agent.stop()


if __name__ == "__main__":
    # Run basic tests
    print("Running unified architecture tests...")
    
    # Test base agent
    asyncio.run(test_base_agent_lifecycle())
    print("✓ Base agent lifecycle test passed")
    
    # Test capability
    asyncio.run(test_capability_execution())
    print("✓ Capability execution test passed")
    
    # Test message handling
    asyncio.run(test_message_handling())
    print("✓ Message handling test passed")
    
    # Test legacy adapter
    asyncio.run(test_legacy_adapter())
    print("✓ Legacy adapter test passed")
    
    # Test PDF processor
    asyncio.run(test_pdf_processor())
    print("✓ PDF processor test passed")
    
    # Test config migration
    test_config_migration()
    print("✓ Config migration test passed")
    
    # Test config format detection
    test_config_format_detection()
    print("✓ Config format detection test passed")
    
    # Test health check
    asyncio.run(test_health_check())
    print("✓ Health check test passed")
    
    print("\nAll tests passed! ✓")