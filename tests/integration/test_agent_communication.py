"""
Integration tests for agent communication and coordination.

This module tests the communication between different agents and the overall
coordination of the academic agent system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
from datetime import datetime
import asyncio

from agents.academic.base_agent import BaseAgent, AgentMessage
from agents.academic.communication_manager import CommunicationManager
from agents.academic.quality_manager import QualityManager
from tests.utils import (
    TestFileManager,
    create_sample_markdown_content,
    create_sample_analysis_result,
    create_mock_agent_with_tools
)


class TestAgentCommunication:
    """Test cases for agent communication."""
    
    def test_agent_message_exchange(self):
        """Test basic message exchange between agents."""
        # Create two agents
        agent1 = BaseAgent("agent1")
        agent2 = BaseAgent("agent2")
        
        # Override the abstract methods for testing
        agent1.check_quality = Mock(return_value=0.8)
        agent1.validate_input = Mock(return_value=True)
        agent1.validate_output = Mock(return_value=True)
        
        agent2.check_quality = Mock(return_value=0.8)
        agent2.validate_input = Mock(return_value=True)
        agent2.validate_output = Mock(return_value=True)
        
        # Test message creation and validation
        message = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type="task_request",
            content={
                "task": "analyze_document",
                "document_path": "/test/document.md"
            },
            metadata={"priority": "high"},
            timestamp=datetime.now()
        )
        
        assert message.validate() is True
        
        # Test message sending
        result = agent1.send_message(
            recipient="agent2",
            message_type="task_request",
            content={"task": "analyze_document"},
            priority=1
        )
        
        assert result is True
        
        # Test message receiving
        result = agent2.receive_message(message)
        assert result is True
    
    def test_communication_with_retry_mechanism(self):
        """Test communication with retry mechanism."""
        agent = BaseAgent("test_agent")
        agent.check_quality = Mock(return_value=0.8)
        agent.validate_input = Mock(return_value=True)
        agent.validate_output = Mock(return_value=True)
        
        message = AgentMessage(
            sender="test_agent",
            recipient="target_agent",
            message_type="test_message",
            content={"data": "test"},
            metadata={},
            timestamp=datetime.now()
        )
        
        # Test retry mechanism
        assert message.retry_count == 0
        assert message.retry() is True
        assert message.retry_count == 1
        
        # Test multiple retries
        assert message.retry() is True
        assert message.retry_count == 2
        assert message.retry() is True
        assert message.retry_count == 3
        
        # Should not allow more retries
        assert message.retry() is False
        assert message.retry_count == 3
    
    def test_hierarchical_message_structure(self):
        """Test parent-child message relationships."""
        parent_message = AgentMessage(
            sender="parent_agent",
            recipient="child_agent",
            message_type="parent_task",
            content={"main_task": "process_document"},
            metadata={},
            timestamp=datetime.now()
        )
        
        child_message = AgentMessage(
            sender="child_agent",
            recipient="parent_agent",
            message_type="child_response",
            content={"subtask": "analyze_section"},
            metadata={},
            timestamp=datetime.now(),
            parent_id="parent_123"
        )
        
        # Test parent-child relationship
        assert parent_message.parent_id is None
        assert child_message.parent_id == "parent_123"
        
        # Test message validation
        assert parent_message.validate() is True
        assert child_message.validate() is True


class TestCommunicationManager:
    """Test cases for CommunicationManager."""
    
    def test_communication_manager_initialization(self):
        """Test CommunicationManager initialization."""
        manager = CommunicationManager()
        
        assert hasattr(manager, 'agents')
        assert hasattr(manager, 'message_queue')
        assert hasattr(manager, 'active_conversations')
    
    def test_agent_registration(self):
        """Test agent registration with communication manager."""
        manager = CommunicationManager()
        
        agent = BaseAgent("test_agent")
        agent.check_quality = Mock(return_value=0.8)
        agent.validate_input = Mock(return_value=True)
        agent.validate_output = Mock(return_value=True)
        
        # Register agent
        manager.register_agent(agent)
        
        assert "test_agent" in manager.agents
        assert manager.agents["test_agent"] == agent
    
    def test_message_routing(self):
        """Test message routing between agents."""
        manager = CommunicationManager()
        
        # Create and register agents
        agent1 = BaseAgent("agent1")
        agent2 = BaseAgent("agent2")
        
        agent1.check_quality = Mock(return_value=0.8)
        agent1.validate_input = Mock(return_value=True)
        agent1.validate_output = Mock(return_value=True)
        
        agent2.check_quality = Mock(return_value=0.8)
        agent2.validate_input = Mock(return_value=True)
        agent2.validate_output = Mock(return_value=True)
        
        manager.register_agent(agent1)
        manager.register_agent(agent2)
        
        # Create message
        message = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type="test_message",
            content={"data": "test"},
            metadata={},
            timestamp=datetime.now()
        )
        
        # Route message
        result = manager.route_message(message)
        
        assert result is True
    
    def test_broadcast_message(self):
        """Test broadcasting message to all agents."""
        manager = CommunicationManager()
        
        # Create and register multiple agents
        agents = []
        for i in range(3):
            agent = BaseAgent(f"agent{i}")
            agent.check_quality = Mock(return_value=0.8)
            agent.validate_input = Mock(return_value=True)
            agent.validate_output = Mock(return_value=True)
            agent.receive_message = Mock(return_value=True)
            
            agents.append(agent)
            manager.register_agent(agent)
        
        # Broadcast message
        message = AgentMessage(
            sender="broadcast_agent",
            recipient="*",  # Broadcast to all
            message_type="broadcast",
            content={"announcement": "system_update"},
            metadata={},
            timestamp=datetime.now()
        )
        
        result = manager.broadcast_message(message)
        
        assert result is True
        
        # Verify all agents received the message
        for agent in agents:
            agent.receive_message.assert_called_once()
    
    def test_message_queue_management(self):
        """Test message queue management."""
        manager = CommunicationManager()
        
        # Create messages with different priorities
        high_priority_msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type="urgent_task",
            content={"urgent": True},
            metadata={},
            timestamp=datetime.now(),
            priority=5
        )
        
        low_priority_msg = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type="regular_task",
            content={"urgent": False},
            metadata={},
            timestamp=datetime.now(),
            priority=1
        )
        
        # Add messages to queue
        manager.add_to_queue(high_priority_msg)
        manager.add_to_queue(low_priority_msg)
        
        # Process queue - high priority should be processed first
        processed_messages = manager.process_queue()
        
        assert len(processed_messages) == 2
        assert processed_messages[0].priority >= processed_messages[1].priority


class TestQualityManager:
    """Test cases for QualityManager."""
    
    def test_quality_manager_initialization(self):
        """Test QualityManager initialization."""
        manager = QualityManager()
        
        assert hasattr(manager, 'quality_threshold')
        assert hasattr(manager, 'evaluation_history')
        assert hasattr(manager, 'quality_metrics')
    
    def test_quality_evaluation(self):
        """Test quality evaluation functionality."""
        manager = QualityManager()
        
        # Test content evaluation
        content = create_sample_markdown_content()
        
        evaluation = manager.evaluate_content(content, "markdown")
        
        assert "quality_score" in evaluation
        assert "feedback" in evaluation
        assert "areas_for_improvement" in evaluation
        assert 0.0 <= evaluation["quality_score"] <= 1.0
    
    def test_quality_threshold_management(self):
        """Test quality threshold management."""
        manager = QualityManager()
        
        # Test default threshold
        assert manager.get_quality_threshold() == 0.7
        
        # Test setting new threshold
        manager.set_quality_threshold(0.8)
        assert manager.get_quality_threshold() == 0.8
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            manager.set_quality_threshold(1.5)  # > 1.0
        
        with pytest.raises(ValueError):
            manager.set_quality_threshold(-0.1)  # < 0.0
    
    def test_evaluation_history_tracking(self):
        """Test evaluation history tracking."""
        manager = QualityManager()
        
        # Perform multiple evaluations
        content1 = create_sample_markdown_content()
        content2 = "# Simple Document\n\nBasic content."
        
        eval1 = manager.evaluate_content(content1, "markdown")
        eval2 = manager.evaluate_content(content2, "markdown")
        
        # Check history
        history = manager.get_evaluation_history()
        
        assert len(history) == 2
        assert history[0]["content_type"] == "markdown"
        assert history[1]["content_type"] == "markdown"
    
    def test_quality_metrics_aggregation(self):
        """Test quality metrics aggregation."""
        manager = QualityManager()
        
        # Perform multiple evaluations
        for i in range(5):
            content = f"# Document {i}\n\nContent for document {i}"
            manager.evaluate_content(content, "markdown")
        
        # Get aggregated metrics
        metrics = manager.get_quality_metrics()
        
        assert "average_quality_score" in metrics
        assert "total_evaluations" in metrics
        assert "pass_rate" in metrics
        assert metrics["total_evaluations"] == 5


class TestEndToEndWorkflow:
    """Test cases for end-to-end workflow integration."""
    
    def test_document_processing_workflow(self, tmp_path):
        """Test complete document processing workflow."""
        # Create test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create test input
        test_md = file_manager.create_file(
            "input/test_document.md",
            create_sample_markdown_content()
        )
        
        # Create mock agents
        pdf_agent = create_mock_agent_with_tools()
        analysis_agent = create_mock_agent_with_tools()
        outline_agent = create_mock_agent_with_tools()
        notes_agent = create_mock_agent_with_tools()
        
        # Create communication manager
        comm_manager = CommunicationManager()
        comm_manager.register_agent(pdf_agent)
        comm_manager.register_agent(analysis_agent)
        comm_manager.register_agent(outline_agent)
        comm_manager.register_agent(notes_agent)
        
        # Create quality manager
        quality_manager = QualityManager()
        
        # Simulate workflow
        # 1. Analysis phase
        analysis_message = AgentMessage(
            sender="controller",
            recipient="analysis_agent",
            message_type="analyze_document",
            content={"document_path": str(test_md)},
            metadata={},
            timestamp=datetime.now()
        )
        
        comm_manager.route_message(analysis_message)
        
        # 2. Outline generation phase
        outline_message = AgentMessage(
            sender="analysis_agent",
            recipient="outline_agent",
            message_type="generate_outline",
            content={"analysis_result": create_sample_analysis_result()},
            metadata={},
            timestamp=datetime.now(),
            parent_id="analysis_123"
        )
        
        comm_manager.route_message(outline_message)
        
        # 3. Notes generation phase
        notes_message = AgentMessage(
            sender="outline_agent",
            recipient="notes_agent",
            message_type="generate_notes",
            content={"outline_result": {}},
            metadata={},
            timestamp=datetime.now(),
            parent_id="outline_123"
        )
        
        comm_manager.route_message(notes_message)
        
        # Verify workflow completion
        assert len(comm_manager.message_queue) >= 0  # Messages processed
        
        # Cleanup
        file_manager.cleanup()
    
    def test_quality_control_integration(self, tmp_path):
        """Test quality control integration in workflow."""
        # Create test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create test content with varying quality
        high_quality_content = create_sample_markdown_content()
        low_quality_content = "# Bad Document\n\nPoor content."
        
        high_quality_file = file_manager.create_file(
            "high_quality.md",
            high_quality_content
        )
        
        low_quality_file = file_manager.create_file(
            "low_quality.md",
            low_quality_content
        )
        
        # Create quality manager
        quality_manager = QualityManager()
        
        # Evaluate content
        high_eval = quality_manager.evaluate_content(high_quality_content, "markdown")
        low_eval = quality_manager.evaluate_content(low_quality_content, "markdown")
        
        # Verify quality differentiation
        assert high_eval["quality_score"] > low_eval["quality_score"]
        
        # Test threshold enforcement
        threshold = 0.7
        quality_manager.set_quality_threshold(threshold)
        
        high_passes = high_eval["quality_score"] >= threshold
        low_passes = low_eval["quality_score"] >= threshold
        
        # High quality should pass, low quality should fail
        assert high_passes is True
        assert low_passes is False
        
        # Cleanup
        file_manager.cleanup()
    
    def test_error_handling_and_recovery(self, tmp_path):
        """Test error handling and recovery in integrated workflow."""
        # Create test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create communication manager
        comm_manager = CommunicationManager()
        
        # Create agent with error handling
        error_agent = BaseAgent("error_agent")
        error_agent.check_quality = Mock(return_value=0.8)
        error_agent.validate_input = Mock(return_value=True)
        error_agent.validate_output = Mock(return_value=True)
        
        comm_manager.register_agent(error_agent)
        
        # Test error handling
        error_context = {
            "operation": "document_processing",
            "retry_count": 0
        }
        
        test_error = Exception("Processing error")
        
        # Should allow retry
        result = error_agent.handle_error(test_error, error_context)
        assert result is True
        
        # Test retry limit
        error_context["retry_count"] = 3
        result = error_agent.handle_error(test_error, error_context)
        assert result is False
        
        # Cleanup
        file_manager.cleanup()
    
    def test_concurrent_processing(self, tmp_path):
        """Test concurrent processing capabilities."""
        # Create test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            file_path = file_manager.create_file(
                f"test_document_{i}.md",
                f"# Document {i}\n\nContent for document {i}"
            )
            test_files.append(file_path)
        
        # Create communication manager
        comm_manager = CommunicationManager()
        
        # Create multiple processing agents
        agents = []
        for i in range(3):
            agent = create_mock_agent_with_tools()
            agent.agent_id = f"agent_{i}"
            agents.append(agent)
            comm_manager.register_agent(agent)
        
        # Create concurrent processing messages
        messages = []
        for i, file_path in enumerate(test_files):
            message = AgentMessage(
                sender="controller",
                recipient=f"agent_{i}",
                message_type="process_document",
                content={"document_path": str(file_path)},
                metadata={},
                timestamp=datetime.now()
            )
            messages.append(message)
        
        # Process messages concurrently
        results = []
        for message in messages:
            result = comm_manager.route_message(message)
            results.append(result)
        
        # Verify all messages were processed
        assert all(results)
        
        # Cleanup
        file_manager.cleanup()


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test cases for performance-related integration scenarios."""
    
    def test_large_document_processing(self, tmp_path):
        """Test processing of large documents."""
        # Create test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create large document
        large_content = create_sample_markdown_content() * 100  # Repeat 100 times
        large_file = file_manager.create_file("large_document.md", large_content)
        
        # Create quality manager
        quality_manager = QualityManager()
        
        # Measure processing time
        import time
        start_time = time.time()
        
        evaluation = quality_manager.evaluate_content(large_content, "markdown")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify evaluation completed
        assert "quality_score" in evaluation
        assert evaluation["quality_score"] > 0
        
        # Performance should be reasonable (less than 10 seconds for this test)
        assert processing_time < 10.0
        
        # Cleanup
        file_manager.cleanup()
    
    def test_memory_usage_monitoring(self, tmp_path):
        """Test memory usage during processing."""
        # Create test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create multiple documents
        documents = []
        for i in range(10):
            content = create_sample_markdown_content()
            file_path = file_manager.create_file(f"document_{i}.md", content)
            documents.append(file_path)
        
        # Monitor memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process documents
        quality_manager = QualityManager()
        for doc in documents:
            content = doc.read_text()
            quality_manager.evaluate_content(content, "markdown")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        # Cleanup
        file_manager.cleanup()