"""
Unit tests for the BaseAgent class.

This module tests the core functionality of the BaseAgent class including
message handling, logging, error handling, and basic agent operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import logging

from agents.academic.base_agent import BaseAgent, AgentMessage
from tests.utils import assert_valid_json_structure


class TestAgentMessage:
    """Test cases for AgentMessage class."""
    
    def test_agent_message_creation(self):
        """Test AgentMessage creation with required fields."""
        message = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            message_type="test_type",
            content={"test": "data"},
            metadata={"meta": "info"},
            timestamp=datetime.now()
        )
        
        assert message.sender == "test_sender"
        assert message.recipient == "test_recipient"
        assert message.message_type == "test_type"
        assert message.content == {"test": "data"}
        assert message.metadata == {"meta": "info"}
        assert isinstance(message.timestamp, datetime)
        assert message.priority == 0  # default value
        assert message.retry_count == 0  # default value
        assert message.parent_id is None  # default value
        assert message.children is None  # default value
    
    def test_agent_message_validation_valid(self):
        """Test AgentMessage validation with valid data."""
        message = AgentMessage(
            sender="test_sender",
            recipient="test_recipient", 
            message_type="test_type",
            content={"test": "data"},
            metadata={},
            timestamp=datetime.now()
        )
        
        assert message.validate() is True
    
    def test_agent_message_validation_missing_fields(self):
        """Test AgentMessage validation with missing required fields."""
        # Create message with missing recipient
        message = AgentMessage(
            sender="test_sender",
            recipient="",  # Empty recipient
            message_type="test_type",
            content={"test": "data"},
            metadata={},
            timestamp=datetime.now()
        )
        
        # Should still validate as empty string is present
        assert message.validate() is True
    
    def test_agent_message_acknowledge(self):
        """Test AgentMessage acknowledge method."""
        message = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            message_type="test_type",
            content={"test": "data"},
            metadata={},
            timestamp=datetime.now()
        )
        
        result = message.acknowledge()
        assert result is True
    
    def test_agent_message_retry(self):
        """Test AgentMessage retry mechanism."""
        message = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            message_type="test_type",
            content={"test": "data"},
            metadata={},
            timestamp=datetime.now()
        )
        
        # Should allow retries up to 3 times
        assert message.retry() is True
        assert message.retry_count == 1
        
        assert message.retry() is True  
        assert message.retry_count == 2
        
        assert message.retry() is True
        assert message.retry_count == 3
        
        # Should not allow more retries after 3
        assert message.retry() is False
        assert message.retry_count == 3
    
    def test_agent_message_with_parent_and_children(self):
        """Test AgentMessage with parent and children relationships."""
        message = AgentMessage(
            sender="test_sender",
            recipient="test_recipient",
            message_type="test_type",
            content={"test": "data"},
            metadata={},
            timestamp=datetime.now(),
            parent_id="parent_123",
            children=["child_1", "child_2"]
        )
        
        assert message.parent_id == "parent_123"
        assert message.children == ["child_1", "child_2"]


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    def test_base_agent_initialization(self):
        """Test BaseAgent initialization."""
        agent = BaseAgent("test_agent")
        
        assert agent.agent_id == "test_agent"
        assert isinstance(agent.logger, logging.Logger)
        assert agent.quality_threshold == 0.7
        assert agent.max_retries == 3
        assert agent.processing_timeout == 300
    
    @patch('agents.academic.base_agent.logging.FileHandler')
    def test_base_agent_logger_setup(self, mock_file_handler):
        """Test logger setup in BaseAgent."""
        mock_handler = Mock()
        mock_file_handler.return_value = mock_handler
        
        agent = BaseAgent("test_agent")
        
        assert agent.logger.name == "test_agent"
        assert agent.logger.level == logging.INFO
        mock_file_handler.assert_called_once_with("logs/test_agent.log")
    
    def test_send_message_valid(self):
        """Test sending a valid message."""
        agent = BaseAgent("test_agent")
        
        result = agent.send_message(
            recipient="target_agent",
            message_type="test_message",
            content={"data": "test_data"},
            priority=1
        )
        
        assert result is True
    
    def test_send_message_with_parent_id(self):
        """Test sending a message with parent ID."""
        agent = BaseAgent("test_agent")
        
        result = agent.send_message(
            recipient="target_agent",
            message_type="test_message",
            content={"data": "test_data"},
            parent_id="parent_123"
        )
        
        assert result is True
    
    def test_receive_message_valid(self):
        """Test receiving a valid message."""
        agent = BaseAgent("test_agent")
        
        message = AgentMessage(
            sender="sender_agent",
            recipient="test_agent",
            message_type="test_message",
            content={"data": "test_data"},
            metadata={},
            timestamp=datetime.now()
        )
        
        result = agent.receive_message(message)
        assert result is True
    
    def test_receive_message_invalid(self):
        """Test receiving an invalid message."""
        agent = BaseAgent("test_agent")
        
        # Create invalid message by removing required field
        message = AgentMessage(
            sender="",  # Empty sender
            recipient="test_agent",
            message_type="test_message",
            content={"data": "test_data"},
            metadata={},
            timestamp=datetime.now()
        )
        
        # Mock the validate method to return False
        with patch.object(message, 'validate', return_value=False):
            result = agent.receive_message(message)
            assert result is False
    
    def test_check_quality_not_implemented(self):
        """Test that check_quality raises NotImplementedError."""
        agent = BaseAgent("test_agent")
        
        with pytest.raises(NotImplementedError):
            agent.check_quality("test_content")
    
    def test_validate_input_not_implemented(self):
        """Test that validate_input raises NotImplementedError."""
        agent = BaseAgent("test_agent")
        
        with pytest.raises(NotImplementedError):
            agent.validate_input("test_input")
    
    def test_validate_output_not_implemented(self):
        """Test that validate_output raises NotImplementedError."""
        agent = BaseAgent("test_agent")
        
        with pytest.raises(NotImplementedError):
            agent.validate_output("test_output")
    
    def test_log_metrics(self):
        """Test logging performance metrics."""
        agent = BaseAgent("test_agent")
        
        metrics = {
            "processing_time": 1.5,
            "memory_usage": 256,
            "cpu_usage": 45.2
        }
        
        with patch.object(agent.logger, 'info') as mock_info:
            agent.log_metrics(metrics)
            mock_info.assert_called_once_with(f"Performance metrics: {metrics}")
    
    def test_handle_error_within_retry_limit(self):
        """Test error handling within retry limit."""
        agent = BaseAgent("test_agent")
        
        error = Exception("Test error")
        context = {
            "operation": "test_operation",
            "retry_count": 1
        }
        
        with patch.object(agent.logger, 'error') as mock_error, \
             patch.object(agent.logger, 'info') as mock_info:
            
            result = agent.handle_error(error, context)
            
            assert result is True
            mock_error.assert_called_once_with(
                "Error in test_operation: Test error"
            )
            mock_info.assert_called_once_with(
                "Retrying operation... (Attempt 2)"
            )
    
    def test_handle_error_exceeds_retry_limit(self):
        """Test error handling when exceeding retry limit."""
        agent = BaseAgent("test_agent")
        
        error = Exception("Test error")
        context = {
            "operation": "test_operation",
            "retry_count": 3  # Exceeds max_retries
        }
        
        with patch.object(agent.logger, 'error') as mock_error:
            result = agent.handle_error(error, context)
            
            assert result is False
            mock_error.assert_called_once_with(
                "Error in test_operation: Test error"
            )
    
    def test_handle_error_no_retry_count(self):
        """Test error handling with no retry count in context."""
        agent = BaseAgent("test_agent")
        
        error = Exception("Test error")
        context = {"operation": "test_operation"}
        
        with patch.object(agent.logger, 'error') as mock_error, \
             patch.object(agent.logger, 'info') as mock_info:
            
            result = agent.handle_error(error, context)
            
            assert result is True
            mock_error.assert_called_once_with(
                "Error in test_operation: Test error"
            )
            mock_info.assert_called_once_with(
                "Retrying operation... (Attempt 1)"
            )
    
    def test_handle_error_no_operation_context(self):
        """Test error handling with no operation in context."""
        agent = BaseAgent("test_agent")
        
        error = Exception("Test error")
        context = {}
        
        with patch.object(agent.logger, 'error') as mock_error:
            result = agent.handle_error(error, context)
            
            assert result is True
            mock_error.assert_called_once_with(
                "Error in unknown operation: Test error"
            )
    
    def test_agent_configuration_defaults(self):
        """Test that agent has proper default configuration."""
        agent = BaseAgent("test_agent")
        
        assert agent.quality_threshold == 0.7
        assert agent.max_retries == 3
        assert agent.processing_timeout == 300
        assert agent.agent_id == "test_agent"
    
    def test_agent_logger_configuration(self):
        """Test that logger is properly configured."""
        agent = BaseAgent("test_agent")
        
        assert agent.logger.name == "test_agent"
        assert agent.logger.level == logging.INFO
        assert len(agent.logger.handlers) >= 1  # At least one handler
    
    def test_message_creation_with_all_fields(self):
        """Test creating message with all optional fields."""
        agent = BaseAgent("test_agent")
        
        result = agent.send_message(
            recipient="target_agent",
            message_type="test_message",
            content={"data": "test_data"},
            priority=2,
            parent_id="parent_123"
        )
        
        assert result is True
    
    def test_message_validation_workflow(self):
        """Test complete message validation workflow."""
        agent = BaseAgent("test_agent")
        
        # Test valid message workflow
        valid_message = AgentMessage(
            sender="sender_agent",
            recipient="test_agent",
            message_type="test_message",
            content={"data": "test_data"},
            metadata={"timestamp": datetime.now().isoformat()},
            timestamp=datetime.now()
        )
        
        assert valid_message.validate() is True
        assert agent.receive_message(valid_message) is True
        
        # Test acknowledgment
        assert valid_message.acknowledge() is True
        
        # Test retry mechanism
        assert valid_message.retry() is True
        assert valid_message.retry_count == 1