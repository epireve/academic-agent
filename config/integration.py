#!/usr/bin/env python3
"""
Integration script for the YAML-based configuration system with existing agents.

This script provides updated base agent classes that work with the new configuration system.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Add the config directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config_manager import ConfigurationManager, AcademicAgentConfig, get_config, get_env_settings
    
    class ConfiguredBaseAgent:
        """
        Enhanced base agent class that integrates with the YAML configuration system.
        
        This class extends the original BaseAgent functionality with configuration-driven
        settings and improved error handling.
        """
        
        def __init__(self, agent_id: str, config_manager: Optional[ConfigurationManager] = None):
            """
            Initialize the configured base agent.
            
            Args:
                agent_id: Unique identifier for the agent
                config_manager: Configuration manager instance (optional)
            """
            self.agent_id = agent_id
            self.config_manager = config_manager or ConfigurationManager()
            
            # Load configuration
            try:
                self.global_config = self.config_manager.get_config()
                self.env_settings = self.config_manager.get_env_settings()
                self.agent_config = self.global_config.get_agent_config(agent_id)
            except Exception as e:
                # Fallback to default configuration if loading fails
                logging.warning(f"Failed to load configuration for agent {agent_id}: {e}")
                self.global_config = None
                self.env_settings = None
                self.agent_config = None
            
            # Set up agent properties from configuration
            self._setup_agent_properties()
            
            # Set up logging
            self.logger = self._setup_logger()
            
            # Initialize message queue and metrics
            self.message_queue = []
            self.metrics = {"messages_sent": 0, "messages_received": 0, "errors": 0}
            
            self.logger.info(f"Configured agent {agent_id} initialized")
        
        def _setup_agent_properties(self):
            """Set up agent properties from configuration."""
            if self.agent_config:
                self.enabled = self.agent_config.enabled
                self.max_retries = self.agent_config.max_retries
                self.processing_timeout = self.agent_config.timeout
                self.quality_threshold = self.agent_config.quality_threshold
                self.specialized_prompt = self.agent_config.specialized_prompt
            else:
                # Fallback defaults
                self.enabled = True
                self.max_retries = 3
                self.processing_timeout = 300
                self.quality_threshold = 0.7
                self.specialized_prompt = None
            
            # Global settings
            if self.global_config:
                self.global_quality_threshold = self.global_config.quality_threshold
                self.improvement_threshold = self.global_config.improvement_threshold
                self.max_improvement_cycles = self.global_config.max_improvement_cycles
                self.communication_interval = self.global_config.communication_interval
            else:
                self.global_quality_threshold = 0.75
                self.improvement_threshold = 0.3
                self.max_improvement_cycles = 3
                self.communication_interval = 30
        
        def _setup_logger(self) -> logging.Logger:
            """Set up logging for the agent using configuration."""
            logger = logging.getLogger(f"academic_agent.{self.agent_id}")
            
            # Get logging configuration
            if self.global_config and self.global_config.logging:
                log_config = self.global_config.logging
                log_level = getattr(logging, log_config.level.upper(), logging.INFO)
                logger.setLevel(log_level)
                
                # Clear existing handlers
                logger.handlers = []
                
                # Console handler
                if log_config.console_enabled:
                    console_handler = logging.StreamHandler()
                    console_formatter = logging.Formatter(log_config.format)
                    console_handler.setFormatter(console_formatter)
                    logger.addHandler(console_handler)
                
                # File handler
                if log_config.file_enabled:
                    log_dir = Path(log_config.log_dir)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_handler = logging.FileHandler(log_dir / f"{self.agent_id}.log")
                    file_formatter = logging.Formatter(log_config.format)
                    file_handler.setFormatter(file_formatter)
                    logger.addHandler(file_handler)
            else:
                # Fallback logging configuration
                logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            
            return logger
        
        def is_enabled(self) -> bool:
            """Check if the agent is enabled."""
            return self.enabled
        
        def get_feedback_loops(self) -> list:
            """Get feedback loops involving this agent."""
            if not self.global_config:
                return []
            
            return self.global_config.get_feedback_loops_for_agent(self.agent_id)
        
        def should_communicate_with(self, target_agent: str) -> bool:
            """Check if this agent should communicate with the target agent."""
            feedback_loops = self.get_feedback_loops()
            
            for loop in feedback_loops:
                if loop.source == self.agent_id and loop.target == target_agent and loop.enabled:
                    return True
            
            return False
        
        def get_communication_interval(self, target_agent: str) -> int:
            """Get the communication interval for a target agent."""
            feedback_loops = self.get_feedback_loops()
            
            for loop in feedback_loops:
                if loop.source == self.agent_id and loop.target == target_agent and loop.enabled:
                    return loop.interval
            
            return self.communication_interval
        
        def send_message(self, recipient: str, message_type: str, content: Dict[str, Any], 
                        priority: int = 0, parent_id: Optional[str] = None) -> bool:
            """
            Send a message to another agent.
            
            Args:
                recipient: Target agent ID
                message_type: Type of message
                content: Message content
                priority: Message priority (0-5)
                parent_id: Parent message ID
                
            Returns:
                Success status
            """
            try:
                # Check if communication is allowed
                if not self.should_communicate_with(recipient):
                    self.logger.warning(f"Communication not allowed with {recipient}")
                    return False
                
                # Check inter-agent communication settings
                if self.global_config and not self.global_config.inter_agent_communication.enabled:
                    self.logger.warning("Inter-agent communication is disabled")
                    return False
                
                # Create message
                message = {
                    "sender": self.agent_id,
                    "recipient": recipient,
                    "message_type": message_type,
                    "content": content,
                    "metadata": {
                        "priority": priority,
                        "parent_id": parent_id,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Add to queue (in a real implementation, this would be sent to a message broker)
                self.message_queue.append(message)
                self.metrics["messages_sent"] += 1
                
                self.logger.info(f"Message sent to {recipient}: {message_type}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to send message to {recipient}: {e}")
                self.metrics["errors"] += 1
                return False
        
        def receive_message(self, message: Dict[str, Any]) -> bool:
            """
            Process a received message.
            
            Args:
                message: Received message
                
            Returns:
                Success status
            """
            try:
                self.metrics["messages_received"] += 1
                
                # Basic message validation
                required_fields = ["sender", "recipient", "message_type", "content"]
                if not all(field in message for field in required_fields):
                    self.logger.error(f"Invalid message format: {message}")
                    return False
                
                # Check if message is for this agent
                if message["recipient"] != self.agent_id:
                    self.logger.warning(f"Message not for this agent: {message['recipient']}")
                    return False
                
                self.logger.info(f"Received message from {message['sender']}: {message['message_type']}")
                
                # Process the message (to be implemented by subclasses)
                return self._process_message(message)
                
            except Exception as e:
                self.logger.error(f"Failed to process message: {e}")
                self.metrics["errors"] += 1
                return False
        
        def _process_message(self, message: Dict[str, Any]) -> bool:
            """
            Process a received message (to be implemented by subclasses).
            
            Args:
                message: Message to process
                
            Returns:
                Success status
            """
            # Default implementation - log and acknowledge
            self.logger.info(f"Processing message: {message['message_type']}")
            return True
        
        def check_quality(self, content: Any) -> float:
            """
            Check the quality of content.
            
            Args:
                content: Content to check
                
            Returns:
                Quality score (0.0 to 1.0)
            """
            # Default implementation - to be overridden by subclasses
            return 0.5
        
        def get_processing_timeout(self) -> int:
            """Get the processing timeout for this agent."""
            return self.processing_timeout
        
        def get_quality_threshold(self) -> float:
            """Get the quality threshold for this agent."""
            return self.quality_threshold
        
        def get_specialized_prompt(self) -> Optional[str]:
            """Get the specialized prompt for this agent."""
            return self.specialized_prompt
        
        def log_metrics(self) -> None:
            """Log performance metrics."""
            self.logger.info(f"Agent metrics: {self.metrics}")
        
        def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
            """
            Handle errors with graceful degradation.
            
            Args:
                error: Exception that occurred
                context: Additional context information
                
            Returns:
                Whether to retry the operation
            """
            context = context or {}
            retry_count = context.get("retry_count", 0)
            
            self.logger.error(f"Error in {context.get('operation', 'unknown operation')}: {str(error)}")
            self.metrics["errors"] += 1
            
            if retry_count < self.max_retries:
                self.logger.info(f"Retrying operation... (Attempt {retry_count + 1})")
                return True
            
            self.logger.error(f"Max retries exceeded for {context.get('operation', 'unknown operation')}")
            return False
        
        def validate_input(self, input_data: Any) -> bool:
            """Validate input data (to be implemented by subclasses)."""
            return True
        
        def validate_output(self, output_data: Any) -> bool:
            """Validate output data (to be implemented by subclasses)."""
            return True
        
        def get_configuration_summary(self) -> Dict[str, Any]:
            """Get a summary of the agent's configuration."""
            return {
                "agent_id": self.agent_id,
                "enabled": self.enabled,
                "max_retries": self.max_retries,
                "processing_timeout": self.processing_timeout,
                "quality_threshold": self.quality_threshold,
                "has_specialized_prompt": self.specialized_prompt is not None,
                "feedback_loops": len(self.get_feedback_loops()),
                "metrics": self.metrics
            }
    
    # Example usage and integration test
    def demo_configuration_integration():
        """Demonstrate the configuration system integration."""
        print("=== Configuration System Integration Demo ===\n")
        
        try:
            # Initialize configuration manager
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("development")
            
            print("1. Configuration loaded successfully")
            print(f"   Environment: {config.environment}")
            print(f"   Debug mode: {config.debug}")
            print(f"   Quality threshold: {config.quality_threshold}")
            print()
            
            # Create configured agents
            agents = {}
            for agent_id in ["ingestion_agent", "outline_agent", "notes_agent"]:
                agent = ConfiguredBaseAgent(agent_id, config_manager)
                agents[agent_id] = agent
                
                print(f"2. Created agent: {agent_id}")
                print(f"   Enabled: {agent.is_enabled()}")
                print(f"   Timeout: {agent.get_processing_timeout()}s")
                print(f"   Quality threshold: {agent.get_quality_threshold()}")
                print(f"   Feedback loops: {len(agent.get_feedback_loops())}")
                print()
            
            # Test inter-agent communication
            print("3. Testing inter-agent communication")
            quality_agent = agents["notes_agent"]
            if quality_agent.should_communicate_with("quality_manager"):
                print("   ✓ Communication allowed with quality_manager")
            else:
                print("   ✗ Communication not allowed with quality_manager")
            
            # Test message sending
            success = quality_agent.send_message(
                "quality_manager",
                "quality_check",
                {"content": "test content", "score": 0.8}
            )
            print(f"   Message sent: {success}")
            print()
            
            # Display configuration summaries
            print("4. Agent configuration summaries:")
            for agent_id, agent in agents.items():
                summary = agent.get_configuration_summary()
                print(f"   {agent_id}: {summary}")
            
            print("\n=== Integration demo completed successfully ===")
            
        except Exception as e:
            print(f"Integration demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    if __name__ == "__main__":
        demo_configuration_integration()

except ImportError as e:
    print(f"Failed to import configuration system: {e}")
    print("Make sure the configuration system is properly installed.")
    sys.exit(1)