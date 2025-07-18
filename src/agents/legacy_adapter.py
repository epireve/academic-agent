"""
Legacy Agent Adapter for Academic Agent System

This module provides adapters to make legacy agents compatible with the
unified agent architecture, allowing gradual migration.
"""

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .base_agent import BaseAgent, AgentCapability, TaskResult, AgentMessage
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError

logger = get_logger(__name__)


class LegacyAgentCapability(AgentCapability):
    """Wraps a legacy agent method as a capability."""
    
    def __init__(
        self,
        name: str,
        legacy_method: Callable,
        input_mapper: Optional[Callable] = None,
        output_mapper: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize legacy capability wrapper.
        
        Args:
            name: Capability name
            legacy_method: The legacy method to wrap
            input_mapper: Optional function to map input data
            output_mapper: Optional function to map output data
            config: Optional configuration
        """
        super().__init__(name, config)
        self.legacy_method = legacy_method
        self.input_mapper = input_mapper or (lambda x: x)
        self.output_mapper = output_mapper or (lambda x: x)
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute the legacy method."""
        task_id = task_data.get("task_id", "unknown")
        
        try:
            # Map input data
            legacy_input = self.input_mapper(task_data)
            
            # Check if legacy method is async
            if asyncio.iscoroutinefunction(self.legacy_method):
                result = await self.legacy_method(legacy_input)
            else:
                # Run sync method in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self.legacy_method,
                    legacy_input
                )
            
            # Map output data
            mapped_result = self.output_mapper(result)
            
            return TaskResult(
                task_id=task_id,
                task_type=self.name,
                success=True,
                result=mapped_result,
            )
            
        except Exception as e:
            self.logger.error(f"Legacy capability {self.name} failed: {e}")
            return TaskResult(
                task_id=task_id,
                task_type=self.name,
                success=False,
                error=e,
            )
    
    def __del__(self):
        """Cleanup executor."""
        self._executor.shutdown(wait=False)


class LegacyAgentAdapter(BaseAgent):
    """
    Adapter to make legacy agents compatible with the unified architecture.
    """
    
    def __init__(
        self,
        agent_id: str,
        legacy_agent: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize legacy agent adapter.
        
        Args:
            agent_id: Unique identifier for the agent
            legacy_agent: The legacy agent instance to wrap
            config: Optional configuration
        """
        super().__init__(agent_id, config)
        self.legacy_agent = legacy_agent
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Auto-discover and wrap capabilities
        self._discover_capabilities()
        
        # Map legacy message handlers if available
        self._setup_message_handlers()
    
    def _discover_capabilities(self):
        """Auto-discover methods that can be exposed as capabilities."""
        # Common method patterns to expose as capabilities
        capability_patterns = [
            ("process", "processing"),
            ("analyze", "analysis"),
            ("generate", "generation"),
            ("extract", "extraction"),
            ("validate", "validation"),
        ]
        
        for attr_name in dir(self.legacy_agent):
            if attr_name.startswith("_"):
                continue
                
            attr = getattr(self.legacy_agent, attr_name)
            if not callable(attr):
                continue
            
            # Check if method matches a capability pattern
            for pattern, capability_type in capability_patterns:
                if pattern in attr_name.lower():
                    capability_name = f"{capability_type}_{attr_name}"
                    
                    # Create capability wrapper
                    capability = LegacyAgentCapability(
                        name=capability_name,
                        legacy_method=attr,
                        config=self.config
                    )
                    
                    self.register_capability(capability)
                    self.logger.info(f"Discovered capability: {capability_name}")
                    break
    
    def _setup_message_handlers(self):
        """Setup message handlers from legacy agent."""
        # Check for common message handling methods
        handler_methods = [
            "handle_message",
            "receive_message",
            "process_message",
            "on_message",
        ]
        
        for method_name in handler_methods:
            if hasattr(self.legacy_agent, method_name):
                method = getattr(self.legacy_agent, method_name)
                if callable(method):
                    # Register a generic handler that delegates to legacy method
                    self.register_message_handler("*", self._handle_legacy_message)
                    self.logger.info(f"Found legacy message handler: {method_name}")
                    break
    
    async def _handle_legacy_message(self, message: AgentMessage):
        """Handle messages using legacy agent's message handler."""
        try:
            # Convert to legacy format if needed
            legacy_message = self._convert_to_legacy_message(message)
            
            # Find the legacy handler method
            handler = None
            for method_name in ["handle_message", "receive_message", "process_message", "on_message"]:
                if hasattr(self.legacy_agent, method_name):
                    handler = getattr(self.legacy_agent, method_name)
                    break
            
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(legacy_message)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self._executor,
                        handler,
                        legacy_message
                    )
            
        except Exception as e:
            self.logger.error(f"Error handling legacy message: {e}")
    
    def _convert_to_legacy_message(self, message: AgentMessage) -> Any:
        """Convert unified message to legacy format."""
        # Check if legacy agent has a specific message class
        if hasattr(self.legacy_agent, "AgentMessage"):
            legacy_class = self.legacy_agent.AgentMessage
            return legacy_class(
                sender=message.sender,
                recipient=message.recipient,
                message_type=message.message_type,
                content=message.content,
                metadata=message.metadata,
                timestamp=message.timestamp,
            )
        else:
            # Return as dict if no specific format
            return message.to_dict()
    
    async def initialize(self):
        """Initialize the legacy agent."""
        # Check for legacy initialization methods
        init_methods = ["initialize", "init", "setup", "start"]
        
        for method_name in init_methods:
            if hasattr(self.legacy_agent, method_name):
                method = getattr(self.legacy_agent, method_name)
                if callable(method):
                    self.logger.info(f"Initializing legacy agent with {method_name}")
                    
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            self._executor,
                            method
                        )
                    break
    
    async def cleanup(self):
        """Cleanup the legacy agent."""
        # Check for legacy cleanup methods
        cleanup_methods = ["cleanup", "shutdown", "stop", "close"]
        
        for method_name in cleanup_methods:
            if hasattr(self.legacy_agent, method_name):
                method = getattr(self.legacy_agent, method_name)
                if callable(method):
                    self.logger.info(f"Cleaning up legacy agent with {method_name}")
                    
                    if asyncio.iscoroutinefunction(method):
                        await method()
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            self._executor,
                            method
                        )
                    break
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
    
    def expose_legacy_method(
        self,
        method_name: str,
        capability_name: Optional[str] = None,
        input_mapper: Optional[Callable] = None,
        output_mapper: Optional[Callable] = None
    ):
        """Manually expose a legacy method as a capability.
        
        Args:
            method_name: Name of the legacy method
            capability_name: Optional capability name (defaults to method name)
            input_mapper: Optional function to map input data
            output_mapper: Optional function to map output data
        """
        if not hasattr(self.legacy_agent, method_name):
            raise ValueError(f"Legacy agent has no method: {method_name}")
        
        method = getattr(self.legacy_agent, method_name)
        if not callable(method):
            raise ValueError(f"{method_name} is not callable")
        
        capability_name = capability_name or method_name
        
        capability = LegacyAgentCapability(
            name=capability_name,
            legacy_method=method,
            input_mapper=input_mapper,
            output_mapper=output_mapper,
            config=self.config
        )
        
        self.register_capability(capability)
        self.logger.info(f"Exposed legacy method as capability: {capability_name}")


def create_legacy_adapter(
    agent_id: str,
    legacy_agent: Any,
    config: Optional[Dict[str, Any]] = None
) -> LegacyAgentAdapter:
    """Create a legacy agent adapter.
    
    Args:
        agent_id: Unique identifier for the agent
        legacy_agent: The legacy agent instance to wrap
        config: Optional configuration
        
    Returns:
        LegacyAgentAdapter instance
    """
    return LegacyAgentAdapter(agent_id, legacy_agent, config)


# Example usage functions

def adapt_academic_agent_v2(agent_instance: Any) -> LegacyAgentAdapter:
    """Adapt an Academic Agent v2 instance."""
    adapter = create_legacy_adapter(
        agent_id=getattr(agent_instance, "agent_id", "academic_agent_v2"),
        legacy_agent=agent_instance
    )
    
    # Expose specific v2 methods
    if hasattr(agent_instance, "execute_task"):
        adapter.expose_legacy_method(
            "execute_task",
            capability_name="task_execution",
            input_mapper=lambda data: data.get("task_data", {}),
            output_mapper=lambda result: result.to_dict() if hasattr(result, "to_dict") else result
        )
    
    if hasattr(agent_instance, "execute_workflow"):
        adapter.expose_legacy_method(
            "execute_workflow",
            capability_name="workflow_execution",
            input_mapper=lambda data: data.get("workflow_data", {}),
            output_mapper=lambda results: [r.to_dict() if hasattr(r, "to_dict") else r for r in results]
        )
    
    return adapter


def adapt_legacy_academic_agent(agent_instance: Any) -> LegacyAgentAdapter:
    """Adapt a legacy academic agent instance."""
    adapter = create_legacy_adapter(
        agent_id=getattr(agent_instance, "agent_id", "legacy_academic_agent"),
        legacy_agent=agent_instance
    )
    
    # Map common legacy methods
    method_mappings = {
        "process_pdf": "pdf_processing",
        "analyze_content": "content_analysis",
        "generate_notes": "note_generation",
        "create_outline": "outline_creation",
    }
    
    for legacy_method, capability_name in method_mappings.items():
        if hasattr(agent_instance, legacy_method):
            adapter.expose_legacy_method(legacy_method, capability_name)
    
    return adapter