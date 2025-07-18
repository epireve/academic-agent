"""
Unified Base Agent Interface for Academic Agent System

This module provides a unified base agent interface that consolidates
the functionality from both v2 and legacy agent systems, providing:
- Common agent interface and lifecycle management
- Standardized message passing and communication
- Unified error handling and logging
- Consistent metrics and monitoring
- Plugin/extension support
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic

# Import from consolidated core modules
from ..core.logging import get_logger
from ..core.exceptions import (
    AcademicAgentError,
    CommunicationError,
    ValidationError,
    ProcessingError,
)
from ..core.simple_monitoring import get_system_monitor
from ..core.simple_config import get_config


T = TypeVar('T')


@dataclass
class AgentMessage:
    """Unified message format for inter-agent communication."""
    
    sender: str
    recipient: str
    message_type: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0
    retry_count: int = 0
    parent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "retry_count": self.retry_count,
            "parent_id": self.parent_id,
            "correlation_id": self.correlation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        data = data.copy()
        if "timestamp" in data and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class TaskResult(Generic[T]):
    """Generic task result container."""
    
    task_id: str
    task_type: str
    success: bool
    result: Optional[T] = None
    error: Optional[Exception] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success": self.success,
            "result": self.result if self.result is None or isinstance(self.result, (dict, list, str, int, float, bool)) else str(self.result),
            "error": str(self.error) if self.error else None,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
        }


class AgentCapability(ABC):
    """Base class for agent capabilities/plugins."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.logger = get_logger(f"capability.{name}")
    
    @abstractmethod
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute capability task."""
        pass
    
    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get capability information."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "config": self.config,
        }


class BaseAgent(ABC):
    """
    Unified base agent class that provides common functionality
    for all agents in the system.
    """
    
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """Initialize base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.logger = get_logger(agent_id)
        self.monitor = get_system_monitor()
        
        # Agent state
        self.status = "initializing"
        self.started_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Capabilities/plugins
        self.capabilities: Dict[str, AgentCapability] = {}
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.sent_messages: List[AgentMessage] = []
        self.received_messages: List[AgentMessage] = []
        
        # Metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "total_processing_time": 0.0,
            "errors": 0,
        }
        
        # Lifecycle hooks
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        self.logger.info(f"Agent {agent_id} initialized")
    
    # Lifecycle methods
    
    async def start(self):
        """Start the agent."""
        if self._running:
            self.logger.warning("Agent already running")
            return
        
        self.logger.info(f"Starting agent {self.agent_id}")
        self._running = True
        self.status = "running"
        
        # Initialize agent-specific components
        await self.initialize()
        
        # Start message processing
        self._tasks.append(
            asyncio.create_task(self._process_messages())
        )
        
        self.logger.info(f"Agent {self.agent_id} started successfully")
    
    async def stop(self):
        """Stop the agent."""
        if not self._running:
            self.logger.warning("Agent not running")
            return
        
        self.logger.info(f"Stopping agent {self.agent_id}")
        self._running = False
        self.status = "stopping"
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Cleanup
        await self.cleanup()
        
        self.status = "stopped"
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific components."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup agent resources."""
        pass
    
    # Capability management
    
    def register_capability(self, capability: AgentCapability):
        """Register a capability with the agent."""
        self.capabilities[capability.name] = capability
        self.logger.info(f"Registered capability: {capability.name}")
    
    def unregister_capability(self, name: str):
        """Unregister a capability."""
        if name in self.capabilities:
            del self.capabilities[name]
            self.logger.info(f"Unregistered capability: {name}")
    
    async def execute_capability(
        self,
        capability_name: str,
        task_data: Dict[str, Any]
    ) -> TaskResult:
        """Execute a capability task."""
        if capability_name not in self.capabilities:
            return TaskResult(
                task_id=task_data.get("task_id", "unknown"),
                task_type=capability_name,
                success=False,
                error=ValueError(f"Unknown capability: {capability_name}"),
            )
        
        capability = self.capabilities[capability_name]
        
        # Validate input
        if not capability.validate_input(task_data):
            return TaskResult(
                task_id=task_data.get("task_id", "unknown"),
                task_type=capability_name,
                success=False,
                error=ValidationError("Invalid input data"),
            )
        
        # Execute capability
        start_time = time.time()
        try:
            result = await capability.execute(task_data)
            result.processing_time = time.time() - start_time
            
            # Update metrics
            if result.success:
                self.metrics["tasks_completed"] += 1
            else:
                self.metrics["tasks_failed"] += 1
            self.metrics["total_processing_time"] += result.processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Capability {capability_name} failed: {e}")
            self.metrics["tasks_failed"] += 1
            self.metrics["errors"] += 1
            
            return TaskResult(
                task_id=task_data.get("task_id", "unknown"),
                task_type=capability_name,
                success=False,
                error=e,
                processing_time=time.time() - start_time,
            )
    
    # Message handling
    
    def register_message_handler(
        self,
        message_type: str,
        handler: Callable[[AgentMessage], None]
    ):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
        self.logger.debug(f"Registered handler for message type: {message_type}")
    
    async def send_message(
        self,
        recipient: str,
        message_type: str,
        content: Dict[str, Any],
        priority: int = 0,
        correlation_id: Optional[str] = None
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            priority=priority,
            correlation_id=correlation_id,
        )
        
        self.sent_messages.append(message)
        self.metrics["messages_sent"] += 1
        self.last_activity = datetime.now()
        
        # In a real system, this would send to a message broker
        # For now, we'll log it
        self.logger.info(
            f"Sent message to {recipient}: {message_type}",
            extra={"message_id": message.message_id}
        )
        
        return message
    
    async def receive_message(self, message: AgentMessage):
        """Receive a message from another agent."""
        self.received_messages.append(message)
        self.metrics["messages_received"] += 1
        self.last_activity = datetime.now()
        
        # Add to processing queue
        await self.message_queue.put(message)
        
        self.logger.info(
            f"Received message from {message.sender}: {message.message_type}",
            extra={"message_id": message.message_id}
        )
    
    async def _process_messages(self):
        """Process messages from the queue."""
        while self._running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Find handler
                handler = self.message_handlers.get(message.message_type)
                if handler:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing message {message.message_id}: {e}"
                        )
                        self.metrics["errors"] += 1
                else:
                    self.logger.warning(
                        f"No handler for message type: {message.message_type}"
                    )
                
            except asyncio.TimeoutError:
                # No messages, continue
                continue
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {e}")
                self.metrics["errors"] += 1
    
    # Utility methods
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        uptime = (datetime.now() - self.started_at).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "uptime_seconds": uptime,
            "last_activity": self.last_activity.isoformat(),
            "capabilities": list(self.capabilities.keys()),
            "metrics": self.metrics.copy(),
            "message_queue_size": self.message_queue.qsize(),
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health = {
            "healthy": self._running and self.status == "running",
            "status": self.status,
            "checks": {}
        }
        
        # Check message processing
        health["checks"]["message_processing"] = {
            "healthy": self.message_queue.qsize() < 1000,
            "queue_size": self.message_queue.qsize()
        }
        
        # Check capabilities
        health["checks"]["capabilities"] = {
            "healthy": len(self.capabilities) > 0,
            "count": len(self.capabilities)
        }
        
        # Overall health
        health["healthy"] = all(
            check["healthy"] for check in health["checks"].values()
        )
        
        return health


class SyncAgentAdapter(BaseAgent):
    """
    Adapter class to make synchronous agents work with the async base agent.
    This allows gradual migration of legacy agents.
    """
    
    def __init__(self, agent_id: str, sync_agent: Any, config: Optional[Dict[str, Any]] = None):
        """Initialize sync adapter.
        
        Args:
            agent_id: Unique identifier for the agent
            sync_agent: The synchronous agent instance to wrap
            config: Optional configuration dictionary
        """
        super().__init__(agent_id, config)
        self.sync_agent = sync_agent
        self._executor = None
    
    async def initialize(self):
        """Initialize the sync agent in an executor."""
        import concurrent.futures
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Initialize sync agent if it has an init method
        if hasattr(self.sync_agent, 'initialize'):
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.sync_agent.initialize
            )
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
    
    async def execute_sync_method(self, method_name: str, *args, **kwargs) -> Any:
        """Execute a synchronous method in the executor."""
        if not hasattr(self.sync_agent, method_name):
            raise AttributeError(f"Sync agent has no method: {method_name}")
        
        method = getattr(self.sync_agent, method_name)
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            method,
            *args,
            **kwargs
        )


# Convenience functions

def create_agent(agent_class: type, agent_id: str, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
    """Factory function to create an agent instance."""
    return agent_class(agent_id=agent_id, config=config)


async def start_agent(agent: BaseAgent) -> BaseAgent:
    """Start an agent and return it."""
    await agent.start()
    return agent