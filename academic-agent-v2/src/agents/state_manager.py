"""
Agent State Management System for Academic Agent v2.

This module provides comprehensive state management and persistence for academic agents,
including task tracking, metrics collection, and recovery capabilities.
"""

import asyncio
import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ..core.exceptions import StateError
from ..core.logging import get_logger
from ..utils.file_utils import ensure_directory

logger = get_logger(__name__)


@dataclass
class TaskState:
    """Represents the state of a task."""
    
    task_id: str
    task_type: str
    status: str  # pending, running, completed, failed, cancelled
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self):
        """Mark task as started."""
        self.status = "running"
        self.started_at = datetime.now()
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None):
        """Mark task as completed."""
        self.status = "completed"
        self.completed_at = datetime.now()
        if output_data:
            self.output_data = output_data
    
    def fail(self, error_message: str):
        """Mark task as failed."""
        self.status = "failed"
        self.completed_at = datetime.now()
        self.error_message = error_message
    
    def cancel(self):
        """Mark task as cancelled."""
        self.status = "cancelled"
        self.completed_at = datetime.now()
    
    def retry(self) -> bool:
        """Attempt to retry the task."""
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.status = "pending"
            self.started_at = None
            self.completed_at = None
            self.error_message = None
            return True
        return False
    
    def get_duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def is_finished(self) -> bool:
        """Check if task is in a finished state."""
        return self.status in ["completed", "failed", "cancelled"]
    
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.status == "failed" and self.retry_count < self.max_retries
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task state to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ["created_at", "started_at", "completed_at"]:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskState":
        """Create TaskState from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ["created_at", "started_at", "completed_at"]:
            if data.get(key) is not None:
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


@dataclass
class AgentMetrics:
    """Represents agent performance metrics."""
    
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    retry_attempts: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    success_rate: float = 0.0
    last_activity: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.now)
    
    def update_task_completed(self, processing_time: Optional[float] = None):
        """Update metrics for completed task."""
        self.total_tasks += 1
        self.completed_tasks += 1
        self.last_activity = datetime.now()
        
        if processing_time:
            self.total_processing_time += processing_time
            self.average_processing_time = (
                self.total_processing_time / self.completed_tasks
            )
        
        self._update_success_rate()
    
    def update_task_failed(self):
        """Update metrics for failed task."""
        self.total_tasks += 1
        self.failed_tasks += 1
        self.last_activity = datetime.now()
        self._update_success_rate()
    
    def update_task_cancelled(self):
        """Update metrics for cancelled task."""
        self.total_tasks += 1
        self.cancelled_tasks += 1
        self.last_activity = datetime.now()
        self._update_success_rate()
    
    def update_retry_attempt(self):
        """Update retry attempt count."""
        self.retry_attempts += 1
    
    def _update_success_rate(self):
        """Update success rate calculation."""
        if self.total_tasks > 0:
            self.success_rate = self.completed_tasks / self.total_tasks
    
    def get_uptime(self) -> float:
        """Get agent uptime in seconds."""
        return (datetime.now() - self.uptime_start).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key in ["last_activity", "uptime_start"]:
            if data[key] is not None:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMetrics":
        """Create AgentMetrics from dictionary."""
        # Convert ISO strings back to datetime objects
        for key in ["last_activity", "uptime_start"]:
            if data.get(key) is not None:
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


@dataclass
class AgentState:
    """Comprehensive agent state."""
    
    agent_id: str
    status: str = "idle"  # idle, busy, error, shutdown
    current_tasks: List[str] = field(default_factory=list)
    task_history: Dict[str, TaskState] = field(default_factory=dict)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    configuration: Dict[str, Any] = field(default_factory=dict)
    plugin_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_checkpoint: Optional[datetime] = None
    version: str = "2.0"
    
    def add_task(self, task: TaskState):
        """Add a new task to the state."""
        self.task_history[task.task_id] = task
        if task.status == "running":
            self.current_tasks.append(task.task_id)
            self.status = "busy"
    
    def update_task(self, task_id: str, task_state: TaskState):
        """Update task state."""
        if task_id in self.task_history:
            old_state = self.task_history[task_id]
            self.task_history[task_id] = task_state
            
            # Update current tasks list
            if old_state.status == "running" and task_state.status != "running":
                if task_id in self.current_tasks:
                    self.current_tasks.remove(task_id)
            elif old_state.status != "running" and task_state.status == "running":
                if task_id not in self.current_tasks:
                    self.current_tasks.append(task_id)
            
            # Update metrics
            if task_state.is_finished():
                if task_state.status == "completed":
                    self.metrics.update_task_completed(task_state.get_duration())
                elif task_state.status == "failed":
                    self.metrics.update_task_failed()
                elif task_state.status == "cancelled":
                    self.metrics.update_task_cancelled()
            
            # Update agent status
            if not self.current_tasks:
                self.status = "idle"
            elif self.status == "idle":
                self.status = "busy"
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task by ID."""
        return self.task_history.get(task_id)
    
    def get_tasks_by_status(self, status: str) -> List[TaskState]:
        """Get tasks by status."""
        return [task for task in self.task_history.values() if task.status == status]
    
    def get_pending_tasks(self) -> List[TaskState]:
        """Get pending tasks."""
        return self.get_tasks_by_status("pending")
    
    def get_running_tasks(self) -> List[TaskState]:
        """Get running tasks."""
        return self.get_tasks_by_status("running")
    
    def get_completed_tasks(self) -> List[TaskState]:
        """Get completed tasks."""
        return self.get_tasks_by_status("completed")
    
    def get_failed_tasks(self) -> List[TaskState]:
        """Get failed tasks."""
        return self.get_tasks_by_status("failed")
    
    def update_plugin_state(self, plugin_name: str, state: Dict[str, Any]):
        """Update plugin state."""
        self.plugin_states[plugin_name] = state
    
    def checkpoint(self):
        """Create a checkpoint of current state."""
        self.last_checkpoint = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "current_tasks": self.current_tasks,
            "task_history": {
                task_id: task.to_dict() for task_id, task in self.task_history.items()
            },
            "metrics": self.metrics.to_dict(),
            "configuration": self.configuration,
            "plugin_states": self.plugin_states,
            "last_checkpoint": self.last_checkpoint.isoformat() if self.last_checkpoint else None,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary."""
        # Convert task history
        task_history = {}
        for task_id, task_data in data.get("task_history", {}).items():
            task_history[task_id] = TaskState.from_dict(task_data)
        
        # Convert metrics
        metrics_data = data.get("metrics", {})
        metrics = AgentMetrics.from_dict(metrics_data)
        
        # Convert last_checkpoint
        last_checkpoint = None
        if data.get("last_checkpoint"):
            last_checkpoint = datetime.fromisoformat(data["last_checkpoint"])
        
        return cls(
            agent_id=data["agent_id"],
            status=data.get("status", "idle"),
            current_tasks=data.get("current_tasks", []),
            task_history=task_history,
            metrics=metrics,
            configuration=data.get("configuration", {}),
            plugin_states=data.get("plugin_states", {}),
            last_checkpoint=last_checkpoint,
            version=data.get("version", "2.0"),
        )


class StateManager:
    """Manages agent state persistence and recovery."""
    
    def __init__(
        self,
        agent_id: str,
        state_dir: Optional[Path] = None,
        auto_save: bool = True,
        save_interval: float = 300.0,  # 5 minutes
    ):
        """Initialize state manager.
        
        Args:
            agent_id: Agent identifier
            state_dir: Directory for state files
            auto_save: Whether to enable automatic saving
            save_interval: Automatic save interval in seconds
        """
        self.agent_id = agent_id
        self.state_dir = state_dir or Path("state")
        self.auto_save = auto_save
        self.save_interval = save_interval
        self.logger = get_logger(__name__)
        
        # Ensure state directory exists
        ensure_directory(self.state_dir)
        
        # Initialize state
        self.state = AgentState(agent_id=agent_id)
        
        # Auto-save task
        self._auto_save_task: Optional[asyncio.Task] = None
        
        # State file paths
        self.json_state_file = self.state_dir / f"{agent_id}_state.json"
        self.binary_state_file = self.state_dir / f"{agent_id}_state.pkl"
        self.backup_state_file = self.state_dir / f"{agent_id}_state_backup.json"
    
    async def start(self):
        """Start the state manager."""
        # Try to load existing state
        await self.load_state()
        
        # Start auto-save if enabled
        if self.auto_save:
            self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        
        self.logger.info(f"State manager started for agent {self.agent_id}")
    
    async def stop(self):
        """Stop the state manager."""
        # Cancel auto-save task
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
        
        # Final save
        await self.save_state()
        
        self.logger.info(f"State manager stopped for agent {self.agent_id}")
    
    async def _auto_save_loop(self):
        """Auto-save loop."""
        while True:
            try:
                await asyncio.sleep(self.save_interval)
                await self.save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-save loop: {e}")
    
    async def save_state(self, backup: bool = True):
        """Save agent state to file.
        
        Args:
            backup: Whether to create a backup of existing state
        """
        try:
            # Create backup if requested
            if backup and self.json_state_file.exists():
                self.json_state_file.replace(self.backup_state_file)
            
            # Create checkpoint
            self.state.checkpoint()
            
            # Save to JSON (human-readable)
            state_data = self.state.to_dict()
            with open(self.json_state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            # Save to binary (faster loading)
            with open(self.binary_state_file, "wb") as f:
                pickle.dump(self.state, f)
            
            self.logger.debug(f"Agent state saved to {self.json_state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}")
            raise StateError(f"Failed to save state: {e}")
    
    async def load_state(self) -> bool:
        """Load agent state from file.
        
        Returns:
            True if state was loaded, False otherwise
        """
        try:
            # Try to load from binary first (faster)
            if self.binary_state_file.exists():
                try:
                    with open(self.binary_state_file, "rb") as f:
                        self.state = pickle.load(f)
                    self.logger.info(f"Agent state loaded from {self.binary_state_file}")
                    return True
                except Exception as e:
                    self.logger.warning(f"Failed to load binary state: {e}")
            
            # Fall back to JSON
            if self.json_state_file.exists():
                with open(self.json_state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                self.state = AgentState.from_dict(state_data)
                self.logger.info(f"Agent state loaded from {self.json_state_file}")
                return True
            
            # Try backup file
            if self.backup_state_file.exists():
                with open(self.backup_state_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                self.state = AgentState.from_dict(state_data)
                self.logger.info(f"Agent state loaded from backup {self.backup_state_file}")
                return True
            
            self.logger.info("No existing state found, starting with clean state")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to load agent state: {e}")
            # Create new clean state
            self.state = AgentState(agent_id=self.agent_id)
            return False
    
    def add_task(self, task: TaskState):
        """Add a task to the state."""
        self.state.add_task(task)
    
    def update_task(self, task_id: str, task_state: TaskState):
        """Update task state."""
        self.state.update_task(task_id, task_state)
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task by ID."""
        return self.state.get_task(task_id)
    
    def get_state(self) -> AgentState:
        """Get current agent state."""
        return self.state
    
    def update_configuration(self, config: Dict[str, Any]):
        """Update agent configuration in state."""
        self.state.configuration = config
    
    def update_plugin_state(self, plugin_name: str, plugin_state: Dict[str, Any]):
        """Update plugin state."""
        self.state.update_plugin_state(plugin_name, plugin_state)
    
    async def export_state(self, export_file: Path, format: str = "json"):
        """Export state to file.
        
        Args:
            export_file: File to export to
            format: Export format ('json', 'yaml', 'pickle')
        """
        try:
            ensure_directory(export_file.parent)
            
            if format == "json":
                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
            elif format == "yaml":
                with open(export_file, "w", encoding="utf-8") as f:
                    yaml.dump(self.state.to_dict(), f, default_flow_style=False)
            elif format == "pickle":
                with open(export_file, "wb") as f:
                    pickle.dump(self.state, f)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"State exported to {export_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export state: {e}")
            raise StateError(f"Failed to export state: {e}")
    
    async def import_state(self, import_file: Path, format: str = "json"):
        """Import state from file.
        
        Args:
            import_file: File to import from
            format: Import format ('json', 'yaml', 'pickle')
        """
        try:
            if not import_file.exists():
                raise StateError(f"Import file not found: {import_file}")
            
            if format == "json":
                with open(import_file, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                self.state = AgentState.from_dict(state_data)
            elif format == "yaml":
                with open(import_file, "r", encoding="utf-8") as f:
                    state_data = yaml.safe_load(f)
                self.state = AgentState.from_dict(state_data)
            elif format == "pickle":
                with open(import_file, "rb") as f:
                    self.state = pickle.load(f)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            self.logger.info(f"State imported from {import_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to import state: {e}")
            raise StateError(f"Failed to import state: {e}")
    
    def get_recovery_info(self) -> Dict[str, Any]:
        """Get information for state recovery."""
        return {
            "agent_id": self.agent_id,
            "state_files": {
                "json": str(self.json_state_file),
                "binary": str(self.binary_state_file),
                "backup": str(self.backup_state_file),
            },
            "last_checkpoint": (
                self.state.last_checkpoint.isoformat() 
                if self.state.last_checkpoint else None
            ),
            "current_status": self.state.status,
            "task_counts": {
                "total": len(self.state.task_history),
                "pending": len(self.state.get_pending_tasks()),
                "running": len(self.state.get_running_tasks()),
                "completed": len(self.state.get_completed_tasks()),
                "failed": len(self.state.get_failed_tasks()),
            },
        }


# Utility functions

async def create_state_manager(
    agent_id: str,
    state_dir: Optional[Path] = None,
    auto_save: bool = True,
    save_interval: float = 300.0,
) -> StateManager:
    """Create and start a state manager.
    
    Args:
        agent_id: Agent identifier
        state_dir: Directory for state files
        auto_save: Whether to enable automatic saving
        save_interval: Automatic save interval in seconds
        
    Returns:
        Started StateManager instance
    """
    manager = StateManager(
        agent_id=agent_id,
        state_dir=state_dir,
        auto_save=auto_save,
        save_interval=save_interval,
    )
    await manager.start()
    return manager


def create_task_state(
    task_id: str,
    task_type: str,
    input_data: Dict[str, Any],
    priority: int = 0,
    max_retries: int = 3,
    dependencies: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TaskState:
    """Create a new task state.
    
    Args:
        task_id: Unique task identifier
        task_type: Type of task
        input_data: Task input data
        priority: Task priority
        max_retries: Maximum retry attempts
        dependencies: List of task dependencies
        metadata: Additional metadata
        
    Returns:
        TaskState instance
    """
    return TaskState(
        task_id=task_id,
        task_type=task_type,
        status="pending",
        created_at=datetime.now(),
        input_data=input_data,
        priority=priority,
        max_retries=max_retries,
        dependencies=dependencies or [],
        metadata=metadata or {},
    )