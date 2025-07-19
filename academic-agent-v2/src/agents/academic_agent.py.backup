"""
Simplified Academic Agent System for Academic Agent v2.

This module provides a streamlined academic agent implementation that replaces
the complex smolagents architecture with a more maintainable system that
leverages the new configuration, logging, and PDF processing capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from ..core.config_manager import ConfigManager
from ..core.exceptions import (
    AcademicAgentError,
    ConfigurationError,
    ProcessingError,
    ValidationError,
)
from ..core.logging import get_logger
from ..core.monitoring import get_system_monitor
from ..processors.pdf_processor import PDFProcessor
from ..utils.file_utils import ensure_directory

logger = get_logger(__name__)


@dataclass
class TaskResult:
    """Represents the result of a task execution."""
    
    task_id: str
    task_type: str
    success: bool
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success": self.success,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat(),
        }


@dataclass
class AgentState:
    """Represents the current state of the academic agent."""
    
    agent_id: str
    status: str = "idle"  # idle, processing, error
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    last_update: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, new_status: str, task_id: Optional[str] = None):
        """Update agent status."""
        self.status = new_status
        self.current_task = task_id
        self.last_update = datetime.now()
    
    def complete_task(self, task_id: str, success: bool):
        """Mark a task as completed."""
        if success:
            self.completed_tasks.append(task_id)
        else:
            self.failed_tasks.append(task_id)
        
        if self.current_task == task_id:
            self.current_task = None
            self.status = "idle"
        
        self.last_update = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "current_task": self.current_task,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "last_update": self.last_update.isoformat(),
            "metrics": self.metrics,
        }


class BasePlugin:
    """Base class for academic agent plugins."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plugin."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        self.enabled = self.config.get("enabled", True)
    
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute plugin task - to be implemented by subclasses."""
        raise NotImplementedError("Plugin must implement execute method")
    
    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Validate input data - to be implemented by subclasses."""
        return True
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.__class__.__name__,
            "enabled": self.enabled,
            "config": self.config,
        }


class PDFProcessorPlugin(BasePlugin):
    """Plugin for PDF processing tasks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PDF processor plugin."""
        super().__init__(config)
        self.pdf_processor = PDFProcessor()
    
    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Validate PDF processing input."""
        required_fields = ["input_path"]
        return all(field in task_data for field in required_fields)
    
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute PDF processing task."""
        task_id = task_data.get("task_id", "pdf_processing")
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_input(task_data):
                raise ValidationError("Invalid input data for PDF processing")
            
            input_path = Path(task_data["input_path"])
            output_dir = Path(task_data.get("output_dir", "output"))
            
            # Ensure output directory exists
            ensure_directory(output_dir)
            
            # Process PDF
            result = self.pdf_processor.process_pdf(input_path)
            
            # Save result
            output_file = output_dir / f"{input_path.stem}_processed.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                task_type="pdf_processing",
                success=True,
                output_data={
                    "output_file": str(output_file),
                    "content": result.get("content_extracted", ""),
                    "metadata": result.get("metadata", {}),
                },
                processing_time=processing_time,
                quality_score=0.8,  # Default quality score
                metadata={"input_file": str(input_path)},
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"PDF processing failed: {str(e)}")
            
            return TaskResult(
                task_id=task_id,
                task_type="pdf_processing",
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                metadata={"input_file": str(task_data.get("input_path", "unknown"))},
            )


class ContentAnalysisPlugin(BasePlugin):
    """Plugin for content analysis tasks."""
    
    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Validate content analysis input."""
        required_fields = ["content"]
        return all(field in task_data for field in required_fields)
    
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute content analysis task."""
        task_id = task_data.get("task_id", "content_analysis")
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_input(task_data):
                raise ValidationError("Invalid input data for content analysis")
            
            content = task_data["content"]
            
            # Perform basic content analysis
            analysis = self._analyze_content(content)
            
            processing_time = time.time() - start_time
            
            return TaskResult(
                task_id=task_id,
                task_type="content_analysis",
                success=True,
                output_data=analysis,
                processing_time=processing_time,
                quality_score=analysis.get("quality_score", 0.7),
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Content analysis failed: {str(e)}")
            
            return TaskResult(
                task_id=task_id,
                task_type="content_analysis",
                success=False,
                error_message=str(e),
                processing_time=processing_time,
            )
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Perform basic content analysis."""
        # Basic text analysis
        word_count = len(content.split())
        char_count = len(content)
        paragraph_count = len([p for p in content.split("\n\n") if p.strip()])
        
        # Simple quality heuristics
        quality_score = min(1.0, max(0.3, word_count / 1000))
        
        return {
            "word_count": word_count,
            "character_count": char_count,
            "paragraph_count": paragraph_count,
            "quality_score": quality_score,
            "has_headings": "#" in content,
            "has_lists": any(marker in content for marker in ["- ", "* ", "1. "]),
            "estimated_reading_time": max(1, word_count // 200),  # minutes
        }


class TaskPlanner:
    """Plans and schedules tasks for the academic agent."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize task planner."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: List[Dict[str, Any]] = []
    
    def plan_workflow(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan a workflow based on input data."""
        tasks = []
        
        # Determine workflow based on input type
        if "pdf_files" in input_data:
            tasks.extend(self._plan_pdf_workflow(input_data["pdf_files"]))
        
        if "content" in input_data:
            tasks.extend(self._plan_content_workflow(input_data["content"]))
        
        return tasks
    
    def _plan_pdf_workflow(self, pdf_files: List[str]) -> List[Dict[str, Any]]:
        """Plan PDF processing workflow."""
        tasks = []
        
        for i, pdf_file in enumerate(pdf_files):
            # PDF processing task
            tasks.append({
                "task_id": f"pdf_processing_{i}",
                "task_type": "pdf_processing",
                "plugin": "PDFProcessorPlugin",
                "input_data": {
                    "input_path": pdf_file,
                    "output_dir": "output/processed",
                },
                "priority": 1,
                "dependencies": [],
            })
            
            # Content analysis task (depends on PDF processing)
            tasks.append({
                "task_id": f"content_analysis_{i}",
                "task_type": "content_analysis",
                "plugin": "ContentAnalysisPlugin",
                "input_data": {
                    "content": f"{{pdf_processing_{i}.output_data.content}}",
                },
                "priority": 2,
                "dependencies": [f"pdf_processing_{i}"],
            })
        
        return tasks
    
    def _plan_content_workflow(self, content: str) -> List[Dict[str, Any]]:
        """Plan content processing workflow."""
        return [{
            "task_id": "content_analysis_direct",
            "task_type": "content_analysis",
            "plugin": "ContentAnalysisPlugin",
            "input_data": {"content": content},
            "priority": 1,
            "dependencies": [],
        }]


class AcademicAgent:
    """
    Simplified Academic Agent that provides a streamlined interface for
    academic content processing with plugin-based extensibility.
    """
    
    def __init__(self, config_path: Optional[Path] = None, agent_id: str = "academic_agent"):
        """Initialize the academic agent.
        
        Args:
            config_path: Path to configuration file
            agent_id: Unique identifier for this agent instance
        """
        self.agent_id = agent_id
        self.logger = get_logger(agent_id)
        self.monitor = get_system_monitor()
        
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.get_default_config()
        
        # Initialize state
        self.state = AgentState(agent_id=agent_id)
        
        # Initialize plugins
        self.plugins: Dict[str, BasePlugin] = {}
        self._register_default_plugins()
        
        # Initialize task planner
        self.task_planner = TaskPlanner(self.config.get("task_planning", {}))
        
        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0,
        }
        
        self.logger.info(f"Academic Agent {agent_id} initialized successfully")
    
    def _register_default_plugins(self):
        """Register default plugins."""
        default_plugins = {
            "PDFProcessorPlugin": PDFProcessorPlugin,
            "ContentAnalysisPlugin": ContentAnalysisPlugin,
        }
        
        for plugin_name, plugin_class in default_plugins.items():
            plugin_config = self.config.get("plugins", {}).get(plugin_name, {})
            self.register_plugin(plugin_name, plugin_class(plugin_config))
    
    def register_plugin(self, name: str, plugin: BasePlugin):
        """Register a plugin with the agent.
        
        Args:
            name: Plugin name
            plugin: Plugin instance
        """
        self.plugins[name] = plugin
        self.logger.info(f"Registered plugin: {name}")
    
    def unregister_plugin(self, name: str):
        """Unregister a plugin.
        
        Args:
            name: Plugin name to remove
        """
        if name in self.plugins:
            del self.plugins[name]
            self.logger.info(f"Unregistered plugin: {name}")
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names."""
        return list(self.plugins.keys())
    
    async def execute_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute a single task.
        
        Args:
            task_data: Task configuration and data
            
        Returns:
            TaskResult with execution results
        """
        task_id = task_data.get("task_id", "unknown")
        plugin_name = task_data.get("plugin")
        
        if not plugin_name or plugin_name not in self.plugins:
            return TaskResult(
                task_id=task_id,
                task_type=task_data.get("task_type", "unknown"),
                success=False,
                error_message=f"Plugin {plugin_name} not found",
            )
        
        # Update state
        self.state.update_status("processing", task_id)
        
        try:
            # Execute task with plugin
            plugin = self.plugins[plugin_name]
            result = await plugin.execute(task_data.get("input_data", {}))
            result.task_id = task_id  # Ensure task ID is set
            
            # Update metrics
            self._update_metrics(result)
            
            # Mark task as completed
            self.state.complete_task(task_id, result.success)
            
            self.logger.info(f"Task {task_id} completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            
            result = TaskResult(
                task_id=task_id,
                task_type=task_data.get("task_type", "unknown"),
                success=False,
                error_message=str(e),
            )
            
            self._update_metrics(result)
            self.state.complete_task(task_id, False)
            
            return result
    
    async def execute_workflow(self, workflow_data: Dict[str, Any]) -> List[TaskResult]:
        """Execute a complete workflow.
        
        Args:
            workflow_data: Workflow configuration and input data
            
        Returns:
            List of TaskResult objects
        """
        self.logger.info("Starting workflow execution")
        
        # Plan tasks
        tasks = self.task_planner.plan_workflow(workflow_data)
        
        if not tasks:
            self.logger.warning("No tasks planned for workflow")
            return []
        
        self.logger.info(f"Planned {len(tasks)} tasks for execution")
        
        # Execute tasks in dependency order
        results = []
        completed_task_outputs = {}
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._sort_tasks_by_dependencies(tasks)
        
        for task in sorted_tasks:
            # Resolve task input data dependencies
            resolved_task = self._resolve_task_dependencies(task, completed_task_outputs)
            
            # Execute task
            result = await self.execute_task(resolved_task)
            results.append(result)
            
            # Store output for dependent tasks
            if result.success and result.output_data:
                completed_task_outputs[task["task_id"]] = result.output_data
        
        self.logger.info(f"Workflow completed with {len(results)} task results")
        return results
    
    def _sort_tasks_by_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort tasks by dependencies and priority."""
        # Simple topological sort
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        completed_task_ids = set()
        
        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = [
                task for task in remaining_tasks
                if all(dep in completed_task_ids for dep in task.get("dependencies", []))
            ]
            
            if not ready_tasks:
                # Break circular dependencies by taking the first task
                ready_tasks = [remaining_tasks[0]]
                self.logger.warning("Possible circular dependency detected")
            
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda x: x.get("priority", 0))
            
            # Add first ready task to sorted list
            task = ready_tasks[0]
            sorted_tasks.append(task)
            remaining_tasks.remove(task)
            completed_task_ids.add(task["task_id"])
        
        return sorted_tasks
    
    def _resolve_task_dependencies(
        self, task: Dict[str, Any], completed_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve task input data dependencies."""
        resolved_task = task.copy()
        
        # Simple dependency resolution - replace template strings
        input_data = resolved_task.get("input_data", {})
        for key, value in input_data.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                # Extract dependency reference
                dep_ref = value[1:-1]  # Remove braces
                if "." in dep_ref:
                    task_id, output_key = dep_ref.split(".", 1)
                    if task_id in completed_outputs:
                        # Navigate nested output data
                        output_value = completed_outputs[task_id]
                        for key_part in output_key.split("."):
                            if isinstance(output_value, dict) and key_part in output_value:
                                output_value = output_value[key_part]
                            else:
                                output_value = None
                                break
                        
                        if output_value is not None:
                            input_data[key] = output_value
        
        resolved_task["input_data"] = input_data
        return resolved_task
    
    def _update_metrics(self, result: TaskResult):
        """Update agent metrics based on task result."""
        if result.success:
            self.metrics["tasks_completed"] += 1
        else:
            self.metrics["tasks_failed"] += 1
        
        if result.processing_time:
            self.metrics["total_processing_time"] += result.processing_time
        
        if result.quality_score:
            # Update average quality score
            total_tasks = self.metrics["tasks_completed"] + self.metrics["tasks_failed"]
            current_avg = self.metrics["average_quality_score"]
            new_avg = ((current_avg * (total_tasks - 1)) + result.quality_score) / total_tasks
            self.metrics["average_quality_score"] = new_avg
    
    def save_state(self, state_file: Path):
        """Save agent state to file.
        
        Args:
            state_file: Path to save state file
        """
        try:
            ensure_directory(state_file.parent)
            
            state_data = {
                "agent_state": self.state.to_dict(),
                "metrics": self.metrics,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Agent state saved to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {str(e)}")
    
    def load_state(self, state_file: Path):
        """Load agent state from file.
        
        Args:
            state_file: Path to state file
        """
        try:
            if not state_file.exists():
                self.logger.warning(f"State file not found: {state_file}")
                return
            
            with open(state_file, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            
            # Restore agent state
            agent_state_data = state_data.get("agent_state", {})
            self.state.agent_id = agent_state_data.get("agent_id", self.agent_id)
            self.state.status = agent_state_data.get("status", "idle")
            self.state.current_task = agent_state_data.get("current_task")
            self.state.completed_tasks = agent_state_data.get("completed_tasks", [])
            self.state.failed_tasks = agent_state_data.get("failed_tasks", [])
            self.state.metrics = agent_state_data.get("metrics", {})
            
            # Restore metrics
            self.metrics.update(state_data.get("metrics", {}))
            
            self.logger.info(f"Agent state loaded from {state_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load agent state: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics.
        
        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_id": self.agent_id,
            "state": self.state.to_dict(),
            "metrics": self.metrics,
            "plugins": {name: plugin.get_plugin_info() for name, plugin in self.plugins.items()},
            "config": self.config,
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent."""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        
        # Update state
        self.state.update_status("shutdown")
        
        # Log final metrics
        self.logger.info(f"Final metrics: {self.metrics}")
        
        self.logger.info("Agent shutdown complete")


# Utility functions for creating and managing academic agents

def create_academic_agent(
    config_path: Optional[Path] = None,
    agent_id: str = "academic_agent"
) -> AcademicAgent:
    """Create and initialize an academic agent.
    
    Args:
        config_path: Optional path to configuration file
        agent_id: Unique identifier for the agent
        
    Returns:
        Initialized AcademicAgent instance
    """
    return AcademicAgent(config_path=config_path, agent_id=agent_id)


async def process_pdfs(
    agent: AcademicAgent,
    pdf_files: List[Union[str, Path]],
    output_dir: Optional[Path] = None
) -> List[TaskResult]:
    """Process a list of PDF files with the academic agent.
    
    Args:
        agent: Academic agent instance
        pdf_files: List of PDF file paths
        output_dir: Optional output directory
        
    Returns:
        List of TaskResult objects
    """
    workflow_data = {
        "pdf_files": [str(pdf) for pdf in pdf_files],
    }
    
    if output_dir:
        workflow_data["output_dir"] = str(output_dir)
    
    return await agent.execute_workflow(workflow_data)


async def analyze_content(
    agent: AcademicAgent,
    content: str
) -> TaskResult:
    """Analyze text content with the academic agent.
    
    Args:
        agent: Academic agent instance
        content: Text content to analyze
        
    Returns:
        TaskResult object
    """
    task_data = {
        "task_id": "content_analysis",
        "task_type": "content_analysis",
        "plugin": "ContentAnalysisPlugin",
        "input_data": {"content": content},
    }
    
    return await agent.execute_task(task_data)