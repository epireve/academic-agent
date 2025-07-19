#!/usr/bin/env python3
"""
Asynchronous Processing Framework for Academic Agent System
Task 15 Implementation - Core async/await patterns and task management

This module provides the foundational async processing capabilities including:
- Task queue system for background processing
- Worker pool management for parallel execution  
- Progress tracking and cancellation support
- Async message passing and communication
- Performance monitoring integration
"""

import asyncio
import time
import logging
import uuid
import weakref
from datetime import datetime, timedelta
from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Set, Union, 
    TypeVar, Generic, Coroutine, Tuple
)
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from ...src.agents.base_agent import AgentMessage, BaseAgent


T = TypeVar('T')


class TaskStatus(Enum):
    """Task execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskMetrics:
    """Metrics for tracking task performance."""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_time: float = 0.0
    worker_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class AsyncTask(Generic[T]):
    """Asynchronous task wrapper with comprehensive tracking."""
    
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    coroutine: Coroutine[Any, Any, T] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[T] = None
    error: Optional[Exception] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    timeout: Optional[float] = None
    retry_max: int = 3
    retry_delay: float = 1.0
    cancellation_token: Optional[asyncio.Event] = None
    progress_callback: Optional[Callable[[float], None]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.cancellation_token is None:
            self.cancellation_token = asyncio.Event()
    
    def cancel(self) -> bool:
        """Cancel the task."""
        if self.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            self.status = TaskStatus.CANCELLED
            self.cancellation_token.set()
            return True
        return False
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task dependencies are satisfied."""
        return self.dependencies.issubset(completed_tasks)
    
    def update_progress(self, progress: float):
        """Update task progress."""
        if self.progress_callback:
            self.progress_callback(progress)
    
    def start_execution(self, worker_id: str):
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.metrics.started_at = datetime.now()
        self.metrics.worker_id = worker_id
    
    def complete_execution(self, result: T = None, error: Exception = None):
        """Mark task as completed."""
        self.metrics.completed_at = datetime.now()
        if self.metrics.started_at:
            self.metrics.duration = (
                self.metrics.completed_at - self.metrics.started_at
            ).total_seconds()
        
        if error:
            self.status = TaskStatus.FAILED
            self.error = error
            self.metrics.error_message = str(error)
        else:
            self.status = TaskStatus.COMPLETED
            self.result = result


class TaskQueue:
    """Priority-based async task queue with dependency management."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queue: List[AsyncTask] = []
        self._pending_tasks: Dict[str, AsyncTask] = {}
        self._completed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger("task_queue")
    
    async def enqueue(self, task: AsyncTask) -> bool:
        """Add task to queue with dependency checking."""
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            
            # Add to pending tasks
            self._pending_tasks[task.task_id] = task
            
            # Insert in priority order
            insert_pos = 0
            for i, existing_task in enumerate(self._queue):
                if task.priority.value > existing_task.priority.value:
                    insert_pos = i
                    break
                insert_pos = i + 1
            
            self._queue.insert(insert_pos, task)
            self.logger.debug(f"Enqueued task {task.task_id} with priority {task.priority}")
            return True
    
    async def dequeue(self) -> Optional[AsyncTask]:
        """Get next ready task from queue."""
        async with self._lock:
            for i, task in enumerate(self._queue):
                if task.is_ready(self._completed_tasks):
                    return self._queue.pop(i)
            return None
    
    async def mark_completed(self, task_id: str):
        """Mark task as completed and remove from pending."""
        async with self._lock:
            self._completed_tasks.add(task_id)
            self._pending_tasks.pop(task_id, None)
    
    async def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        async with self._lock:
            return len(self._queue)
    
    async def get_ready_count(self) -> int:
        """Get count of ready-to-execute tasks."""
        async with self._lock:
            ready_count = 0
            for task in self._queue:
                if task.is_ready(self._completed_tasks):
                    ready_count += 1
            return ready_count


class AsyncWorker:
    """Asynchronous worker for executing tasks."""
    
    def __init__(self, worker_id: str, task_queue: TaskQueue):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.current_task: Optional[AsyncTask] = None
        self.is_running = False
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0
        self.logger = logging.getLogger(f"worker_{worker_id}")
    
    async def start(self):
        """Start the worker loop."""
        self.is_running = True
        self.logger.info(f"Worker {self.worker_id} started")
        
        while self.is_running:
            try:
                # Get next task
                task = await self.task_queue.dequeue()
                if not task:
                    await asyncio.sleep(0.1)  # Brief pause when no tasks
                    continue
                
                # Execute task
                await self._execute_task(task)
                
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(1)  # Error recovery pause
    
    async def stop(self):
        """Stop the worker."""
        self.is_running = False
        if self.current_task:
            self.current_task.cancel()
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    async def _execute_task(self, task: AsyncTask):
        """Execute a single task with monitoring."""
        self.current_task = task
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Start task execution
            task.start_execution(self.worker_id)
            self.logger.debug(f"Executing task {task.task_id}")
            
            # Execute with timeout if specified
            if task.timeout:
                result = await asyncio.wait_for(
                    task.coroutine, 
                    timeout=task.timeout
                )
            else:
                result = await task.coroutine
            
            # Check for cancellation
            if task.cancellation_token.is_set():
                task.status = TaskStatus.CANCELLED
                return
            
            # Complete successfully
            task.complete_execution(result=result)
            await self.task_queue.mark_completed(task.task_id)
            self.tasks_completed += 1
            
            self.logger.debug(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.complete_execution(error=Exception("Task timeout"))
            self.tasks_failed += 1
            self.logger.warning(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.complete_execution(error=e)
            self.tasks_failed += 1
            self.logger.error(f"Task {task.task_id} failed: {e}")
            
        finally:
            # Update metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            task.metrics.cpu_time = end_time - start_time
            task.metrics.memory_usage_mb = max(0, end_memory - start_memory)
            
            self.total_processing_time += task.metrics.cpu_time
            self.current_task = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_processing_time": self.total_processing_time,
            "average_task_time": (
                self.total_processing_time / max(1, self.tasks_completed)
            ),
            "current_task": (
                self.current_task.task_id if self.current_task else None
            )
        }


class WorkerPool:
    """Pool of async workers for parallel task execution."""
    
    def __init__(self, pool_size: int = 4, task_queue: Optional[TaskQueue] = None):
        self.pool_size = pool_size
        self.task_queue = task_queue or TaskQueue()
        self.workers: List[AsyncWorker] = []
        self.worker_tasks: List[asyncio.Task] = []
        self.is_running = False
        self.logger = logging.getLogger("worker_pool")
    
    async def start(self):
        """Start all workers in the pool."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create and start workers
        for i in range(self.pool_size):
            worker = AsyncWorker(f"worker_{i}", self.task_queue)
            self.workers.append(worker)
            
            # Start worker in background
            worker_task = asyncio.create_task(worker.start())
            self.worker_tasks.append(worker_task)
        
        self.logger.info(f"Worker pool started with {self.pool_size} workers")
    
    async def stop(self, timeout: float = 30.0):
        """Stop all workers gracefully."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop all workers
        stop_tasks = [worker.stop() for worker in self.workers]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.worker_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Some workers did not stop within timeout")
        
        self.workers.clear()
        self.worker_tasks.clear()
        
        self.logger.info("Worker pool stopped")
    
    async def submit_task(self, task: AsyncTask) -> str:
        """Submit a task to the pool."""
        success = await self.task_queue.enqueue(task)
        if success:
            self.logger.debug(f"Task {task.task_id} submitted to pool")
            return task.task_id
        else:
            raise RuntimeError("Task queue is full")
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for a specific task to complete."""
        start_time = time.time()
        
        while True:
            # Check if task is completed
            if task_id in self.task_queue._completed_tasks:
                # Find the completed task
                for worker in self.workers:
                    if (worker.current_task and 
                        worker.current_task.task_id == task_id):
                        if worker.current_task.status == TaskStatus.COMPLETED:
                            return worker.current_task.result
                        else:
                            raise RuntimeError(f"Task failed: {worker.current_task.error}")
                
                # Task completed but not in current tasks (already cleaned up)
                return None
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} did not complete within timeout")
            
            await asyncio.sleep(0.1)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        worker_stats = [worker.get_stats() for worker in self.workers]
        
        total_completed = sum(w["tasks_completed"] for w in worker_stats)
        total_failed = sum(w["tasks_failed"] for w in worker_stats)
        total_processing_time = sum(w["total_processing_time"] for w in worker_stats)
        
        return {
            "pool_size": self.pool_size,
            "is_running": self.is_running,
            "workers": worker_stats,
            "totals": {
                "tasks_completed": total_completed,
                "tasks_failed": total_failed,
                "total_processing_time": total_processing_time,
                "average_task_time": (
                    total_processing_time / max(1, total_completed)
                )
            },
            "queue_stats": {
                "pending_tasks": len(self.task_queue._pending_tasks),
                "completed_tasks": len(self.task_queue._completed_tasks)
            }
        }


class AsyncCommunicationManager:
    """Async version of the communication manager for inter-agent messaging."""
    
    def __init__(self):
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.active_conversations: Dict[str, List[AgentMessage]] = {}
        self.logger = logging.getLogger("async_comm_manager")
        self._processing_task: Optional[asyncio.Task] = None
        self.is_running = False
    
    async def start(self):
        """Start the async communication manager."""
        if self.is_running:
            return
        
        self.is_running = True
        self._processing_task = asyncio.create_task(self._process_messages())
        self.logger.info("Async communication manager started")
    
    async def stop(self):
        """Stop the communication manager."""
        self.is_running = False
        
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Async communication manager stopped")
    
    async def register_agent(self, agent_id: str, agent: BaseAgent):
        """Register an agent for communication."""
        self.agent_registry[agent_id] = agent
        self.logger.debug(f"Registered agent {agent_id}")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message asynchronously."""
        if not message.validate():
            return False
        
        await self.message_queue.put(message)
        
        # Track conversation
        conversation_id = f"{message.sender}_{message.recipient}"
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = []
        self.active_conversations[conversation_id].append(message)
        
        return True
    
    async def _process_messages(self):
        """Process messages from the queue."""
        while self.is_running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Deliver message
                await self._deliver_message(message)
                
            except asyncio.TimeoutError:
                continue  # No messages to process
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _deliver_message(self, message: AgentMessage):
        """Deliver message to recipient agent."""
        recipient = self.agent_registry.get(message.recipient)
        if not recipient:
            self.logger.warning(f"Recipient {message.recipient} not found")
            return
        
        try:
            # Call the agent's receive_message method
            # This could be made async in the future
            await asyncio.get_event_loop().run_in_executor(
                None, recipient.receive_message, message
            )
            self.logger.debug(f"Delivered message from {message.sender} to {message.recipient}")
            
        except Exception as e:
            self.logger.error(f"Failed to deliver message: {e}")


class AsyncProgressTracker:
    """Track progress of async operations with callbacks."""
    
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger("progress_tracker")
    
    def start_operation(self, operation_id: str, total_steps: int, description: str = ""):
        """Start tracking an operation."""
        self.operations[operation_id] = {
            "total_steps": total_steps,
            "current_step": 0,
            "description": description,
            "started_at": datetime.now(),
            "progress": 0.0,
            "status": "running"
        }
        self.logger.debug(f"Started tracking operation {operation_id}")
    
    def update_progress(self, operation_id: str, step: int, message: str = ""):
        """Update operation progress."""
        if operation_id not in self.operations:
            return
        
        operation = self.operations[operation_id]
        operation["current_step"] = step
        operation["progress"] = step / operation["total_steps"]
        operation["last_message"] = message
        operation["updated_at"] = datetime.now()
        
        # Call callbacks
        callbacks = self.callbacks.get(operation_id, [])
        for callback in callbacks:
            try:
                callback(operation["progress"], message)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    def complete_operation(self, operation_id: str, success: bool = True, message: str = ""):
        """Mark operation as completed."""
        if operation_id not in self.operations:
            return
        
        operation = self.operations[operation_id]
        operation["status"] = "completed" if success else "failed"
        operation["completed_at"] = datetime.now()
        operation["final_message"] = message
        operation["progress"] = 1.0 if success else operation["progress"]
        
        # Final callback
        callbacks = self.callbacks.get(operation_id, [])
        for callback in callbacks:
            try:
                callback(operation["progress"], message)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    def add_callback(self, operation_id: str, callback: Callable):
        """Add progress callback for an operation."""
        if operation_id not in self.callbacks:
            self.callbacks[operation_id] = []
        self.callbacks[operation_id].append(callback)
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an operation."""
        return self.operations.get(operation_id)


class AsyncResourceManager:
    """Manage shared resources for async operations."""
    
    def __init__(self):
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.resource_usage: Dict[str, int] = {}
        self.logger = logging.getLogger("resource_manager")
    
    def create_semaphore(self, name: str, limit: int):
        """Create a named semaphore for resource limiting."""
        self.semaphores[name] = asyncio.Semaphore(limit)
        self.resource_usage[name] = 0
        self.logger.debug(f"Created semaphore '{name}' with limit {limit}")
    
    def create_lock(self, name: str):
        """Create a named lock for resource synchronization."""
        self.locks[name] = asyncio.Lock()
        self.logger.debug(f"Created lock '{name}'")
    
    @asynccontextmanager
    async def acquire_resource(self, semaphore_name: str):
        """Acquire a resource with automatic release."""
        if semaphore_name not in self.semaphores:
            raise ValueError(f"Semaphore '{semaphore_name}' not found")
        
        semaphore = self.semaphores[semaphore_name]
        await semaphore.acquire()
        self.resource_usage[semaphore_name] += 1
        
        try:
            yield
        finally:
            semaphore.release()
            self.resource_usage[semaphore_name] -= 1
    
    @asynccontextmanager
    async def acquire_lock(self, lock_name: str):
        """Acquire a lock with automatic release."""
        if lock_name not in self.locks:
            raise ValueError(f"Lock '{lock_name}' not found")
        
        lock = self.locks[lock_name]
        async with lock:
            yield
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource usage statistics."""
        return {
            "semaphores": {
                name: {
                    "limit": sem._value + self.resource_usage.get(name, 0),
                    "available": sem._value,
                    "in_use": self.resource_usage.get(name, 0)
                }
                for name, sem in self.semaphores.items()
            },
            "locks": {
                name: {
                    "locked": lock.locked()
                }
                for name, lock in self.locks.items()
            }
        }


# Factory functions for easy initialization
def create_async_framework(
    worker_pool_size: int = 4,
    task_queue_size: int = 1000,
    enable_monitoring: bool = True
) -> Tuple[WorkerPool, AsyncCommunicationManager, AsyncProgressTracker, AsyncResourceManager]:
    """Create and configure the async framework components."""
    
    # Create task queue and worker pool
    task_queue = TaskQueue(max_size=task_queue_size)
    worker_pool = WorkerPool(pool_size=worker_pool_size, task_queue=task_queue)
    
    # Create communication manager
    comm_manager = AsyncCommunicationManager()
    
    # Create progress tracker
    progress_tracker = AsyncProgressTracker()
    
    # Create resource manager with default resources
    resource_manager = AsyncResourceManager()
    
    # Set up default resource limits
    resource_manager.create_semaphore("pdf_processing", 2)  # Limit concurrent PDF processing
    resource_manager.create_semaphore("memory_intensive", 1)  # Limit memory-intensive operations
    resource_manager.create_semaphore("api_calls", 5)  # Limit API calls
    
    # Create locks for shared resources
    resource_manager.create_lock("file_operations")
    resource_manager.create_lock("model_access")
    
    return worker_pool, comm_manager, progress_tracker, resource_manager


# Utility decorators for async operations
def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for automatic retry of async operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        await asyncio.sleep(wait_time)
                    else:
                        raise last_exception
            
        return wrapper
    return decorator


def async_timeout(timeout_seconds: float):
    """Decorator to add timeout to async operations."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=timeout_seconds
            )
        return wrapper
    return decorator


def async_rate_limit(calls_per_second: float):
    """Decorator to rate limit async operations."""
    min_interval = 1.0 / calls_per_second
    last_call_time = {}
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            now = time.time()
            func_key = f"{func.__name__}_{id(func)}"
            
            if func_key in last_call_time:
                elapsed = now - last_call_time[func_key]
                if elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)
            
            last_call_time[func_key] = time.time()
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator