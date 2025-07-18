#!/usr/bin/env python3
"""
Asynchronous Main Academic Agent
Task 15 Implementation - High-performance async agent orchestration

This module provides the async version of the main academic agent with:
- Asynchronous workflow coordination
- Parallel processing capabilities
- Advanced progress tracking
- Real-time monitoring integration
- Cancellation and timeout support
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging

from .base_agent import BaseAgent, AgentMessage
from .async_framework import (
    AsyncTask, TaskPriority, TaskStatus, WorkerPool, 
    AsyncCommunicationManager, AsyncProgressTracker, AsyncResourceManager,
    create_async_framework, async_retry, async_timeout
)

try:
    from ..monitoring.integration import get_monitoring_integration
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class AsyncMainAcademicAgent(BaseAgent):
    """Asynchronous main academic agent for orchestrating parallel workflows."""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__("async_main_academic_agent")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize async framework components
        (
            self.worker_pool,
            self.comm_manager, 
            self.progress_tracker,
            self.resource_manager
        ) = create_async_framework(
            worker_pool_size=self.config.get("worker_pool_size", 4),
            task_queue_size=self.config.get("task_queue_size", 1000)
        )
        
        # Agent registry for async communication
        self.agent_registry: Dict[str, BaseAgent] = {}
        
        # Workflow tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.completed_workflows: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.performance_metrics = {
            "workflows_completed": 0,
            "workflows_failed": 0,
            "total_processing_time": 0.0,
            "average_workflow_time": 0.0,
            "peak_concurrent_tasks": 0,
            "total_tasks_processed": 0
        }
        
        # Monitoring integration
        self.monitoring = None
        if MONITORING_AVAILABLE:
            try:
                self.monitoring = get_monitoring_integration()
            except Exception as e:
                self.logger.warning(f"Could not initialize monitoring: {e}")
        
        self.logger.info("Async main academic agent initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with async-specific defaults."""
        default_config = {
            "worker_pool_size": 4,
            "task_queue_size": 1000,
            "max_concurrent_workflows": 3,
            "workflow_timeout": 3600,  # 1 hour
            "task_timeout": 600,  # 10 minutes
            "quality_threshold": 0.7,
            "max_improvement_cycles": 3,
            "progress_update_interval": 5.0,
            "resource_limits": {
                "pdf_processing": 2,
                "memory_intensive": 1,
                "api_calls": 5
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        
        return default_config
    
    async def start(self):
        """Start the async agent and all components."""
        try:
            # Start worker pool
            await self.worker_pool.start()
            
            # Start communication manager
            await self.comm_manager.start()
            
            # Start monitoring if available
            if self.monitoring:
                self.monitoring.start_monitoring()
            
            self.logger.info("Async main academic agent started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start async agent: {e}")
            raise
    
    async def stop(self):
        """Stop the async agent and cleanup resources."""
        try:
            # Stop communication manager
            await self.comm_manager.stop()
            
            # Stop worker pool
            await self.worker_pool.stop()
            
            # Stop monitoring if available
            if self.monitoring:
                self.monitoring.stop_monitoring()
            
            self.logger.info("Async main academic agent stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping async agent: {e}")
    
    async def register_agent(self, agent_id: str, agent: BaseAgent):
        """Register an agent for async communication."""
        self.agent_registry[agent_id] = agent
        await self.comm_manager.register_agent(agent_id, agent)
        self.logger.debug(f"Registered agent {agent_id}")
    
    @async_timeout(3600)  # 1 hour timeout
    async def process_workflow_async(
        self, 
        workflow_id: str,
        input_files: List[str],
        workflow_config: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """Process a complete workflow asynchronously with parallel execution."""
        
        start_time = time.time()
        workflow_config = workflow_config or {}
        
        # Initialize workflow tracking
        self.active_workflows[workflow_id] = {
            "status": "running",
            "start_time": start_time,
            "input_files": input_files,
            "stages": {},
            "tasks": [],
            "progress": 0.0
        }
        
        # Start progress tracking
        total_stages = 5  # ingestion, outline, notes, quality, improvement
        self.progress_tracker.start_operation(
            workflow_id, 
            total_stages, 
            f"Processing workflow with {len(input_files)} files"
        )
        
        if progress_callback:
            self.progress_tracker.add_callback(workflow_id, progress_callback)
        
        try:
            # Process stages in parallel where possible
            results = await self._execute_workflow_stages(workflow_id, input_files, workflow_config)
            
            # Mark workflow as completed
            processing_time = time.time() - start_time
            await self._complete_workflow(workflow_id, True, processing_time, results)
            
            return results
            
        except Exception as e:
            # Mark workflow as failed
            processing_time = time.time() - start_time
            await self._complete_workflow(workflow_id, False, processing_time, None, str(e))
            raise
    
    async def _execute_workflow_stages(
        self, 
        workflow_id: str, 
        input_files: List[str],
        workflow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow stages with optimal parallelization."""
        
        results = {}
        workflow = self.active_workflows[workflow_id]
        
        # Stage 1: Parallel PDF Ingestion
        self.logger.info(f"Starting parallel ingestion for workflow {workflow_id}")
        ingestion_tasks = []
        
        for i, file_path in enumerate(input_files):
            task = AsyncTask(
                coroutine=self._process_single_pdf_async(file_path, workflow_id),
                priority=TaskPriority.HIGH,
                timeout=self.config.get("task_timeout", 600),
                context={"workflow_id": workflow_id, "file_index": i}
            )
            ingestion_tasks.append(task)
            workflow["tasks"].append(task.task_id)
            await self.worker_pool.submit_task(task)
        
        # Wait for all ingestion tasks to complete
        ingestion_results = []
        for task in ingestion_tasks:
            try:
                result = await self.worker_pool.wait_for_task(task.task_id, timeout=600)
                ingestion_results.append(result)
            except Exception as e:
                self.logger.error(f"Ingestion task failed: {e}")
                ingestion_results.append(None)
        
        # Filter successful results
        successful_ingestions = [r for r in ingestion_results if r is not None]
        results["ingestion"] = successful_ingestions
        
        self.progress_tracker.update_progress(workflow_id, 1, "PDF ingestion completed")
        
        if not successful_ingestions:
            raise RuntimeError("All PDF ingestion tasks failed")
        
        # Stage 2: Parallel Content Analysis
        self.logger.info(f"Starting parallel analysis for workflow {workflow_id}")
        analysis_tasks = []
        
        for i, ingestion_result in enumerate(successful_ingestions):
            task = AsyncTask(
                coroutine=self._analyze_content_async(ingestion_result, workflow_id),
                priority=TaskPriority.NORMAL,
                timeout=self.config.get("task_timeout", 600),
                context={"workflow_id": workflow_id, "content_index": i}
            )
            analysis_tasks.append(task)
            workflow["tasks"].append(task.task_id)
            await self.worker_pool.submit_task(task)
        
        # Wait for analysis completion
        analysis_results = []
        for task in analysis_tasks:
            try:
                result = await self.worker_pool.wait_for_task(task.task_id, timeout=600)
                analysis_results.append(result)
            except Exception as e:
                self.logger.error(f"Analysis task failed: {e}")
                analysis_results.append(None)
        
        results["analysis"] = [r for r in analysis_results if r is not None]
        self.progress_tracker.update_progress(workflow_id, 2, "Content analysis completed")
        
        # Stage 3: Outline Generation (can be partially parallel)
        self.logger.info(f"Starting outline generation for workflow {workflow_id}")
        outline_result = await self._generate_outlines_async(results["analysis"], workflow_id)
        results["outlines"] = outline_result
        
        self.progress_tracker.update_progress(workflow_id, 3, "Outline generation completed")
        
        # Stage 4: Notes Generation with Dependencies
        self.logger.info(f"Starting notes generation for workflow {workflow_id}")
        notes_result = await self._generate_notes_async(
            results["analysis"], 
            results["outlines"], 
            workflow_id
        )
        results["notes"] = notes_result
        
        self.progress_tracker.update_progress(workflow_id, 4, "Notes generation completed")
        
        # Stage 5: Quality Assessment and Improvement
        self.logger.info(f"Starting quality assessment for workflow {workflow_id}")
        quality_result = await self._assess_and_improve_quality_async(
            results["notes"], 
            workflow_id,
            workflow_config.get("quality_threshold", self.config.get("quality_threshold", 0.7))
        )
        results["quality"] = quality_result
        
        self.progress_tracker.update_progress(workflow_id, 5, "Quality assessment completed")
        
        return results
    
    async def _process_single_pdf_async(self, file_path: str, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Process a single PDF file asynchronously."""
        
        async with self.resource_manager.acquire_resource("pdf_processing"):
            try:
                # This would use the async PDF processor
                # For now, simulate the processing
                await asyncio.sleep(1)  # Simulate processing time
                
                result = {
                    "source_file": file_path,
                    "markdown_path": f"{Path(file_path).stem}.md",
                    "processing_time": 1.0,
                    "pages_processed": 10,
                    "success": True
                }
                
                self.logger.debug(f"Processed PDF: {file_path}")
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to process PDF {file_path}: {e}")
                return None
    
    async def _analyze_content_async(self, ingestion_result: Dict[str, Any], workflow_id: str) -> Optional[Dict[str, Any]]:
        """Analyze content asynchronously."""
        
        try:
            # Simulate content analysis
            await asyncio.sleep(0.5)
            
            result = {
                "source": ingestion_result["source_file"],
                "topics": ["Topic 1", "Topic 2", "Topic 3"],
                "key_concepts": ["Concept A", "Concept B", "Concept C"],
                "structure": {"sections": 5, "complexity": "medium"},
                "success": True
            }
            
            self.logger.debug(f"Analyzed content from: {ingestion_result['source_file']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze content: {e}")
            return None
    
    async def _generate_outlines_async(self, analysis_results: List[Dict[str, Any]], workflow_id: str) -> Dict[str, Any]:
        """Generate outlines asynchronously."""
        
        try:
            # Create outline generation task
            task = AsyncTask(
                coroutine=self._create_consolidated_outline(analysis_results),
                priority=TaskPriority.NORMAL,
                timeout=300,
                context={"workflow_id": workflow_id, "stage": "outline"}
            )
            
            await self.worker_pool.submit_task(task)
            result = await self.worker_pool.wait_for_task(task.task_id, timeout=300)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate outlines: {e}")
            raise
    
    async def _create_consolidated_outline(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a consolidated outline from analysis results."""
        
        # Simulate outline creation
        await asyncio.sleep(1)
        
        return {
            "consolidated_outline": "# Main Topics\n## Section 1\n## Section 2",
            "source_count": len(analysis_results),
            "structure_depth": 3,
            "success": True
        }
    
    async def _generate_notes_async(
        self, 
        analysis_results: List[Dict[str, Any]], 
        outline_result: Dict[str, Any],
        workflow_id: str
    ) -> Dict[str, Any]:
        """Generate notes asynchronously."""
        
        try:
            # Create notes generation task
            task = AsyncTask(
                coroutine=self._create_comprehensive_notes(analysis_results, outline_result),
                priority=TaskPriority.NORMAL,
                timeout=600,
                context={"workflow_id": workflow_id, "stage": "notes"}
            )
            
            await self.worker_pool.submit_task(task)
            result = await self.worker_pool.wait_for_task(task.task_id, timeout=600)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate notes: {e}")
            raise
    
    async def _create_comprehensive_notes(
        self, 
        analysis_results: List[Dict[str, Any]], 
        outline_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive notes from analysis and outline."""
        
        # Simulate notes creation
        await asyncio.sleep(2)
        
        return {
            "comprehensive_notes": "# Comprehensive Notes\n\nDetailed content...",
            "word_count": 5000,
            "sections": 8,
            "quality_indicators": {"completeness": 0.9, "coherence": 0.8},
            "success": True
        }
    
    @async_retry(max_retries=2, delay=1.0)
    async def _assess_and_improve_quality_async(
        self, 
        notes_result: Dict[str, Any], 
        workflow_id: str,
        quality_threshold: float
    ) -> Dict[str, Any]:
        """Assess and improve quality asynchronously with retry logic."""
        
        try:
            # Quality assessment task
            assessment_task = AsyncTask(
                coroutine=self._assess_quality(notes_result),
                priority=TaskPriority.HIGH,
                timeout=300,
                context={"workflow_id": workflow_id, "stage": "quality_assessment"}
            )
            
            await self.worker_pool.submit_task(assessment_task)
            assessment_result = await self.worker_pool.wait_for_task(assessment_task.task_id, timeout=300)
            
            # Check if improvement is needed
            quality_score = assessment_result.get("overall_score", 0.0)
            
            if quality_score < quality_threshold:
                self.logger.info(f"Quality score {quality_score:.2f} below threshold {quality_threshold}, starting improvement")
                
                # Improvement task
                improvement_task = AsyncTask(
                    coroutine=self._improve_content(notes_result, assessment_result),
                    priority=TaskPriority.HIGH,
                    timeout=600,
                    context={"workflow_id": workflow_id, "stage": "improvement"}
                )
                
                await self.worker_pool.submit_task(improvement_task)
                improvement_result = await self.worker_pool.wait_for_task(improvement_task.task_id, timeout=600)
                
                # Re-assess improved content
                reassessment_task = AsyncTask(
                    coroutine=self._assess_quality(improvement_result),
                    priority=TaskPriority.HIGH,
                    timeout=300,
                    context={"workflow_id": workflow_id, "stage": "reassessment"}
                )
                
                await self.worker_pool.submit_task(reassessment_task)
                final_assessment = await self.worker_pool.wait_for_task(reassessment_task.task_id, timeout=300)
                
                return {
                    "initial_assessment": assessment_result,
                    "improvement_applied": True,
                    "improved_content": improvement_result,
                    "final_assessment": final_assessment,
                    "quality_improved": final_assessment.get("overall_score", 0.0) > quality_score
                }
            else:
                return {
                    "assessment": assessment_result,
                    "improvement_applied": False,
                    "quality_threshold_met": True
                }
                
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            raise
    
    async def _assess_quality(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Assess content quality."""
        
        # Simulate quality assessment
        await asyncio.sleep(1)
        
        return {
            "overall_score": 0.85,
            "completeness": 0.9,
            "coherence": 0.8,
            "accuracy": 0.85,
            "suggestions": ["Improve section transitions", "Add more examples"],
            "success": True
        }
    
    async def _improve_content(self, content: Dict[str, Any], assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Improve content based on assessment."""
        
        # Simulate content improvement
        await asyncio.sleep(2)
        
        return {
            "improved_notes": "# Improved Comprehensive Notes\n\nEnhanced content...",
            "improvements_applied": assessment.get("suggestions", []),
            "word_count": content.get("word_count", 0) + 500,
            "success": True
        }
    
    async def _complete_workflow(
        self, 
        workflow_id: str, 
        success: bool, 
        processing_time: float,
        results: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Complete workflow and update metrics."""
        
        workflow = self.active_workflows.pop(workflow_id, {})
        workflow.update({
            "status": "completed" if success else "failed",
            "end_time": time.time(),
            "processing_time": processing_time,
            "results": results,
            "error_message": error_message
        })
        
        self.completed_workflows.append(workflow)
        
        # Update performance metrics
        if success:
            self.performance_metrics["workflows_completed"] += 1
            self.progress_tracker.complete_operation(workflow_id, True, "Workflow completed successfully")
        else:
            self.performance_metrics["workflows_failed"] += 1
            self.progress_tracker.complete_operation(workflow_id, False, f"Workflow failed: {error_message}")
        
        # Update timing metrics
        self.performance_metrics["total_processing_time"] += processing_time
        total_workflows = (
            self.performance_metrics["workflows_completed"] + 
            self.performance_metrics["workflows_failed"]
        )
        
        if total_workflows > 0:
            self.performance_metrics["average_workflow_time"] = (
                self.performance_metrics["total_processing_time"] / total_workflows
            )
        
        # Track concurrent tasks peak
        current_tasks = len(workflow.get("tasks", []))
        if current_tasks > self.performance_metrics["peak_concurrent_tasks"]:
            self.performance_metrics["peak_concurrent_tasks"] = current_tasks
        
        self.performance_metrics["total_tasks_processed"] += current_tasks
        
        # Record monitoring metrics
        if self.monitoring:
            try:
                with self.monitoring.track_operation("workflow", workflow_id):
                    pass  # Context manager handles the timing
            except:
                pass  # Ignore monitoring errors
        
        self.logger.info(
            f"Workflow {workflow_id} {'completed' if success else 'failed'} "
            f"in {processing_time:.2f}s"
        )
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        # Cancel all tasks in the workflow
        cancelled_count = 0
        for task_id in workflow.get("tasks", []):
            # Cancel tasks in worker pool
            # This would need to be implemented in the worker pool
            cancelled_count += 1
        
        # Mark workflow as cancelled
        await self._complete_workflow(
            workflow_id, 
            False, 
            time.time() - workflow.get("start_time", time.time()),
            None,
            f"Workflow cancelled ({cancelled_count} tasks cancelled)"
        )
        
        self.logger.info(f"Cancelled workflow {workflow_id} with {cancelled_count} tasks")
        return True
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow."""
        
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id].copy()
            workflow["progress"] = self.progress_tracker.get_operation_status(workflow_id)
            return workflow
        
        # Check completed workflows
        for workflow in self.completed_workflows:
            if workflow.get("workflow_id") == workflow_id:
                return workflow
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            "agent_info": {
                "agent_id": self.agent_id,
                "status": "running" if self.worker_pool.is_running else "stopped"
            },
            "worker_pool": self.worker_pool.get_pool_stats(),
            "resource_usage": self.resource_manager.get_resource_stats(),
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "performance_metrics": self.performance_metrics,
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check worker pool
        try:
            pool_stats = self.worker_pool.get_pool_stats()
            health_status["components"]["worker_pool"] = {
                "status": "healthy" if self.worker_pool.is_running else "stopped",
                "workers_active": pool_stats["pool_size"],
                "tasks_completed": pool_stats["totals"]["tasks_completed"]
            }
        except Exception as e:
            health_status["components"]["worker_pool"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check communication manager
        try:
            health_status["components"]["communication"] = {
                "status": "healthy" if self.comm_manager.is_running else "stopped",
                "registered_agents": len(self.agent_registry)
            }
        except Exception as e:
            health_status["components"]["communication"] = {
                "status": "error", 
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check resource usage
        try:
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent()
            
            resource_status = "healthy"
            if memory_percent > 90 or cpu_percent > 90:
                resource_status = "critical"
                health_status["status"] = "critical"
            elif memory_percent > 75 or cpu_percent > 75:
                resource_status = "warning"
                if health_status["status"] == "healthy":
                    health_status["status"] = "warning"
            
            health_status["components"]["resources"] = {
                "status": resource_status,
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent
            }
        except Exception as e:
            health_status["components"]["resources"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status


# Utility functions for async workflow management
async def create_and_start_async_agent(config_path: Optional[str] = None) -> AsyncMainAcademicAgent:
    """Create and start an async main academic agent."""
    agent = AsyncMainAcademicAgent(config_path)
    await agent.start()
    return agent


async def run_batch_workflow_async(
    agent: AsyncMainAcademicAgent,
    input_batches: List[List[str]],
    max_concurrent: int = 3,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    """Run multiple workflows concurrently with limited concurrency."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    async def process_batch(batch_index: int, files: List[str]):
        async with semaphore:
            workflow_id = f"batch_{batch_index}_{int(time.time())}"
            try:
                result = await agent.process_workflow_async(workflow_id, files)
                if progress_callback:
                    progress_callback(batch_index + 1, len(input_batches))
                return {"batch_index": batch_index, "result": result, "success": True}
            except Exception as e:
                agent.logger.error(f"Batch {batch_index} failed: {e}")
                return {"batch_index": batch_index, "error": str(e), "success": False}
    
    # Create tasks for all batches
    tasks = [
        process_batch(i, files) 
        for i, files in enumerate(input_batches)
    ]
    
    # Wait for all batches to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [r for r in results if isinstance(r, dict)]


# Main async entry point
async def main():
    """Main async entry point for testing."""
    
    # Create and start agent
    agent = await create_and_start_async_agent()
    
    try:
        # Example workflow
        test_files = ["test1.pdf", "test2.pdf", "test3.pdf"]
        
        def progress_callback(progress: float, message: str):
            print(f"Progress: {progress:.1%} - {message}")
        
        workflow_id = f"test_workflow_{int(time.time())}"
        result = await agent.process_workflow_async(
            workflow_id, 
            test_files,
            progress_callback=progress_callback
        )
        
        print(f"Workflow completed: {result}")
        
        # Print system status
        status = agent.get_system_status()
        print(f"System status: {json.dumps(status, indent=2)}")
        
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())