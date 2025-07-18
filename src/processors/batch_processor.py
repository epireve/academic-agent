#!/usr/bin/env python3
"""
Batch Processing System for Academic Agent v2
Task 27 Implementation - Handle multiple PDFs simultaneously with optimal resource usage

This module provides comprehensive batch processing capabilities including:
- Job queue system for managing multiple PDF processing tasks
- Worker processes for parallel processing with resource management
- Adaptive batch sizing based on system resources
- Progress tracking and real-time reporting
- Integration with existing MemoryManager and monitoring systems
- Automatic retry mechanisms with exponential backoff
"""

import asyncio
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
from queue import Queue, Empty
import signal
import sys

from ..core.config import MarkerConfig, get_config
from ..core.exceptions import ProcessingError, ValidationError
from ..core.logging import get_logger
from ..core.memory_manager import get_memory_manager, MemoryStats
from ..core.monitoring import get_system_monitor, Metric
from ..utils.file_utils import get_file_size_mb
from .pdf_processor import PDFProcessor


class JobStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class BatchStrategy(Enum):
    """Batch processing strategy."""
    SEQUENTIAL = "sequential"        # Process one at a time
    CONCURRENT = "concurrent"        # Process multiple concurrently
    ADAPTIVE = "adaptive"            # Adapt based on resources
    MEMORY_AWARE = "memory_aware"    # Adjust based on memory usage


@dataclass
class BatchJob:
    """Represents a single batch processing job."""
    job_id: str
    pdf_path: Path
    output_path: Path
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: float = 0.0
    file_size_mb: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    worker_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "pdf_path": str(self.pdf_path),
            "output_path": str(self.output_path),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "processing_time": self.processing_time,
            "file_size_mb": self.file_size_mb,
            "retry_count": self.retry_count,
            "worker_id": self.worker_id,
            "error": self.error
        }


@dataclass
class BatchProgress:
    """Track progress of batch processing."""
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    current_jobs: Dict[int, str] = field(default_factory=dict)  # worker_id -> job_id
    start_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    throughput_mbps: float = 0.0
    active_workers: int = 0
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_jobs == 0:
            return 0.0
        return (self.completed_jobs + self.failed_jobs) / self.total_jobs * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total_processed = self.completed_jobs + self.failed_jobs
        if total_processed == 0:
            return 0.0
        return self.completed_jobs / total_processed * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary."""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        return {
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "progress_percent": self.progress_percent,
            "success_rate": self.success_rate,
            "elapsed_time": elapsed_time,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "throughput_mbps": self.throughput_mbps,
            "active_workers": self.active_workers,
            "current_jobs": self.current_jobs
        }


class BatchWorker(mp.Process):
    """Worker process for batch processing."""
    
    def __init__(self, 
                 worker_id: int,
                 job_queue: mp.Queue,
                 result_queue: mp.Queue,
                 control_queue: mp.Queue,
                 config: MarkerConfig):
        super().__init__()
        self.worker_id = worker_id
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.control_queue = control_queue
        self.config = config
        self.pdf_processor = None
        self.is_running = True
        self.logger = None
        
    def run(self):
        """Worker main loop."""
        # Set up logging in subprocess
        self.logger = get_logger(f"batch_worker_{self.worker_id}")
        self.logger.info(f"Worker {self.worker_id} started")
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(self.config)
        
        # Signal handler for clean shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        while self.is_running:
            try:
                # Check for control messages
                try:
                    control_msg = self.control_queue.get_nowait()
                    if control_msg == "stop":
                        self.logger.info(f"Worker {self.worker_id} received stop signal")
                        break
                except Empty:
                    pass
                
                # Get next job from queue with timeout
                try:
                    job_dict = self.job_queue.get(timeout=1)
                except Empty:
                    continue
                
                # Process the job
                self._process_job(job_dict)
                
            except Exception as e:
                self.logger.error(f"Worker {self.worker_id} error: {e}")
                
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        self.is_running = False
    
    def _process_job(self, job_dict: Dict[str, Any]):
        """Process a single job."""
        start_time = time.time()
        
        # Reconstruct job from dict
        job = BatchJob(
            job_id=job_dict["job_id"],
            pdf_path=Path(job_dict["pdf_path"]),
            output_path=Path(job_dict["output_path"]),
            file_size_mb=job_dict["file_size_mb"],
            retry_count=job_dict.get("retry_count", 0),
            max_retries=job_dict.get("max_retries", 3)
        )
        
        job.worker_id = self.worker_id
        job.started_at = datetime.now()
        job.status = JobStatus.RUNNING
        
        # Notify start
        self.result_queue.put({
            "type": "job_started",
            "job_id": job.job_id,
            "worker_id": self.worker_id,
            "timestamp": job.started_at.isoformat()
        })
        
        try:
            # Process PDF
            result = self.pdf_processor.process_pdf(job.pdf_path)
            
            # Save result to output path
            output_file = job.output_path / f"{job.pdf_path.stem}_processed.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.result = result
            job.completed_at = datetime.now()
            job.processing_time = time.time() - start_time
            
            # Send success result
            self.result_queue.put({
                "type": "job_completed",
                "job": job.to_dict(),
                "worker_id": self.worker_id,
                "output_file": str(output_file)
            })
            
            self.logger.info(f"Worker {self.worker_id} completed job {job.job_id} in {job.processing_time:.2f}s")
            
        except Exception as e:
            # Handle processing error
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            job.processing_time = time.time() - start_time
            
            self.logger.error(f"Worker {self.worker_id} failed job {job.job_id}: {e}")
            
            # Send failure result
            self.result_queue.put({
                "type": "job_failed",
                "job": job.to_dict(),
                "worker_id": self.worker_id,
                "error": str(e),
                "can_retry": job.retry_count < job.max_retries
            })


class BatchProcessor:
    """High-performance batch processor for multiple PDFs."""
    
    def __init__(self, config: Optional[MarkerConfig] = None):
        """Initialize batch processor.
        
        Args:
            config: Optional MarkerConfig. If None, uses default config.
        """
        self.config = config or MarkerConfig()
        self.logger = get_logger("batch_processor")
        
        # Get managers
        self.memory_manager = get_memory_manager()
        self.system_monitor = get_system_monitor()
        
        # Processing configuration
        self.max_workers = self._calculate_optimal_workers()
        self.batch_strategy = BatchStrategy.ADAPTIVE
        self.adaptive_batch_size = True
        self.memory_threshold_percent = 80.0
        
        # Job management
        self.job_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.control_queue: mp.Queue = mp.Queue()
        self.workers: List[BatchWorker] = []
        self.jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: List[BatchJob] = []
        self.failed_jobs: List[BatchJob] = []
        
        # Progress tracking
        self.progress = BatchProgress(total_jobs=0)
        self.progress_callbacks: List[Callable[[BatchProgress], None]] = []
        self.result_handler_thread = None
        self.is_processing = False
        
        # Resource monitoring
        self.resource_monitor_thread = None
        self.resource_stats = {
            "memory_usage_mb": 0,
            "cpu_percent": 0,
            "active_workers": 0,
            "jobs_per_second": 0
        }
        
        self.logger.info(f"BatchProcessor initialized with {self.max_workers} max workers")
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources."""
        cpu_count = mp.cpu_count()
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Conservative approach: use 75% of CPUs
        optimal_cpus = max(1, int(cpu_count * 0.75))
        
        # Adjust based on available memory (assume 500MB per worker)
        available_memory_mb = memory_stats.available_memory_mb
        memory_based_workers = max(1, int(available_memory_mb / 500))
        
        # Take the minimum to avoid overloading
        optimal_workers = min(optimal_cpus, memory_based_workers, 8)  # Cap at 8 workers
        
        self.logger.info(f"Calculated optimal workers: {optimal_workers} (CPUs: {cpu_count}, Memory: {available_memory_mb:.1f}MB)")
        return optimal_workers
    
    def add_progress_callback(self, callback: Callable[[BatchProgress], None]):
        """Add a callback for progress updates."""
        self.progress_callbacks.append(callback)
    
    def _update_progress(self):
        """Update progress and notify callbacks."""
        # Calculate throughput
        elapsed_time = (datetime.now() - self.progress.start_time).total_seconds()
        if elapsed_time > 0 and self.progress.completed_jobs > 0:
            total_mb_processed = sum(job.file_size_mb for job in self.completed_jobs)
            self.progress.throughput_mbps = total_mb_processed / elapsed_time
            
            # Estimate completion time
            remaining_jobs = self.progress.total_jobs - self.progress.completed_jobs - self.progress.failed_jobs
            if remaining_jobs > 0 and self.progress.throughput_mbps > 0:
                avg_file_size = total_mb_processed / self.progress.completed_jobs
                estimated_seconds = (remaining_jobs * avg_file_size) / self.progress.throughput_mbps
                from datetime import timedelta
                self.progress.estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        # Update active workers
        self.progress.active_workers = len([w for w in self.workers if w.is_alive()])
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(self.progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update monitoring metrics."""
        metrics = [
            Metric(name="batch.total_jobs", value=self.progress.total_jobs, unit="count"),
            Metric(name="batch.completed_jobs", value=self.progress.completed_jobs, unit="count"),
            Metric(name="batch.failed_jobs", value=self.progress.failed_jobs, unit="count"),
            Metric(name="batch.progress_percent", value=self.progress.progress_percent, unit="percent"),
            Metric(name="batch.success_rate", value=self.progress.success_rate, unit="percent"),
            Metric(name="batch.throughput", value=self.progress.throughput_mbps, unit="MB/s"),
            Metric(name="batch.active_workers", value=self.progress.active_workers, unit="count")
        ]
        
        for metric in metrics:
            self.system_monitor.metrics_collector.collect_metric(metric)
    
    def _handle_results(self):
        """Handle results from worker processes."""
        while self.is_processing:
            try:
                result = self.result_queue.get(timeout=1)
                
                if result["type"] == "job_started":
                    job_id = result["job_id"]
                    worker_id = result["worker_id"]
                    self.progress.current_jobs[worker_id] = job_id
                    self.logger.debug(f"Job {job_id} started on worker {worker_id}")
                
                elif result["type"] == "job_completed":
                    job_dict = result["job"]
                    job_id = job_dict["job_id"]
                    worker_id = result["worker_id"]
                    
                    # Update job in registry
                    if job_id in self.jobs:
                        job = self.jobs[job_id]
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now()
                        job.processing_time = job_dict["processing_time"]
                        self.completed_jobs.append(job)
                    
                    # Update progress
                    self.progress.completed_jobs += 1
                    if worker_id in self.progress.current_jobs:
                        del self.progress.current_jobs[worker_id]
                    
                    self._update_progress()
                    self.logger.info(f"Job {job_id} completed successfully")
                
                elif result["type"] == "job_failed":
                    job_dict = result["job"]
                    job_id = job_dict["job_id"]
                    worker_id = result["worker_id"]
                    can_retry = result["can_retry"]
                    
                    # Handle failure
                    if job_id in self.jobs:
                        job = self.jobs[job_id]
                        job.status = JobStatus.FAILED
                        job.error = result["error"]
                        
                        if can_retry:
                            # Retry the job
                            job.retry_count += 1
                            job.status = JobStatus.RETRYING
                            self.logger.warning(f"Retrying job {job_id} (attempt {job.retry_count + 1})")
                            
                            # Re-queue with exponential backoff
                            backoff = 2 ** job.retry_count
                            threading.Timer(backoff, self._requeue_job, args=[job]).start()
                        else:
                            # Max retries exceeded
                            self.failed_jobs.append(job)
                            self.progress.failed_jobs += 1
                            self.logger.error(f"Job {job_id} failed after {job.retry_count} retries")
                    
                    # Update progress
                    if worker_id in self.progress.current_jobs:
                        del self.progress.current_jobs[worker_id]
                    
                    self._update_progress()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error handling result: {e}")
    
    def _requeue_job(self, job: BatchJob):
        """Re-queue a job for retry."""
        job_dict = {
            "job_id": job.job_id,
            "pdf_path": str(job.pdf_path),
            "output_path": str(job.output_path),
            "file_size_mb": job.file_size_mb,
            "retry_count": job.retry_count,
            "max_retries": job.max_retries
        }
        self.job_queue.put(job_dict)
    
    def _monitor_resources(self):
        """Monitor system resources during processing."""
        while self.is_processing:
            try:
                # Get memory stats
                memory_stats = self.memory_manager.get_memory_stats()
                self.resource_stats["memory_usage_mb"] = memory_stats.process_memory_mb
                
                # Check memory threshold
                if memory_stats.memory_percent > self.memory_threshold_percent:
                    self.logger.warning(f"High memory usage: {memory_stats.memory_percent:.1f}%")
                    
                    # Adaptive strategy: reduce workers if memory is high
                    if self.batch_strategy == BatchStrategy.ADAPTIVE:
                        active_workers = len([w for w in self.workers if w.is_alive()])
                        if active_workers > 1:
                            self._reduce_workers(1)
                
                # Sleep for monitoring interval
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
    
    def _start_workers(self, num_workers: int):
        """Start worker processes."""
        for i in range(num_workers):
            worker = BatchWorker(
                worker_id=i,
                job_queue=self.job_queue,
                result_queue=self.result_queue,
                control_queue=self.control_queue,
                config=self.config
            )
            worker.start()
            self.workers.append(worker)
        
        self.logger.info(f"Started {num_workers} workers")
    
    def _stop_workers(self):
        """Stop all worker processes."""
        # Send stop signal to all workers
        for _ in self.workers:
            self.control_queue.put("stop")
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                self.logger.warning(f"Worker {worker.worker_id} did not stop gracefully, terminating")
                worker.terminate()
                worker.join()
        
        self.workers.clear()
        self.logger.info("All workers stopped")
    
    def _reduce_workers(self, num_to_reduce: int):
        """Reduce number of active workers."""
        active_workers = [w for w in self.workers if w.is_alive()]
        workers_to_stop = min(num_to_reduce, len(active_workers) - 1)  # Keep at least 1
        
        if workers_to_stop > 0:
            self.logger.info(f"Reducing workers by {workers_to_stop} due to resource constraints")
            for i in range(workers_to_stop):
                self.control_queue.put("stop")
    
    async def process_batch_async(self, 
                                  pdf_paths: List[Path],
                                  output_dir: Path,
                                  batch_size: Optional[int] = None,
                                  strategy: Optional[BatchStrategy] = None) -> Dict[str, Any]:
        """Process a batch of PDFs asynchronously.
        
        Args:
            pdf_paths: List of PDF file paths to process
            output_dir: Output directory for results
            batch_size: Optional batch size (None for adaptive)
            strategy: Optional batch strategy (None for default)
            
        Returns:
            Dictionary with processing results and statistics
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.process_batch,
            pdf_paths,
            output_dir,
            batch_size,
            strategy
        )
    
    def process_batch(self,
                      pdf_paths: List[Path],
                      output_dir: Path,
                      batch_size: Optional[int] = None,
                      strategy: Optional[BatchStrategy] = None) -> Dict[str, Any]:
        """Process a batch of PDFs.
        
        Args:
            pdf_paths: List of PDF file paths to process
            output_dir: Output directory for results
            batch_size: Optional batch size (None for adaptive)
            strategy: Optional batch strategy (None for default)
            
        Returns:
            Dictionary with processing results and statistics
        """
        if self.is_processing:
            raise RuntimeError("Batch processing already in progress")
        
        start_time = time.time()
        self.is_processing = True
        self.batch_strategy = strategy or self.batch_strategy
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress
        self.progress = BatchProgress(total_jobs=len(pdf_paths))
        self.jobs.clear()
        self.completed_jobs.clear()
        self.failed_jobs.clear()
        
        try:
            # Create jobs
            for i, pdf_path in enumerate(pdf_paths):
                if not pdf_path.exists():
                    self.logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                
                job = BatchJob(
                    job_id=f"job_{i:04d}_{pdf_path.stem}",
                    pdf_path=pdf_path,
                    output_path=output_dir,
                    file_size_mb=get_file_size_mb(pdf_path)
                )
                self.jobs[job.job_id] = job
                
                # Queue job
                job_dict = {
                    "job_id": job.job_id,
                    "pdf_path": str(job.pdf_path),
                    "output_path": str(job.output_path),
                    "file_size_mb": job.file_size_mb,
                    "retry_count": 0,
                    "max_retries": job.max_retries
                }
                self.job_queue.put(job_dict)
            
            # Determine number of workers
            if batch_size:
                num_workers = min(batch_size, self.max_workers, len(self.jobs))
            else:
                # Adaptive sizing based on resources and job count
                num_workers = self._calculate_adaptive_workers(len(self.jobs))
            
            self.logger.info(f"Processing {len(self.jobs)} PDFs with {num_workers} workers using {self.batch_strategy.value} strategy")
            
            # Start result handler thread
            self.result_handler_thread = threading.Thread(target=self._handle_results, daemon=True)
            self.result_handler_thread.start()
            
            # Start resource monitor thread
            self.resource_monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.resource_monitor_thread.start()
            
            # Start workers
            self._start_workers(num_workers)
            
            # Wait for all jobs to complete
            while (self.progress.completed_jobs + self.progress.failed_jobs) < len(self.jobs):
                time.sleep(1)
                self._update_progress()
            
            # Final progress update
            self._update_progress()
            
            # Generate results
            processing_time = time.time() - start_time
            results = self._generate_batch_results(processing_time)
            
            self.logger.info(f"Batch processing completed in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            raise
            
        finally:
            # Cleanup
            self.is_processing = False
            self._stop_workers()
            
            # Wait for threads
            if self.result_handler_thread:
                self.result_handler_thread.join(timeout=5)
            if self.resource_monitor_thread:
                self.resource_monitor_thread.join(timeout=5)
    
    def _calculate_adaptive_workers(self, job_count: int) -> int:
        """Calculate adaptive number of workers based on job count and resources."""
        # Start with optimal workers
        workers = self.max_workers
        
        # Adjust based on job count
        if job_count < workers:
            workers = job_count
        
        # Adjust based on current memory usage
        memory_stats = self.memory_manager.get_memory_stats()
        if memory_stats.memory_percent > 70:
            workers = max(1, workers // 2)
        elif memory_stats.memory_percent > 50:
            workers = max(1, int(workers * 0.75))
        
        return workers
    
    def _generate_batch_results(self, processing_time: float) -> Dict[str, Any]:
        """Generate comprehensive batch processing results."""
        total_size_mb = sum(job.file_size_mb for job in self.jobs.values())
        processed_size_mb = sum(job.file_size_mb for job in self.completed_jobs)
        
        results = {
            "summary": {
                "total_jobs": len(self.jobs),
                "completed_jobs": len(self.completed_jobs),
                "failed_jobs": len(self.failed_jobs),
                "success_rate": self.progress.success_rate,
                "total_processing_time": processing_time,
                "average_time_per_job": processing_time / max(1, len(self.completed_jobs)),
                "total_size_mb": total_size_mb,
                "processed_size_mb": processed_size_mb,
                "throughput_mbps": processed_size_mb / max(1, processing_time),
                "workers_used": self.max_workers
            },
            "completed_jobs": [job.to_dict() for job in self.completed_jobs],
            "failed_jobs": [job.to_dict() for job in self.failed_jobs],
            "resource_stats": self.resource_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add performance metrics
        if self.completed_jobs:
            processing_times = [job.processing_time for job in self.completed_jobs]
            results["performance_metrics"] = {
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "jobs_per_minute": len(self.completed_jobs) / (processing_time / 60)
            }
        
        return results
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Get current batch processing status."""
        return {
            "is_processing": self.is_processing,
            "progress": self.progress.to_dict() if self.progress else None,
            "active_workers": len([w for w in self.workers if w.is_alive()]),
            "queued_jobs": self.job_queue.qsize() if hasattr(self.job_queue, 'qsize') else "unknown",
            "resource_stats": self.resource_stats
        }
    
    def cancel_batch(self):
        """Cancel current batch processing."""
        if not self.is_processing:
            return
        
        self.logger.warning("Cancelling batch processing")
        self.is_processing = False
        
        # Mark remaining jobs as cancelled
        for job in self.jobs.values():
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.status = JobStatus.CANCELLED
        
        # Stop workers
        self._stop_workers()


# Convenience functions
def create_batch_processor(config: Optional[MarkerConfig] = None) -> BatchProcessor:
    """Create a batch processor instance."""
    return BatchProcessor(config)


async def process_pdfs_batch(pdf_paths: List[Union[str, Path]],
                            output_dir: Union[str, Path],
                            config: Optional[MarkerConfig] = None,
                            progress_callback: Optional[Callable[[BatchProgress], None]] = None) -> Dict[str, Any]:
    """Process multiple PDFs in batch with progress tracking.
    
    Args:
        pdf_paths: List of PDF file paths
        output_dir: Output directory for results
        config: Optional configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with processing results
    """
    # Convert paths
    pdf_paths = [Path(p) for p in pdf_paths]
    output_dir = Path(output_dir)
    
    # Create processor
    processor = create_batch_processor(config)
    
    # Add progress callback if provided
    if progress_callback:
        processor.add_progress_callback(progress_callback)
    
    # Process batch
    return await processor.process_batch_async(pdf_paths, output_dir)