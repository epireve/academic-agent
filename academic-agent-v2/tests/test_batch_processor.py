#!/usr/bin/env python3
"""
Comprehensive tests for the Batch Processing System
Task 27 Implementation - Test suite for batch processor
"""

import asyncio
import json
import multiprocessing as mp
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from academic_agent_v2.src.processors.batch_processor import (
    BatchProcessor, BatchJob, BatchProgress, BatchWorker,
    JobStatus, BatchStrategy, create_batch_processor, process_pdfs_batch
)
from academic_agent_v2.src.core.config import MarkerConfig
from academic_agent_v2.src.core.exceptions import ProcessingError


class TestBatchJob:
    """Test BatchJob data class."""
    
    def test_batch_job_creation(self):
        """Test creating a batch job."""
        job = BatchJob(
            job_id="test_job_001",
            pdf_path=Path("/test/file.pdf"),
            output_path=Path("/test/output")
        )
        
        assert job.job_id == "test_job_001"
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert job.worker_id is None
    
    def test_batch_job_to_dict(self):
        """Test converting job to dictionary."""
        job = BatchJob(
            job_id="test_job_002",
            pdf_path=Path("/test/file.pdf"),
            output_path=Path("/test/output"),
            file_size_mb=10.5
        )
        
        job_dict = job.to_dict()
        assert job_dict["job_id"] == "test_job_002"
        assert job_dict["pdf_path"] == "/test/file.pdf"
        assert job_dict["output_path"] == "/test/output"
        assert job_dict["status"] == "pending"
        assert job_dict["file_size_mb"] == 10.5
        assert job_dict["retry_count"] == 0


class TestBatchProgress:
    """Test BatchProgress tracking."""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        progress = BatchProgress(total_jobs=10)
        
        assert progress.total_jobs == 10
        assert progress.completed_jobs == 0
        assert progress.failed_jobs == 0
        assert progress.progress_percent == 0.0
        assert progress.success_rate == 0.0
    
    def test_progress_calculation(self):
        """Test progress calculations."""
        progress = BatchProgress(total_jobs=10)
        progress.completed_jobs = 6
        progress.failed_jobs = 2
        
        assert progress.progress_percent == 80.0  # (6+2)/10 * 100
        assert progress.success_rate == 75.0      # 6/(6+2) * 100
    
    def test_progress_to_dict(self):
        """Test converting progress to dictionary."""
        progress = BatchProgress(total_jobs=5)
        progress.completed_jobs = 3
        progress.failed_jobs = 1
        progress.throughput_mbps = 25.5
        progress.active_workers = 4
        
        progress_dict = progress.to_dict()
        assert progress_dict["total_jobs"] == 5
        assert progress_dict["completed_jobs"] == 3
        assert progress_dict["failed_jobs"] == 1
        assert progress_dict["progress_percent"] == 80.0
        assert progress_dict["success_rate"] == 75.0
        assert progress_dict["throughput_mbps"] == 25.5
        assert progress_dict["active_workers"] == 4


class TestBatchWorker:
    """Test BatchWorker process."""
    
    def test_worker_creation(self):
        """Test creating a batch worker."""
        job_queue = mp.Queue()
        result_queue = mp.Queue()
        control_queue = mp.Queue()
        config = MarkerConfig()
        
        worker = BatchWorker(
            worker_id=1,
            job_queue=job_queue,
            result_queue=result_queue,
            control_queue=control_queue,
            config=config
        )
        
        assert worker.worker_id == 1
        assert worker.is_running is True
    
    @patch('academic_agent_v2.src.processors.batch_processor.PDFProcessor')
    def test_worker_process_job(self, mock_pdf_processor):
        """Test worker processing a job."""
        # Setup
        job_queue = mp.Queue()
        result_queue = mp.Queue()
        control_queue = mp.Queue()
        config = MarkerConfig()
        
        # Mock PDF processor
        mock_processor_instance = Mock()
        mock_processor_instance.process_pdf.return_value = {
            "success": True,
            "pages_processed": 10
        }
        mock_pdf_processor.return_value = mock_processor_instance
        
        # Create worker
        worker = BatchWorker(
            worker_id=1,
            job_queue=job_queue,
            result_queue=result_queue,
            control_queue=control_queue,
            config=config
        )
        
        # Create test job
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_file = temp_path / "test.pdf"
            pdf_file.write_text("dummy pdf content")
            
            job_dict = {
                "job_id": "test_job",
                "pdf_path": str(pdf_file),
                "output_path": str(temp_path),
                "file_size_mb": 1.0,
                "retry_count": 0,
                "max_retries": 3
            }
            
            # Process job directly (not in subprocess)
            worker.pdf_processor = mock_processor_instance
            worker.logger = Mock()
            worker._process_job(job_dict)
            
            # Check results
            result = result_queue.get_nowait()
            assert result["type"] == "job_started"
            
            result = result_queue.get_nowait()
            assert result["type"] == "job_completed"
            assert result["job"]["job_id"] == "test_job"
            assert result["worker_id"] == 1


class TestBatchProcessor:
    """Test BatchProcessor main class."""
    
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_processor_initialization(self, mock_monitor, mock_memory):
        """Test batch processor initialization."""
        # Mock memory manager
        mock_memory_instance = Mock()
        mock_memory_instance.get_memory_stats.return_value = Mock(
            available_memory_mb=8192,
            memory_percent=50.0
        )
        mock_memory.return_value = mock_memory_instance
        
        # Mock system monitor
        mock_monitor.return_value = Mock()
        
        processor = BatchProcessor()
        
        assert processor.max_workers > 0
        assert processor.batch_strategy == BatchStrategy.ADAPTIVE
        assert processor.is_processing is False
        assert len(processor.jobs) == 0
    
    @patch('academic_agent_v2.src.processors.batch_processor.mp.cpu_count')
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_calculate_optimal_workers(self, mock_monitor, mock_memory, mock_cpu_count):
        """Test optimal worker calculation."""
        # Setup mocks
        mock_cpu_count.return_value = 8
        mock_memory_instance = Mock()
        mock_memory_instance.get_memory_stats.return_value = Mock(
            available_memory_mb=4096  # 4GB
        )
        mock_memory.return_value = mock_memory_instance
        mock_monitor.return_value = Mock()
        
        processor = BatchProcessor()
        
        # Should be min of:
        # - 75% of CPUs = 6
        # - Memory-based (4096/500) = 8
        # - Cap of 8
        # So should be 6
        assert processor.max_workers == 6
    
    def test_add_progress_callback(self):
        """Test adding progress callbacks."""
        with patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager'), \
             patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor'):
            
            processor = BatchProcessor()
            callback = Mock()
            
            processor.add_progress_callback(callback)
            assert callback in processor.progress_callbacks
    
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_update_progress(self, mock_monitor, mock_memory):
        """Test progress updates."""
        # Setup mocks
        mock_memory.return_value = Mock()
        mock_monitor_instance = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        processor = BatchProcessor()
        processor.progress = BatchProgress(total_jobs=5)
        processor.progress.completed_jobs = 3
        
        # Add mock callback
        callback = Mock()
        processor.add_progress_callback(callback)
        
        # Update progress
        processor._update_progress()
        
        # Check callback was called
        callback.assert_called_once()
        progress_arg = callback.call_args[0][0]
        assert progress_arg.completed_jobs == 3
    
    @pytest.mark.asyncio
    @patch('academic_agent_v2.src.processors.batch_processor.BatchWorker')
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    async def test_process_batch_async(self, mock_monitor, mock_memory, mock_worker):
        """Test async batch processing."""
        # Setup mocks
        mock_memory_instance = Mock()
        mock_memory_instance.get_memory_stats.return_value = Mock(
            available_memory_mb=8192,
            memory_percent=50.0
        )
        mock_memory.return_value = mock_memory_instance
        mock_monitor.return_value = Mock()
        
        # Mock worker
        mock_worker_instance = Mock()
        mock_worker_instance.is_alive.return_value = False
        mock_worker.return_value = mock_worker_instance
        
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test PDFs
            pdf_files = []
            for i in range(3):
                pdf_file = temp_path / f"test_{i}.pdf"
                pdf_file.write_text(f"dummy pdf content {i}")
                pdf_files.append(pdf_file)
            
            output_dir = temp_path / "output"
            
            # Mock the synchronous process_batch to return quickly
            with patch.object(processor, 'process_batch') as mock_process:
                mock_process.return_value = {
                    "summary": {
                        "total_jobs": 3,
                        "completed_jobs": 3,
                        "failed_jobs": 0,
                        "success_rate": 100.0
                    }
                }
                
                # Process batch
                results = await processor.process_batch_async(
                    pdf_files,
                    output_dir
                )
                
                assert results["summary"]["total_jobs"] == 3
                assert results["summary"]["success_rate"] == 100.0
    
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_batch_status(self, mock_monitor, mock_memory):
        """Test getting batch status."""
        # Setup mocks
        mock_memory.return_value = Mock()
        mock_monitor.return_value = Mock()
        
        processor = BatchProcessor()
        
        # Initial status
        status = processor.get_batch_status()
        assert status["is_processing"] is False
        assert status["progress"] is None
        assert status["active_workers"] == 0
    
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_cancel_batch(self, mock_monitor, mock_memory):
        """Test cancelling batch processing."""
        # Setup mocks
        mock_memory.return_value = Mock()
        mock_monitor.return_value = Mock()
        
        processor = BatchProcessor()
        processor.is_processing = True
        
        # Add some test jobs
        job1 = BatchJob("job1", Path("test1.pdf"), Path("output"))
        job1.status = JobStatus.RUNNING
        job2 = BatchJob("job2", Path("test2.pdf"), Path("output"))
        job2.status = JobStatus.PENDING
        
        processor.jobs = {"job1": job1, "job2": job2}
        
        # Cancel batch
        processor.cancel_batch()
        
        assert processor.is_processing is False
        assert job1.status == JobStatus.CANCELLED
        assert job2.status == JobStatus.CANCELLED


class TestBatchProcessingIntegration:
    """Integration tests for batch processing."""
    
    @pytest.mark.integration
    @patch('academic_agent_v2.src.processors.batch_processor.PDFProcessor')
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_end_to_end_batch_processing(self, mock_monitor, mock_memory, mock_pdf_processor):
        """Test end-to-end batch processing."""
        # Setup mocks
        mock_memory_instance = Mock()
        mock_memory_instance.get_memory_stats.return_value = Mock(
            available_memory_mb=8192,
            memory_percent=50.0,
            total_memory_mb=16384,
            used_memory_mb=8192,
            process_memory_mb=512
        )
        mock_memory.return_value = mock_memory_instance
        
        mock_monitor_instance = Mock()
        mock_monitor_instance.metrics_collector = Mock()
        mock_monitor.return_value = mock_monitor_instance
        
        # Mock PDF processor
        mock_processor_instance = Mock()
        mock_processor_instance.process_pdf.return_value = {
            "success": True,
            "pages_processed": 10,
            "processing_time": 1.0
        }
        mock_pdf_processor.return_value = mock_processor_instance
        
        # Create processor
        processor = BatchProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test PDFs
            pdf_files = []
            for i in range(5):
                pdf_file = temp_path / f"test_{i}.pdf"
                pdf_file.write_text(f"dummy pdf content {i}")
                pdf_files.append(pdf_file)
            
            output_dir = temp_path / "output"
            
            # Add progress callback
            progress_updates = []
            def progress_callback(progress):
                progress_updates.append(progress.to_dict())
            
            processor.add_progress_callback(progress_callback)
            
            # Process batch with small test batch
            # We'll mock the worker behavior to complete quickly
            with patch.object(processor, '_start_workers'), \
                 patch.object(processor, '_stop_workers'), \
                 patch.object(processor, '_handle_results'), \
                 patch.object(processor, '_monitor_resources'):
                
                # Simulate jobs completing
                processor.progress.total_jobs = len(pdf_files)
                for i, pdf_file in enumerate(pdf_files):
                    job = BatchJob(
                        job_id=f"job_{i}",
                        pdf_path=pdf_file,
                        output_path=output_dir,
                        file_size_mb=1.0
                    )
                    job.status = JobStatus.COMPLETED
                    job.processing_time = 1.0
                    processor.jobs[job.job_id] = job
                    processor.completed_jobs.append(job)
                    processor.progress.completed_jobs += 1
                
                # Get results
                results = processor._generate_batch_results(5.0)
                
                assert results["summary"]["total_jobs"] == 5
                assert results["summary"]["completed_jobs"] == 5
                assert results["summary"]["failed_jobs"] == 0
                assert results["summary"]["success_rate"] == 100.0
                assert results["summary"]["throughput_mbps"] == 1.0  # 5 MB in 5 seconds


class TestBatchUtilities:
    """Test utility functions."""
    
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_create_batch_processor(self, mock_monitor, mock_memory):
        """Test creating batch processor with utility function."""
        # Setup mocks
        mock_memory.return_value = Mock()
        mock_monitor.return_value = Mock()
        
        processor = create_batch_processor()
        assert isinstance(processor, BatchProcessor)
    
    @pytest.mark.asyncio
    @patch('academic_agent_v2.src.processors.batch_processor.BatchProcessor')
    async def test_process_pdfs_batch_utility(self, mock_processor_class):
        """Test process_pdfs_batch utility function."""
        # Mock processor instance
        mock_processor = Mock()
        mock_processor.process_batch_async = Mock(return_value={
            "summary": {"total_jobs": 2, "completed_jobs": 2}
        })
        mock_processor_class.return_value = mock_processor
        
        # Test paths
        pdf_paths = ["test1.pdf", "test2.pdf"]
        output_dir = "output"
        
        # Mock callback
        callback = Mock()
        
        # Process
        results = await process_pdfs_batch(
            pdf_paths,
            output_dir,
            progress_callback=callback
        )
        
        # Verify
        assert results["summary"]["total_jobs"] == 2
        mock_processor.add_progress_callback.assert_called_once_with(callback)


class TestBatchStrategies:
    """Test different batch processing strategies."""
    
    @patch('academic_agent_v2.src.processors.batch_processor.get_memory_manager')
    @patch('academic_agent_v2.src.processors.batch_processor.get_system_monitor')
    def test_adaptive_strategy(self, mock_monitor, mock_memory):
        """Test adaptive batch strategy."""
        # Setup mocks
        mock_memory_instance = Mock()
        mock_memory_instance.get_memory_stats.return_value = Mock(
            available_memory_mb=8192,
            memory_percent=50.0
        )
        mock_memory.return_value = mock_memory_instance
        mock_monitor.return_value = Mock()
        
        processor = BatchProcessor()
        processor.batch_strategy = BatchStrategy.ADAPTIVE
        
        # Test adaptive worker calculation
        # With 50% memory, should use full workers
        workers = processor._calculate_adaptive_workers(10)
        assert workers == processor.max_workers
        
        # With high memory usage, should reduce workers
        mock_memory_instance.get_memory_stats.return_value = Mock(
            available_memory_mb=2048,
            memory_percent=75.0
        )
        workers = processor._calculate_adaptive_workers(10)
        assert workers < processor.max_workers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])