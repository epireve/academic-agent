#!/usr/bin/env python3
"""
Test Suite for High-Performance PDF Processor
Academic Agent v2 - Task 11 Implementation

This module provides comprehensive tests for the Marker-based PDF processor
including unit tests, integration tests, and performance benchmarks.
"""

import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil
import logging
from datetime import datetime
import os

# Import the modules to test
from marker_pdf_processor import (
    MarkerPDFProcessor,
    ProcessingResult,
    ProcessingMetrics,
    ChapterSplit,
    ChapterSplitter,
    PerformanceMonitor,
    create_pdf_processor
)
from monitoring import (
    MonitoringSystem,
    ProcessingEvent,
    MetricsCollector,
    AlertManager,
    AlertLevel,
    MetricType
)


class TestFixtures:
    """Test fixtures and utilities."""
    
    @staticmethod
    def create_sample_pdf(content: str = "Sample PDF content") -> Path:
        """Create a sample PDF file for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        pdf_path = temp_dir / "sample.pdf"
        
        # Create a mock PDF file (just a text file with .pdf extension for testing)
        pdf_path.write_text(content)
        return pdf_path
        
    @staticmethod
    def create_test_config() -> dict:
        """Create a test configuration."""
        return {
            'device': 'cpu',
            'batch_size': 1,
            'max_workers': 2,
            'extract_images': True,
            'split_chapters': True,
            'enable_editor_model': False,
            'enable_ocr': False,
            'max_pages': 5
        }
        
    @staticmethod
    def create_sample_markdown_content() -> str:
        """Create sample markdown content for testing."""
        return """# Chapter 1: Introduction

This is the first chapter of the document.

## 1.1 Overview

Some overview content here.

# Chapter 2: Methods

This is the second chapter.

## 2.1 Methodology

Some methodology content here.

# Chapter 3: Results

This is the third chapter.

## 3.1 Findings

Some findings here.
"""

    @staticmethod
    def cleanup_temp_files(temp_paths: list):
        """Clean up temporary files and directories."""
        for path in temp_paths:
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


class TestProcessingMetrics:
    """Test ProcessingMetrics class."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ProcessingMetrics()
        assert metrics.start_time > 0
        assert metrics.end_time is None
        assert metrics.processing_time == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.pages_processed == 0
        assert metrics.success is False
        assert metrics.error_message is None
        assert metrics.file_size_mb == 0.0
        assert metrics.device_used == "cpu"
        assert metrics.batch_size == 1
        
    def test_metrics_with_values(self):
        """Test metrics with custom values."""
        metrics = ProcessingMetrics(
            processing_time=5.2,
            memory_usage_mb=150.0,
            pages_processed=25,
            success=True,
            file_size_mb=2.5,
            device_used="mps",
            batch_size=3
        )
        
        assert metrics.processing_time == 5.2
        assert metrics.memory_usage_mb == 150.0
        assert metrics.pages_processed == 25
        assert metrics.success is True
        assert metrics.file_size_mb == 2.5
        assert metrics.device_used == "mps"
        assert metrics.batch_size == 3


class TestProcessingResult:
    """Test ProcessingResult class."""
    
    def test_result_initialization(self):
        """Test result initialization."""
        source_path = Path("test.pdf")
        result = ProcessingResult(source_path=source_path)
        
        assert result.source_path == source_path
        assert result.output_path is None
        assert result.markdown_content == ""
        assert result.images == []
        assert result.chapters == {}
        assert result.metadata == {}
        assert isinstance(result.metrics, ProcessingMetrics)
        assert result.success is False
        assert result.error_message is None
        
    def test_result_with_success(self):
        """Test successful result."""
        source_path = Path("test.pdf")
        output_path = Path("output/test.md")
        content = "# Test Content"
        
        result = ProcessingResult(
            source_path=source_path,
            output_path=output_path,
            markdown_content=content,
            success=True
        )
        
        assert result.success is True
        assert result.output_path == output_path
        assert result.markdown_content == content
        assert result.error_message is None


class TestChapterSplitter:
    """Test ChapterSplitter class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.splitter = ChapterSplitter()
        self.sample_content = TestFixtures.create_sample_markdown_content()
        
    def test_chapter_detection(self):
        """Test chapter detection from content."""
        chapters = self.splitter.split_content(self.sample_content, Path("test.pdf"))
        
        assert len(chapters) == 3
        assert "chapter_01" in chapters
        assert "chapter_02" in chapters
        assert "chapter_03" in chapters
        
        # Check first chapter
        chapter1 = chapters["chapter_01"]
        assert chapter1.chapter_number == 1
        assert "Introduction" in chapter1.title
        assert "first chapter" in chapter1.content
        
    def test_chapter_pattern_matching(self):
        """Test various chapter header patterns."""
        test_patterns = [
            "# Chapter 1: Introduction",
            "## Chapter 2: Methods",
            "# CHAPTER 3: RESULTS",
            "## CHAPTER 4: DISCUSSION",
            "# 5. Conclusion",
            "## 6. References"
        ]
        
        for pattern in test_patterns:
            match = self.splitter._find_chapter_match(pattern)
            assert match is not None
            chapter_num, title = match
            assert isinstance(chapter_num, int)
            assert chapter_num > 0
            
    def test_empty_content(self):
        """Test with empty content."""
        chapters = self.splitter.split_content("", Path("test.pdf"))
        assert len(chapters) == 0
        
    def test_no_chapters(self):
        """Test content with no chapter headers."""
        content = "This is just regular content with no chapters."
        chapters = self.splitter.split_content(content, Path("test.pdf"))
        assert len(chapters) == 0


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.monitor = PerformanceMonitor()
        
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.metrics_history == []
        assert isinstance(self.monitor.system_stats, dict)
        
    def test_start_monitoring(self):
        """Test starting monitoring."""
        metrics = self.monitor.start_monitoring("test_operation")
        
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.start_time > 0
        assert metrics.memory_usage_mb > 0
        
    def test_end_monitoring(self):
        """Test ending monitoring."""
        metrics = self.monitor.start_monitoring("test_operation")
        time.sleep(0.1)  # Small delay
        self.monitor.end_monitoring(metrics, True)
        
        assert metrics.end_time > metrics.start_time
        assert metrics.processing_time > 0
        assert metrics.success is True
        assert len(self.monitor.metrics_history) == 1
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Create some test metrics
        for i in range(3):
            metrics = self.monitor.start_monitoring(f"test_operation_{i}")
            time.sleep(0.01)
            self.monitor.end_monitoring(metrics, i % 2 == 0)  # Alternate success/failure
            
        summary = self.monitor.get_performance_summary()
        
        assert summary["total_operations"] == 3
        assert summary["successful_operations"] == 2
        assert summary["failed_operations"] == 1
        assert summary["success_rate"] == 2/3
        assert summary["average_processing_time"] > 0


class TestMarkerPDFProcessor:
    """Test MarkerPDFProcessor class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = TestFixtures.create_test_config()
        self.processor = MarkerPDFProcessor(self.config)
        self.temp_files = []
        
    def teardown_method(self):
        """Cleanup test environment."""
        TestFixtures.cleanup_temp_files(self.temp_files)
        
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.device == 'cpu'
        assert self.processor.batch_size == 1
        assert self.processor.max_workers == 2
        assert self.processor.extract_images is True
        assert self.processor.split_chapters is True
        assert self.processor.models_loaded is False
        
    def test_processor_with_default_config(self):
        """Test processor with default configuration."""
        processor = MarkerPDFProcessor()
        
        assert processor.batch_size == 2
        assert processor.max_workers == 4
        assert processor.extract_images is True
        
    @pytest.mark.asyncio
    async def test_initialize_models(self):
        """Test model initialization."""
        # Mock the model loading since we're testing without actual Marker
        with patch('marker_pdf_processor.MARKER_AVAILABLE', False):
            await self.processor.initialize_models()
            # Should not crash even when Marker is not available
            
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self):
        """Test processing non-existent file."""
        nonexistent_path = Path("nonexistent.pdf")
        output_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(output_dir)
        
        result = await self.processor.process_single_pdf(nonexistent_path, output_dir)
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
        
    @pytest.mark.asyncio
    async def test_process_non_pdf_file(self):
        """Test processing non-PDF file."""
        # Create a text file with .txt extension
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("This is not a PDF")
        
        output_dir = temp_dir / "output"
        
        result = await self.processor.process_single_pdf(txt_file, output_dir)
        
        assert result.success is False
        assert "not a pdf" in result.error_message.lower()
        
    @pytest.mark.asyncio
    async def test_process_sample_pdf(self):
        """Test processing a sample PDF file."""
        # Create a sample PDF file
        pdf_path = TestFixtures.create_sample_pdf("Sample PDF content")
        self.temp_files.append(pdf_path.parent)
        
        output_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(output_dir)
        
        result = await self.processor.process_single_pdf(pdf_path, output_dir)
        
        # Should succeed with simulation
        assert result.success is True
        assert result.output_path is not None
        assert result.output_path.exists()
        assert len(result.markdown_content) > 0
        assert result.metrics.processing_time > 0
        
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing of multiple PDFs."""
        # Create multiple sample PDFs
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        pdf_paths = []
        for i in range(3):
            pdf_path = temp_dir / f"test_{i}.pdf"
            pdf_path.write_text(f"Sample PDF content {i}")
            pdf_paths.append(pdf_path)
            
        output_dir = temp_dir / "output"
        
        results = await self.processor.process_batch(pdf_paths, output_dir)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Check that summary file was created
        summary_file = output_dir / "batch_processing_summary.json"
        assert summary_file.exists()
        
        # Check summary content
        summary_data = json.loads(summary_file.read_text())
        assert summary_data["processing_summary"]["total_files"] == 3
        assert summary_data["processing_summary"]["successful"] == 3
        assert summary_data["processing_summary"]["failed"] == 0
        
    @pytest.mark.asyncio
    async def test_empty_batch_processing(self):
        """Test batch processing with empty list."""
        results = await self.processor.process_batch([], Path("output"))
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_get_processing_stats(self):
        """Test getting processing statistics."""
        stats = await self.processor.get_processing_stats()
        
        assert "processor_info" in stats
        assert "system_info" in stats
        assert "performance_metrics" in stats
        assert "cache_info" in stats
        
        proc_info = stats["processor_info"]
        assert proc_info["device"] == "cpu"
        assert proc_info["batch_size"] == 1
        assert proc_info["max_workers"] == 2
        
    @pytest.mark.asyncio
    async def test_cache_functionality(self):
        """Test caching functionality."""
        # Create a sample PDF file
        pdf_path = TestFixtures.create_sample_pdf("Sample PDF content")
        self.temp_files.append(pdf_path.parent)
        
        output_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(output_dir)
        
        # Process the file twice
        result1 = await self.processor.process_single_pdf(pdf_path, output_dir)
        result2 = await self.processor.process_single_pdf(pdf_path, output_dir)
        
        # Both should succeed
        assert result1.success is True
        assert result2.success is True
        
        # Check cache statistics
        stats = await self.processor.get_processing_stats()
        assert stats["cache_info"]["cached_files"] >= 1
        
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        # Process a file to populate cache
        pdf_path = TestFixtures.create_sample_pdf("Sample PDF content")
        self.temp_files.append(pdf_path.parent)
        
        output_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(output_dir)
        
        await self.processor.process_single_pdf(pdf_path, output_dir)
        
        # Clear cache
        await self.processor.clear_cache()
        
        # Check cache is empty
        stats = await self.processor.get_processing_stats()
        assert stats["cache_info"]["cached_files"] == 0


class TestCreatePDFProcessor:
    """Test the create_pdf_processor factory function."""
    
    def test_create_with_default_config(self):
        """Test creating processor with default configuration."""
        processor = create_pdf_processor()
        
        assert isinstance(processor, MarkerPDFProcessor)
        assert processor.batch_size == 2
        assert processor.max_workers == 4
        assert processor.extract_images is True
        
    def test_create_with_custom_config(self):
        """Test creating processor with custom configuration."""
        config = {
            'batch_size': 5,
            'max_workers': 8,
            'extract_images': False
        }
        
        processor = create_pdf_processor(config)
        
        assert processor.batch_size == 5
        assert processor.max_workers == 8
        assert processor.extract_images is False


class TestMonitoringSystem:
    """Test MonitoringSystem class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.monitoring = MonitoringSystem()
        
    def teardown_method(self):
        """Cleanup test environment."""
        self.monitoring.stop_monitoring()
        
    def test_monitoring_initialization(self):
        """Test monitoring system initialization."""
        assert self.monitoring.metrics_collector is not None
        assert self.monitoring.system_monitor is not None
        assert self.monitoring.alert_manager is not None
        assert self.monitoring.processing_logger is not None
        
    def test_record_processing_event(self):
        """Test recording processing events."""
        event = ProcessingEvent(
            event_type='pdf_processing',
            timestamp=datetime.now(),
            file_path='test.pdf',
            status='success',
            processing_time=5.2,
            memory_usage=150.0,
            pages_processed=25,
            file_size=2.5
        )
        
        self.monitoring.record_processing_event(event)
        
        # Check that event was recorded
        events = self.monitoring.processing_logger.get_events()
        assert len(events) == 1
        assert events[0].file_path == 'test.pdf'
        assert events[0].status == 'success'
        
    def test_dashboard_data(self):
        """Test dashboard data generation."""
        # Record some events
        event = ProcessingEvent(
            event_type='pdf_processing',
            timestamp=datetime.now(),
            file_path='test.pdf',
            status='success',
            processing_time=5.2,
            memory_usage=150.0,
            pages_processed=25,
            file_size=2.5
        )
        
        self.monitoring.record_processing_event(event)
        
        # Get dashboard data
        dashboard = self.monitoring.get_dashboard_data()
        
        assert 'system_metrics' in dashboard
        assert 'processing_summary' in dashboard
        assert 'active_alerts' in dashboard
        assert 'performance_profile' in dashboard
        assert 'timestamp' in dashboard
        
    def test_export_metrics(self):
        """Test metrics export functionality."""
        # Record an event
        event = ProcessingEvent(
            event_type='pdf_processing',
            timestamp=datetime.now(),
            file_path='test.pdf',
            status='success',
            processing_time=5.2,
            memory_usage=150.0,
            pages_processed=25,
            file_size=2.5
        )
        
        self.monitoring.record_processing_event(event)
        
        # Export as JSON
        json_export = self.monitoring.export_metrics('json')
        assert json_export is not None
        
        # Verify it's valid JSON
        data = json.loads(json_export)
        assert 'system_metrics' in data


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_files = []
        
    def teardown_method(self):
        """Cleanup test environment."""
        TestFixtures.cleanup_temp_files(self.temp_files)
        
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test complete end-to-end processing pipeline."""
        # Create test environment
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample PDFs
        pdf_paths = []
        for i in range(2):
            pdf_path = temp_dir / f"test_{i}.pdf"
            pdf_path.write_text(f"Sample PDF content {i}")
            pdf_paths.append(pdf_path)
            
        output_dir = temp_dir / "output"
        
        # Initialize monitoring
        monitoring = MonitoringSystem()
        
        # Create processor
        config = TestFixtures.create_test_config()
        processor = MarkerPDFProcessor(config)
        
        try:
            # Process files
            results = await processor.process_batch(pdf_paths, output_dir)
            
            # Verify results
            assert len(results) == 2
            assert all(result.success for result in results)
            
            # Check output files exist
            for result in results:
                assert result.output_path.exists()
                assert len(result.markdown_content) > 0
                
            # Check batch summary
            summary_file = output_dir / "batch_processing_summary.json"
            assert summary_file.exists()
            
            # Get processing stats
            stats = await processor.get_processing_stats()
            assert stats["processor_info"]["device"] == "cpu"
            
            # Get monitoring data
            dashboard = monitoring.get_dashboard_data()
            assert 'system_metrics' in dashboard
            
        finally:
            monitoring.stop_monitoring()
            
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in processing pipeline."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create mixed files (some valid, some invalid)
        files = []
        
        # Valid PDF
        pdf_path = temp_dir / "valid.pdf"
        pdf_path.write_text("Valid PDF content")
        files.append(pdf_path)
        
        # Invalid file (non-PDF)
        txt_path = temp_dir / "invalid.txt"
        txt_path.write_text("Invalid file")
        files.append(txt_path)
        
        # Non-existent file
        nonexistent_path = temp_dir / "nonexistent.pdf"
        files.append(nonexistent_path)
        
        output_dir = temp_dir / "output"
        
        # Create processor
        config = TestFixtures.create_test_config()
        processor = MarkerPDFProcessor(config)
        
        # Process files
        results = await processor.process_batch(files, output_dir)
        
        # Verify results
        assert len(results) == 3
        
        # Check that valid file succeeded
        valid_result = next((r for r in results if r.source_path.name == "valid.pdf"), None)
        assert valid_result is not None
        assert valid_result.success is True
        
        # Check that invalid files failed
        invalid_results = [r for r in results if not r.success]
        assert len(invalid_results) == 2
        
        # Check batch summary
        summary_file = output_dir / "batch_processing_summary.json"
        assert summary_file.exists()
        
        summary_data = json.loads(summary_file.read_text())
        assert summary_data["processing_summary"]["successful"] == 1
        assert summary_data["processing_summary"]["failed"] == 2


class TestPerformanceBenchmarks:
    """Performance benchmarks for the PDF processor."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_files = []
        
    def teardown_method(self):
        """Cleanup test environment."""
        TestFixtures.cleanup_temp_files(self.temp_files)
        
    @pytest.mark.asyncio
    async def test_processing_performance(self):
        """Test processing performance with different batch sizes."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create test files
        pdf_paths = []
        for i in range(5):
            pdf_path = temp_dir / f"test_{i}.pdf"
            pdf_path.write_text(f"Sample PDF content {i}" * 100)  # Larger content
            pdf_paths.append(pdf_path)
            
        output_dir = temp_dir / "output"
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5]
        results = {}
        
        for batch_size in batch_sizes:
            config = TestFixtures.create_test_config()
            config['batch_size'] = batch_size
            
            processor = MarkerPDFProcessor(config)
            
            start_time = time.time()
            processing_results = await processor.process_batch(pdf_paths, output_dir / f"batch_{batch_size}")
            end_time = time.time()
            
            results[batch_size] = {
                'processing_time': end_time - start_time,
                'successful': len([r for r in processing_results if r.success]),
                'failed': len([r for r in processing_results if not r.success])
            }
            
        # Verify all processing completed successfully
        for batch_size, result in results.items():
            assert result['successful'] == 5
            assert result['failed'] == 0
            assert result['processing_time'] > 0
            
        # Log performance results
        print(f"Performance results: {results}")
        
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create test files
        pdf_paths = []
        for i in range(3):
            pdf_path = temp_dir / f"test_{i}.pdf"
            pdf_path.write_text(f"Sample PDF content {i}" * 1000)  # Larger content
            pdf_paths.append(pdf_path)
            
        output_dir = temp_dir / "output"
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process files
        config = TestFixtures.create_test_config()
        processor = MarkerPDFProcessor(config)
        
        results = await processor.process_batch(pdf_paths, output_dir)
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify processing succeeded
        assert all(result.success for result in results)
        
        # Memory increase should be reasonable (less than 100MB for small test files)
        assert memory_increase < 100
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB, Increase: {memory_increase:.1f}MB")


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])