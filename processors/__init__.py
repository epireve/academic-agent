"""
High-Performance PDF Processor Package
Academic Agent v2 - Task 11 Implementation

This package provides a comprehensive PDF processing system using the Marker library
for superior PDF-to-markdown conversion with advanced features for academic documents.

Main Components:
    - MarkerPDFProcessor: Core processing engine
    - ChapterSplitter: Intelligent chapter detection and splitting
    - PerformanceMonitor: Performance tracking and metrics
    - MonitoringSystem: Comprehensive monitoring and alerting
    - CLI Interface: Command-line interface for easy usage

Usage:
    from processors import create_pdf_processor
    
    processor = create_pdf_processor()
    result = await processor.process_single_pdf(pdf_path, output_dir)
"""

from .marker_pdf_processor import (
    MarkerPDFProcessor,
    ProcessingResult,
    ProcessingMetrics,
    ChapterSplit,
    ChapterSplitter,
    PerformanceMonitor,
    create_pdf_processor
)

from .monitoring import (
    MonitoringSystem,
    ProcessingEvent,
    MetricsCollector,
    SystemMonitor,
    AlertManager,
    AlertLevel,
    MetricType,
    PerformanceProfiler,
    MonitoringContext
)

from .pdf_processor_cli import PDFProcessorCLI

__version__ = "1.0.0"
__author__ = "Academic Agent Team"
__description__ = "High-Performance PDF Processor using Marker Library"

__all__ = [
    # Core processing
    "MarkerPDFProcessor",
    "ProcessingResult", 
    "ProcessingMetrics",
    "ChapterSplit",
    "ChapterSplitter",
    "PerformanceMonitor",
    "create_pdf_processor",
    
    # Monitoring
    "MonitoringSystem",
    "ProcessingEvent",
    "MetricsCollector", 
    "SystemMonitor",
    "AlertManager",
    "AlertLevel",
    "MetricType",
    "PerformanceProfiler",
    "MonitoringContext",
    
    # CLI
    "PDFProcessorCLI"
]