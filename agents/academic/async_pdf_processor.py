#!/usr/bin/env python3
"""
Asynchronous PDF Processing Agent
Task 15 Implementation - High-performance async PDF processing

This module provides async PDF processing capabilities with:
- Concurrent PDF processing
- Progress tracking and cancellation
- Resource management and throttling
- Memory-efficient batch processing
- Integration with monitoring systems
"""

import asyncio
import time
import aiofiles
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging

from ...src.agents.base_agent import BaseAgent
from .async_framework import (
    AsyncTask, TaskPriority, TaskStatus, async_retry, async_timeout,
    AsyncResourceManager, AsyncProgressTracker
)

try:
    from ..processors.marker_pdf_processor import MarkerPDFProcessor, ProcessingResult, ProcessingMetrics
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    
try:
    from ..tools.pdf_processor.processor import DoclingProcessor
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


@dataclass
class AsyncProcessingResult:
    """Enhanced processing result for async operations."""
    source_path: Path
    output_path: Optional[Path] = None
    markdown_content: str = ""
    images: List[Path] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    processor_used: str = ""
    pages_processed: int = 0
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.metadata is None:
            self.metadata = {}


class AsyncPDFProcessor(BaseAgent):
    """Asynchronous PDF processor with advanced features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("async_pdf_processor")
        
        self.config = config or {}
        self.setup_config()
        
        # Initialize processors
        self.marker_processor = None
        self.docling_processor = None
        self._initialize_processors()
        
        # Resource management
        self.resource_manager = AsyncResourceManager()
        self._setup_resources()
        
        # Progress tracking
        self.progress_tracker = AsyncProgressTracker()
        
        # Processing cache
        self.processing_cache: Dict[str, AsyncProcessingResult] = {}
        self.cache_enabled = self.config.get("enable_cache", True)
        
        # Metrics
        self.processing_metrics = {
            "total_files_processed": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "total_processing_time": 0.0,
            "total_pages_processed": 0,
            "average_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger.info("Async PDF processor initialized")
    
    def setup_config(self):
        """Setup configuration with defaults."""
        defaults = {
            "max_concurrent_processes": 3,
            "max_file_size_mb": 100,
            "timeout_seconds": 600,
            "enable_cache": True,
            "cache_ttl_hours": 24,
            "preferred_processor": "marker",  # marker, docling, auto
            "output_format": "markdown",
            "extract_images": True,
            "batch_size": 5,
            "memory_limit_mb": 2048,
            "progress_update_interval": 1.0
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _initialize_processors(self):
        """Initialize available PDF processors."""
        if MARKER_AVAILABLE:
            try:
                marker_config = {
                    'device': self.config.get('device', 'cpu'),
                    'batch_size': 1,  # For individual file processing
                    'max_workers': 1,
                    'extract_images': self.config.get('extract_images', True)
                }
                self.marker_processor = MarkerPDFProcessor(marker_config)
                self.logger.info("Marker processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Marker processor: {e}")
        
        if DOCLING_AVAILABLE:
            try:
                device = self.config.get('device', 'cpu')
                self.docling_processor = DoclingProcessor(device=device)
                self.logger.info("Docling processor initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Docling processor: {e}")
        
        if not self.marker_processor and not self.docling_processor:
            self.logger.warning("No PDF processors available - using simulation mode")
    
    def _setup_resources(self):
        """Setup resource management."""
        max_concurrent = self.config.get("max_concurrent_processes", 3)
        self.resource_manager.create_semaphore("pdf_processing", max_concurrent)
        self.resource_manager.create_semaphore("memory_intensive", 1)
        self.resource_manager.create_lock("cache_access")
    
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate cache key for a file."""
        try:
            stat = file_path.stat()
            return f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
        except Exception:
            return f"{file_path.name}_{time.time()}"
    
    async def _check_cache(self, file_path: Path) -> Optional[AsyncProcessingResult]:
        """Check if file result is cached."""
        if not self.cache_enabled:
            return None
        
        cache_key = self._get_cache_key(file_path)
        
        async with self.resource_manager.acquire_lock("cache_access"):
            if cache_key in self.processing_cache:
                result = self.processing_cache[cache_key]
                # Check if cache is still valid (TTL check could be added here)
                self.processing_metrics["cache_hits"] += 1
                self.logger.debug(f"Cache hit for {file_path.name}")
                return result
        
        self.processing_metrics["cache_misses"] += 1
        return None
    
    async def _cache_result(self, file_path: Path, result: AsyncProcessingResult):
        """Cache processing result."""
        if not self.cache_enabled:
            return
        
        cache_key = self._get_cache_key(file_path)
        
        async with self.resource_manager.acquire_lock("cache_access"):
            self.processing_cache[cache_key] = result
            self.logger.debug(f"Cached result for {file_path.name}")
    
    @async_retry(max_retries=2, delay=1.0)
    @async_timeout(600)  # 10 minute timeout
    async def process_single_pdf_async(
        self, 
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> AsyncProcessingResult:
        """Process a single PDF file asynchronously."""
        
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        
        # Validate input
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # Check file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        max_size = self.config.get("max_file_size_mb", 100)
        if file_size_mb > max_size:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {max_size}MB")
        
        # Check cache first
        cached_result = await self._check_cache(pdf_path)
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        # Create progress tracking
        operation_id = f"pdf_{pdf_path.stem}_{int(time.time())}"
        self.progress_tracker.start_operation(operation_id, 100, f"Processing {pdf_path.name}")
        
        if progress_callback:
            self.progress_tracker.add_callback(operation_id, progress_callback)
        
        try:
            # Acquire processing resource
            async with self.resource_manager.acquire_resource("pdf_processing"):
                
                # Update progress
                self.progress_tracker.update_progress(operation_id, 10, "Starting PDF processing")
                
                # Create output directory
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Choose processor
                processor_name = self._choose_processor()
                self.logger.info(f"Processing {pdf_path.name} with {processor_name}")
                
                # Process based on chosen processor
                if processor_name == "marker" and self.marker_processor:
                    result = await self._process_with_marker(pdf_path, output_dir, operation_id)
                elif processor_name == "docling" and self.docling_processor:
                    result = await self._process_with_docling(pdf_path, output_dir, operation_id)
                else:
                    result = await self._process_with_simulation(pdf_path, output_dir, operation_id)
                
                # Update metrics
                processing_time = time.time() - start_time
                result.processing_time = processing_time
                result.processor_used = processor_name
                
                if result.success:
                    self.processing_metrics["successful_processes"] += 1
                    self.progress_tracker.complete_operation(operation_id, True, "PDF processing completed")
                else:
                    self.processing_metrics["failed_processes"] += 1
                    self.progress_tracker.complete_operation(operation_id, False, f"PDF processing failed: {result.error_message}")
                
                # Update global metrics
                self.processing_metrics["total_files_processed"] += 1
                self.processing_metrics["total_processing_time"] += processing_time
                self.processing_metrics["total_pages_processed"] += result.pages_processed
                
                if self.processing_metrics["successful_processes"] > 0:
                    self.processing_metrics["average_processing_time"] = (
                        self.processing_metrics["total_processing_time"] / 
                        self.processing_metrics["successful_processes"]
                    )
                
                # Cache result
                await self._cache_result(pdf_path, result)
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {pdf_path.name}: {str(e)}"
            
            self.logger.error(error_msg)
            self.progress_tracker.complete_operation(operation_id, False, error_msg)
            
            # Update metrics
            self.processing_metrics["failed_processes"] += 1
            self.processing_metrics["total_files_processed"] += 1
            
            return AsyncProcessingResult(
                source_path=pdf_path,
                processing_time=processing_time,
                success=False,
                error_message=str(e),
                processor_used="error"
            )
    
    def _choose_processor(self) -> str:
        """Choose the best available processor."""
        preferred = self.config.get("preferred_processor", "marker")
        
        if preferred == "auto":
            # Auto-select based on availability and performance
            if self.marker_processor:
                return "marker"
            elif self.docling_processor:
                return "docling"
            else:
                return "simulation"
        elif preferred == "marker" and self.marker_processor:
            return "marker"
        elif preferred == "docling" and self.docling_processor:
            return "docling"
        else:
            # Fallback to any available processor
            if self.marker_processor:
                return "marker"
            elif self.docling_processor:
                return "docling"
            else:
                return "simulation"
    
    async def _process_with_marker(
        self, 
        pdf_path: Path, 
        output_dir: Path, 
        operation_id: str
    ) -> AsyncProcessingResult:
        """Process PDF using Marker processor."""
        
        try:
            self.progress_tracker.update_progress(operation_id, 20, "Initializing Marker processor")
            
            # Initialize Marker models if needed
            await self.marker_processor.initialize_models()
            
            self.progress_tracker.update_progress(operation_id, 40, "Processing with Marker")
            
            # Process the PDF
            result = await self.marker_processor.process_single_pdf(
                pdf_path, 
                output_dir,
                chapter_splitting=True
            )
            
            self.progress_tracker.update_progress(operation_id, 90, "Finalizing Marker results")
            
            # Convert to our result format
            async_result = AsyncProcessingResult(
                source_path=pdf_path,
                output_path=result.output_path,
                markdown_content=result.markdown_content,
                images=result.images,
                metadata=result.metadata,
                success=result.success,
                error_message=result.error_message,
                pages_processed=result.metrics.pages_processed if result.metrics else 0
            )
            
            return async_result
            
        except Exception as e:
            return AsyncProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=f"Marker processing error: {str(e)}"
            )
    
    async def _process_with_docling(
        self, 
        pdf_path: Path, 
        output_dir: Path, 
        operation_id: str
    ) -> AsyncProcessingResult:
        """Process PDF using Docling processor."""
        
        try:
            self.progress_tracker.update_progress(operation_id, 30, "Processing with Docling")
            
            # Run Docling in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            markdown_path, metadata = await loop.run_in_executor(
                None,
                self.docling_processor.process_pdf,
                str(pdf_path),
                str(output_dir),
                True  # rename_smartly
            )
            
            self.progress_tracker.update_progress(operation_id, 80, "Reading Docling results")
            
            # Read the generated content
            markdown_content = ""
            if markdown_path and Path(markdown_path).exists():
                async with aiofiles.open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_content = await f.read()
            
            # Extract images (if any)
            images = []
            if markdown_path:
                images_dir = Path(markdown_path).parent / f"{Path(markdown_path).stem}_images"
                if images_dir.exists():
                    images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            
            return AsyncProcessingResult(
                source_path=pdf_path,
                output_path=Path(markdown_path) if markdown_path else None,
                markdown_content=markdown_content,
                images=images,
                metadata=metadata or {},
                success=bool(markdown_path),
                error_message=None if markdown_path else "No output generated",
                pages_processed=10  # Estimate since Docling doesn't provide this
            )
            
        except Exception as e:
            return AsyncProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=f"Docling processing error: {str(e)}"
            )
    
    async def _process_with_simulation(
        self, 
        pdf_path: Path, 
        output_dir: Path, 
        operation_id: str
    ) -> AsyncProcessingResult:
        """Simulate PDF processing when no real processors are available."""
        
        try:
            file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
            
            # Simulate processing time based on file size
            processing_time = min(file_size_mb * 0.1, 3.0)
            
            for i in range(10):
                await asyncio.sleep(processing_time / 10)
                progress = 20 + (i * 7)  # 20% to 90%
                self.progress_tracker.update_progress(operation_id, progress, f"Simulating processing step {i+1}/10")
            
            # Create simulated output
            output_file = output_dir / f"{pdf_path.stem}.md"
            
            simulated_content = f"""# {pdf_path.stem}

## Simulated Processing Result

This is a simulated result for demonstration purposes.

**File Information:**
- Source: {pdf_path.name}
- Size: {file_size_mb:.2f} MB
- Processed: {datetime.now().isoformat()}

## Content Summary

This would contain the actual extracted content from the PDF.
The simulation indicates that the async processing framework is working correctly.

### Processing Details
- Estimated pages: {max(1, int(file_size_mb * 10))}
- Processing time: {processing_time:.2f}s
- Status: Simulation completed successfully
"""
            
            # Write simulated content
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write(simulated_content)
            
            return AsyncProcessingResult(
                source_path=pdf_path,
                output_path=output_file,
                markdown_content=simulated_content,
                images=[],
                metadata={"simulated": True, "file_size_mb": file_size_mb},
                success=True,
                pages_processed=max(1, int(file_size_mb * 10))
            )
            
        except Exception as e:
            return AsyncProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=f"Simulation error: {str(e)}"
            )
    
    async def process_batch_async(
        self, 
        pdf_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[AsyncProcessingResult]:
        """Process multiple PDFs concurrently."""
        
        output_dir = Path(output_dir)
        pdf_paths = [Path(p) for p in pdf_paths]
        
        self.logger.info(f"Starting batch processing of {len(pdf_paths)} PDFs")
        
        # Create semaphore for batch concurrency control
        batch_size = self.config.get("batch_size", 5)
        semaphore = asyncio.Semaphore(min(batch_size, len(pdf_paths)))
        
        async def process_single(index: int, pdf_path: Path):
            async with semaphore:
                result = await self.process_single_pdf_async(
                    pdf_path, 
                    output_dir / pdf_path.stem
                )
                
                if progress_callback:
                    progress_callback(index + 1, len(pdf_paths))
                
                return result
        
        # Create tasks for all PDFs
        tasks = [
            process_single(i, pdf_path) 
            for i, pdf_path in enumerate(pdf_paths)
        ]
        
        # Process all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(AsyncProcessingResult(
                    source_path=pdf_paths[i],
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        # Generate batch summary
        await self._generate_batch_summary(final_results, output_dir)
        
        self.logger.info(f"Batch processing completed: {len(final_results)} files processed")
        return final_results
    
    async def _generate_batch_summary(
        self, 
        results: List[AsyncProcessingResult], 
        output_dir: Path
    ):
        """Generate batch processing summary."""
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        summary = {
            "batch_summary": {
                "total_files": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0,
                "total_processing_time": sum(r.processing_time for r in results),
                "average_processing_time": (
                    sum(r.processing_time for r in results) / len(results) 
                    if results else 0
                ),
                "total_pages": sum(r.pages_processed for r in successful),
                "timestamp": datetime.now().isoformat()
            },
            "successful_files": [
                {
                    "filename": r.source_path.name,
                    "output_path": str(r.output_path) if r.output_path else None,
                    "pages": r.pages_processed,
                    "processing_time": r.processing_time,
                    "processor": r.processor_used,
                    "images_extracted": len(r.images)
                }
                for r in successful
            ],
            "failed_files": [
                {
                    "filename": r.source_path.name,
                    "error": r.error_message,
                    "processing_time": r.processing_time
                }
                for r in failed
            ],
            "metrics": self.get_processing_metrics()
        }
        
        summary_file = output_dir / "async_batch_summary.json"
        async with aiofiles.open(summary_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(summary, indent=2))
        
        self.logger.info(f"Batch summary saved to {summary_file}")
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics."""
        return {
            **self.processing_metrics,
            "cache_efficiency": (
                self.processing_metrics["cache_hits"] / 
                max(1, self.processing_metrics["cache_hits"] + self.processing_metrics["cache_misses"])
            ),
            "cached_results": len(self.processing_cache),
            "resource_usage": self.resource_manager.get_resource_stats()
        }
    
    async def clear_cache(self):
        """Clear processing cache."""
        async with self.resource_manager.acquire_lock("cache_access"):
            self.processing_cache.clear()
            self.logger.info("Processing cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the processor."""
        
        health = {
            "status": "healthy",
            "processors": {
                "marker": "available" if self.marker_processor else "unavailable",
                "docling": "available" if self.docling_processor else "unavailable"
            },
            "metrics": self.get_processing_metrics(),
            "config": {
                "max_concurrent": self.config.get("max_concurrent_processes", 3),
                "cache_enabled": self.cache_enabled,
                "preferred_processor": self.config.get("preferred_processor", "marker")
            }
        }
        
        # Check if any processors are available
        if not self.marker_processor and not self.docling_processor:
            health["status"] = "degraded"
            health["warnings"] = ["No PDF processors available - using simulation mode"]
        
        return health


# Utility functions
async def process_pdfs_async(
    pdf_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[AsyncProcessingResult]:
    """Convenience function to process PDFs asynchronously."""
    
    processor = AsyncPDFProcessor(config)
    return await processor.process_batch_async(pdf_paths, output_dir, progress_callback)


# Example usage
async def main():
    """Example usage of async PDF processor."""
    
    # Configuration
    config = {
        "max_concurrent_processes": 2,
        "preferred_processor": "simulation",  # Use simulation for demo
        "enable_cache": True,
        "batch_size": 3
    }
    
    # Create processor
    processor = AsyncPDFProcessor(config)
    
    # Example files (these would be real PDF files)
    test_files = [
        "test1.pdf",
        "test2.pdf", 
        "test3.pdf"
    ]
    
    output_dir = Path("output/async_processed")
    
    def progress_callback(completed: int, total: int):
        print(f"Progress: {completed}/{total} files processed ({completed/total:.1%})")
    
    try:
        # Process batch
        results = await processor.process_batch_async(
            test_files, 
            output_dir,
            progress_callback
        )
        
        # Print results
        for result in results:
            if result.success:
                print(f"✓ {result.source_path.name} processed in {result.processing_time:.2f}s")
            else:
                print(f"✗ {result.source_path.name} failed: {result.error_message}")
        
        # Print metrics
        metrics = processor.get_processing_metrics()
        print(f"\nMetrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())