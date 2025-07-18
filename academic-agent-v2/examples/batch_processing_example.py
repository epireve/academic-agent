#!/usr/bin/env python3
"""
Example usage of the Batch Processing System
Task 27 Implementation - Demonstrates batch processing capabilities
"""

import asyncio
import time
from pathlib import Path
from typing import List
import json

from academic_agent_v2.src.processors.batch_processor import (
    BatchProcessor, BatchStrategy, BatchProgress, create_batch_processor, process_pdfs_batch
)
from academic_agent_v2.src.core.config import MarkerConfig
from academic_agent_v2.src.core.logging import get_logger


# Setup logger
logger = get_logger("batch_example")


def progress_callback(progress: BatchProgress):
    """Example progress callback function."""
    print(f"\rProgress: {progress.progress_percent:.1f}% | "
          f"Completed: {progress.completed_jobs}/{progress.total_jobs} | "
          f"Failed: {progress.failed_jobs} | "
          f"Throughput: {progress.throughput_mbps:.2f} MB/s", end="")


async def example_basic_batch_processing():
    """Example 1: Basic batch processing with default settings."""
    print("\n=== Example 1: Basic Batch Processing ===")
    
    # Setup paths
    pdf_dir = Path("data/pdfs")  # Your PDF directory
    output_dir = Path("output/batch_basic")
    
    # Get PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))[:5]  # Process first 5 PDFs
    
    if not pdf_files:
        logger.warning("No PDF files found in data/pdfs directory")
        return
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    # Create batch processor
    processor = create_batch_processor()
    
    # Add progress callback
    processor.add_progress_callback(progress_callback)
    
    # Process batch
    start_time = time.time()
    results = await processor.process_batch_async(pdf_files, output_dir)
    
    print(f"\n\nBatch processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Success rate: {results['summary']['success_rate']:.1f}%")
    print(f"Throughput: {results['summary']['throughput_mbps']:.2f} MB/s")
    
    # Save results
    results_file = output_dir / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


async def example_custom_configuration():
    """Example 2: Batch processing with custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Setup paths
    pdf_dir = Path("data/pdfs")
    output_dir = Path("output/batch_custom")
    
    # Get PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))[:10]
    
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    # Create custom configuration
    config = MarkerConfig(
        batch_size=4,           # Process 4 files concurrently
        device="cpu",           # Use CPU instead of GPU
        enable_filtering=True,  # Enable advanced filtering
        enable_equations=True   # Enable equation processing
    )
    
    # Create processor with custom config
    processor = BatchProcessor(config)
    processor.max_workers = 4  # Limit to 4 workers
    processor.memory_threshold_percent = 70.0  # More aggressive memory management
    
    print(f"Processing {len(pdf_files)} PDFs with custom configuration...")
    print(f"- Max workers: {processor.max_workers}")
    print(f"- Memory threshold: {processor.memory_threshold_percent}%")
    print(f"- Device: {config.device}")
    
    # Process with specific strategy
    results = await processor.process_batch_async(
        pdf_files,
        output_dir,
        strategy=BatchStrategy.MEMORY_AWARE
    )
    
    print(f"\nCompleted: {results['summary']['completed_jobs']} files")
    print(f"Failed: {results['summary']['failed_jobs']} files")
    
    # Display failed jobs if any
    if results['failed_jobs']:
        print("\nFailed jobs:")
        for job in results['failed_jobs']:
            print(f"  - {job['pdf_path']}: {job['error']}")


async def example_adaptive_processing():
    """Example 3: Adaptive batch processing based on system resources."""
    print("\n=== Example 3: Adaptive Processing ===")
    
    # Setup paths
    pdf_dir = Path("data/pdfs")
    output_dir = Path("output/batch_adaptive")
    
    # Get all PDF files
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    print(f"Processing {len(pdf_files)} PDFs with adaptive strategy...")
    
    # Use utility function with progress callback
    def detailed_progress(progress: BatchProgress):
        """Detailed progress reporting."""
        print(f"\rProgress: {progress.progress_percent:.1f}% | "
              f"Active workers: {progress.active_workers} | "
              f"Current jobs: {len(progress.current_jobs)} | "
              f"Est. completion: {progress.estimated_completion.strftime('%H:%M:%S') if progress.estimated_completion else 'calculating...'}", 
              end="")
    
    # Process with adaptive strategy
    results = await process_pdfs_batch(
        pdf_files,
        output_dir,
        progress_callback=detailed_progress
    )
    
    print(f"\n\nAdaptive processing completed!")
    print(f"Total processing time: {results['summary']['total_processing_time']:.2f}s")
    print(f"Average time per job: {results['summary']['average_time_per_job']:.2f}s")
    
    # Display performance metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print("\nPerformance Metrics:")
        print(f"  - Min processing time: {metrics['min_processing_time']:.2f}s")
        print(f"  - Max processing time: {metrics['max_processing_time']:.2f}s")
        print(f"  - Avg processing time: {metrics['avg_processing_time']:.2f}s")
        print(f"  - Jobs per minute: {metrics['jobs_per_minute']:.1f}")


async def example_batch_with_retry():
    """Example 4: Batch processing with retry mechanism."""
    print("\n=== Example 4: Batch Processing with Retry ===")
    
    # Setup paths
    pdf_dir = Path("data/pdfs")
    output_dir = Path("output/batch_retry")
    
    # Include some problematic files to test retry
    pdf_files = list(pdf_dir.glob("*.pdf"))[:5]
    
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    # Create processor with custom retry settings
    processor = create_batch_processor()
    
    # Modify retry settings for all jobs
    for job in processor.jobs.values():
        job.max_retries = 5  # Increase retry attempts
    
    print(f"Processing {len(pdf_files)} PDFs with enhanced retry mechanism...")
    
    # Track retries
    retry_count = 0
    
    def retry_progress(progress: BatchProgress):
        nonlocal retry_count
        # This is a simplified example - in real implementation, 
        # you would track retries through the result queue
        print(f"\rProgress: {progress.progress_percent:.1f}% | "
              f"Success rate: {progress.success_rate:.1f}%", end="")
    
    processor.add_progress_callback(retry_progress)
    
    results = await processor.process_batch_async(
        pdf_files,
        output_dir,
        strategy=BatchStrategy.CONCURRENT
    )
    
    print(f"\n\nProcessing completed with retry mechanism")
    print(f"Final success rate: {results['summary']['success_rate']:.1f}%")
    
    # Analyze retry patterns
    if results['failed_jobs']:
        print(f"\nFailed after retries: {len(results['failed_jobs'])} files")
        for job in results['failed_jobs']:
            print(f"  - {Path(job['pdf_path']).name}: {job['retry_count']} retries")


async def example_real_time_monitoring():
    """Example 5: Real-time monitoring and status checking."""
    print("\n=== Example 5: Real-time Monitoring ===")
    
    # Setup paths
    pdf_dir = Path("data/pdfs")
    output_dir = Path("output/batch_monitoring")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))[:20]  # Process 20 files
    
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    print(f"Processing {len(pdf_files)} PDFs with real-time monitoring...")
    
    # Create processor
    processor = create_batch_processor()
    
    # Start processing in background
    process_task = asyncio.create_task(
        processor.process_batch_async(pdf_files, output_dir)
    )
    
    # Monitor status in real-time
    print("\nMonitoring batch status (press Ctrl+C to stop):")
    try:
        while not process_task.done():
            status = processor.get_batch_status()
            
            if status['is_processing'] and status['progress']:
                progress = status['progress']
                print(f"\rStatus: Processing | "
                      f"Progress: {progress['progress_percent']:.1f}% | "
                      f"Active workers: {progress['active_workers']} | "
                      f"Throughput: {progress['throughput_mbps']:.2f} MB/s | "
                      f"Queue: {status['queued_jobs']}     ", end="")
            
            await asyncio.sleep(1)
        
        # Get final results
        results = await process_task
        print(f"\n\nBatch processing completed!")
        print(f"Total jobs: {results['summary']['total_jobs']}")
        print(f"Completed: {results['summary']['completed_jobs']}")
        print(f"Failed: {results['summary']['failed_jobs']}")
        
    except KeyboardInterrupt:
        print("\n\nCancelling batch processing...")
        processor.cancel_batch()
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass
        print("Batch processing cancelled")


async def example_resource_aware_processing():
    """Example 6: Resource-aware batch processing."""
    print("\n=== Example 6: Resource-aware Processing ===")
    
    # Import psutil for resource monitoring
    import psutil
    
    # Setup paths
    pdf_dir = Path("data/pdfs")
    output_dir = Path("output/batch_resource_aware")
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found")
        return
    
    print(f"Processing {len(pdf_files)} PDFs with resource awareness...")
    print(f"System resources:")
    print(f"  - CPU cores: {psutil.cpu_count()}")
    print(f"  - Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"  - Current CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    # Create processor with resource-aware settings
    processor = create_batch_processor()
    
    # Track resource usage
    resource_samples = []
    
    def resource_monitor_callback(progress: BatchProgress):
        # Collect resource usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        resource_samples.append({
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'active_workers': progress.active_workers
        })
        
        print(f"\rProgress: {progress.progress_percent:.1f}% | "
              f"CPU: {cpu_percent:.1f}% | "
              f"Memory: {memory_percent:.1f}% | "
              f"Workers: {progress.active_workers}", end="")
    
    processor.add_progress_callback(resource_monitor_callback)
    
    # Process with memory-aware strategy
    results = await processor.process_batch_async(
        pdf_files,
        output_dir,
        strategy=BatchStrategy.MEMORY_AWARE
    )
    
    print(f"\n\nResource-aware processing completed!")
    print(f"Success rate: {results['summary']['success_rate']:.1f}%")
    
    # Analyze resource usage
    if resource_samples:
        avg_cpu = sum(s['cpu_percent'] for s in resource_samples) / len(resource_samples)
        avg_memory = sum(s['memory_percent'] for s in resource_samples) / len(resource_samples)
        max_cpu = max(s['cpu_percent'] for s in resource_samples)
        max_memory = max(s['memory_percent'] for s in resource_samples)
        
        print("\nResource usage statistics:")
        print(f"  - Average CPU: {avg_cpu:.1f}%")
        print(f"  - Maximum CPU: {max_cpu:.1f}%")
        print(f"  - Average Memory: {avg_memory:.1f}%")
        print(f"  - Maximum Memory: {max_memory:.1f}%")
        
        # Save resource data
        resource_file = output_dir / "resource_usage.json"
        with open(resource_file, 'w') as f:
            json.dump(resource_samples, f, indent=2)
        print(f"\nResource usage data saved to {resource_file}")


async def main():
    """Run all examples."""
    print("Batch Processing System Examples")
    print("=" * 50)
    
    # Create test data directory
    test_data_dir = Path("data/pdfs")
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some dummy PDF files for testing if none exist
    if not list(test_data_dir.glob("*.pdf")):
        print("Creating test PDF files...")
        for i in range(10):
            pdf_file = test_data_dir / f"test_document_{i:03d}.pdf"
            # Create minimal valid PDF
            pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 6 0 R >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document {i}) Tj
ET
endstream
endobj
6 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000257 00000 n
0000000359 00000 n
trailer
<< /Size 7 /Root 1 0 R >>
startxref
444
%%EOF"""
            pdf_file.write_text(pdf_content)
        print(f"Created {i+1} test PDF files in {test_data_dir}")
    
    # Run examples
    examples = [
        ("Basic Batch Processing", example_basic_batch_processing),
        ("Custom Configuration", example_custom_configuration),
        ("Adaptive Processing", example_adaptive_processing),
        ("Batch with Retry", example_batch_with_retry),
        ("Real-time Monitoring", example_real_time_monitoring),
        ("Resource-aware Processing", example_resource_aware_processing)
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\nRunning Example {i}: {name}")
        print("-" * 50)
        
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example failed: {e}")
        
        if i < len(examples):
            print("\nPress Enter to continue to next example...")
            input()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())