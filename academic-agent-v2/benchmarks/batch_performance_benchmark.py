#!/usr/bin/env python3
"""
Performance benchmarking script for Batch Processing System
Task 27 Implementation - Measure and optimize batch processing performance
"""

import asyncio
import time
import json
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from academic_agent_v2.src.processors.batch_processor import (
    BatchProcessor, BatchStrategy, create_batch_processor
)
from academic_agent_v2.src.core.config import MarkerConfig
from academic_agent_v2.src.core.logging import get_logger


class BatchPerformanceBenchmark:
    """Comprehensive performance benchmarking for batch processor."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("batch_benchmark")
        self.results = {
            "benchmarks": [],
            "system_info": self._get_system_info(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        cpu_info = {
            "count": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
        
        memory_info = psutil.virtual_memory()
        
        return {
            "cpu": cpu_info,
            "memory": {
                "total_gb": memory_info.total / (1024**3),
                "available_gb": memory_info.available / (1024**3)
            },
            "platform": {
                "python_version": sys.version,
                "system": platform.system(),
                "release": platform.release()
            }
        }
    
    async def benchmark_worker_scaling(self, pdf_files: List[Path], output_dir: Path):
        """Benchmark performance with different numbers of workers."""
        self.logger.info("Starting worker scaling benchmark")
        
        worker_counts = [1, 2, 4, 8, 12, 16]
        results = []
        
        for worker_count in worker_counts:
            if worker_count > psutil.cpu_count():
                continue
                
            self.logger.info(f"Testing with {worker_count} workers")
            
            # Create processor with specific worker count
            config = MarkerConfig()
            processor = BatchProcessor(config)
            processor.max_workers = worker_count
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)
            
            try:
                batch_results = await processor.process_batch_async(
                    pdf_files,
                    output_dir / f"workers_{worker_count}",
                    batch_size=worker_count
                )
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**2)
                
                results.append({
                    "worker_count": worker_count,
                    "total_time": end_time - start_time,
                    "throughput_mbps": batch_results["summary"]["throughput_mbps"],
                    "jobs_per_minute": batch_results["summary"]["completed_jobs"] / ((end_time - start_time) / 60),
                    "memory_used_mb": end_memory - start_memory,
                    "success_rate": batch_results["summary"]["success_rate"],
                    "completed_jobs": batch_results["summary"]["completed_jobs"],
                    "failed_jobs": batch_results["summary"]["failed_jobs"]
                })
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {worker_count} workers: {e}")
                results.append({
                    "worker_count": worker_count,
                    "error": str(e)
                })
        
        self.results["benchmarks"].append({
            "type": "worker_scaling",
            "results": results
        })
        
        return results
    
    async def benchmark_batch_strategies(self, pdf_files: List[Path], output_dir: Path):
        """Benchmark different batch processing strategies."""
        self.logger.info("Starting batch strategy benchmark")
        
        strategies = [
            BatchStrategy.SEQUENTIAL,
            BatchStrategy.CONCURRENT,
            BatchStrategy.ADAPTIVE,
            BatchStrategy.MEMORY_AWARE
        ]
        
        results = []
        
        for strategy in strategies:
            self.logger.info(f"Testing {strategy.value} strategy")
            
            processor = create_batch_processor()
            
            start_time = time.time()
            cpu_percent_start = psutil.cpu_percent(interval=1)
            
            try:
                batch_results = await processor.process_batch_async(
                    pdf_files,
                    output_dir / f"strategy_{strategy.value}",
                    strategy=strategy
                )
                
                end_time = time.time()
                cpu_percent_avg = psutil.cpu_percent(interval=1)
                
                results.append({
                    "strategy": strategy.value,
                    "total_time": end_time - start_time,
                    "throughput_mbps": batch_results["summary"]["throughput_mbps"],
                    "cpu_usage_percent": cpu_percent_avg,
                    "workers_used": batch_results["summary"]["workers_used"],
                    "success_rate": batch_results["summary"]["success_rate"],
                    "resource_efficiency": batch_results["summary"]["throughput_mbps"] / max(1, cpu_percent_avg)
                })
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {strategy.value} strategy: {e}")
                results.append({
                    "strategy": strategy.value,
                    "error": str(e)
                })
        
        self.results["benchmarks"].append({
            "type": "batch_strategies",
            "results": results
        })
        
        return results
    
    async def benchmark_file_sizes(self, output_dir: Path):
        """Benchmark performance with different file sizes."""
        self.logger.info("Starting file size benchmark")
        
        # Create test files of different sizes
        file_sizes_mb = [1, 5, 10, 25, 50, 100]
        test_files = []
        
        for size_mb in file_sizes_mb:
            file_path = self.output_dir / f"test_file_{size_mb}mb.pdf"
            # Create dummy file (in real scenario, these would be actual PDFs)
            with open(file_path, 'wb') as f:
                f.write(b'%PDF-1.4\n' + b'x' * (size_mb * 1024 * 1024 - 9))
            test_files.append((file_path, size_mb))
        
        results = []
        processor = create_batch_processor()
        
        for file_path, size_mb in test_files:
            self.logger.info(f"Testing {size_mb}MB file")
            
            start_time = time.time()
            
            try:
                batch_results = await processor.process_batch_async(
                    [file_path],
                    output_dir / f"size_{size_mb}mb"
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                results.append({
                    "file_size_mb": size_mb,
                    "processing_time": processing_time,
                    "throughput_mbps": size_mb / processing_time,
                    "success": batch_results["summary"]["completed_jobs"] == 1
                })
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {size_mb}MB file: {e}")
                results.append({
                    "file_size_mb": size_mb,
                    "error": str(e)
                })
        
        # Cleanup test files
        for file_path, _ in test_files:
            try:
                file_path.unlink()
            except:
                pass
        
        self.results["benchmarks"].append({
            "type": "file_sizes",
            "results": results
        })
        
        return results
    
    async def benchmark_memory_pressure(self, pdf_files: List[Path], output_dir: Path):
        """Benchmark performance under memory pressure."""
        self.logger.info("Starting memory pressure benchmark")
        
        # Test with different memory limits
        memory_limits_percent = [90, 75, 50, 25]
        results = []
        
        for limit_percent in memory_limits_percent:
            self.logger.info(f"Testing with {limit_percent}% memory limit")
            
            processor = create_batch_processor()
            processor.memory_threshold_percent = limit_percent
            
            # Monitor memory during processing
            memory_samples = []
            
            async def memory_monitor():
                while processor.is_processing:
                    memory_percent = psutil.virtual_memory().percent
                    memory_samples.append(memory_percent)
                    await asyncio.sleep(1)
            
            # Start memory monitoring
            monitor_task = asyncio.create_task(memory_monitor())
            
            start_time = time.time()
            
            try:
                batch_results = await processor.process_batch_async(
                    pdf_files,
                    output_dir / f"memory_{limit_percent}pct",
                    strategy=BatchStrategy.MEMORY_AWARE
                )
                
                end_time = time.time()
                
                # Stop monitoring
                await monitor_task
                
                results.append({
                    "memory_limit_percent": limit_percent,
                    "total_time": end_time - start_time,
                    "throughput_mbps": batch_results["summary"]["throughput_mbps"],
                    "avg_memory_usage": statistics.mean(memory_samples) if memory_samples else 0,
                    "max_memory_usage": max(memory_samples) if memory_samples else 0,
                    "workers_used": batch_results["summary"]["workers_used"],
                    "success_rate": batch_results["summary"]["success_rate"]
                })
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {limit_percent}% memory limit: {e}")
                results.append({
                    "memory_limit_percent": limit_percent,
                    "error": str(e)
                })
        
        self.results["benchmarks"].append({
            "type": "memory_pressure",
            "results": results
        })
        
        return results
    
    def generate_visualizations(self):
        """Generate performance visualization charts."""
        self.logger.info("Generating performance visualizations")
        
        # Worker scaling chart
        worker_results = next((b["results"] for b in self.results["benchmarks"] 
                              if b["type"] == "worker_scaling"), [])
        
        if worker_results:
            self._plot_worker_scaling(worker_results)
        
        # Strategy comparison chart
        strategy_results = next((b["results"] for b in self.results["benchmarks"]
                                if b["type"] == "batch_strategies"), [])
        
        if strategy_results:
            self._plot_strategy_comparison(strategy_results)
        
        # File size performance chart
        size_results = next((b["results"] for b in self.results["benchmarks"]
                            if b["type"] == "file_sizes"), [])
        
        if size_results:
            self._plot_file_size_performance(size_results)
    
    def _plot_worker_scaling(self, results: List[Dict[str, Any]]):
        """Plot worker scaling performance."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return
        
        workers = [r["worker_count"] for r in valid_results]
        throughput = [r["throughput_mbps"] for r in valid_results]
        jobs_per_minute = [r["jobs_per_minute"] for r in valid_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Throughput vs Workers
        ax1.plot(workers, throughput, 'b-o', linewidth=2, markersize=8)
        ax1.set_xlabel("Number of Workers")
        ax1.set_ylabel("Throughput (MB/s)")
        ax1.set_title("Throughput vs Worker Count")
        ax1.grid(True, alpha=0.3)
        
        # Jobs per minute vs Workers
        ax2.plot(workers, jobs_per_minute, 'g-s', linewidth=2, markersize=8)
        ax2.set_xlabel("Number of Workers")
        ax2.set_ylabel("Jobs per Minute")
        ax2.set_title("Processing Rate vs Worker Count")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "worker_scaling_performance.png", dpi=300)
        plt.close()
    
    def _plot_strategy_comparison(self, results: List[Dict[str, Any]]):
        """Plot strategy comparison."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return
        
        strategies = [r["strategy"] for r in valid_results]
        throughput = [r["throughput_mbps"] for r in valid_results]
        efficiency = [r["resource_efficiency"] for r in valid_results]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, throughput, width, label='Throughput (MB/s)')
        bars2 = ax.bar(x + width/2, efficiency, width, label='Resource Efficiency')
        
        ax.set_xlabel("Batch Strategy")
        ax.set_ylabel("Performance Metric")
        ax.set_title("Batch Strategy Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "strategy_comparison.png", dpi=300)
        plt.close()
    
    def _plot_file_size_performance(self, results: List[Dict[str, Any]]):
        """Plot file size vs performance."""
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return
        
        sizes = [r["file_size_mb"] for r in valid_results]
        times = [r["processing_time"] for r in valid_results]
        throughput = [r["throughput_mbps"] for r in valid_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Processing time vs file size
        ax1.plot(sizes, times, 'r-o', linewidth=2, markersize=8)
        ax1.set_xlabel("File Size (MB)")
        ax1.set_ylabel("Processing Time (seconds)")
        ax1.set_title("Processing Time vs File Size")
        ax1.grid(True, alpha=0.3)
        
        # Throughput vs file size
        ax2.plot(sizes, throughput, 'b-s', linewidth=2, markersize=8)
        ax2.set_xlabel("File Size (MB)")
        ax2.set_ylabel("Throughput (MB/s)")
        ax2.set_title("Throughput vs File Size")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "file_size_performance.png", dpi=300)
        plt.close()
    
    def save_results(self):
        """Save benchmark results to file."""
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {results_file}")
    
    def generate_report(self):
        """Generate comprehensive benchmark report."""
        report_lines = [
            "# Batch Processing Performance Benchmark Report",
            f"\nGenerated: {self.results['timestamp']}",
            "\n## System Information",
            f"- CPUs: {self.results['system_info']['cpu']['count']} cores ({self.results['system_info']['cpu']['count_logical']} logical)",
            f"- Memory: {self.results['system_info']['memory']['total_gb']:.1f} GB total",
            f"- Platform: {self.results['system_info']['platform']['system']} {self.results['system_info']['platform']['release']}",
            "\n## Benchmark Results Summary"
        ]
        
        for benchmark in self.results["benchmarks"]:
            report_lines.append(f"\n### {benchmark['type'].replace('_', ' ').title()}")
            
            if benchmark["type"] == "worker_scaling":
                valid_results = [r for r in benchmark["results"] if "error" not in r]
                if valid_results:
                    best_throughput = max(valid_results, key=lambda x: x["throughput_mbps"])
                    report_lines.append(f"- Best throughput: {best_throughput['throughput_mbps']:.2f} MB/s with {best_throughput['worker_count']} workers")
                    
                    best_efficiency = max(valid_results, key=lambda x: x["jobs_per_minute"] / x["worker_count"])
                    report_lines.append(f"- Most efficient: {best_efficiency['worker_count']} workers ({best_efficiency['jobs_per_minute'] / best_efficiency['worker_count']:.2f} jobs/min/worker)")
            
            elif benchmark["type"] == "batch_strategies":
                valid_results = [r for r in benchmark["results"] if "error" not in r]
                if valid_results:
                    best_strategy = max(valid_results, key=lambda x: x["throughput_mbps"])
                    report_lines.append(f"- Best strategy: {best_strategy['strategy']} ({best_strategy['throughput_mbps']:.2f} MB/s)")
                    
                    most_efficient = max(valid_results, key=lambda x: x["resource_efficiency"])
                    report_lines.append(f"- Most resource efficient: {most_efficient['strategy']} (efficiency: {most_efficient['resource_efficiency']:.2f})")
        
        report_lines.append("\n## Recommendations")
        report_lines.append("Based on the benchmark results:")
        
        # Generate recommendations based on results
        worker_results = next((b["results"] for b in self.results["benchmarks"] 
                              if b["type"] == "worker_scaling"), [])
        
        if worker_results:
            valid_results = [r for r in worker_results if "error" not in r]
            if valid_results:
                throughputs = [r["throughput_mbps"] for r in valid_results]
                workers = [r["worker_count"] for r in valid_results]
                
                # Find diminishing returns point
                for i in range(1, len(throughputs)):
                    if i > 0:
                        improvement = (throughputs[i] - throughputs[i-1]) / throughputs[i-1]
                        if improvement < 0.1:  # Less than 10% improvement
                            report_lines.append(f"- Optimal worker count: {workers[i-1]} (diminishing returns beyond this)")
                            break
        
        strategy_results = next((b["results"] for b in self.results["benchmarks"]
                                if b["type"] == "batch_strategies"), [])
        
        if strategy_results:
            valid_results = [r for r in strategy_results if "error" not in r]
            if valid_results:
                best_strategy = max(valid_results, key=lambda x: x.get("resource_efficiency", 0))
                report_lines.append(f"- Recommended strategy: {best_strategy['strategy']} for best resource utilization")
        
        # Write report
        report_file = self.output_dir / "benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Benchmark report saved to {report_file}")


async def run_comprehensive_benchmark(pdf_files: List[Path], output_dir: Path):
    """Run comprehensive performance benchmark."""
    benchmark = BatchPerformanceBenchmark(output_dir)
    
    # Run all benchmarks
    await benchmark.benchmark_worker_scaling(pdf_files, output_dir / "worker_scaling")
    await benchmark.benchmark_batch_strategies(pdf_files, output_dir / "strategies")
    await benchmark.benchmark_file_sizes(output_dir / "file_sizes")
    await benchmark.benchmark_memory_pressure(pdf_files, output_dir / "memory_pressure")
    
    # Generate outputs
    benchmark.generate_visualizations()
    benchmark.save_results()
    benchmark.generate_report()
    
    return benchmark.results


# Import required modules at the top
import sys
import platform


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Processing Performance Benchmark")
    parser.add_argument("--pdf-dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    # Get PDF files
    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))[:args.max_files]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        sys.exit(1)
    
    print(f"Running benchmark with {len(pdf_files)} PDF files")
    
    # Run benchmark
    asyncio.run(run_comprehensive_benchmark(pdf_files, Path(args.output_dir)))