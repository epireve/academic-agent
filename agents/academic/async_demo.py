#!/usr/bin/env python3
"""
Asynchronous Academic Agent System Demo
Task 15 Implementation - Complete demonstration of async capabilities

This script demonstrates the full async academic agent system with:
- Parallel PDF processing
- Concurrent content generation
- Real-time monitoring and progress tracking
- Performance optimization and resource management
- Health checks and system diagnostics
"""

import asyncio
import time
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Import async components
from .async_main_agent import AsyncMainAcademicAgent, create_and_start_async_agent
from .async_pdf_processor import AsyncPDFProcessor
from .async_content_generator import AsyncContentGenerator
from .async_monitoring import AsyncMonitoringSystem, create_monitoring_system


class AsyncAcademicAgentDemo:
    """Comprehensive demo of the async academic agent system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.temp_dir = None
        self.demo_files = []
        
        # Components
        self.main_agent: Optional[AsyncMainAcademicAgent] = None
        self.pdf_processor: Optional[AsyncPDFProcessor] = None
        self.content_generator: Optional[AsyncContentGenerator] = None
        self.monitoring_system: Optional[AsyncMonitoringSystem] = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger("async_demo")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for demo."""
        return {
            "worker_pool_size": 3,
            "task_queue_size": 100,
            "max_concurrent_workflows": 2,
            "pdf_processor": {
                "max_concurrent_processes": 2,
                "preferred_processor": "simulation",
                "enable_cache": True,
                "batch_size": 3
            },
            "content_generator": {
                "max_concurrent_generations": 2,
                "enable_cache": True,
                "ai_model": "simulation"
            },
            "monitoring": {
                "metrics_buffer_size": 1000,
                "metrics_flush_interval": 10.0,
                "system_check_interval": 5.0
            }
        }
    
    def setup_logging(self):
        """Setup logging for demo."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('async_demo.log')
            ]
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize demo components."""
        self.logger.info("Initializing async academic agent demo...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="async_demo_"))
        self.logger.info(f"Created temp directory: {self.temp_dir}")
        
        # Create demo files
        await self._create_demo_files()
        
        # Initialize monitoring first
        self.monitoring_system = await create_monitoring_system(self.config.get("monitoring"))
        self.logger.info("Monitoring system initialized")
        
        # Initialize main agent
        self.main_agent = await create_and_start_async_agent(None)
        self.logger.info("Main agent initialized")
        
        # Initialize PDF processor
        self.pdf_processor = AsyncPDFProcessor(self.config.get("pdf_processor"))
        self.logger.info("PDF processor initialized")
        
        # Initialize content generator
        self.content_generator = AsyncContentGenerator(self.config.get("content_generator"))
        self.logger.info("Content generator initialized")
        
        # Start monitoring the main agent's worker pool
        if self.main_agent and self.monitoring_system:
            asyncio.create_task(
                self.monitoring_system.monitor_worker_pool(self.main_agent.worker_pool)
            )
        
        self.logger.info("Demo initialization completed")
    
    async def cleanup(self):
        """Cleanup demo resources."""
        self.logger.info("Cleaning up demo resources...")
        
        # Stop components
        if self.main_agent:
            await self.main_agent.stop()
        
        if self.monitoring_system:
            await self.monitoring_system.stop()
        
        # Clean up temp directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        
        self.logger.info("Demo cleanup completed")
    
    async def _create_demo_files(self):
        """Create demo PDF files (simulated with text files)."""
        demo_content = [
            {
                "filename": "chapter1_introduction.pdf",
                "content": """# Introduction to Academic Processing

## Overview
This chapter introduces the fundamental concepts of academic document processing
and the importance of efficient content extraction and analysis.

## Key Concepts
- Document Structure Analysis
- Content Extraction Techniques
- Quality Assessment Metrics
- Processing Optimization

## Learning Objectives
By the end of this chapter, you will understand:
1. The importance of structured document processing
2. Various extraction methodologies
3. Quality control mechanisms
4. Performance optimization strategies

## Summary
Academic document processing requires a systematic approach that balances
accuracy with efficiency. This introduction sets the foundation for
understanding advanced processing techniques."""
            },
            {
                "filename": "chapter2_methodology.pdf", 
                "content": """# Methodology and Approach

## Research Framework
This chapter outlines the methodological approach used in academic
content processing and analysis systems.

## Core Principles
- **Accuracy**: Ensuring high fidelity content extraction
- **Efficiency**: Optimizing processing speed and resource usage
- **Scalability**: Supporting large-scale document processing
- **Reliability**: Maintaining consistent results across diverse inputs

## Processing Pipeline
1. **Ingestion**: Document acquisition and preprocessing
2. **Analysis**: Content structure identification and parsing
3. **Extraction**: Key information and concept identification
4. **Synthesis**: Comprehensive content consolidation
5. **Quality Control**: Validation and improvement processes

## Implementation Considerations
- Resource management and optimization
- Error handling and recovery mechanisms
- Progress tracking and user feedback
- Monitoring and performance metrics

## Conclusion
A well-designed methodology ensures consistent, high-quality results
while maintaining system efficiency and user satisfaction."""
            },
            {
                "filename": "chapter3_implementation.pdf",
                "content": """# Implementation Details

## System Architecture
This chapter describes the technical implementation of the async
academic processing system.

## Asynchronous Processing Benefits
- **Concurrency**: Multiple documents processed simultaneously
- **Responsiveness**: Non-blocking operations maintain system availability
- **Scalability**: Efficient resource utilization for large workloads
- **Fault Tolerance**: Isolated failures don't affect entire system

## Key Components
### Task Management
- Priority-based task queuing
- Dependency resolution
- Progress tracking and cancellation support

### Worker Pool Management
- Dynamic worker allocation
- Load balancing across processing units
- Resource monitoring and optimization

### Content Processing Pipeline
- Parallel PDF ingestion
- Concurrent analysis and outline generation
- Asynchronous notes creation with quality assessment

## Performance Optimization
- Caching mechanisms for improved efficiency
- Resource-aware processing limits
- Memory management and garbage collection
- Real-time monitoring and alerting

## Monitoring and Diagnostics
- Comprehensive metrics collection
- Health checks and system status reporting
- Performance analysis and optimization recommendations

## Summary
The async implementation provides significant performance improvements
while maintaining high quality standards and system reliability."""
            }
        ]
        
        for content_info in demo_content:
            file_path = self.temp_dir / content_info["filename"]
            
            # For demo purposes, we'll create text files instead of PDFs
            # In real usage, these would be actual PDF files
            async with asyncio.to_thread(open, file_path, 'w', encoding='utf-8') as f:
                await asyncio.to_thread(f.write, content_info["content"])
            
            self.demo_files.append(str(file_path))
        
        self.logger.info(f"Created {len(self.demo_files)} demo files")
    
    async def run_pdf_processing_demo(self) -> Dict[str, Any]:
        """Demonstrate async PDF processing capabilities."""
        self.logger.info("\n" + "="*60)
        self.logger.info("ASYNC PDF PROCESSING DEMO")
        self.logger.info("="*60)
        
        start_time = time.time()
        output_dir = self.temp_dir / "pdf_output"
        
        def progress_callback(completed: int, total: int):
            progress = completed / total
            self.logger.info(f"PDF Processing Progress: {completed}/{total} ({progress:.1%})")
        
        try:
            # Monitor the operation
            async with self.monitoring_system.monitor_operation("pdf_batch_processing"):
                results = await self.pdf_processor.process_batch_async(
                    self.demo_files,
                    output_dir,
                    progress_callback
                )
            
            processing_time = time.time() - start_time
            
            # Analyze results
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            total_pages = sum(r.pages_processed for r in successful)
            avg_processing_time = sum(r.processing_time for r in successful) / len(successful) if successful else 0
            
            self.logger.info(f"\nPDF Processing Results:")
            self.logger.info(f"  Files processed: {len(results)}")
            self.logger.info(f"  Successful: {len(successful)}")
            self.logger.info(f"  Failed: {len(failed)}")
            self.logger.info(f"  Total pages: {total_pages}")
            self.logger.info(f"  Total time: {processing_time:.2f}s")
            self.logger.info(f"  Average time per file: {avg_processing_time:.2f}s")
            
            # Get processor metrics
            metrics = self.pdf_processor.get_processing_metrics()
            self.logger.info(f"  Cache hit rate: {metrics.get('cache_efficiency', 0):.1%}")
            
            return {
                "results": results,
                "metrics": metrics,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"PDF processing demo failed: {e}")
            return {"error": str(e), "success": False}
    
    async def run_content_generation_demo(self, pdf_results: List[Any]) -> Dict[str, Any]:
        """Demonstrate async content generation capabilities."""
        self.logger.info("\n" + "="*60)
        self.logger.info("ASYNC CONTENT GENERATION DEMO")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        def progress_callback(progress: float, message: str):
            self.logger.info(f"Content Generation Progress: {progress:.1%} - {message}")
        
        try:
            # Extract markdown content from PDF results
            markdown_contents = []
            for result in pdf_results:
                if result.success and result.markdown_content:
                    markdown_contents.append(result.markdown_content)
            
            if not markdown_contents:
                raise ValueError("No markdown content available for processing")
            
            # Step 1: Analyze content
            self.logger.info("Starting content analysis...")
            
            analysis_tasks = []
            for i, content in enumerate(markdown_contents):
                source_file = f"demo_file_{i+1}.md"
                task = self.content_generator.analyze_content_async(
                    content, 
                    source_file,
                    progress_callback
                )
                analysis_tasks.append(task)
            
            async with self.monitoring_system.monitor_operation("content_analysis"):
                analysis_results = await asyncio.gather(*analysis_tasks)
            
            successful_analyses = [r for r in analysis_results if r.success]
            self.logger.info(f"Analyzed {len(successful_analyses)}/{len(analysis_results)} files successfully")
            
            if not successful_analyses:
                raise ValueError("All content analysis failed")
            
            # Step 2: Generate outline
            self.logger.info("Generating consolidated outline...")
            
            async with self.monitoring_system.monitor_operation("outline_generation"):
                outline_result = await self.content_generator.generate_outline_async(
                    successful_analyses,
                    target_depth=3,
                    progress_callback=progress_callback
                )
            
            if not outline_result.success:
                raise ValueError(f"Outline generation failed: {outline_result.error_message}")
            
            self.logger.info(f"Generated outline with {outline_result.estimated_sections} sections")
            
            # Step 3: Generate comprehensive notes
            self.logger.info("Generating comprehensive notes...")
            
            async with self.monitoring_system.monitor_operation("notes_generation"):
                notes_result = await self.content_generator.generate_notes_async(
                    successful_analyses,
                    outline_result,
                    target_length=3000,
                    progress_callback=progress_callback
                )
            
            if not notes_result.success:
                raise ValueError(f"Notes generation failed: {notes_result.error_message}")
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"\nContent Generation Results:")
            self.logger.info(f"  Analysis completed: {len(successful_analyses)}")
            self.logger.info(f"  Outline sections: {outline_result.estimated_sections}")
            self.logger.info(f"  Notes word count: {notes_result.word_count}")
            self.logger.info(f"  Notes sections: {notes_result.sections_count}")
            self.logger.info(f"  Total processing time: {processing_time:.2f}s")
            
            # Get generator metrics
            metrics = await self.content_generator.get_generation_metrics()
            self.logger.info(f"  Success rate: {metrics.get('success_rate', 0):.1%}")
            self.logger.info(f"  AI API calls: {metrics.get('ai_api_calls', 0)}")
            
            # Save generated content
            output_dir = self.temp_dir / "content_output"
            output_dir.mkdir(exist_ok=True)
            
            outline_file = output_dir / "generated_outline.md"
            notes_file = output_dir / "comprehensive_notes.md"
            
            async with asyncio.to_thread(open, outline_file, 'w', encoding='utf-8') as f:
                await asyncio.to_thread(f.write, outline_result.outline_content)
            
            async with asyncio.to_thread(open, notes_file, 'w', encoding='utf-8') as f:
                await asyncio.to_thread(f.write, notes_result.notes_content)
            
            self.logger.info(f"  Generated content saved to: {output_dir}")
            
            return {
                "analysis_results": successful_analyses,
                "outline_result": outline_result,
                "notes_result": notes_result,
                "metrics": metrics,
                "processing_time": processing_time,
                "output_files": {
                    "outline": str(outline_file),
                    "notes": str(notes_file)
                },
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Content generation demo failed: {e}")
            return {"error": str(e), "success": False}
    
    async def run_workflow_demo(self) -> Dict[str, Any]:
        """Demonstrate complete async workflow processing."""
        self.logger.info("\n" + "="*60)
        self.logger.info("ASYNC WORKFLOW DEMO")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        def progress_callback(progress: float, message: str):
            self.logger.info(f"Workflow Progress: {progress:.1%} - {message}")
        
        try:
            # Use main agent to process complete workflow
            workflow_id = f"demo_workflow_{int(time.time())}"
            
            workflow_config = {
                "quality_threshold": 0.6,  # Lower for demo
                "max_improvement_cycles": 2
            }
            
            async with self.monitoring_system.monitor_operation("complete_workflow"):
                workflow_result = await self.main_agent.process_workflow_async(
                    workflow_id,
                    self.demo_files,
                    workflow_config,
                    progress_callback
                )
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"\nWorkflow Results:")
            self.logger.info(f"  Workflow ID: {workflow_id}")
            self.logger.info(f"  Input files: {len(self.demo_files)}")
            self.logger.info(f"  Processing time: {processing_time:.2f}s")
            
            # Display workflow stages
            for stage, result in workflow_result.items():
                if isinstance(result, list):
                    self.logger.info(f"  {stage.capitalize()}: {len(result)} items processed")
                elif isinstance(result, dict):
                    if 'success' in result:
                        status = "✓" if result['success'] else "✗"
                        self.logger.info(f"  {stage.capitalize()}: {status}")
            
            return {
                "workflow_result": workflow_result,
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Workflow demo failed: {e}")
            return {"error": str(e), "success": False}
    
    async def run_performance_monitoring_demo(self) -> Dict[str, Any]:
        """Demonstrate performance monitoring capabilities."""
        self.logger.info("\n" + "="*60)
        self.logger.info("PERFORMANCE MONITORING DEMO")
        self.logger.info("="*60)
        
        try:
            # Get comprehensive system status
            status = await self.monitoring_system.get_comprehensive_status()
            
            self.logger.info("System Status:")
            self.logger.info(f"  Monitoring active: {status['monitoring_system']['status']}")
            self.logger.info(f"  Metrics collected: {status['monitoring_system']['metrics_collected']}")
            
            if 'task_monitoring' in status:
                task_stats = status['task_monitoring']
                self.logger.info(f"  Active tasks: {task_stats.get('active_tasks', 0)}")
                self.logger.info(f"  Completed tasks: {task_stats.get('total_completed', 0)}")
                self.logger.info(f"  Success rate: {task_stats.get('success_rate', 0):.1%}")
            
            if 'system_performance' in status:
                perf = status['system_performance']
                self.logger.info(f"  CPU usage: {perf.get('cpu_percent', 0):.1f}%")
                self.logger.info(f"  Memory usage: {perf.get('memory_percent', 0):.1f}%")
            
            # Generate performance report
            report = await self.monitoring_system.generate_performance_report(1)  # 1 hour
            
            if 'error' not in report:
                self.logger.info("\nPerformance Report:")
                self.logger.info(f"  Report period: {report['report_period']['duration_hours']} hours")
                self.logger.info(f"  Data points: {report['report_period']['data_points']}")
                
                if 'cpu_statistics' in report:
                    cpu_stats = report['cpu_statistics']
                    self.logger.info(f"  CPU - Avg: {cpu_stats['average']:.1f}%, Max: {cpu_stats['max']:.1f}%")
                
                if 'memory_statistics' in report:
                    mem_stats = report['memory_statistics']
                    self.logger.info(f"  Memory - Avg: {mem_stats['average']:.1f}%, Max: {mem_stats['max']:.1f}%")
            
            # Check health of individual components
            health_checks = {}
            
            if self.pdf_processor:
                health_checks['pdf_processor'] = await self.pdf_processor.health_check()
            
            if self.content_generator:
                health_checks['content_generator'] = await self.content_generator.health_check()
            
            if self.main_agent:
                health_checks['main_agent'] = await self.main_agent.health_check()
            
            self.logger.info("\nComponent Health Checks:")
            for component, health in health_checks.items():
                status_indicator = "✓" if health.get('status') == 'healthy' else "⚠" if health.get('status') == 'degraded' else "✗"
                self.logger.info(f"  {component}: {status_indicator} {health.get('status', 'unknown')}")
            
            return {
                "system_status": status,
                "performance_report": report,
                "health_checks": health_checks,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Performance monitoring demo failed: {e}")
            return {"error": str(e), "success": False}
    
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete demonstration."""
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING COMPLETE ASYNC ACADEMIC AGENT DEMO")
        self.logger.info("="*80)
        
        demo_start_time = time.time()
        results = {}
        
        try:
            # Run PDF processing demo
            pdf_results = await self.run_pdf_processing_demo()
            results['pdf_processing'] = pdf_results
            
            if pdf_results.get('success'):
                # Run content generation demo
                content_results = await self.run_content_generation_demo(pdf_results['results'])
                results['content_generation'] = content_results
            
            # Run workflow demo
            workflow_results = await self.run_workflow_demo()
            results['workflow'] = workflow_results
            
            # Run monitoring demo
            monitoring_results = await self.run_performance_monitoring_demo()
            results['monitoring'] = monitoring_results
            
            total_demo_time = time.time() - demo_start_time
            
            self.logger.info("\n" + "="*80)
            self.logger.info("DEMO SUMMARY")
            self.logger.info("="*80)
            self.logger.info(f"Total demo time: {total_demo_time:.2f}s")
            
            for demo_name, demo_result in results.items():
                status = "✓" if demo_result.get('success') else "✗"
                self.logger.info(f"{demo_name}: {status}")
                
                if demo_result.get('processing_time'):
                    self.logger.info(f"  Processing time: {demo_result['processing_time']:.2f}s")
            
            # Save comprehensive results
            results_file = self.temp_dir / "demo_results.json"
            async with asyncio.to_thread(open, results_file, 'w', encoding='utf-8') as f:
                await asyncio.to_thread(
                    f.write, 
                    json.dumps(results, indent=2, default=str)
                )
            
            self.logger.info(f"\nDemo results saved to: {results_file}")
            self.logger.info("Demo completed successfully!")
            
            return {
                "results": results,
                "total_time": total_demo_time,
                "results_file": str(results_file),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            return {
                "error": str(e),
                "partial_results": results,
                "success": False
            }


async def main():
    """Main demo entry point."""
    
    # Demo configuration
    config = {
        "worker_pool_size": 3,
        "pdf_processor": {
            "max_concurrent_processes": 2,
            "preferred_processor": "simulation",
            "enable_cache": True
        },
        "content_generator": {
            "max_concurrent_generations": 2,
            "enable_cache": True
        },
        "monitoring": {
            "metrics_flush_interval": 5.0,
            "system_check_interval": 3.0
        }
    }
    
    # Run demo
    async with AsyncAcademicAgentDemo(config) as demo:
        results = await demo.run_complete_demo()
        
        if results.get('success'):
            print(f"\n🎉 Demo completed successfully in {results['total_time']:.2f}s!")
            print(f"📊 Results saved to: {results.get('results_file', 'N/A')}")
        else:
            print(f"\n❌ Demo failed: {results.get('error', 'Unknown error')}")
            if 'partial_results' in results:
                print(f"📊 Partial results available")


if __name__ == "__main__":
    asyncio.run(main())