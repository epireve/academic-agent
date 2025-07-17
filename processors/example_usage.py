#!/usr/bin/env python3
"""
Example Usage Script for High-Performance PDF Processor
Academic Agent v2 - Task 11 Implementation

This script demonstrates various features of the PDF processor including:
- Basic processing
- Batch processing
- Chapter splitting
- Image extraction
- Performance monitoring
- Error handling
- Configuration management
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

# Import our PDF processor modules
from marker_pdf_processor import create_pdf_processor, ProcessingResult
from monitoring import MonitoringSystem, ProcessingEvent
from pdf_processor_cli import PDFProcessorCLI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessorDemo:
    """Demonstration of PDF processor capabilities."""
    
    def __init__(self):
        self.temp_files = []
        self.monitoring = MonitoringSystem()
        
    def cleanup(self):
        """Clean up temporary files."""
        for temp_path in self.temp_files:
            if temp_path.exists():
                if temp_path.is_dir():
                    shutil.rmtree(temp_path)
                else:
                    temp_path.unlink()
        self.monitoring.stop_monitoring()
        
    def create_sample_pdfs(self, output_dir: Path) -> List[Path]:
        """Create sample PDF files for demonstration."""
        sample_pdfs = []
        
        # Sample academic paper
        paper_content = """# Research Paper: AI in Education
        
## Abstract
This paper explores the application of artificial intelligence in educational settings.

## 1. Introduction
Artificial intelligence has transformed many industries, and education is no exception.

### 1.1 Background
The integration of AI technologies in learning environments has shown promising results.

### 1.2 Objectives
- Analyze current AI applications in education
- Identify challenges and opportunities
- Propose future research directions

## 2. Literature Review
Previous studies have shown that AI can enhance personalized learning experiences.

### 2.1 Machine Learning in Education
Machine learning algorithms can adapt to individual learning patterns.

### 2.2 Natural Language Processing
NLP techniques can improve automated grading and feedback systems.

## 3. Methodology
We conducted a comprehensive review of existing literature and case studies.

## 4. Results
Our analysis reveals significant potential for AI in educational applications.

### 4.1 Personalized Learning
AI systems can customize content delivery based on student performance.

### 4.2 Automated Assessment
Intelligent systems can provide immediate feedback on student work.

## 5. Discussion
The findings suggest that AI integration requires careful consideration of ethical implications.

## 6. Conclusion
AI technologies offer substantial opportunities to enhance educational outcomes.

## References
[1] Smith, J. (2023). AI in Education: A Comprehensive Review.
[2] Johnson, M. (2022). Machine Learning for Personalized Learning.
"""
        
        paper_path = output_dir / "ai_education_paper.pdf"
        paper_path.write_text(paper_content)
        sample_pdfs.append(paper_path)
        
        # Sample textbook chapter
        textbook_content = """# Chapter 1: Introduction to Computer Science

## Learning Objectives
By the end of this chapter, students will be able to:
- Define computer science and its core areas
- Understand the role of algorithms in problem-solving
- Identify different programming paradigms

## 1.1 What is Computer Science?
Computer science is the study of algorithmic processes and computational systems.

### 1.1.1 Core Areas
- Programming and software development
- Computer systems and architecture
- Data structures and algorithms
- Human-computer interaction

### 1.1.2 Applications
Computer science applications span numerous fields including:
- Healthcare informatics
- Financial technology
- Entertainment and gaming
- Scientific research

## 1.2 Problem-Solving with Algorithms
An algorithm is a step-by-step procedure for solving a problem.

### 1.2.1 Algorithm Design
Good algorithms have the following characteristics:
- Correctness: produces the right output
- Efficiency: uses resources optimally
- Clarity: easy to understand and implement

### 1.2.2 Algorithm Analysis
We analyze algorithms based on:
- Time complexity
- Space complexity
- Scalability

## 1.3 Programming Paradigms
Different approaches to programming include:

### 1.3.1 Imperative Programming
- Procedural programming
- Object-oriented programming

### 1.3.2 Declarative Programming
- Functional programming
- Logic programming

## Chapter Summary
This chapter introduced the fundamental concepts of computer science, including the definition of the field, the role of algorithms in problem-solving, and various programming paradigms.

## Review Questions
1. What are the core areas of computer science?
2. What makes an algorithm efficient?
3. Compare imperative and declarative programming paradigms.

# Chapter 2: Data Structures and Algorithms

## Learning Objectives
Students will learn about:
- Basic data structures
- Algorithm complexity analysis
- Common algorithmic techniques

## 2.1 Fundamental Data Structures
Data structures organize and store data for efficient access and modification.

### 2.1.1 Arrays
Arrays store elements in contiguous memory locations.

### 2.1.2 Linked Lists
Linked lists use pointers to connect elements.

### 2.1.3 Stacks and Queues
- Stack: Last-In-First-Out (LIFO)
- Queue: First-In-First-Out (FIFO)

## 2.2 Algorithm Complexity
We measure algorithm efficiency using Big O notation.

### 2.2.1 Time Complexity
- O(1): Constant time
- O(log n): Logarithmic time
- O(n): Linear time
- O(n²): Quadratic time

### 2.2.2 Space Complexity
The amount of memory an algorithm uses.

## 2.3 Common Algorithms
Important algorithmic techniques include:

### 2.3.1 Sorting Algorithms
- Bubble sort: O(n²)
- Quick sort: O(n log n)
- Merge sort: O(n log n)

### 2.3.2 Search Algorithms
- Linear search: O(n)
- Binary search: O(log n)

## Chapter Summary
This chapter covered fundamental data structures and algorithmic analysis techniques essential for computer science problem-solving.
"""
        
        textbook_path = output_dir / "computer_science_textbook.pdf"
        textbook_path.write_text(textbook_content)
        sample_pdfs.append(textbook_path)
        
        # Sample lecture slides
        slides_content = """# Lecture 5: Database Systems

## Slide 1: Introduction to Databases
- What is a database?
- Database Management Systems (DBMS)
- Why use databases?

## Slide 2: Database Models
- Relational model
- NoSQL models
- Graph databases

## Slide 3: Relational Databases
- Tables, rows, and columns
- Primary keys and foreign keys
- Relationships between tables

## Slide 4: SQL Basics
- SELECT statements
- INSERT, UPDATE, DELETE
- JOIN operations

## Slide 5: Database Design
- Normalization
- Entity-Relationship (ER) diagrams
- Design principles

## Slide 6: Transactions
- ACID properties
- Concurrency control
- Deadlock prevention

## Slide 7: Performance Optimization
- Indexing strategies
- Query optimization
- Database tuning

## Slide 8: NoSQL Databases
- Document databases
- Key-value stores
- Column-family databases

## Slide 9: Big Data and Databases
- Distributed databases
- Data warehousing
- Analytics platforms

## Slide 10: Summary
- Key concepts covered
- Next lecture preview
- Reading assignments
"""
        
        slides_path = output_dir / "database_lecture_slides.pdf"
        slides_path.write_text(slides_content)
        sample_pdfs.append(slides_path)
        
        return sample_pdfs
        
    async def demo_basic_processing(self):
        """Demonstrate basic PDF processing."""
        print("🔧 Demo 1: Basic PDF Processing")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample PDFs
        sample_pdfs = self.create_sample_pdfs(temp_dir / "input")
        output_dir = temp_dir / "output"
        
        # Create processor with basic configuration
        config = {
            'batch_size': 1,
            'extract_images': True,
            'split_chapters': True,
            'device': 'cpu'  # Use CPU for demo
        }
        
        processor = create_pdf_processor(config)
        
        # Process first PDF
        pdf_path = sample_pdfs[0]
        print(f"📄 Processing: {pdf_path.name}")
        
        start_time = time.time()
        result = await processor.process_single_pdf(pdf_path, output_dir)
        end_time = time.time()
        
        if result.success:
            print(f"✅ Processing successful!")
            print(f"   📝 Output file: {result.output_path}")
            print(f"   ⏱️  Processing time: {result.metrics.processing_time:.2f}s")
            print(f"   📊 Pages processed: {result.metrics.pages_processed}")
            print(f"   💾 Memory usage: {result.metrics.memory_usage_mb:.1f}MB")
            print(f"   🖼️  Images extracted: {len(result.images)}")
            
            # Show content preview
            if result.output_path.exists():
                content = result.output_path.read_text()
                print(f"\n📖 Content preview (first 300 chars):")
                print(content[:300] + "..." if len(content) > 300 else content)
        else:
            print(f"❌ Processing failed: {result.error_message}")
            
        print(f"\n⏰ Total demo time: {end_time - start_time:.2f}s\n")
        
    async def demo_batch_processing(self):
        """Demonstrate batch processing capabilities."""
        print("🔧 Demo 2: Batch Processing")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample PDFs
        sample_pdfs = self.create_sample_pdfs(temp_dir / "input")
        output_dir = temp_dir / "output"
        
        # Create processor with batch configuration
        config = {
            'batch_size': 2,
            'max_workers': 3,
            'extract_images': True,
            'split_chapters': True,
            'device': 'cpu'
        }
        
        processor = create_pdf_processor(config)
        
        print(f"📄 Processing {len(sample_pdfs)} PDFs in batch mode...")
        
        # Progress callback
        def progress_callback(processed, total):
            percentage = (processed / total) * 100
            print(f"\r📊 Progress: {processed}/{total} files processed ({percentage:.1f}%)", end='', flush=True)
        
        start_time = time.time()
        results = await processor.process_batch(sample_pdfs, output_dir, progress_callback)
        end_time = time.time()
        
        print("\n")  # New line after progress
        
        # Analyze results
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"✅ Batch processing completed!")
        print(f"   📊 Total files: {len(results)}")
        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   📈 Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            total_time = sum(r.metrics.processing_time for r in successful)
            total_pages = sum(r.metrics.pages_processed for r in successful)
            print(f"   ⏱️  Total processing time: {total_time:.2f}s")
            print(f"   📄 Total pages processed: {total_pages}")
            print(f"   🚀 Average processing speed: {total_pages/total_time:.1f} pages/second")
            
        # Check batch summary
        summary_file = output_dir / "batch_processing_summary.json"
        if summary_file.exists():
            summary_data = json.loads(summary_file.read_text())
            print(f"   📋 Batch summary saved to: {summary_file}")
            print(f"   📊 Processing summary available in JSON format")
            
        print(f"\n⏰ Total demo time: {end_time - start_time:.2f}s\n")
        
    async def demo_chapter_splitting(self):
        """Demonstrate chapter splitting functionality."""
        print("🔧 Demo 3: Chapter Splitting")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample textbook
        sample_pdfs = self.create_sample_pdfs(temp_dir / "input")
        textbook_pdf = sample_pdfs[1]  # The textbook sample
        output_dir = temp_dir / "output"
        
        # Create processor with chapter splitting enabled
        config = {
            'batch_size': 1,
            'split_chapters': True,
            'extract_images': True,
            'device': 'cpu'
        }
        
        processor = create_pdf_processor(config)
        
        print(f"📚 Processing textbook: {textbook_pdf.name}")
        print(f"🔍 Chapter splitting enabled")
        
        start_time = time.time()
        result = await processor.process_single_pdf(textbook_pdf, output_dir, chapter_splitting=True)
        end_time = time.time()
        
        if result.success:
            print(f"✅ Processing successful!")
            print(f"   📝 Main output: {result.output_path}")
            print(f"   📚 Chapters detected: {len(result.chapters)}")
            
            # Show chapter information
            if result.chapters:
                print(f"\n📖 Chapter breakdown:")
                for chapter_name, chapter_content in result.chapters.items():
                    word_count = len(chapter_content.split())
                    print(f"   • {chapter_name}: {word_count} words")
                    
                # Check for chapter files
                chapter_dir = output_dir / f"{textbook_pdf.stem}_chapters"
                if chapter_dir.exists():
                    chapter_files = list(chapter_dir.glob("*.md"))
                    print(f"   📁 Individual chapter files: {len(chapter_files)}")
                    for chapter_file in chapter_files:
                        print(f"     - {chapter_file.name}")
            else:
                print(f"   ⚠️  No chapters detected")
                
        else:
            print(f"❌ Processing failed: {result.error_message}")
            
        print(f"\n⏰ Total demo time: {end_time - start_time:.2f}s\n")
        
    async def demo_monitoring(self):
        """Demonstrate monitoring and performance tracking."""
        print("🔧 Demo 4: Monitoring and Performance Tracking")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample PDFs
        sample_pdfs = self.create_sample_pdfs(temp_dir / "input")
        output_dir = temp_dir / "output"
        
        # Create processor with monitoring
        config = {
            'batch_size': 1,
            'extract_images': True,
            'split_chapters': True,
            'device': 'cpu'
        }
        
        processor = create_pdf_processor(config)
        
        print("📊 Starting monitoring system...")
        
        # Process files with monitoring
        for i, pdf_path in enumerate(sample_pdfs):
            print(f"\n📄 Processing {i+1}/{len(sample_pdfs)}: {pdf_path.name}")
            
            start_time = time.time()
            result = await processor.process_single_pdf(pdf_path, output_dir / f"file_{i}")
            end_time = time.time()
            
            # Record processing event
            event = ProcessingEvent(
                event_type='pdf_processing',
                timestamp=start_time,
                file_path=str(pdf_path),
                status='success' if result.success else 'failed',
                processing_time=result.metrics.processing_time,
                memory_usage=result.metrics.memory_usage_mb,
                pages_processed=result.metrics.pages_processed,
                file_size=result.metrics.file_size_mb,
                error_message=result.error_message
            )
            
            self.monitoring.record_processing_event(event)
            
            if result.success:
                print(f"   ✅ Success - {result.metrics.processing_time:.2f}s")
            else:
                print(f"   ❌ Failed - {result.error_message}")
                
        # Get monitoring data
        print(f"\n📊 Monitoring Dashboard:")
        dashboard = self.monitoring.get_dashboard_data()
        
        # Processing summary
        proc_summary = dashboard['processing_summary']
        print(f"   📈 Processing Statistics:")
        print(f"     • Total events: {proc_summary['total_events']}")
        print(f"     • Successful: {proc_summary['successful_events']}")
        print(f"     • Failed: {proc_summary['failed_events']}")
        print(f"     • Success rate: {proc_summary['success_rate']*100:.1f}%")
        print(f"     • Average processing time: {proc_summary['processing_time']['avg']:.2f}s")
        print(f"     • Total pages processed: {proc_summary['pages_processed']['total']}")
        
        # System metrics
        if dashboard['system_metrics']:
            print(f"   💻 System Metrics:")
            gauges = dashboard['system_metrics'].get('gauges', {})
            if gauges:
                cpu_usage = gauges.get('system.cpu.usage_percent', 0)
                memory_usage = gauges.get('system.memory.usage_percent', 0)
                print(f"     • CPU Usage: {cpu_usage:.1f}%")
                print(f"     • Memory Usage: {memory_usage:.1f}%")
                
        # Alerts
        active_alerts = dashboard['active_alerts']
        if active_alerts:
            print(f"   🚨 Active Alerts: {len(active_alerts)}")
            for alert in active_alerts:
                print(f"     • {alert['level']}: {alert['message']}")
        else:
            print(f"   ✅ No active alerts")
            
        print()
        
    async def demo_error_handling(self):
        """Demonstrate error handling capabilities."""
        print("🔧 Demo 5: Error Handling")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create mixed test files
        test_files = []
        
        # Valid PDF
        valid_pdf = temp_dir / "valid.pdf"
        valid_pdf.write_text("Valid PDF content")
        test_files.append(valid_pdf)
        
        # Invalid file (not PDF)
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_text("This is not a PDF")
        test_files.append(invalid_file)
        
        # Non-existent file
        nonexistent_file = temp_dir / "nonexistent.pdf"
        test_files.append(nonexistent_file)
        
        output_dir = temp_dir / "output"
        
        # Create processor
        config = {
            'batch_size': 1,
            'extract_images': True,
            'split_chapters': True,
            'device': 'cpu'
        }
        
        processor = create_pdf_processor(config)
        
        print(f"🧪 Testing error handling with {len(test_files)} files:")
        
        for i, file_path in enumerate(test_files):
            print(f"\n📄 Test {i+1}: {file_path.name}")
            
            if file_path.exists():
                print(f"   📁 File exists: Yes")
                print(f"   📋 File type: {file_path.suffix}")
            else:
                print(f"   📁 File exists: No")
                
            try:
                result = await processor.process_single_pdf(file_path, output_dir)
                
                if result.success:
                    print(f"   ✅ Status: Success")
                    print(f"   ⏱️  Processing time: {result.metrics.processing_time:.2f}s")
                    print(f"   📊 Pages processed: {result.metrics.pages_processed}")
                else:
                    print(f"   ❌ Status: Failed")
                    print(f"   🔍 Error: {result.error_message}")
                    print(f"   ⏱️  Processing time: {result.metrics.processing_time:.2f}s")
                    
            except Exception as e:
                print(f"   💥 Exception: {type(e).__name__}: {str(e)}")
                
        print(f"\n✅ Error handling demo completed\n")
        
    async def demo_performance_comparison(self):
        """Demonstrate performance comparison with different configurations."""
        print("🔧 Demo 6: Performance Comparison")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample PDFs
        sample_pdfs = self.create_sample_pdfs(temp_dir / "input")
        
        # Test different configurations
        configurations = [
            {
                'name': 'Basic CPU',
                'config': {
                    'batch_size': 1,
                    'max_workers': 1,
                    'extract_images': False,
                    'split_chapters': False,
                    'device': 'cpu'
                }
            },
            {
                'name': 'Optimized CPU',
                'config': {
                    'batch_size': 2,
                    'max_workers': 4,
                    'extract_images': True,
                    'split_chapters': True,
                    'device': 'cpu'
                }
            },
            {
                'name': 'Full Features',
                'config': {
                    'batch_size': 3,
                    'max_workers': 4,
                    'extract_images': True,
                    'split_chapters': True,
                    'enable_editor_model': True,
                    'enable_ocr': True,
                    'device': 'cpu'
                }
            }
        ]
        
        results = {}
        
        for config_info in configurations:
            config_name = config_info['name']
            config = config_info['config']
            
            print(f"\n⚡ Testing configuration: {config_name}")
            print(f"   📋 Settings: {json.dumps(config, indent=2)}")
            
            processor = create_pdf_processor(config)
            output_dir = temp_dir / f"output_{config_name.lower().replace(' ', '_')}"
            
            start_time = time.time()
            processing_results = await processor.process_batch(sample_pdfs, output_dir)
            end_time = time.time()
            
            successful = [r for r in processing_results if r.success]
            failed = [r for r in processing_results if not r.success]
            
            total_pages = sum(r.metrics.pages_processed for r in successful)
            total_processing_time = sum(r.metrics.processing_time for r in successful)
            
            results[config_name] = {
                'total_time': end_time - start_time,
                'processing_time': total_processing_time,
                'successful': len(successful),
                'failed': len(failed),
                'pages_processed': total_pages,
                'pages_per_second': total_pages / total_processing_time if total_processing_time > 0 else 0
            }
            
            print(f"   ✅ Results:")
            print(f"     • Total time: {end_time - start_time:.2f}s")
            print(f"     • Processing time: {total_processing_time:.2f}s")
            print(f"     • Successful: {len(successful)}/{len(processing_results)}")
            print(f"     • Pages processed: {total_pages}")
            print(f"     • Speed: {results[config_name]['pages_per_second']:.1f} pages/second")
            
        # Performance comparison summary
        print(f"\n📊 Performance Comparison Summary:")
        print(f"{'Configuration':<15} {'Time (s)':<10} {'Pages/s':<10} {'Success':<10}")
        print("-" * 50)
        
        for config_name, result in results.items():
            print(f"{config_name:<15} {result['total_time']:<10.2f} {result['pages_per_second']:<10.1f} {result['successful']:<10}")
            
        # Find best performing configuration
        best_config = max(results.items(), key=lambda x: x[1]['pages_per_second'])
        print(f"\n🏆 Best performing configuration: {best_config[0]}")
        print(f"   Speed: {best_config[1]['pages_per_second']:.1f} pages/second")
        
        print()
        
    async def demo_cli_interface(self):
        """Demonstrate CLI interface capabilities."""
        print("🔧 Demo 7: CLI Interface")
        print("=" * 50)
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_files.append(temp_dir)
        
        # Create sample PDFs
        sample_pdfs = self.create_sample_pdfs(temp_dir / "input")
        output_dir = temp_dir / "output"
        
        print("💻 Command Line Interface Examples:")
        print()
        
        # Show example commands
        examples = [
            {
                'description': 'Process a single PDF',
                'command': f'python pdf_processor_cli.py process "{sample_pdfs[0]}" "{output_dir}"'
            },
            {
                'description': 'Process multiple PDFs in batch',
                'command': f'python pdf_processor_cli.py batch "{temp_dir}/input" "{output_dir}" --batch-size 2'
            },
            {
                'description': 'Process with custom settings',
                'command': f'python pdf_processor_cli.py process "{sample_pdfs[0]}" "{output_dir}" --no-split-chapters --device cpu'
            },
            {
                'description': 'Show processing statistics',
                'command': 'python pdf_processor_cli.py stats'
            },
            {
                'description': 'Get help information',
                'command': 'python pdf_processor_cli.py --help'
            }
        ]
        
        for example in examples:
            print(f"📝 {example['description']}:")
            print(f"   {example['command']}")
            print()
            
        # Actually run a simple CLI command
        print("🚀 Running CLI demo...")
        
        # Create a simple CLI instance and simulate processing
        cli = PDFProcessorCLI()
        
        # Mock command line arguments
        class MockArgs:
            def __init__(self):
                self.input = str(sample_pdfs[0])
                self.output = str(output_dir)
                self.batch_size = 1
                self.extract_images = True
                self.split_chapters = True
                self.max_pages = None
                self.device = 'cpu'
                self.enable_ocr = False
                self.enable_editor_model = False
                
        args = MockArgs()
        
        result = await cli.process_single_file(args)
        
        if result['success']:
            print("✅ CLI processing successful!")
        else:
            print(f"❌ CLI processing failed: {result['error']}")
            
        print()
        
    async def run_all_demos(self):
        """Run all demonstration examples."""
        print("🎯 High-Performance PDF Processor - Complete Demo")
        print("=" * 60)
        print("📋 This demo showcases all features of the PDF processor:")
        print("   • Basic PDF processing")
        print("   • Batch processing")
        print("   • Chapter splitting")
        print("   • Monitoring and performance tracking")
        print("   • Error handling")
        print("   • Performance comparison")
        print("   • CLI interface")
        print("=" * 60)
        print()
        
        demos = [
            self.demo_basic_processing,
            self.demo_batch_processing,
            self.demo_chapter_splitting,
            self.demo_monitoring,
            self.demo_error_handling,
            self.demo_performance_comparison,
            self.demo_cli_interface
        ]
        
        total_start_time = time.time()
        
        for demo in demos:
            try:
                await demo()
            except Exception as e:
                logger.error(f"Demo failed: {e}")
                print(f"❌ Demo failed: {e}\n")
                
        total_end_time = time.time()
        
        print("🎉 All demos completed!")
        print(f"⏰ Total demo time: {total_end_time - total_start_time:.2f}s")
        print("🔗 For more information, see the README.md file")
        print("📧 For support, check the troubleshooting section")


async def main():
    """Main function to run the demo."""
    demo = PDFProcessorDemo()
    
    try:
        await demo.run_all_demos()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"💥 Demo failed: {e}")
    finally:
        demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())