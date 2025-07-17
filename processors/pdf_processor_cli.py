#!/usr/bin/env python3
"""
Command Line Interface for High-Performance PDF Processor
Academic Agent v2 - Task 11 Implementation

This module provides a CLI interface for the Marker-based PDF processor
with support for single file processing, batch processing, and monitoring.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import signal
import os

from marker_pdf_processor import MarkerPDFProcessor, create_pdf_processor


class PDFProcessorCLI:
    """Command-line interface for PDF processing."""
    
    def __init__(self):
        self.processor: Optional[MarkerPDFProcessor] = None
        self.interrupted = False
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        self.interrupted = True
        if self.processor:
            self.logger.info("Received interrupt signal, cleaning up...")
            
    def _setup_logging(self, log_level: str, log_file: Optional[str] = None):
        """Setup logging configuration."""
        level = getattr(logging, log_level.upper())
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # File handler if specified
        handlers = [console_handler]
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            handlers=handlers,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def _progress_callback(self, processed: int, total: int):
        """Progress callback for batch processing."""
        percentage = (processed / total) * 100
        print(f"\rProgress: {processed}/{total} files processed ({percentage:.1f}%)", end='', flush=True)
        
    def _validate_input_path(self, path: str) -> Path:
        """Validate and convert input path."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        return path_obj
        
    def _validate_output_path(self, path: str) -> Path:
        """Validate and create output path."""
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
        
    def _collect_pdf_files(self, input_path: Path) -> List[Path]:
        """Collect PDF files from input path."""
        pdf_files = []
        
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                pdf_files.append(input_path)
            else:
                raise ValueError(f"Input file is not a PDF: {input_path}")
        else:
            # Collect all PDFs from directory
            for pdf_file in input_path.rglob('*.pdf'):
                pdf_files.append(pdf_file)
                
        return sorted(pdf_files)
        
    async def process_single_file(self, args) -> Dict[str, Any]:
        """Process a single PDF file."""
        try:
            # Validate paths
            input_path = self._validate_input_path(args.input)
            output_path = self._validate_output_path(args.output)
            
            # Create processor
            config = {
                'batch_size': args.batch_size,
                'extract_images': args.extract_images,
                'split_chapters': args.split_chapters,
                'max_pages': args.max_pages,
                'device': args.device,
                'enable_ocr': args.enable_ocr,
                'enable_editor_model': args.enable_editor_model,
            }
            
            self.processor = create_pdf_processor(config)
            
            print(f"Processing: {input_path.name}")
            print(f"Output directory: {output_path}")
            print(f"Configuration: {json.dumps(config, indent=2)}")
            
            # Process file
            result = await self.processor.process_single_pdf(
                input_path, 
                output_path,
                chapter_splitting=args.split_chapters
            )
            
            if result.success:
                print(f"\n✅ Processing completed successfully!")
                print(f"   Output file: {result.output_path}")
                print(f"   Processing time: {result.metrics.processing_time:.2f}s")
                print(f"   Pages processed: {result.metrics.pages_processed}")
                print(f"   Memory usage: {result.metrics.memory_usage_mb:.1f}MB")
                
                if result.chapters:
                    print(f"   Chapters extracted: {len(result.chapters)}")
                    
                if result.images:
                    print(f"   Images extracted: {len(result.images)}")
                    
            else:
                print(f"\n❌ Processing failed: {result.error_message}")
                return {"success": False, "error": result.error_message}
                
            return {"success": True, "result": result}
            
        except Exception as e:
            error_msg = f"Error processing file: {str(e)}"
            print(f"\n❌ {error_msg}")
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
    async def process_batch(self, args) -> Dict[str, Any]:
        """Process multiple PDF files in batch."""
        try:
            # Validate paths
            input_path = self._validate_input_path(args.input)
            output_path = self._validate_output_path(args.output)
            
            # Collect PDF files
            pdf_files = self._collect_pdf_files(input_path)
            
            if not pdf_files:
                print("No PDF files found in input path")
                return {"success": False, "error": "No PDF files found"}
                
            print(f"Found {len(pdf_files)} PDF files to process")
            
            # Create processor
            config = {
                'batch_size': args.batch_size,
                'extract_images': args.extract_images,
                'split_chapters': args.split_chapters,
                'max_pages': args.max_pages,
                'device': args.device,
                'enable_ocr': args.enable_ocr,
                'enable_editor_model': args.enable_editor_model,
                'max_workers': args.max_workers,
            }
            
            self.processor = create_pdf_processor(config)
            
            print(f"Output directory: {output_path}")
            print(f"Configuration: {json.dumps(config, indent=2)}")
            print("\nStarting batch processing...")
            
            # Process files
            results = await self.processor.process_batch(
                pdf_files,
                output_path,
                progress_callback=self._progress_callback
            )
            
            print("\n")  # New line after progress
            
            # Summary
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            print(f"✅ Batch processing completed!")
            print(f"   Total files: {len(results)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")
            print(f"   Success rate: {len(successful)/len(results)*100:.1f}%")
            
            if successful:
                total_time = sum(r.metrics.processing_time for r in successful)
                total_pages = sum(r.metrics.pages_processed for r in successful)
                print(f"   Total processing time: {total_time:.2f}s")
                print(f"   Average time per file: {total_time/len(successful):.2f}s")
                print(f"   Total pages processed: {total_pages}")
                
            if failed:
                print(f"\n❌ Failed files:")
                for result in failed:
                    print(f"   - {result.source_path.name}: {result.error_message}")
                    
            return {"success": True, "results": results}
            
        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            print(f"\n❌ {error_msg}")
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
    async def show_stats(self, args) -> Dict[str, Any]:
        """Show processing statistics."""
        try:
            if not self.processor:
                config = {'device': args.device}
                self.processor = create_pdf_processor(config)
                
            stats = await self.processor.get_processing_stats()
            
            print("📊 PDF Processor Statistics")
            print("=" * 50)
            
            # Processor info
            proc_info = stats['processor_info']
            print(f"Device: {proc_info['device']}")
            print(f"GPU Available: {proc_info['gpu_available']}")
            print(f"Marker Available: {proc_info['marker_available']}")
            print(f"Batch Size: {proc_info['batch_size']}")
            print(f"Max Workers: {proc_info['max_workers']}")
            print(f"Models Loaded: {proc_info['models_loaded']}")
            
            # System info
            sys_info = stats['system_info']
            print(f"\nSystem Information:")
            print(f"CPU Count: {sys_info['cpu_count']}")
            print(f"Total Memory: {sys_info['memory_total_gb']:.1f} GB")
            print(f"Available Memory: {sys_info['memory_available_gb']:.1f} GB")
            print(f"Memory Usage: {sys_info['memory_usage_percent']:.1f}%")
            
            # Performance metrics
            perf_metrics = stats['performance_metrics']
            if perf_metrics:
                print(f"\nPerformance Metrics:")
                print(f"Total Operations: {perf_metrics['total_operations']}")
                print(f"Successful Operations: {perf_metrics['successful_operations']}")
                print(f"Success Rate: {perf_metrics['success_rate']*100:.1f}%")
                print(f"Average Processing Time: {perf_metrics['average_processing_time']:.2f}s")
                print(f"Total Pages Processed: {perf_metrics['total_pages_processed']}")
                print(f"Average Memory Usage: {perf_metrics['average_memory_usage']:.1f}MB")
                
            # Cache info
            cache_info = stats['cache_info']
            print(f"\nCache Information:")
            print(f"Cached Files: {cache_info['cached_files']}")
            print(f"Cache Size: {cache_info['cache_size_mb']:.2f} MB")
            
            return {"success": True, "stats": stats}
            
        except Exception as e:
            error_msg = f"Error getting statistics: {str(e)}"
            print(f"\n❌ {error_msg}")
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
    async def run(self):
        """Run the CLI application."""
        parser = argparse.ArgumentParser(
            description="High-Performance PDF Processor using Marker Library",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process single file
  python pdf_processor_cli.py process input.pdf output/

  # Process batch with custom settings
  python pdf_processor_cli.py batch input_dir/ output/ --batch-size 3 --max-workers 4

  # Show statistics
  python pdf_processor_cli.py stats

  # Process with chapter splitting disabled
  python pdf_processor_cli.py process textbook.pdf output/ --no-split-chapters
            """
        )
        
        # Global arguments
        parser.add_argument('--log-level', default='INFO', 
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          help='Logging level')
        parser.add_argument('--log-file', help='Log file path')
        parser.add_argument('--device', default='auto',
                          choices=['auto', 'cpu', 'cuda', 'mps'],
                          help='Processing device')
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Process single file
        process_parser = subparsers.add_parser('process', help='Process a single PDF file')
        process_parser.add_argument('input', help='Input PDF file path')
        process_parser.add_argument('output', help='Output directory path')
        process_parser.add_argument('--batch-size', type=int, default=1,
                                  help='Batch size for processing (default: 1)')
        process_parser.add_argument('--max-pages', type=int,
                                  help='Maximum pages to process')
        process_parser.add_argument('--no-extract-images', action='store_false',
                                  dest='extract_images', default=True,
                                  help='Disable image extraction')
        process_parser.add_argument('--no-split-chapters', action='store_false',
                                  dest='split_chapters', default=True,
                                  help='Disable chapter splitting')
        process_parser.add_argument('--no-ocr', action='store_false',
                                  dest='enable_ocr', default=True,
                                  help='Disable OCR processing')
        process_parser.add_argument('--no-editor-model', action='store_false',
                                  dest='enable_editor_model', default=True,
                                  help='Disable editor model')
        
        # Process batch
        batch_parser = subparsers.add_parser('batch', help='Process multiple PDF files')
        batch_parser.add_argument('input', help='Input directory path')
        batch_parser.add_argument('output', help='Output directory path')
        batch_parser.add_argument('--batch-size', type=int, default=2,
                                help='Batch size for processing (default: 2)')
        batch_parser.add_argument('--max-workers', type=int, default=4,
                                help='Maximum worker threads (default: 4)')
        batch_parser.add_argument('--max-pages', type=int,
                                help='Maximum pages to process per file')
        batch_parser.add_argument('--no-extract-images', action='store_false',
                                dest='extract_images', default=True,
                                help='Disable image extraction')
        batch_parser.add_argument('--no-split-chapters', action='store_false',
                                dest='split_chapters', default=True,
                                help='Disable chapter splitting')
        batch_parser.add_argument('--no-ocr', action='store_false',
                                dest='enable_ocr', default=True,
                                help='Disable OCR processing')
        batch_parser.add_argument('--no-editor-model', action='store_false',
                                dest='enable_editor_model', default=True,
                                help='Disable editor model')
        
        # Show statistics
        stats_parser = subparsers.add_parser('stats', help='Show processing statistics')
        
        # Parse arguments
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
            
        # Setup logging
        self._setup_logging(args.log_level, args.log_file)
        
        # Handle device selection
        if args.device == 'auto':
            # Auto-detect device (handled by processor)
            args.device = None
        
        try:
            # Execute command
            if args.command == 'process':
                result = await self.process_single_file(args)
            elif args.command == 'batch':
                result = await self.process_batch(args)
            elif args.command == 'stats':
                result = await self.show_stats(args)
            else:
                print(f"Unknown command: {args.command}")
                parser.print_help()
                return
                
            # Exit with appropriate code
            if result['success']:
                sys.exit(0)
            else:
                sys.exit(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Processing interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"\n💥 Unexpected error: {str(e)}")
            self.logger.error(f"Unexpected error: {str(e)}")
            sys.exit(1)
        finally:
            # Cleanup
            if self.processor:
                await self.processor.clear_cache()


def main():
    """Main entry point."""
    cli = PDFProcessorCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()