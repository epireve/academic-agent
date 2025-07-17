#!/usr/bin/env python
"""
Batch Consolidation Script - Process multiple content sources in batches
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the agents directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from academic.consolidation_agent import ContentConsolidationAgent, ConsolidationResult


class BatchConsolidationProcessor:
    """Handles batch processing of content consolidation"""
    
    def __init__(self, config_path: str = None):
        self.agent = ContentConsolidationAgent()
        self.config = self._load_config(config_path)
        self.batch_results = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "batch_size": 10,
                "parallel_processing": False,
                "retry_failed": True,
                "backup_enabled": True,
                "progress_reporting": True
            }
    
    def process_batch(self, batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single batch of consolidation tasks"""
        batch_id = batch_config.get("batch_id", f"batch_{int(time.time())}")
        search_paths = batch_config.get("search_paths", [])
        output_path = batch_config.get("output_path", "")
        
        self.agent.logger.info(f"Starting batch processing: {batch_id}")
        
        start_time = datetime.now()
        
        try:
            # Run consolidation workflow
            result = self.agent.consolidate_workflow(search_paths, output_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            batch_result = {
                "batch_id": batch_id,
                "success": result.success,
                "processing_time": processing_time,
                "processed_files": len(result.processed_files),
                "skipped_files": len(result.skipped_files),
                "errors": len(result.errors),
                "consolidation_result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.batch_results.append(batch_result)
            
            # Log batch completion
            self.agent.logger.info(f"Batch {batch_id} completed in {processing_time:.2f} seconds")
            
            return batch_result
            
        except Exception as e:
            error_result = {
                "batch_id": batch_id,
                "success": False,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.batch_results.append(error_result)
            self.agent.logger.error(f"Batch {batch_id} failed: {str(e)}")
            
            return error_result
    
    def process_multiple_batches(self, batch_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple batches sequentially"""
        results = []
        
        for i, batch_config in enumerate(batch_configs):
            batch_config["batch_id"] = batch_config.get("batch_id", f"batch_{i+1}")
            
            self.agent.logger.info(f"Processing batch {i+1}/{len(batch_configs)}")
            
            result = self.process_batch(batch_config)
            results.append(result)
            
            # Progress reporting
            if self.config.get("progress_reporting", True):
                self._report_progress(i+1, len(batch_configs), result)
        
        return results
    
    def _report_progress(self, current: int, total: int, result: Dict[str, Any]):
        """Report progress of batch processing"""
        progress = (current / total) * 100
        status = "SUCCESS" if result["success"] else "FAILED"
        
        print(f"Batch {current}/{total} ({progress:.1f}%) - {status}")
        
        if result["success"]:
            print(f"  Processed: {result['processed_files']} files")
            print(f"  Skipped: {result['skipped_files']} files")
            print(f"  Time: {result['processing_time']:.2f}s")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print()
    
    def generate_batch_report(self, output_path: str) -> str:
        """Generate comprehensive batch processing report"""
        report = {
            "batch_processing_summary": {
                "total_batches": len(self.batch_results),
                "successful_batches": len([r for r in self.batch_results if r["success"]]),
                "failed_batches": len([r for r in self.batch_results if not r["success"]]),
                "total_processing_time": sum(r["processing_time"] for r in self.batch_results),
                "report_generated": datetime.now().isoformat()
            },
            "batch_details": self.batch_results,
            "aggregate_statistics": self._calculate_aggregate_stats()
        }
        
        report_path = os.path.join(output_path, "batch_consolidation_report.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown summary
        markdown_report = self._generate_markdown_report(report)
        markdown_path = os.path.join(output_path, "batch_consolidation_summary.md")
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        return report_path
    
    def _calculate_aggregate_stats(self) -> Dict[str, Any]:
        """Calculate aggregate statistics across all batches"""
        successful_batches = [r for r in self.batch_results if r["success"]]
        
        if not successful_batches:
            return {"error": "No successful batches to analyze"}
        
        total_files = sum(r["processed_files"] for r in successful_batches)
        total_skipped = sum(r["skipped_files"] for r in successful_batches)
        total_errors = sum(r["errors"] for r in successful_batches)
        
        return {
            "total_files_processed": total_files,
            "total_files_skipped": total_skipped,
            "total_errors": total_errors,
            "average_processing_time": sum(r["processing_time"] for r in successful_batches) / len(successful_batches),
            "success_rate": len(successful_batches) / len(self.batch_results) * 100
        }
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown summary report"""
        lines = [
            "# Batch Consolidation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total Batches: {report['batch_processing_summary']['total_batches']}",
            f"- Successful: {report['batch_processing_summary']['successful_batches']}",
            f"- Failed: {report['batch_processing_summary']['failed_batches']}",
            f"- Total Processing Time: {report['batch_processing_summary']['total_processing_time']:.2f}s",
            ""
        ]
        
        # Add aggregate statistics
        if "aggregate_statistics" in report and "error" not in report["aggregate_statistics"]:
            stats = report["aggregate_statistics"]
            lines.extend([
                "## Aggregate Statistics",
                f"- Files Processed: {stats['total_files_processed']}",
                f"- Files Skipped: {stats['total_files_skipped']}",
                f"- Total Errors: {stats['total_errors']}",
                f"- Average Processing Time: {stats['average_processing_time']:.2f}s",
                f"- Success Rate: {stats['success_rate']:.1f}%",
                ""
            ])
        
        # Add batch details
        lines.extend([
            "## Batch Details",
            "| Batch ID | Status | Files Processed | Files Skipped | Processing Time |",
            "|----------|--------|----------------|---------------|-----------------|"
        ])
        
        for batch in report["batch_details"]:
            status = "✅ SUCCESS" if batch["success"] else "❌ FAILED"
            processed = batch.get("processed_files", 0)
            skipped = batch.get("skipped_files", 0)
            time_taken = batch.get("processing_time", 0)
            
            lines.append(f"| {batch['batch_id']} | {status} | {processed} | {skipped} | {time_taken:.2f}s |")
        
        return "\n".join(lines)


def create_batch_config_from_existing_structure() -> List[Dict[str, Any]]:
    """Create batch configuration based on existing project structure"""
    base_path = "/Users/invoture/dev.local/academic-agent"
    
    # Define batch configurations for different content types
    batch_configs = [
        {
            "batch_id": "transcripts_batch",
            "search_paths": [
                f"{base_path}/output/sra/transcripts/markdown",
                f"{base_path}/output/sra/transcripts/standardized"
            ],
            "output_path": f"{base_path}/output/consolidated/transcripts",
            "content_type_filter": "transcript"
        },
        {
            "batch_id": "lectures_batch", 
            "search_paths": [
                f"{base_path}/markdown",
                f"{base_path}/output/Lecture Notes Chapter 6"
            ],
            "output_path": f"{base_path}/output/consolidated/lectures",
            "content_type_filter": "lecture"
        },
        {
            "batch_id": "textbook_batch",
            "search_paths": [
                f"{base_path}/output/sra/textbook/markdown"
            ],
            "output_path": f"{base_path}/output/consolidated/textbook",
            "content_type_filter": "textbook"
        },
        {
            "batch_id": "external_notes_batch",
            "search_paths": [
                "/Users/invoture/dev.local/mse-st/sra"
            ],
            "output_path": f"{base_path}/output/consolidated/external_notes",
            "content_type_filter": "notes"
        }
    ]
    
    return batch_configs


def main():
    """Main function for batch consolidation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Content Consolidation")
    parser.add_argument("--config", help="Path to batch configuration file")
    parser.add_argument("--output", default="/Users/invoture/dev.local/academic-agent/output/batch_consolidated",
                       help="Output directory for consolidated content")
    parser.add_argument("--dry-run", action="store_true", help="Run without making changes")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load batch configurations
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            batch_configs = json.load(f)
    else:
        print("No config file provided, using default batch configuration based on project structure")
        batch_configs = create_batch_config_from_existing_structure()
    
    # Create batch processor
    processor = BatchConsolidationProcessor(args.config)
    
    print("Batch Content Consolidation")
    print("=" * 50)
    print(f"Number of batches: {len(batch_configs)}")
    print(f"Output directory: {args.output}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
        print("\nBatch Configuration:")
        for i, config in enumerate(batch_configs):
            print(f"\nBatch {i+1}: {config.get('batch_id', 'unnamed')}")
            print(f"  Search paths: {config.get('search_paths', [])}")
            print(f"  Output path: {config.get('output_path', 'not specified')}")
            print(f"  Content filter: {config.get('content_type_filter', 'none')}")
        return
    
    # Process batches
    try:
        results = processor.process_multiple_batches(batch_configs)
        
        # Generate report
        report_path = processor.generate_batch_report(args.output)
        
        print("\nBatch Processing Complete!")
        print("=" * 50)
        print(f"Report saved to: {report_path}")
        print(f"Summary available at: {os.path.join(args.output, 'batch_consolidation_summary.md')}")
        
        # Show final statistics
        successful = len([r for r in results if r["success"]])
        failed = len([r for r in results if not r["success"]])
        
        print(f"\nFinal Statistics:")
        print(f"Successful batches: {successful}")
        print(f"Failed batches: {failed}")
        print(f"Success rate: {(successful/len(results)*100):.1f}%")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()