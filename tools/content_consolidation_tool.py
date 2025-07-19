#!/usr/bin/env python
"""
Content Consolidation Tool - Main script to run the consolidation workflow
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.core.output_manager import get_output_manager, get_final_output_path, get_processed_output_path, get_analysis_output_path
from src.core.output_manager import OutputCategory, ContentType


# Add the agents directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from academic.consolidation_agent import ContentConsolidationAgent, ConsolidationResult


def main():
    """Main function to run content consolidation"""
    parser = argparse.ArgumentParser(
        description="Content Consolidation Tool - Merge transcripts and resolve naming conflicts"
    )
    
    parser.add_argument(
        "--search-paths", 
        nargs='+', 
        default=[
            get_processed_output_path(ContentType.MARKDOWN, subdirectory="transcripts"),
            get_processed_output_path(ContentType.MARKDOWN), 
            "/Users/invoture/dev.local/mse-st/sra"
        ],
        help="Paths to search for content files"
    )
    
    parser.add_argument(
        "--output-path", 
        default=get_final_output_path(ContentType.REPORTS, subdirectory="consolidated"),
        help="Path for consolidated output"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Run without making changes (discovery only)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--config-file",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration if provided
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Create consolidation agent
    agent = ContentConsolidationAgent()
    
    print("Content Consolidation Tool")
    print("=" * 50)
    print(f"Search paths: {args.search_paths}")
    print(f"Output path: {args.output_path}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    try:
        if args.dry_run:
            # Discovery only mode
            print("Running discovery only (dry run)...")
            discovered_files = agent.scan_locations(args.search_paths)
            
            print(f"\nDiscovered {len(discovered_files)} files:")
            print("-" * 40)
            
            # Group by content type
            content_groups = {}
            for file_info in discovered_files:
                content_type = file_info.get("content_type", "unknown")
                if content_type not in content_groups:
                    content_groups[content_type] = []
                content_groups[content_type].append(file_info)
            
            for content_type, files in content_groups.items():
                print(f"\n{content_type.upper()} ({len(files)} files):")
                for file_info in sorted(files, key=lambda x: x.get("week_number", 0)):
                    week = file_info.get("week_number", "?")
                    confidence = file_info.get("confidence", 0.0)
                    filename = file_info.get("filename", "")
                    print(f"  Week {week:2}: {filename} (confidence: {confidence:.2f})")
            
            # Show potential conflicts
            print("\nPotential naming conflicts:")
            print("-" * 40)
            file_mappings = agent.resolve_naming_conflicts(discovered_files)
            
            week_conflicts = {}
            for mapping in file_mappings:
                week = mapping.week_number
                if week not in week_conflicts:
                    week_conflicts[week] = []
                week_conflicts[week].append(mapping)
            
            for week, mappings in sorted(week_conflicts.items()):
                if len(mappings) > 1:
                    print(f"Week {week}: {len(mappings)} files")
                    for mapping in mappings:
                        print(f"  - {os.path.basename(mapping.source_path)} -> {mapping.target_path}")
                        if mapping.metadata and "duplicates" in mapping.metadata:
                            for dup in mapping.metadata["duplicates"]:
                                print(f"    (skipping: {dup['filename']})")
            
        else:
            # Full consolidation workflow
            print("Running full consolidation workflow...")
            result = agent.consolidate_workflow(args.search_paths, args.output_path)
            
            print("\nConsolidation Results:")
            print("=" * 50)
            print(f"Success: {result.success}")
            print(f"Processed files: {len(result.processed_files)}")
            print(f"Skipped files: {len(result.skipped_files)}")
            print(f"Errors: {len(result.errors)}")
            
            if result.processed_files:
                print("\nProcessed Files:")
                print("-" * 30)
                for mapping in result.processed_files:
                    week = mapping.week_number or "?"
                    content_type = mapping.content_type or "unknown"
                    confidence = mapping.confidence or 0.0
                    source_file = os.path.basename(mapping.source_path)
                    target_file = os.path.basename(mapping.target_path)
                    print(f"Week {week:2} ({content_type}): {source_file} -> {target_file} (conf: {confidence:.2f})")
            
            if result.skipped_files:
                print(f"\nSkipped Files ({len(result.skipped_files)}):")
                print("-" * 30)
                for skipped in result.skipped_files:
                    print(f"  - {os.path.basename(skipped)}")
            
            if result.errors:
                print(f"\nErrors ({len(result.errors)}):")
                print("-" * 30)
                for error in result.errors:
                    print(f"  - {error.get('source_path', 'unknown')}: {error.get('error', 'unknown error')}")
            
            # Show unified structure
            if result.unified_structure:
                print("\nUnified Structure:")
                print("-" * 30)
                for content_type, path in result.unified_structure.items():
                    # Count files in directory
                    try:
                        files = [f for f in os.listdir(path) if f.endswith('.md')]
                        print(f"{content_type}: {len(files)} files in {path}")
                    except:
                        print(f"{content_type}: {path}")
            
            # Show quality metrics
            if result.consolidation_report:
                quality_score = agent.check_quality({"consolidation_result": result})
                print(f"\nQuality Score: {quality_score:.2f}")
                
                # Show confidence distribution
                conf_dist = result.consolidation_report.get("confidence_distribution", {})
                if conf_dist:
                    print("Confidence Distribution:")
                    for level, count in conf_dist.items():
                        print(f"  {level}: {count} files")
            
            print(f"\nConsolidation report saved to: {os.path.join(args.output_path, 'consolidation_report.json')}")
            print(f"Master index available at: {os.path.join(args.output_path, 'master_index.md')}")
            
    except Exception as e:
        print(f"Error during consolidation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()