#!/usr/bin/env python
"""
Week Resolver Tool - Command line interface for week numbering resolution

This tool provides a convenient interface to detect and resolve week numbering
discrepancies in academic content, particularly addressing the week-13/14 issue.

Usage:
    python tools/week_resolver_tool.py --search-paths /path/to/content --output-path /path/to/resolved
    python tools/week_resolver_tool.py --analyze-only --search-paths /path/to/content
    python tools/week_resolver_tool.py --fix-week-13-14 --search-paths /path/to/content --output-path /path/to/resolved
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the processors module to the path
sys.path.append(str(Path(__file__).parent.parent / "academic-agent-v2" / "src"))

try:
    from processors.week_resolver import WeekResolver, WeekResolutionResult
except ImportError as e:
    print(f"Error importing WeekResolver: {e}")
    print("Please ensure the week_resolver.py file is in the correct location.")
    sys.exit(1)


class WeekResolverTool:
    """Command line tool for week number resolution"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def setup_logging(self, verbose: bool = False, log_file: str = None):
        """Setup logging configuration"""
        level = logging.DEBUG if verbose else logging.INFO
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def analyze_week_issues(self, search_paths: List[str], config_path: str = None) -> Dict[str, Any]:
        """Analyze week numbering issues without making changes"""
        self.logger.info("Analyzing week numbering issues...")
        
        # Create resolver
        resolver = WeekResolver(config_path=config_path)
        
        # Discover and analyze files
        file_detections = {}
        total_files = 0
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                self.logger.warning(f"Search path does not exist: {search_path}")
                continue
            
            self.logger.info(f"Scanning: {search_path}")
            
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(('.md', '.txt')):
                        file_path = os.path.join(root, file)
                        total_files += 1
                        
                        # Read content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception as e:
                            self.logger.warning(f"Could not read {file_path}: {e}")
                            content = None
                        
                        # Detect week numbers
                        detections = resolver.detect_week_numbers(file_path, content)
                        if detections:
                            file_detections[file_path] = detections
        
        # Identify discrepancies
        discrepancies = resolver.identify_discrepancies(file_detections)
        
        # Create analysis report
        analysis_report = {
            'summary': {
                'total_files_scanned': total_files,
                'files_with_week_info': len(file_detections),
                'discrepancies_found': len(discrepancies),
                'analysis_date': resolver.logger.handlers[0].formatter.formatTime(
                    logging.LogRecord('', 0, '', 0, '', (), None)
                ) if resolver.logger.handlers else 'unknown'
            },
            'file_detections': {
                os.path.basename(path): [
                    {
                        'week_number': d.week_number,
                        'confidence': d.confidence,
                        'source': d.source,
                        'context': d.context
                    }
                    for d in detections
                ]
                for path, detections in file_detections.items()
            },
            'discrepancies': discrepancies,
            'recommendations': self._generate_recommendations(discrepancies)
        }
        
        return analysis_report
    
    def _generate_recommendations(self, discrepancies: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on identified discrepancies"""
        recommendations = []
        
        for discrepancy in discrepancies:
            if discrepancy['type'] == 'missing_weeks':
                missing = discrepancy['weeks']
                recommendations.append(f"Missing weeks {missing}: Check if content exists under different naming")
            
            elif discrepancy['type'] == 'suspicious_gap':
                week = discrepancy['week']
                recommendations.append(f"Week {week} missing but week {week+1} exists: Likely mislabeling, consider renaming week {week+1} to week {week}")
            
            elif discrepancy['type'] == 'duplicate_weeks':
                duplicates = discrepancy['weeks']
                recommendations.append(f"Duplicate weeks found: {list(duplicates.keys())}: Review and merge or rename duplicates")
            
            elif discrepancy['type'] == 'extra_weeks':
                extra = discrepancy['weeks']
                recommendations.append(f"Unexpected weeks {extra}: Review if these are valid or mislabeled")
        
        return recommendations
    
    def fix_week_13_14_issue(self, search_paths: List[str], output_path: str, config_path: str = None) -> WeekResolutionResult:
        """Specifically address the week 13/14 issue"""
        self.logger.info("Addressing week 13/14 numbering issue...")
        
        # Load config with specific week 13/14 overrides
        config = self._load_week_13_14_config(config_path)
        
        # Create resolver with updated config
        config_file = self._create_temp_config(config)
        resolver = WeekResolver(config_path=config_file)
        
        # Run resolution workflow
        result = resolver.resolve_academic_content(search_paths, output_path)
        
        # Clean up temp config
        if os.path.exists(config_file):
            os.remove(config_file)
        
        return result
    
    def _load_week_13_14_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration with specific week 13/14 fixes"""
        default_config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'week_resolver_config.json'
        )
        
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        elif os.path.exists(default_config_path):
            with open(default_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # Add specific week 13/14 overrides
        if 'manual_overrides' not in config:
            config['manual_overrides'] = {}
        
        # Common week 14 -> week 13 mappings
        week_14_patterns = [
            'week-14.md',
            'week_14.md',
            'week-14_transcript.md',
            'week_14_transcript.md',
            'week-14_notes.md',
            'week_14_notes.md',
            'week-14_lecture.md',
            'week_14_lecture.md'
        ]
        
        for pattern in week_14_patterns:
            config['manual_overrides'][pattern] = 13
        
        return config
    
    def _create_temp_config(self, config: Dict[str, Any]) -> str:
        """Create a temporary config file"""
        import tempfile
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_file, indent=2)
        temp_file.close()
        
        return temp_file.name
    
    def run_full_resolution(self, search_paths: List[str], output_path: str, config_path: str = None) -> WeekResolutionResult:
        """Run complete week number resolution workflow"""
        self.logger.info("Running complete week number resolution...")
        
        resolver = WeekResolver(config_path=config_path)
        result = resolver.resolve_academic_content(search_paths, output_path)
        
        return result
    
    def print_analysis_report(self, report: Dict[str, Any]):
        """Print formatted analysis report"""
        print("\n" + "="*80)
        print("WEEK NUMBERING ANALYSIS REPORT")
        print("="*80)
        
        summary = report['summary']
        print(f"\nSummary:")
        print(f"  Total files scanned: {summary['total_files_scanned']}")
        print(f"  Files with week info: {summary['files_with_week_info']}")
        print(f"  Discrepancies found: {summary['discrepancies_found']}")
        
        if report['discrepancies']:
            print(f"\nDiscrepancies Found:")
            for i, disc in enumerate(report['discrepancies'], 1):
                print(f"  {i}. {disc['type'].replace('_', ' ').title()}")
                print(f"     {disc['description']}")
                if 'suggested_resolution' in disc:
                    print(f"     Suggested: {disc['suggested_resolution']}")
                print()
        
        if report['recommendations']:
            print(f"Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\nDetailed File Analysis:")
        for filename, detections in report['file_detections'].items():
            print(f"  {filename}:")
            for detection in detections:
                print(f"    Week {detection['week_number']} (confidence: {detection['confidence']:.2f}) - {detection['source']}: {detection['context']}")
        
        print("="*80)
    
    def print_resolution_report(self, result: WeekResolutionResult):
        """Print formatted resolution report"""
        print("\n" + "="*80)
        print("WEEK RESOLUTION RESULTS")
        print("="*80)
        
        print(f"\nOverall Success: {'✓' if result.success else '✗'}")
        print(f"Files processed: {len(result.mappings)}")
        print(f"Errors encountered: {len(result.errors)}")
        print(f"Files needing manual review: {len(result.manual_review_needed)}")
        
        if result.resolution_report:
            summary = result.resolution_report['summary']
            print(f"\nProcessing Summary:")
            print(f"  Files with changes: {summary['files_with_changes']}")
            print(f"  Files unchanged: {summary['files_unchanged']}")
            
            if 'week_changes' in result.resolution_report and result.resolution_report['week_changes']:
                print(f"\nWeek Number Changes:")
                for change in result.resolution_report['week_changes']:
                    print(f"  {change['file']}: week {change['original_week']} → week {change['resolved_week']} ({change['method']})")
        
        if result.discrepancies:
            print(f"\nDiscrepancies Addressed:")
            for disc in result.discrepancies:
                print(f"  - {disc['description']}")
        
        if result.manual_review_needed:
            print(f"\nFiles Requiring Manual Review:")
            for file_path in result.manual_review_needed:
                print(f"  - {os.path.basename(file_path)}")
        
        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error.get('file_path', 'unknown')}: {error.get('error', 'unknown error')}")
        
        print("="*80)


def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(
        description="Week Number Resolution Tool for Academic Content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze week numbering issues (no changes made)
  python week_resolver_tool.py --analyze-only --search-paths /path/to/content
  
  # Fix week 13/14 issue specifically
  python week_resolver_tool.py --fix-week-13-14 --search-paths /path/to/content --output-path /path/to/resolved
  
  # Full resolution workflow
  python week_resolver_tool.py --search-paths /path/to/content --output-path /path/to/resolved
  
  # Use custom configuration
  python week_resolver_tool.py --search-paths /path/to/content --output-path /path/to/resolved --config custom_config.json
        """
    )
    
    # Required arguments
    parser.add_argument("--search-paths", nargs='+', required=True,
                       help="Paths to search for academic content files")
    
    # Optional arguments
    parser.add_argument("--output-path", 
                       help="Output directory for resolved files (required unless --analyze-only)")
    parser.add_argument("--config", 
                       help="Path to week resolver configuration file")
    
    # Mode selection
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze issues without making changes")
    parser.add_argument("--fix-week-13-14", action="store_true",
                       help="Specifically address the week 13/14 numbering issue")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--log-file", 
                       help="Log file path (default: logs/week_resolver.log)")
    
    # Output options
    parser.add_argument("--save-analysis", 
                       help="Save analysis report to specified file")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress console output except errors")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.analyze_only and not args.output_path:
        parser.error("--output-path is required unless --analyze-only is specified")
    
    # Create tool instance
    tool = WeekResolverTool()
    
    # Setup logging
    log_file = args.log_file or "logs/week_resolver.log"
    tool.setup_logging(verbose=args.verbose, log_file=log_file)
    
    try:
        if args.analyze_only:
            # Analysis mode
            print("Analyzing week numbering issues...")
            report = tool.analyze_week_issues(args.search_paths, args.config)
            
            if not args.quiet:
                tool.print_analysis_report(report)
            
            # Save analysis if requested
            if args.save_analysis:
                os.makedirs(os.path.dirname(args.save_analysis), exist_ok=True)
                with open(args.save_analysis, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\nAnalysis report saved to: {args.save_analysis}")
        
        elif args.fix_week_13_14:
            # Week 13/14 specific fix
            print("Fixing week 13/14 numbering issue...")
            result = tool.fix_week_13_14_issue(args.search_paths, args.output_path, args.config)
            
            if not args.quiet:
                tool.print_resolution_report(result)
        
        else:
            # Full resolution workflow
            print("Running complete week number resolution...")
            result = tool.run_full_resolution(args.search_paths, args.output_path, args.config)
            
            if not args.quiet:
                tool.print_resolution_report(result)
        
        print(f"\nCompleted successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()