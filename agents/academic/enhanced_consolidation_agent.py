#!/usr/bin/env python
"""
Enhanced Content Consolidation Agent - Integrates week resolver functionality
with the existing consolidation system to provide comprehensive content organization
and week numbering resolution.

This builds on the original consolidation agent and adds week resolver capabilities.
"""

import os
import sys
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

# Import base consolidation functionality
from .consolidation_agent import ContentConsolidationAgent, FileMapping, ConsolidationResult
# Use unified BaseAgent for standardized interface
from ...src.agents.base_agent import BaseAgent, AgentMessage

# Import from unified architecture - no more path manipulation!
from ...src.core.output_manager import get_output_manager, OutputCategory, ContentType

try:
    from ...src.processors.week_resolver import WeekResolver, WeekMapping, WeekResolutionResult
except ImportError as e:
    print(f"Warning: WeekResolver not available: {e}")
    WeekResolver = None
    WeekMapping = None
    WeekResolutionResult = None


@dataclass 
class EnhancedConsolidationResult:
    """Extended consolidation result with week resolution information"""
    consolidation_result: ConsolidationResult
    week_resolution_result: Optional[WeekResolutionResult] = None
    week_discrepancies_resolved: int = 0
    week_conflicts_found: int = 0
    integration_success: bool = True
    integration_errors: List[str] = None


class EnhancedContentConsolidationAgent(ContentConsolidationAgent):
    """
    Enhanced consolidation agent that integrates week numbering resolution
    with content consolidation functionality.
    """
    
    def __init__(self, enable_week_resolution: bool = True):
        super().__init__()
        self.agent_id = "enhanced_consolidation_agent"
        self.enable_week_resolution = enable_week_resolution and WeekResolver is not None
        
        # Week resolver configuration
        self.week_resolver_config_path = None
        self.week_resolver = None
        
        # Initialize week resolver if available
        if self.enable_week_resolution:
            self._initialize_week_resolver()
        else:
            self.logger.warning("Week resolution disabled or WeekResolver not available")
    
    def _initialize_week_resolver(self):
        """Initialize the week resolver component"""
        try:
            # Look for week resolver config
            config_paths = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'week_resolver_config.json'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'consolidation_config.json')
            ]
            
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            self.week_resolver_config_path = config_path
            self.week_resolver = WeekResolver(config_path=config_path)
            
            self.logger.info(f"Week resolver initialized with config: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize week resolver: {e}")
            self.enable_week_resolution = False
    
    async def initialize(self):
        """Initialize agent-specific resources, including parent initialization."""
        try:
            # Initialize parent (ContentConsolidationAgent)
            await super().initialize()
            
            # Setup enhanced consolidation directories
            enhanced_dir = self.output_manager.get_output_path(
                OutputCategory.PROCESSED,
                ContentType.JSON,
                subdirectory="enhanced_consolidation"
            )
            enhanced_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup week resolution reports directory
            week_reports_dir = self.output_manager.get_output_path(
                OutputCategory.REPORTS,
                ContentType.JSON,
                subdirectory="week_resolution"
            )
            week_reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Re-initialize week resolver if needed after output manager is available
            if self.enable_week_resolution and self.week_resolver:
                # Week resolver is already initialized in __init__
                pass
            
            self.logger.info(f"{self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.agent_id}: {e}")
            raise

    async def cleanup(self):
        """Cleanup agent resources, including parent cleanup."""
        try:
            # Cleanup week resolver
            if hasattr(self, 'week_resolver') and self.week_resolver:
                # WeekResolver doesn't need explicit cleanup in current version
                pass
            
            # Cleanup parent (ContentConsolidationAgent)
            await super().cleanup()
            
            self.logger.info(f"{self.agent_id} cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during {self.agent_id} cleanup: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for enhanced consolidation."""
        # Use parent validation and add enhanced checks
        if hasattr(super(), 'validate_input'):
            parent_valid = super().validate_input(input_data)
        else:
            parent_valid = True  # Fallback if parent doesn't have method
        
        if isinstance(input_data, dict):
            # Check for enhanced consolidation specific fields
            enhanced_fields = ["search_paths", "output_path"]
            enhanced_valid = all(field in input_data for field in enhanced_fields)
            return parent_valid and enhanced_valid
        
        return parent_valid

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data from enhanced consolidation."""
        # Use parent validation and add enhanced checks
        if hasattr(super(), 'validate_output'):
            parent_valid = super().validate_output(output_data)
        else:
            parent_valid = True  # Fallback if parent doesn't have method
        
        if isinstance(output_data, EnhancedConsolidationResult):
            enhanced_valid = (
                hasattr(output_data, 'consolidation_result') and
                hasattr(output_data, 'integration_success')
            )
            return parent_valid and enhanced_valid
        
        return parent_valid
    
    def consolidate_with_week_resolution(self, search_paths: List[str], output_path: str,
                                       resolve_weeks_first: bool = True) -> EnhancedConsolidationResult:
        """
        Enhanced consolidation workflow that includes week numbering resolution
        
        Args:
            search_paths: List of paths to search for content
            output_path: Path for consolidated output
            resolve_weeks_first: Whether to resolve week numbers before consolidation
            
        Returns:
            Enhanced consolidation result with week resolution information
        """
        self.logger.info("Starting enhanced consolidation workflow with week resolution")
        
        week_resolution_result = None
        integration_errors = []
        
        try:
            if self.enable_week_resolution and resolve_weeks_first:
                # Step 1: Resolve week numbering issues first
                self.logger.info("Step 1: Resolving week numbering discrepancies")
                week_resolution_result = self._resolve_week_numbers(search_paths, output_path)
                
                if week_resolution_result.success:
                    # Use resolved content as input for consolidation
                    resolved_paths = self._get_resolved_content_paths(output_path)
                    consolidation_input_paths = resolved_paths if resolved_paths else search_paths
                else:
                    self.logger.warning("Week resolution had issues, proceeding with original paths")
                    consolidation_input_paths = search_paths
            else:
                consolidation_input_paths = search_paths
            
            # Step 2: Run standard consolidation workflow
            self.logger.info("Step 2: Running content consolidation")
            consolidation_result = self.consolidate_workflow(consolidation_input_paths, output_path)
            
            # Step 3: Post-consolidation week resolution (if not done before)
            if self.enable_week_resolution and not resolve_weeks_first:
                self.logger.info("Step 3: Post-consolidation week resolution")
                week_resolution_result = self._resolve_week_numbers([output_path], output_path)
            
            # Step 4: Integration and final organization
            self.logger.info("Step 4: Final integration and organization")
            self._integrate_week_resolution_with_consolidation(
                consolidation_result, week_resolution_result, output_path
            )
            
        except Exception as e:
            self.logger.error(f"Error in enhanced consolidation workflow: {e}")
            integration_errors.append(str(e))
            
            # Fallback to basic consolidation
            consolidation_result = self.consolidate_workflow(search_paths, output_path)
        
        # Calculate integration metrics
        week_conflicts_found = 0
        week_discrepancies_resolved = 0
        
        if week_resolution_result:
            week_conflicts_found = len(week_resolution_result.discrepancies)
            week_discrepancies_resolved = len([
                m for m in week_resolution_result.mappings 
                if m.original_week != m.resolved_week
            ])
        
        return EnhancedConsolidationResult(
            consolidation_result=consolidation_result,
            week_resolution_result=week_resolution_result,
            week_discrepancies_resolved=week_discrepancies_resolved,
            week_conflicts_found=week_conflicts_found,
            integration_success=len(integration_errors) == 0,
            integration_errors=integration_errors or []
        )
    
    def _resolve_week_numbers(self, search_paths: List[str], output_path: str) -> WeekResolutionResult:
        """Run week number resolution on the specified paths"""
        if not self.week_resolver:
            raise RuntimeError("Week resolver not initialized")
        
        # Create separate directory for week resolution
        week_resolution_path = os.path.join(output_path, "week_resolved")
        os.makedirs(week_resolution_path, exist_ok=True)
        
        # Run week resolution
        result = self.week_resolver.resolve_academic_content(search_paths, week_resolution_path)
        
        # Log results
        self.logger.info(f"Week resolution completed: {len(result.mappings)} files processed")
        if result.discrepancies:
            self.logger.info(f"Resolved {len(result.discrepancies)} week numbering discrepancies")
        
        return result
    
    def _get_resolved_content_paths(self, output_path: str) -> List[str]:
        """Get paths to week-resolved content for use in consolidation"""
        week_resolution_path = os.path.join(output_path, "week_resolved")
        
        if not os.path.exists(week_resolution_path):
            return []
        
        # Find content type directories
        resolved_paths = []
        for item in os.listdir(week_resolution_path):
            item_path = os.path.join(week_resolution_path, item)
            if os.path.isdir(item_path):
                resolved_paths.append(item_path)
        
        return resolved_paths
    
    def _integrate_week_resolution_with_consolidation(self, 
                                                    consolidation_result: ConsolidationResult,
                                                    week_resolution_result: Optional[WeekResolutionResult],
                                                    output_path: str):
        """Integrate week resolution results with consolidation results"""
        
        if not week_resolution_result:
            return
        
        # Create integrated metadata
        integration_metadata = {
            'consolidation_summary': consolidation_result.consolidation_report,
            'week_resolution_summary': week_resolution_result.resolution_report,
            'integration_timestamp': datetime.now().isoformat(),
            'integration_agent': self.agent_id
        }
        
        # Save integrated metadata
        metadata_path = os.path.join(output_path, "integration_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(integration_metadata, f, indent=2, default=str)
        
        # Create cross-reference index
        self._create_cross_reference_index(consolidation_result, week_resolution_result, output_path)
        
        # Generate final master index
        self._generate_enhanced_master_index(output_path)
    
    def _create_cross_reference_index(self, 
                                    consolidation_result: ConsolidationResult,
                                    week_resolution_result: WeekResolutionResult,
                                    output_path: str):
        """Create cross-reference index linking consolidation and week resolution results"""
        
        cross_reference = {
            'file_mappings': {},
            'week_mappings': {},
            'content_type_mappings': {}
        }
        
        # Build file mapping cross-reference
        for mapping in week_resolution_result.mappings:
            filename = os.path.basename(mapping.file_path)
            cross_reference['file_mappings'][filename] = {
                'original_week': mapping.original_week,
                'resolved_week': mapping.resolved_week,
                'content_type': mapping.content_type,
                'resolution_method': mapping.resolution_method,
                'confidence': mapping.confidence
            }
        
        # Build week mapping cross-reference
        week_files = defaultdict(list)
        for mapping in week_resolution_result.mappings:
            week_files[mapping.resolved_week].append({
                'filename': os.path.basename(mapping.file_path),
                'content_type': mapping.content_type,
                'original_week': mapping.original_week
            })
        
        cross_reference['week_mappings'] = dict(week_files)
        
        # Build content type cross-reference
        content_type_files = defaultdict(list)
        for mapping in week_resolution_result.mappings:
            content_type_files[mapping.content_type].append({
                'filename': os.path.basename(mapping.file_path),
                'week': mapping.resolved_week,
                'original_week': mapping.original_week
            })
        
        cross_reference['content_type_mappings'] = dict(content_type_files)
        
        # Save cross-reference index
        cross_ref_path = os.path.join(output_path, "cross_reference_index.json")
        with open(cross_ref_path, 'w', encoding='utf-8') as f:
            json.dump(cross_reference, f, indent=2, default=str)
    
    def _generate_enhanced_master_index(self, output_path: str):
        """Generate enhanced master index with week resolution information"""
        
        index_content = [
            "# Enhanced Academic Content Index",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This index includes both consolidated content and week numbering resolution results.",
            "",
            "## Content Organization",
            ""
        ]
        
        # Scan output directory for organized content
        for item in os.listdir(output_path):
            item_path = os.path.join(output_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                index_content.append(f"### {item.title().replace('_', ' ')}")
                
                # Count files in directory
                try:
                    files = [f for f in os.listdir(item_path) if f.endswith('.md')]
                    index_content.append(f"Files: {len(files)}")
                    
                    if files:
                        index_content.append("Files:")
                        for file in sorted(files)[:10]:  # Show first 10 files
                            index_content.append(f"- [{file}]({item}/{file})")
                        if len(files) > 10:
                            index_content.append(f"- ... and {len(files) - 10} more files")
                    
                except Exception as e:
                    index_content.append(f"Error listing files: {e}")
                
                index_content.append("")
        
        # Add week resolution summary if available
        week_resolution_report_path = os.path.join(output_path, "week_resolution_report.json")
        if os.path.exists(week_resolution_report_path):
            try:
                with open(week_resolution_report_path, 'r', encoding='utf-8') as f:
                    week_report = json.load(f)
                
                index_content.extend([
                    "## Week Resolution Summary",
                    "",
                    f"Files processed: {week_report.get('files_processed', 'unknown')}",
                    f"Discrepancies found: {len(week_report.get('discrepancies', []))}",
                    f"Files changed: {week_report.get('resolution_result', {}).get('summary', {}).get('files_with_changes', 'unknown')}",
                    ""
                ])
                
                if week_report.get('discrepancies'):
                    index_content.append("### Discrepancies Resolved:")
                    for disc in week_report['discrepancies']:
                        index_content.append(f"- {disc.get('description', 'Unknown discrepancy')}")
                    index_content.append("")
                
            except Exception as e:
                index_content.append(f"Error reading week resolution report: {e}")
        
        # Add consolidation summary if available
        consolidation_report_path = os.path.join(output_path, "consolidation_report.json")
        if os.path.exists(consolidation_report_path):
            try:
                with open(consolidation_report_path, 'r', encoding='utf-8') as f:
                    consolidation_report = json.load(f)
                
                summary = consolidation_report.get('consolidation_summary', {})
                index_content.extend([
                    "## Consolidation Summary",
                    "",
                    f"Files processed: {summary.get('total_files_processed', 'unknown')}",
                    f"Files skipped: {summary.get('total_files_skipped', 'unknown')}",
                    f"Errors: {summary.get('total_errors', 'unknown')}",
                    ""
                ])
                
            except Exception as e:
                index_content.append(f"Error reading consolidation report: {e}")
        
        # Add usage instructions
        index_content.extend([
            "## Usage Instructions",
            "",
            "### Accessing Content",
            "- Navigate to content type directories (transcripts, lectures, notes, textbook)",
            "- Files are organized by week number and content type",
            "- Check `cross_reference_index.json` for detailed file mappings",
            "",
            "### Understanding Week Resolution",
            "- Original week numbering issues have been identified and resolved",
            "- Check `week_resolution_log.json` for detailed resolution history",
            "- Files with resolved week numbers include metadata headers with resolution information",
            "",
            "### Reports and Metadata",
            "- `integration_metadata.json`: Combined consolidation and week resolution metadata",
            "- `week_resolution_report.json`: Detailed week resolution results",
            "- `consolidation_report.json`: Content consolidation results",
            ""
        ])
        
        # Save enhanced master index
        index_path = os.path.join(output_path, "enhanced_master_index.md")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(index_content))
    
    def analyze_content_quality_with_weeks(self, search_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze content quality including week numbering consistency
        
        Args:
            search_paths: Paths to analyze
            
        Returns:
            Comprehensive quality analysis including week numbering
        """
        quality_analysis = {
            'content_quality': {},
            'week_numbering_quality': {},
            'integration_quality': {},
            'recommendations': []
        }
        
        # Basic content quality analysis from parent class
        try:
            # This would integrate with existing quality checks
            content_files = []
            for search_path in search_paths:
                if os.path.exists(search_path):
                    for root, dirs, files in os.walk(search_path):
                        for file in files:
                            if file.endswith(('.md', '.txt')):
                                content_files.append(os.path.join(root, file))
            
            quality_analysis['content_quality'] = {
                'total_files': len(content_files),
                'accessible_files': len(content_files),  # Simplified
                'quality_score': 0.8  # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Error in content quality analysis: {e}")
            quality_analysis['content_quality'] = {'error': str(e)}
        
        # Week numbering quality analysis
        if self.enable_week_resolution:
            try:
                week_quality = self._analyze_week_numbering_quality(search_paths)
                quality_analysis['week_numbering_quality'] = week_quality
                
                # Generate recommendations based on week analysis
                if week_quality.get('discrepancies_found', 0) > 0:
                    quality_analysis['recommendations'].append(
                        "Week numbering discrepancies detected. Run week resolution workflow."
                    )
                
                if week_quality.get('missing_weeks'):
                    quality_analysis['recommendations'].append(
                        f"Missing weeks detected: {week_quality['missing_weeks']}. Check for mislabeled content."
                    )
                
            except Exception as e:
                self.logger.error(f"Error in week numbering analysis: {e}")
                quality_analysis['week_numbering_quality'] = {'error': str(e)}
        
        # Integration quality assessment
        quality_analysis['integration_quality'] = {
            'week_resolver_available': self.enable_week_resolution,
            'config_files_found': self.week_resolver_config_path is not None,
            'system_ready': self.enable_week_resolution and self.week_resolver is not None
        }
        
        return quality_analysis
    
    def _analyze_week_numbering_quality(self, search_paths: List[str]) -> Dict[str, Any]:
        """Analyze the quality of week numbering across content"""
        
        if not self.week_resolver:
            return {'error': 'Week resolver not available'}
        
        # Discover files and detect week numbers
        file_detections = {}
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(('.md', '.txt')):
                        file_path = os.path.join(root, file)
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except:
                            continue
                        
                        detections = self.week_resolver.detect_week_numbers(file_path, content)
                        if detections:
                            file_detections[file_path] = detections
        
        # Identify discrepancies
        discrepancies = self.week_resolver.identify_discrepancies(file_detections)
        
        # Calculate week distribution
        week_distribution = defaultdict(int)
        confidence_scores = []
        
        for file_path, detections in file_detections.items():
            week = detections[0].week_number
            confidence = detections[0].confidence
            
            week_distribution[week] += 1
            confidence_scores.append(confidence)
        
        # Calculate metrics
        detected_weeks = set(week_distribution.keys())
        expected_weeks = set(range(1, 16))
        missing_weeks = sorted(expected_weeks - detected_weeks)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        quality_metrics = {
            'total_files_with_weeks': len(file_detections),
            'unique_weeks_detected': len(detected_weeks),
            'missing_weeks': missing_weeks,
            'week_distribution': dict(week_distribution),
            'discrepancies_found': len(discrepancies),
            'discrepancies': discrepancies,
            'average_confidence': avg_confidence,
            'quality_score': self._calculate_week_quality_score(
                len(file_detections), len(discrepancies), avg_confidence, len(missing_weeks)
            )
        }
        
        return quality_metrics
    
    def _calculate_week_quality_score(self, total_files: int, discrepancies: int, 
                                    avg_confidence: float, missing_weeks: int) -> float:
        """Calculate overall week numbering quality score"""
        if total_files == 0:
            return 0.0
        
        # Base score from confidence
        confidence_score = avg_confidence
        
        # Penalty for discrepancies
        discrepancy_penalty = min(0.5, (discrepancies / total_files) * 2)
        
        # Penalty for missing weeks
        missing_penalty = min(0.3, (missing_weeks / 15) * 0.5)
        
        quality_score = confidence_score - discrepancy_penalty - missing_penalty
        
        return max(0.0, min(1.0, quality_score))
    
    def check_quality(self, content: Dict[str, Any]) -> float:
        """Enhanced quality check that includes week numbering quality"""
        
        # Get base quality score from parent class
        base_quality = super().check_quality(content)
        
        # Add week numbering quality if available
        if 'week_quality_analysis' in content:
            week_quality = content['week_quality_analysis'].get('quality_score', 0.7)
            # Weighted average: 70% content quality, 30% week quality
            enhanced_quality = (base_quality * 0.7) + (week_quality * 0.3)
        else:
            enhanced_quality = base_quality
        
        return enhanced_quality


def main():
    """Main entry point for the enhanced consolidation agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Content Consolidation Agent with Week Resolution")
    parser.add_argument("--search-paths", nargs='+', required=True,
                       help="Paths to search for content files")
    parser.add_argument("--output-path", required=True,
                       help="Path for consolidated output")
    parser.add_argument("--disable-week-resolution", action="store_true",
                       help="Disable week numbering resolution")
    parser.add_argument("--resolve-weeks-first", action="store_true", default=True,
                       help="Resolve week numbers before consolidation")
    parser.add_argument("--analysis-only", action="store_true",
                       help="Only perform analysis without making changes")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Create enhanced consolidation agent
    agent = EnhancedContentConsolidationAgent(
        enable_week_resolution=not args.disable_week_resolution
    )
    
    if args.analysis_only:
        # Run analysis only
        print("Running content quality analysis...")
        analysis = agent.analyze_content_quality_with_weeks(args.search_paths)
        
        print("\nAnalysis Results:")
        print("="*50)
        
        # Content quality
        content_quality = analysis.get('content_quality', {})
        print(f"Content Quality:")
        print(f"  Total files: {content_quality.get('total_files', 'unknown')}")
        print(f"  Quality score: {content_quality.get('quality_score', 'unknown')}")
        
        # Week numbering quality
        week_quality = analysis.get('week_numbering_quality', {})
        if 'error' not in week_quality:
            print(f"\nWeek Numbering Quality:")
            print(f"  Files with week info: {week_quality.get('total_files_with_weeks', 0)}")
            print(f"  Unique weeks: {week_quality.get('unique_weeks_detected', 0)}")
            print(f"  Missing weeks: {week_quality.get('missing_weeks', [])}")
            print(f"  Discrepancies: {week_quality.get('discrepancies_found', 0)}")
            print(f"  Quality score: {week_quality.get('quality_score', 'unknown'):.2f}")
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Save analysis
        analysis_path = os.path.join(args.output_path, "content_analysis.json")
        os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis saved to: {analysis_path}")
        
    else:
        # Run enhanced consolidation workflow
        print("Running enhanced consolidation workflow...")
        result = agent.consolidate_with_week_resolution(
            args.search_paths, 
            args.output_path,
            resolve_weeks_first=args.resolve_weeks_first
        )
        
        # Print results
        print(f"\nEnhanced Consolidation Results:")
        print("="*50)
        print(f"Integration Success: {result.integration_success}")
        print(f"Week Conflicts Found: {result.week_conflicts_found}")
        print(f"Week Discrepancies Resolved: {result.week_discrepancies_resolved}")
        
        # Consolidation results
        consolidation = result.consolidation_result
        print(f"\nConsolidation:")
        print(f"  Success: {consolidation.success}")
        print(f"  Processed files: {len(consolidation.processed_files)}")
        print(f"  Skipped files: {len(consolidation.skipped_files)}")
        print(f"  Errors: {len(consolidation.errors)}")
        
        # Week resolution results
        if result.week_resolution_result:
            week_res = result.week_resolution_result
            print(f"\nWeek Resolution:")
            print(f"  Success: {week_res.success}")
            print(f"  Files processed: {len(week_res.mappings)}")
            print(f"  Manual review needed: {len(week_res.manual_review_needed)}")
            print(f"  Errors: {len(week_res.errors)}")
        
        # Integration errors
        if result.integration_errors:
            print(f"\nIntegration Errors:")
            for error in result.integration_errors:
                print(f"  - {error}")
        
        print(f"\nOutput saved to: {args.output_path}")


if __name__ == "__main__":
    main()