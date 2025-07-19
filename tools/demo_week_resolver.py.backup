#!/usr/bin/env python
"""
Week Resolver Demonstration Script

This script demonstrates the week resolver functionality on the actual
academic-agent project data to identify and resolve week numbering issues.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the processors module to the path
sys.path.append(str(Path(__file__).parent.parent / "academic-agent-v2" / "src"))

try:
    from src.processors.week_resolver import WeekResolver
except ImportError as e:
    print(f"Error importing WeekResolver: {e}")
    sys.exit(1)


class WeekResolverDemo:
    """Demonstration of week resolver functionality"""
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.search_paths = [
            os.path.join(project_root, "output/sra/transcripts/markdown"),
            os.path.join(project_root, "output/sra/transcripts/standardized"),
            os.path.join(project_root, "output/sra/integrated_notes"),
            os.path.join(project_root, "output/sra/enhanced_integrated_notes"),
            os.path.join(project_root, "output/sra/ai_enhanced_study_notes"),
        ]
        
        # Configuration path
        self.config_path = os.path.join(project_root, "config/week_resolver_config.json")
        
    def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of week numbering in the project"""
        print("Analyzing current week numbering state...")
        print("="*50)
        
        resolver = WeekResolver(config_path=self.config_path)
        
        # Discover all files with week information
        all_file_detections = {}
        path_analysis = {}
        
        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                print(f"Skipping non-existent path: {search_path}")
                continue
            
            print(f"\nAnalyzing: {os.path.relpath(search_path, self.project_root)}")
            path_files = []
            
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(('.md', '.txt')):
                        file_path = os.path.join(root, file)
                        
                        # Read content
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception as e:
                            print(f"  Warning: Could not read {file}: {e}")
                            continue
                        
                        # Detect week numbers
                        detections = resolver.detect_week_numbers(file_path, content)
                        if detections:
                            all_file_detections[file_path] = detections
                            path_files.append({
                                'file': file,
                                'week': detections[0].week_number,
                                'confidence': detections[0].confidence,
                                'source': detections[0].source
                            })
                            
                            print(f"  {file}: week {detections[0].week_number} (confidence: {detections[0].confidence:.2f}, source: {detections[0].source})")
            
            path_analysis[search_path] = path_files
        
        # Identify discrepancies
        print(f"\nDiscrepancy Analysis:")
        print("-" * 30)
        discrepancies = resolver.identify_discrepancies(all_file_detections)
        
        if discrepancies:
            for disc in discrepancies:
                print(f"• {disc['type'].replace('_', ' ').title()}: {disc['description']}")
                if 'suggested_resolution' in disc:
                    print(f"  → Suggested: {disc['suggested_resolution']}")
        else:
            print("No discrepancies found.")
        
        # Week distribution analysis
        week_distribution = {}
        for file_path, detections in all_file_detections.items():
            week = detections[0].week_number
            if week not in week_distribution:
                week_distribution[week] = []
            week_distribution[week].append(os.path.basename(file_path))
        
        print(f"\nWeek Distribution:")
        print("-" * 20)
        for week in sorted(week_distribution.keys()):
            files = week_distribution[week]
            print(f"Week {week:2d}: {len(files)} files")
            for file in files:
                print(f"         {file}")
        
        # Identify missing weeks
        detected_weeks = set(week_distribution.keys())
        expected_weeks = set(range(1, 16))  # Weeks 1-15
        missing_weeks = expected_weeks - detected_weeks
        
        if missing_weeks:
            print(f"\nMissing Weeks: {sorted(missing_weeks)}")
        
        return {
            'total_files_analyzed': len(all_file_detections),
            'path_analysis': path_analysis,
            'discrepancies': discrepancies,
            'week_distribution': week_distribution,
            'missing_weeks': sorted(missing_weeks),
            'detected_weeks': sorted(detected_weeks)
        }
    
    def demonstrate_week_13_14_fix(self) -> Dict[str, Any]:
        """Demonstrate the specific week 13/14 issue resolution"""
        print("\n" + "="*50)
        print("DEMONSTRATING WEEK 13/14 ISSUE RESOLUTION")
        print("="*50)
        
        # Find files that might be affected by the week 13/14 issue
        week_14_files = []
        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                continue
            
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if 'week-14' in file.lower() or 'week_14' in file.lower():
                        file_path = os.path.join(root, file)
                        week_14_files.append(file_path)
        
        print(f"Found {len(week_14_files)} files with 'week-14' naming:")
        for file_path in week_14_files:
            print(f"  {os.path.relpath(file_path, self.project_root)}")
        
        if not week_14_files:
            print("No week-14 files found. The issue may already be resolved or not present.")
            return {'week_14_files': [], 'resolution_applied': False}
        
        # Analyze content of week-14 files
        print(f"\nAnalyzing content of week-14 files:")
        resolver = WeekResolver(config_path=self.config_path)
        
        for file_path in week_14_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                print(f"\n{os.path.basename(file_path)}:")
                
                # Look for content that suggests this is actually week 13
                week_13_indicators = [
                    'final lecture',
                    'exam briefing', 
                    'review',
                    'q&a',
                    'question and answer',
                    'final concepts',
                    'last class'
                ]
                
                content_lower = content.lower()
                found_indicators = [indicator for indicator in week_13_indicators if indicator in content_lower]
                
                if found_indicators:
                    print(f"  Week 13 content indicators found: {found_indicators}")
                    print(f"  → This file likely contains week 13 content mislabeled as week 14")
                else:
                    print(f"  No clear week 13 indicators found")
                
                # Check for explicit week mentions in content
                import re
                week_mentions = re.findall(r'week\s*(\d+)', content_lower)
                if week_mentions:
                    print(f"  Content mentions weeks: {set(week_mentions)}")
                
            except Exception as e:
                print(f"  Error reading file: {e}")
        
        return {
            'week_14_files': week_14_files,
            'resolution_applied': False,
            'recommendation': 'Use --fix-week-13-14 flag to apply automatic resolution'
        }
    
    def show_resolution_preview(self) -> Dict[str, Any]:
        """Show what changes would be made without applying them"""
        print("\n" + "="*50)
        print("RESOLUTION PREVIEW (NO CHANGES MADE)")
        print("="*50)
        
        resolver = WeekResolver(config_path=self.config_path)
        
        # Discover files
        file_detections = {}
        for search_path in self.search_paths:
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
                        
                        detections = resolver.detect_week_numbers(file_path, content)
                        if detections:
                            file_detections[file_path] = detections
        
        # Get discrepancies and resolutions
        discrepancies = resolver.identify_discrepancies(file_detections)
        mappings = resolver.resolve_week_numbers(file_detections, discrepancies)
        
        # Show what would change
        changes = []
        no_changes = []
        
        for mapping in mappings:
            if mapping.original_week != mapping.resolved_week:
                changes.append({
                    'file': os.path.basename(mapping.file_path),
                    'original_week': mapping.original_week,
                    'resolved_week': mapping.resolved_week,
                    'method': mapping.resolution_method,
                    'confidence': mapping.confidence
                })
            else:
                no_changes.append(os.path.basename(mapping.file_path))
        
        print(f"Files that would be changed: {len(changes)}")
        for change in changes:
            print(f"  {change['file']}: week {change['original_week']} → week {change['resolved_week']} ({change['method']})")
        
        print(f"\nFiles that would remain unchanged: {len(no_changes)}")
        if len(no_changes) <= 10:
            for file in no_changes:
                print(f"  {file}")
        else:
            print(f"  (showing first 10 of {len(no_changes)})")
            for file in no_changes[:10]:
                print(f"  {file}")
            print(f"  ... and {len(no_changes) - 10} more")
        
        return {
            'total_files': len(mappings),
            'files_to_change': len(changes),
            'files_unchanged': len(no_changes),
            'changes': changes
        }
    
    def run_demo(self) -> Dict[str, Any]:
        """Run the complete demonstration"""
        print("WEEK RESOLVER DEMONSTRATION")
        print("="*50)
        print(f"Project root: {self.project_root}")
        print(f"Search paths:")
        for path in self.search_paths:
            exists = os.path.exists(path)
            print(f"  {'✓' if exists else '✗'} {os.path.relpath(path, self.project_root)}")
        
        results = {}
        
        # Step 1: Analyze current state
        results['current_state'] = self.analyze_current_state()
        
        # Step 2: Demonstrate week 13/14 specific issue
        results['week_13_14_demo'] = self.demonstrate_week_13_14_fix()
        
        # Step 3: Show resolution preview
        results['resolution_preview'] = self.show_resolution_preview()
        
        # Summary
        print("\n" + "="*50)
        print("DEMONSTRATION SUMMARY")
        print("="*50)
        
        current = results['current_state']
        preview = results['resolution_preview']
        
        print(f"Total files with week information: {current['total_files_analyzed']}")
        print(f"Discrepancies found: {len(current['discrepancies'])}")
        print(f"Missing weeks: {current['missing_weeks']}")
        print(f"Files that would be changed: {preview['files_to_change']}")
        
        if current['discrepancies']:
            print(f"\nTo resolve issues, run:")
            print(f"  python tools/week_resolver_tool.py \\")
            print(f"    --search-paths {' '.join(self.search_paths)} \\")
            print(f"    --output-path output/resolved_weeks")
            print(f"\nOr for week 13/14 specific fix:")
            print(f"  python tools/week_resolver_tool.py \\")
            print(f"    --fix-week-13-14 \\")
            print(f"    --search-paths {' '.join(self.search_paths)} \\")
            print(f"    --output-path output/resolved_weeks")
        else:
            print(f"\nNo critical issues found. Week numbering appears consistent.")
        
        return results


def main():
    """Main demonstration script"""
    # Get project root
    script_path = Path(__file__).parent.parent
    project_root = str(script_path.resolve())
    
    print(f"Academic Agent Week Resolver Demonstration")
    print(f"Project root: {project_root}")
    
    # Run demonstration
    demo = WeekResolverDemo(project_root)
    results = demo.run_demo()
    
    # Save results
    results_file = os.path.join(project_root, "week_resolver_demo_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDemo results saved to: {os.path.relpath(results_file, project_root)}")


if __name__ == "__main__":
    main()