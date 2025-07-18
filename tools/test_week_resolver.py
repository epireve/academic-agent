#!/usr/bin/env python
"""
Week Resolver Test Suite

This script tests the week resolver functionality to ensure it correctly
identifies and resolves week numbering discrepancies.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add the unified src module to the path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.processors.week_resolver import WeekResolver, WeekDetection, WeekMapping
except ImportError as e:
    print(f"Error importing WeekResolver: {e}")
    sys.exit(1)


class WeekResolverTester:
    """Test suite for week resolver functionality"""
    
    def __init__(self):
        self.test_data_dir = None
        self.resolver = None
        
    def setup_test_data(self) -> str:
        """Create test data structure with known week numbering issues"""
        # Create temporary directory
        self.test_data_dir = tempfile.mkdtemp(prefix="week_resolver_test_")
        
        # Create test file structure
        test_files = {
            "transcripts/week-6.md": self._create_test_content("week-6", "transcript", 6),
            "transcripts/week-8.md": self._create_test_content("week-8", "transcript", 8),
            "transcripts/week-9.md": self._create_test_content("week-9", "transcript", 9),
            "transcripts/week-10.md": self._create_test_content("week-10", "transcript", 10),
            "transcripts/week-11.md": self._create_test_content("week-11", "transcript", 11),
            "transcripts/week-12.md": self._create_test_content("week-12", "transcript", 12),
            "transcripts/week-14.md": self._create_test_content("week-14", "transcript", 14, actual_week=13),  # This should be week 13
            "lectures/lecture_1.md": self._create_test_content("lecture_1", "lecture", 1),
            "lectures/lecture_2.md": self._create_test_content("lecture_2", "lecture", 2),
            "lectures/lecture_3.md": self._create_test_content("lecture_3", "lecture", 3),
            "notes/week_01_notes.md": self._create_test_content("week_01_notes", "notes", 1),
            "notes/week_02_notes.md": self._create_test_content("week_02_notes", "notes", 2),
            "notes/w3_summary.md": self._create_test_content("w3_summary", "notes", 3),
            "textbook/chapter_1.md": self._create_test_content("chapter_1", "textbook", 1),
            "textbook/chapter_2.md": self._create_test_content("chapter_2", "textbook", 2),
            "mixed/thirteenth_week_content.md": self._create_test_content("thirteenth_week_content", "mixed", 13),
            "mixed/final_lecture.md": self._create_test_content("final_lecture", "mixed", 14, content_markers=["exam prep", "final review"]),
            "no_week_info/random_content.md": "# Random Content\n\nThis file has no week information.",
        }
        
        # Create directories and files
        for relative_path, content in test_files.items():
            full_path = os.path.join(self.test_data_dir, relative_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create metadata files
        metadata_files = {
            "transcripts/week-6_meta.json": {"week": 6, "type": "transcript"},
            "transcripts/week-14_meta.json": {"week": 14, "type": "transcript", "note": "Actually week 13 content"}
        }
        
        for relative_path, metadata in metadata_files.items():
            full_path = os.path.join(self.test_data_dir, relative_path)
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        return self.test_data_dir
    
    def _create_test_content(self, filename: str, content_type: str, week_number: int, 
                           actual_week: int = None, content_markers: List[str] = None) -> str:
        """Create test content with known week information"""
        actual = actual_week or week_number
        markers = content_markers or [f"week {actual} content", f"topic for week {actual}"]
        
        content = f"""---
week: {actual}
content_type: {content_type}
source_file: {filename}
---

# Week {actual} - {content_type.title()}

This is {content_type} content for week {actual}.

## Topics Covered
"""
        
        for marker in markers:
            content += f"- {marker}\n"
        
        content += f"""
## Content Details

Previous week: week {actual - 1 if actual > 1 else 'none'}
Current week: week {actual}
Next week: week {actual + 1 if actual < 15 else 'none'}

This content demonstrates week numbering patterns for testing the week resolver.
"""
        
        return content
    
    def cleanup_test_data(self):
        """Clean up test data directory"""
        if self.test_data_dir and os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    def test_week_detection(self) -> Dict[str, Any]:
        """Test week number detection from various sources"""
        print("Testing week number detection...")
        
        # Setup test data
        test_dir = self.setup_test_data()
        self.resolver = WeekResolver()
        
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "details": []
        }
        
        # Test cases: (file_path, expected_week, expected_confidence_min)
        test_cases = [
            ("transcripts/week-6.md", 6, 0.8),
            ("transcripts/week-14.md", 14, 0.8),  # Filename detection (will be resolved later)
            ("lectures/lecture_1.md", 1, 0.7),
            ("notes/week_01_notes.md", 1, 0.8),
            ("notes/w3_summary.md", 3, 0.7),
            ("textbook/chapter_1.md", 1, 0.5),
            ("mixed/thirteenth_week_content.md", 13, 0.6),
        ]
        
        for relative_path, expected_week, min_confidence in test_cases:
            results["total_tests"] += 1
            full_path = os.path.join(test_dir, relative_path)
            
            # Read content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Detect week numbers
            detections = self.resolver.detect_week_numbers(full_path, content)
            
            # Check results
            if detections and detections[0].week_number == expected_week and detections[0].confidence >= min_confidence:
                results["passed"] += 1
                status = "PASS"
            else:
                results["failed"] += 1
                status = "FAIL"
            
            results["details"].append({
                "file": relative_path,
                "expected_week": expected_week,
                "detected_week": detections[0].week_number if detections else None,
                "confidence": detections[0].confidence if detections else 0,
                "status": status
            })
        
        return results
    
    def test_discrepancy_detection(self) -> Dict[str, Any]:
        """Test identification of week numbering discrepancies"""
        print("Testing discrepancy detection...")
        
        if not self.test_data_dir:
            self.setup_test_data()
        if not self.resolver:
            self.resolver = WeekResolver()
        
        # Scan all test files
        file_detections = {}
        for root, dirs, files in os.walk(self.test_data_dir):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    detections = self.resolver.detect_week_numbers(file_path, content)
                    if detections:
                        file_detections[file_path] = detections
        
        # Identify discrepancies
        discrepancies = self.resolver.identify_discrepancies(file_detections)
        
        # Expected discrepancies:
        # 1. Missing weeks (4, 5, 7, 13)
        # 2. Suspicious gap (missing 13 but has 14)
        
        expected_discrepancy_types = {'missing_weeks', 'suspicious_gap'}
        found_types = {disc['type'] for disc in discrepancies}
        
        results = {
            "expected_discrepancies": expected_discrepancy_types,
            "found_discrepancies": found_types,
            "discrepancy_details": discrepancies,
            "test_passed": expected_discrepancy_types.issubset(found_types)
        }
        
        return results
    
    def test_week_resolution(self) -> Dict[str, Any]:
        """Test week number resolution logic"""
        print("Testing week number resolution...")
        
        if not self.test_data_dir:
            self.setup_test_data()
        
        # Create resolver with test configuration
        config = {
            "manual_overrides": {
                "week-14.md": 13  # Override week-14.md to be week 13
            },
            "confidence_thresholds": {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            },
            "academic_calendar": {
                "valid_range": [1, 15],
                "common_gaps": [13]
            }
        }
        
        # Create temporary config file
        config_file = os.path.join(self.test_data_dir, "test_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        self.resolver = WeekResolver(config_path=config_file)
        
        # Run resolution workflow
        output_dir = os.path.join(self.test_data_dir, "resolved")
        result = self.resolver.resolve_academic_content([self.test_data_dir], output_dir)
        
        # Check if week-14.md was correctly resolved to week 13
        week_13_transcript = None
        for mapping in result.mappings:
            if "week-14.md" in mapping.file_path and mapping.content_type == "transcript":
                week_13_transcript = mapping
                break
        
        results = {
            "resolution_success": result.success,
            "total_mappings": len(result.mappings),
            "files_with_changes": len([m for m in result.mappings if m.original_week != m.resolved_week]),
            "week_14_to_13_resolved": (
                week_13_transcript is not None and 
                week_13_transcript.original_week == 14 and 
                week_13_transcript.resolved_week == 13
            ),
            "discrepancies_found": len(result.discrepancies),
            "errors": len(result.errors),
            "output_files_created": len(os.listdir(output_dir)) if os.path.exists(output_dir) else 0
        }
        
        return results
    
    def test_content_analysis(self) -> Dict[str, Any]:
        """Test content-based week inference"""
        print("Testing content analysis...")
        
        # Create test content with sequential references
        test_content = """---
week: 10
---

# Week 10 Lecture

Today we continue from last week (week 9) and will cover:
- Risk mitigation strategies
- Following up on week 9's risk analysis

Next week (week 11) we will discuss reporting.
"""
        
        # Create temporary file
        test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        test_file.write(test_content)
        test_file.close()
        
        try:
            resolver = WeekResolver()
            detections = resolver.detect_week_numbers(test_file.name, test_content)
            
            # Should detect week 10 from frontmatter
            frontmatter_detection = next((d for d in detections if d.source == "metadata"), None)
            
            results = {
                "detections_found": len(detections),
                "frontmatter_detected": frontmatter_detection is not None,
                "frontmatter_week": frontmatter_detection.week_number if frontmatter_detection else None,
                "content_analysis_working": len(detections) > 0
            }
            
        finally:
            os.unlink(test_file.name)
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print("="*60)
        print("WEEK RESOLVER TEST SUITE")
        print("="*60)
        
        all_results = {}
        
        try:
            # Test 1: Week Detection
            all_results["week_detection"] = self.test_week_detection()
            
            # Test 2: Discrepancy Detection
            all_results["discrepancy_detection"] = self.test_discrepancy_detection()
            
            # Test 3: Week Resolution
            all_results["week_resolution"] = self.test_week_resolution()
            
            # Test 4: Content Analysis
            all_results["content_analysis"] = self.test_content_analysis()
            
            # Calculate overall results
            total_tests = 4
            passed_tests = sum([
                all_results["week_detection"]["passed"] == all_results["week_detection"]["total_tests"],
                all_results["discrepancy_detection"]["test_passed"],
                all_results["week_resolution"]["week_14_to_13_resolved"],
                all_results["content_analysis"]["content_analysis_working"]
            ])
            
            all_results["summary"] = {
                "total_test_categories": total_tests,
                "passed_categories": passed_tests,
                "overall_success": passed_tests == total_tests,
                "test_data_location": self.test_data_dir
            }
            
        finally:
            # Cleanup
            # self.cleanup_test_data()  # Comment out for debugging
            pass
        
        return all_results
    
    def print_test_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        print("\nTEST RESULTS:")
        print("-" * 40)
        
        # Week Detection Results
        wd = results["week_detection"]
        print(f"Week Detection: {wd['passed']}/{wd['total_tests']} passed")
        for detail in wd["details"]:
            status_symbol = "✓" if detail["status"] == "PASS" else "✗"
            print(f"  {status_symbol} {detail['file']}: expected week {detail['expected_week']}, got {detail['detected_week']}")
        
        # Discrepancy Detection
        dd = results["discrepancy_detection"]
        print(f"\nDiscrepancy Detection: {'PASS' if dd['test_passed'] else 'FAIL'}")
        print(f"  Expected: {dd['expected_discrepancies']}")
        print(f"  Found: {dd['found_discrepancies']}")
        
        # Week Resolution
        wr = results["week_resolution"]
        print(f"\nWeek Resolution: {'PASS' if wr['week_14_to_13_resolved'] else 'FAIL'}")
        print(f"  Files processed: {wr['total_mappings']}")
        print(f"  Files changed: {wr['files_with_changes']}")
        print(f"  Week 14→13 resolved: {wr['week_14_to_13_resolved']}")
        
        # Content Analysis
        ca = results["content_analysis"]
        print(f"\nContent Analysis: {'PASS' if ca['content_analysis_working'] else 'FAIL'}")
        print(f"  Detections found: {ca['detections_found']}")
        print(f"  Frontmatter detected: {ca['frontmatter_detected']}")
        
        # Overall Summary
        summary = results["summary"]
        print(f"\nOVERALL RESULT: {'PASS' if summary['overall_success'] else 'FAIL'}")
        print(f"Categories passed: {summary['passed_categories']}/{summary['total_test_categories']}")
        
        if not summary['overall_success']:
            print("\nSome tests failed. Check implementation for issues.")
        else:
            print("\nAll tests passed! Week resolver is working correctly.")


def main():
    """Run the test suite"""
    tester = WeekResolverTester()
    
    try:
        results = tester.run_all_tests()
        tester.print_test_results(results)
        
        # Save results to file
        results_file = "week_resolver_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Return appropriate exit code
        return 0 if results["summary"]["overall_success"] else 1
        
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())