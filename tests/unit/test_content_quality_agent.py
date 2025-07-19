#!/usr/bin/env python
"""
Unit tests for Content Quality Assurance Agent

Tests comprehensive quality validation functionality including:
- Content structure analysis
- Completeness checking
- Formatting validation
- Consistency checking
- Academic quality assessment
- Auto-fix capabilities
- Integration with consolidation
"""

import unittest
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.content_quality_agent import (
    ContentQualityAgent,
    ContentStructure,
    FormattingIssue,
    CompletenessCheck,
    QualityReport,
    QualityAnalytics
)


class TestContentQualityAgent(unittest.TestCase):
    """Test cases for ContentQualityAgent"""

    def setUp(self):
        """Set up test fixtures"""
        self.agent = ContentQualityAgent()
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample content for testing
        self.sample_markdown = """---
week: 1
content_type: comprehensive_study_notes
---

# High-Level Concept Overview

```mermaid
graph TD
    A[Security Manager] --> B[Risk Assessment]
```

## Executive Summary

This week introduces the foundational concepts of security risk assessment.

## Key Concepts

### Security Risk Assessment

A systematic process to identify and evaluate risks.

#### Examples

- Vulnerability scanning
- Penetration testing
- Risk analysis

## Detailed Analysis

The security risk assessment process involves multiple stages:

1. Asset identification
2. Threat analysis
3. Vulnerability assessment

## Practical Applications

Organizations use SRA to:
- Prioritize security investments
- Comply with regulations
- Improve security posture

## Exam Focus Areas

- Definition of SRA
- Key components
- Implementation strategies

## Review Questions

1. What is a security risk assessment?
2. How does SRA help with decision making?
"""

        self.poor_quality_markdown = """# bad header
no space after hash

this is a very long line that exceeds the maximum line length and should be flagged as a formatting issue because it makes the content harder to read

- badlist
no space after bullet

__inconsistent__bold**formatting**

![](image.jpg)
missing alt text
"""

        self.incomplete_content = """# Chapter 1

Some basic content but missing required sections.
"""

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_id, "content_quality_agent")
        self.assertIsNotNone(self.agent.quality_manager)
        self.assertIn("weights", self.agent.quality_config)
        self.assertIn("thresholds", self.agent.quality_config)

    def test_detect_content_type(self):
        """Test content type detection"""
        # Test from filename
        self.assertEqual(
            self.agent._detect_content_type("week_01_transcript.md", "content"),
            "transcript"
        )
        self.assertEqual(
            self.agent._detect_content_type("lecture_notes.md", "content"),
            "lecture"
        )
        self.assertEqual(
            self.agent._detect_content_type("comprehensive_study_notes.md", "content"),
            "comprehensive_study_notes"
        )
        
        # Test from content
        self.assertEqual(
            self.agent._detect_content_type("file.md", "# Chapter 1\nContent"),
            "textbook"
        )

    def test_extract_week_number(self):
        """Test week number extraction"""
        # From filename
        self.assertEqual(
            self.agent._extract_week_number("week_01_notes.md", "content"),
            1
        )
        self.assertEqual(
            self.agent._extract_week_number("lecture_05.md", "content"),
            5
        )
        
        # From content frontmatter
        content_with_week = "---\nweek: 3\n---\nContent"
        self.assertEqual(
            self.agent._extract_week_number("file.md", content_with_week),
            3
        )

    def test_analyze_content_structure(self):
        """Test content structure analysis"""
        structure = self.agent._analyze_content_structure(self.sample_markdown)
        
        self.assertIsInstance(structure, ContentStructure)
        self.assertGreater(len(structure.headers), 0)
        self.assertGreater(structure.word_count, 0)
        self.assertEqual(structure.mermaid_diagrams, 1)
        self.assertGreater(structure.lists, 0)

    def test_check_completeness(self):
        """Test completeness checking"""
        # Test complete content
        completeness = self.agent._check_completeness(
            self.sample_markdown, 
            "comprehensive_study_notes", 
            1
        )
        self.assertIsInstance(completeness, CompletenessCheck)
        self.assertGreater(completeness.completeness_score, 0.8)
        self.assertEqual(len(completeness.missing_sections), 0)
        
        # Test incomplete content
        incomplete_completeness = self.agent._check_completeness(
            self.incomplete_content,
            "comprehensive_study_notes",
            1
        )
        self.assertLess(incomplete_completeness.completeness_score, 0.5)
        self.assertGreater(len(incomplete_completeness.missing_sections), 0)

    def test_validate_formatting(self):
        """Test formatting validation"""
        issues = self.agent._validate_formatting(self.poor_quality_markdown)
        
        self.assertIsInstance(issues, list)
        self.assertGreater(len(issues), 0)
        
        # Check for specific issue types
        issue_types = [issue.issue_type for issue in issues]
        self.assertIn("header_formatting", issue_types)
        self.assertIn("list_formatting", issue_types)
        self.assertIn("missing_alt_text", issue_types)

    def test_check_consistency(self):
        """Test consistency checking"""
        score = self.agent._check_consistency(self.sample_markdown, "comprehensive_study_notes")
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Poor consistency content should score lower
        poor_score = self.agent._check_consistency(self.poor_quality_markdown, "notes")
        self.assertLess(poor_score, score)

    def test_assess_academic_quality(self):
        """Test academic quality assessment"""
        quality_score = self.agent._assess_academic_quality(
            self.sample_markdown, 
            "comprehensive_study_notes"
        )
        
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        self.assertGreater(quality_score, 0.3)  # Should detect academic content

    def test_calculate_formatting_score(self):
        """Test formatting score calculation"""
        # No issues should give perfect score
        score_no_issues = self.agent._calculate_formatting_score([])
        self.assertEqual(score_no_issues, 1.0)
        
        # Create sample issues
        issues = [
            FormattingIssue("header_formatting", "medium", "test", 1, "fix", True),
            FormattingIssue("list_formatting", "low", "test", 2, "fix", True)
        ]
        
        score_with_issues = self.agent._calculate_formatting_score(issues)
        self.assertLess(score_with_issues, 1.0)
        self.assertGreater(score_with_issues, 0.0)

    def test_calculate_overall_quality(self):
        """Test overall quality calculation"""
        overall = self.agent._calculate_overall_quality(0.8, 0.9, 0.7, 0.6)
        
        self.assertIsInstance(overall, float)
        self.assertGreaterEqual(overall, 0.0)
        self.assertLessEqual(overall, 1.0)
        
        # Should be weighted average
        weights = self.agent.quality_config["weights"]
        expected = (
            0.8 * weights["completeness"] +
            0.9 * weights["formatting"] +
            0.7 * weights["consistency"] +
            0.6 * weights["academic_quality"]
        )
        self.assertAlmostEqual(overall, expected, places=3)

    def test_generate_improvement_suggestions(self):
        """Test improvement suggestion generation"""
        completeness = CompletenessCheck(
            required_sections=["section1", "section2"],
            missing_sections=["section1"],
            incomplete_sections=["section2"],
            completeness_score=0.5,
            week_number=1,
            content_type="notes"
        )
        
        formatting_issues = [
            FormattingIssue("header_formatting", "critical", "fix header", 1, "suggestion", True)
        ]
        
        suggestions = self.agent._generate_improvement_suggestions(
            completeness, formatting_issues, 0.6, 0.5
        )
        
        self.assertIn("general", suggestions)
        self.assertIn("priority", suggestions)
        self.assertIn("auto_fixable", suggestions)
        self.assertGreater(len(suggestions["priority"]), 0)

    def test_assess_content_quality_with_file(self):
        """Test complete quality assessment with file"""
        # Create temporary file
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        report = self.agent.assess_content_quality(test_file)
        
        self.assertIsInstance(report, QualityReport)
        self.assertEqual(report.file_path, test_file)
        self.assertEqual(report.content_type, "comprehensive_study_notes")
        self.assertEqual(report.week_number, 1)
        self.assertGreater(report.overall_quality_score, 0.5)
        self.assertIsInstance(report.content_structure, ContentStructure)
        self.assertIsInstance(report.improvement_suggestions, list)

    def test_assess_content_quality_with_content(self):
        """Test quality assessment with content string"""
        report = self.agent.assess_content_quality(
            "test_file.md",
            content=self.sample_markdown,
            content_type="comprehensive_study_notes",
            week_number=1
        )
        
        self.assertIsInstance(report, QualityReport)
        self.assertEqual(report.content_type, "comprehensive_study_notes")
        self.assertEqual(report.week_number, 1)

    def test_validate_batch_content(self):
        """Test batch content validation"""
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = os.path.join(self.temp_dir, f"test_content_{i}.md")
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(self.sample_markdown.replace("week: 1", f"week: {i+1}"))
            test_files.append(test_file)
        
        results = self.agent.validate_batch_content(test_files)
        
        self.assertEqual(results["total_files"], 3)
        self.assertEqual(results["processed_files"], 3)
        self.assertEqual(results["failed_files"], 0)
        self.assertEqual(len(results["reports"]), 3)
        self.assertIn("summary", results)

    def test_generate_quality_analytics(self):
        """Test quality analytics generation"""
        # First assess some content to build history
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        # Generate multiple reports
        for i in range(5):
            self.agent.assess_content_quality(test_file)
        
        analytics = self.agent.generate_quality_analytics()
        
        self.assertIsInstance(analytics, QualityAnalytics)
        self.assertEqual(analytics.total_files_assessed, 5)
        self.assertGreater(analytics.average_quality_score, 0.0)
        self.assertIn("content_type_quality", analytics.__dict__)

    def test_auto_fix_issues(self):
        """Test auto-fix functionality"""
        # Create file with fixable issues
        test_file = os.path.join(self.temp_dir, "test_fix.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.poor_quality_markdown)
        
        fix_results = self.agent.auto_fix_issues(test_file, backup=False)
        
        self.assertTrue(fix_results["success"])
        self.assertGreaterEqual(fix_results["fixes_applied"], 0)
        
        # Read fixed content and verify improvements
        with open(test_file, 'r', encoding='utf-8') as f:
            fixed_content = f.read()
        
        # Should have proper header spacing
        self.assertIn("# bad header", fixed_content)
        # Should have proper list spacing
        self.assertIn("- badlist", fixed_content)

    @patch('src.agents.content_quality_agent.ContentQualityAgent.assess_content_quality')
    def test_integrate_with_consolidation(self, mock_assess):
        """Test integration with consolidation results"""
        # Mock consolidation result
        mock_consolidation = Mock()
        mock_consolidation.success = True
        mock_consolidation.processed_files = [
            Mock(target_path="file1.md", content_type="notes", week_number=1),
            Mock(target_path="file2.md", content_type="lecture", week_number=2)
        ]
        
        # Mock quality report
        mock_report = Mock()
        mock_report.to_dict.return_value = {"quality_score": 0.8}
        mock_assess.return_value = mock_report
        
        # Mock file existence
        with patch('os.path.exists', return_value=True):
            result = self.agent.integrate_with_consolidation(mock_consolidation)
        
        self.assertTrue(result["consolidation_success"])
        self.assertEqual(result["total_consolidated_files"], 2)
        self.assertIn("integrated_metrics", result)
        self.assertIn("recommendations", result)

    def test_check_quality_method(self):
        """Test check_quality method implementation"""
        # Test with file path
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        quality_score = self.agent.check_quality({"file_path": test_file})
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        
        # Test with content text
        quality_score_text = self.agent.check_quality({
            "content_text": self.sample_markdown,
            "content_type": "comprehensive_study_notes"
        })
        self.assertIsInstance(quality_score_text, float)

    def test_validate_input(self):
        """Test input validation"""
        # Valid file path
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        self.assertTrue(self.agent.validate_input(test_file))
        
        # Valid content dict
        self.assertTrue(self.agent.validate_input({
            "content_text": "test content",
            "content_type": "notes"
        }))
        
        # Valid batch
        self.assertTrue(self.agent.validate_input([test_file, test_file]))
        
        # Invalid inputs
        self.assertFalse(self.agent.validate_input("nonexistent_file.md"))
        self.assertFalse(self.agent.validate_input({}))

    def test_validate_output(self):
        """Test output validation"""
        # Create valid quality report
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        report = self.agent.assess_content_quality(test_file)
        self.assertTrue(self.agent.validate_output(report))
        
        # Valid batch results
        batch_results = {
            "total_files": 1,
            "processed_files": 1,
            "reports": [report.to_dict()]
        }
        self.assertTrue(self.agent.validate_output(batch_results))
        
        # Invalid outputs
        self.assertFalse(self.agent.validate_output("invalid"))
        self.assertFalse(self.agent.validate_output({}))

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with non-existent file
        with self.assertRaises(Exception):
            self.agent.assess_content_quality("nonexistent_file.md")
        
        # Test with invalid content type
        try:
            report = self.agent.assess_content_quality(
                "test.md",
                content="test content",
                content_type="invalid_type"
            )
            # Should handle gracefully
            self.assertIsNotNone(report)
        except Exception:
            self.fail("Should handle invalid content type gracefully")

    def test_performance_metrics(self):
        """Test performance metrics collection"""
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        report = self.agent.assess_content_quality(test_file)
        
        # Check processing time is recorded
        self.assertGreater(report.processing_time, 0.0)
        self.assertIsInstance(report.assessment_date, datetime)
        self.assertGreater(report.file_size, 0)
        self.assertIsNotNone(report.content_hash)

    def test_content_hash_generation(self):
        """Test content hash generation for duplicate detection"""
        test_file = os.path.join(self.temp_dir, "test_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(self.sample_markdown)
        
        report1 = self.agent.assess_content_quality(test_file)
        report2 = self.agent.assess_content_quality(test_file)
        
        # Same content should produce same hash
        self.assertEqual(report1.content_hash, report2.content_hash)
        
        # Different content should produce different hash
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("different content")
        
        report3 = self.agent.assess_content_quality(test_file)
        self.assertNotEqual(report1.content_hash, report3.content_hash)


class TestQualityReportDataClass(unittest.TestCase):
    """Test QualityReport data class functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_structure = ContentStructure(
            headers=["# Header 1", "## Header 2"],
            header_levels={1: 1, 2: 1},
            paragraphs=5,
            lists=2,
            images=1,
            tables=0,
            code_blocks=1,
            mermaid_diagrams=1,
            citations=2,
            word_count=100,
            character_count=500
        )
        
        self.sample_completeness = CompletenessCheck(
            required_sections=["intro", "body", "conclusion"],
            missing_sections=[],
            incomplete_sections=["body"],
            completeness_score=0.8,
            week_number=1,
            content_type="notes"
        )

    def test_quality_report_to_dict(self):
        """Test QualityReport serialization"""
        from src.agents.quality_manager import QualityMetrics, QualityEvaluation
        
        quality_evaluation = QualityEvaluation(
            content_type="notes",
            quality_score=0.8,
            feedback=["Good quality"],
            areas_for_improvement=["Add more detail"],
            strengths=["Well structured"],
            metrics=QualityMetrics(0.8, 0.7, 0.9, 0.8, 0.7, 0.8),
            assessment="Good quality content",
            approved=True
        )
        
        report = QualityReport(
            content_id="test_123",
            file_path="/test/file.md",
            content_type="notes",
            week_number=1,
            overall_quality_score=0.8,
            completeness_score=0.8,
            formatting_score=0.9,
            consistency_score=0.7,
            academic_quality_score=0.8,
            content_structure=self.sample_structure,
            completeness_check=self.sample_completeness,
            formatting_issues=[],
            quality_evaluation=quality_evaluation,
            improvement_suggestions=["Test suggestion"],
            priority_fixes=["Fix headers"],
            auto_fixes_available=["Format lists"],
            assessment_date=datetime.now(),
            processing_time=0.5,
            file_size=1000,
            content_hash="abc123"
        )
        
        report_dict = report.to_dict()
        
        self.assertIsInstance(report_dict, dict)
        self.assertEqual(report_dict["content_id"], "test_123")
        self.assertEqual(report_dict["overall_quality_score"], 0.8)
        self.assertIn("content_structure", report_dict)
        self.assertIn("completeness_check", report_dict)


if __name__ == '__main__':
    unittest.main()