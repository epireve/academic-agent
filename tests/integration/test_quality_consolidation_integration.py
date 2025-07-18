#!/usr/bin/env python
"""
Integration tests for Content Quality Assurance and Consolidation

Tests the integration between ContentQualityAgent and ContentConsolidationAgent
to ensure quality validation works seamlessly with content consolidation workflows.
"""

import unittest
import tempfile
import os
import json
import shutil
from datetime import datetime
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.academic.content_quality_agent import ContentQualityAgent
from agents.academic.consolidation_agent import ContentConsolidationAgent, ConsolidationResult, FileMapping


class TestQualityConsolidationIntegration(unittest.TestCase):
    """Integration tests for quality assurance and consolidation"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.quality_agent = ContentQualityAgent()
        self.consolidation_agent = ContentConsolidationAgent()
        
        # Create test directory structure
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sample content for different quality levels
        self.high_quality_content = """---
week: 1
content_type: comprehensive_study_notes
---

# High-Level Concept Overview

```mermaid
graph TD
    A[Security Manager] --> B[Risk Assessment]
    B --> C[Risk Mitigation]
```

## Executive Summary

This comprehensive study guide covers the fundamental concepts of security risk assessment,
providing a systematic approach to identifying, analyzing, and mitigating security risks
in organizational environments.

## Key Concepts

### Security Risk Assessment (SRA)

A Security Risk Assessment is a systematic evaluation of potential security threats and
vulnerabilities that could impact an organization's information assets.

#### Definition and Purpose

The primary purpose of an SRA is to:
- Identify potential security risks
- Assess the likelihood and impact of threats
- Determine appropriate risk mitigation strategies
- Support informed decision-making

### Risk Components

#### Assets
Information, systems, and resources that have value to the organization.

#### Threats
Potential causes of incidents that could harm assets.

#### Vulnerabilities
Weaknesses that could be exploited by threats.

## Detailed Analysis

### Risk Assessment Methodologies

1. **Qualitative Assessment**
   - Uses descriptive scales (High, Medium, Low)
   - Easier to understand and communicate
   - Less precise than quantitative methods

2. **Quantitative Assessment**
   - Uses numerical values and calculations
   - More precise and objective
   - Requires more data and resources

3. **Hybrid Approach**
   - Combines qualitative and quantitative methods
   - Balances precision with practicality
   - Most commonly used in practice

## Practical Applications

### Industry Implementation

Organizations across various industries implement SRA to:
- Meet regulatory compliance requirements
- Improve overall security posture
- Optimize security investments
- Reduce operational risks

### Case Studies

**Financial Services**: Banks use SRA to protect customer data and comply with regulations.
**Healthcare**: Hospitals conduct SRA to secure patient information under HIPAA.
**Manufacturing**: Companies assess risks to intellectual property and operational systems.

## Exam Focus Areas

### Key Topics for Assessment

1. **SRA Fundamentals**
   - Definition and importance
   - Components of risk
   - Assessment methodologies

2. **Implementation Process**
   - Planning and scoping
   - Data gathering techniques
   - Risk analysis methods

3. **Risk Treatment**
   - Risk acceptance strategies
   - Mitigation techniques
   - Transfer mechanisms

## Review Questions

1. What are the three main components of security risk?
2. How do qualitative and quantitative risk assessments differ?
3. What factors should be considered when scoping an SRA?
4. Describe the relationship between threats, vulnerabilities, and risks.
5. What are the advantages and disadvantages of different risk assessment methodologies?

### Practice Scenarios

**Scenario 1**: A company wants to implement cloud storage. What security risks should be assessed?
**Scenario 2**: An organization experiences a data breach. How would SRA help prevent future incidents?
"""

        self.medium_quality_content = """# Week 2 Notes

## Security Risk Analysis

Risk analysis is important for security.

### Types of Analysis

- Qualitative
- Quantitative

### Process

1. Identify assets
2. Find threats
3. Assess vulnerabilities

## Examples

Banks do risk assessment.
Hospitals need security too.

## Summary

Risk analysis helps organizations.
"""

        self.poor_quality_content = """#badheader
no space

this is a very long line that exceeds reasonable limits and should be considered poor formatting that impacts readability and maintenance
-badlist
no space

missing sections
"""

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def create_test_files(self):
        """Create test files with varying quality levels"""
        test_files = []
        
        # High quality file
        high_quality_file = os.path.join(self.input_dir, "week_01_comprehensive_study_notes.md")
        with open(high_quality_file, 'w', encoding='utf-8') as f:
            f.write(self.high_quality_content)
        test_files.append(high_quality_file)
        
        # Medium quality file
        medium_quality_file = os.path.join(self.input_dir, "week_02_notes.md")
        with open(medium_quality_file, 'w', encoding='utf-8') as f:
            f.write(self.medium_quality_content)
        test_files.append(medium_quality_file)
        
        # Poor quality file
        poor_quality_file = os.path.join(self.input_dir, "week_03_notes.md")
        with open(poor_quality_file, 'w', encoding='utf-8') as f:
            f.write(self.poor_quality_content)
        test_files.append(poor_quality_file)
        
        return test_files

    def create_mock_consolidation_result(self, test_files):
        """Create mock consolidation result"""
        processed_files = []
        
        for i, file_path in enumerate(test_files):
            # Copy files to output directory for testing
            target_filename = f"week_{i+1:02d}_standardized.md"
            target_path = os.path.join(self.output_dir, target_filename)
            shutil.copy2(file_path, target_path)
            
            file_mapping = FileMapping(
                source_path=file_path,
                target_path=target_path,
                confidence=0.8,
                week_number=i+1,
                content_type="notes" if i > 0 else "comprehensive_study_notes",
                metadata={"original_file": os.path.basename(file_path)}
            )
            processed_files.append(file_mapping)
        
        return ConsolidationResult(
            success=True,
            processed_files=processed_files,
            skipped_files=[],
            errors=[],
            consolidation_report={
                "total_files_processed": len(processed_files),
                "consolidation_date": datetime.now().isoformat()
            },
            unified_structure={
                "notes": self.output_dir,
                "comprehensive_study_notes": self.output_dir
            }
        )

    def test_post_consolidation_quality_assessment(self):
        """Test quality assessment after consolidation"""
        # Create test files
        test_files = self.create_test_files()
        
        # Create mock consolidation result
        consolidation_result = self.create_mock_consolidation_result(test_files)
        
        # Perform quality assessment on consolidated content
        integration_result = self.quality_agent.integrate_with_consolidation(consolidation_result)
        
        # Verify integration results
        self.assertTrue(integration_result["consolidation_success"])
        self.assertEqual(integration_result["total_consolidated_files"], 3)
        self.assertEqual(integration_result["quality_assessed_files"], 3)
        self.assertIn("integrated_metrics", integration_result)
        self.assertIn("recommendations", integration_result)
        
        # Check quality reports
        quality_reports = integration_result["quality_reports"]
        self.assertEqual(len(quality_reports), 3)
        
        # Verify quality scores vary appropriately
        scores = [report["overall_quality_score"] for report in quality_reports]
        self.assertEqual(len(scores), 3)
        
        # High quality content should score highest
        high_quality_score = scores[0]  # First file is high quality
        self.assertGreater(high_quality_score, 0.7)

    def test_quality_metrics_integration(self):
        """Test integration of quality metrics with consolidation metrics"""
        test_files = self.create_test_files()
        consolidation_result = self.create_mock_consolidation_result(test_files)
        
        integration_result = self.quality_agent.integrate_with_consolidation(consolidation_result)
        
        # Check integrated metrics
        metrics = integration_result["integrated_metrics"]
        
        self.assertIn("overall_quality_score", metrics)
        self.assertIn("overall_completeness_score", metrics)
        self.assertIn("content_type_distribution", metrics)
        self.assertIn("content_type_quality_scores", metrics)
        self.assertIn("consolidation_efficiency", metrics)
        self.assertIn("quality_pass_rate", metrics)
        
        # Verify metric values are reasonable
        self.assertGreaterEqual(metrics["overall_quality_score"], 0.0)
        self.assertLessEqual(metrics["overall_quality_score"], 1.0)
        self.assertEqual(metrics["consolidation_efficiency"], 1.0)  # All files assessed

    def test_quality_recommendations_generation(self):
        """Test generation of quality-based recommendations"""
        test_files = self.create_test_files()
        consolidation_result = self.create_mock_consolidation_result(test_files)
        
        integration_result = self.quality_agent.integrate_with_consolidation(consolidation_result)
        
        # Check recommendations
        recommendations = integration_result["recommendations"]
        self.assertIsInstance(recommendations, list)
        
        # Should have recommendations for quality improvements
        quality_recommendations = [r for r in recommendations if "quality" in r.lower()]
        self.assertGreater(len(quality_recommendations), 0)

    def test_batch_quality_validation_workflow(self):
        """Test complete batch quality validation workflow"""
        # Create test files
        test_files = self.create_test_files()
        
        # Perform batch quality assessment
        batch_results = self.quality_agent.validate_batch_content(
            test_files, 
            output_dir=self.output_dir
        )
        
        # Verify batch results
        self.assertEqual(batch_results["total_files"], 3)
        self.assertEqual(batch_results["processed_files"], 3)
        self.assertEqual(batch_results["failed_files"], 0)
        
        # Check summary statistics
        summary = batch_results["summary"]
        self.assertIn("average_quality", summary)
        self.assertIn("pass_rate", summary)
        self.assertIn("content_types", summary)
        
        # Verify quality distribution
        scores = [report["overall_quality_score"] for report in batch_results["reports"]]
        self.assertEqual(len(scores), 3)
        
        # High quality file should score well
        self.assertTrue(any(score > 0.8 for score in scores))

    def test_auto_fix_integration(self):
        """Test auto-fix functionality with consolidation workflow"""
        # Create file with fixable issues
        fixable_content = """#badheader
content with formatting issues

-badlist
no space

__inconsistent__bold**formatting**
"""
        
        test_file = os.path.join(self.input_dir, "fixable_content.md")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(fixable_content)
        
        # Assess quality before fix
        initial_report = self.quality_agent.assess_content_quality(test_file)
        initial_score = initial_report.overall_quality_score
        
        # Apply auto-fixes
        fix_results = self.quality_agent.auto_fix_issues(test_file, backup=True)
        
        # Assess quality after fix
        final_report = self.quality_agent.assess_content_quality(test_file)
        final_score = final_report.overall_quality_score
        
        # Verify improvements
        self.assertTrue(fix_results["success"])
        self.assertGreater(fix_results["fixes_applied"], 0)
        self.assertGreater(final_score, initial_score)

    def test_quality_analytics_generation(self):
        """Test generation of quality analytics across multiple assessments"""
        test_files = self.create_test_files()
        
        # Perform multiple assessments to build history
        for file_path in test_files:
            self.quality_agent.assess_content_quality(file_path)
        
        # Generate analytics
        analytics = self.quality_agent.generate_quality_analytics()
        
        # Verify analytics structure
        self.assertEqual(analytics.total_files_assessed, 3)
        self.assertGreater(analytics.average_quality_score, 0.0)
        self.assertIn("content_type_quality", analytics.__dict__)
        self.assertIn("common_issues", analytics.__dict__)
        self.assertGreater(len(analytics.improvement_opportunities), 0)

    def test_quality_gate_enforcement(self):
        """Test quality gate enforcement in consolidation workflow"""
        test_files = self.create_test_files()
        
        # Set strict quality threshold
        original_threshold = self.quality_agent.quality_threshold
        self.quality_agent.quality_threshold = 0.9  # Very high threshold
        
        try:
            # Assess files with high threshold
            reports = []
            for file_path in test_files:
                report = self.quality_agent.assess_content_quality(file_path)
                reports.append(report)
            
            # Check which files pass the quality gate
            passing_files = [r for r in reports if r.overall_quality_score >= 0.9]
            failing_files = [r for r in reports if r.overall_quality_score < 0.9]
            
            # Should have some files that don't meet high threshold
            self.assertGreater(len(failing_files), 0)
            
            # High quality content should still pass
            self.assertTrue(any(r.content_type == "comprehensive_study_notes" for r in passing_files))
            
        finally:
            # Restore original threshold
            self.quality_agent.quality_threshold = original_threshold

    def test_content_type_specific_validation(self):
        """Test content type specific validation rules"""
        # Test comprehensive study notes validation
        high_quality_report = self.quality_agent.assess_content_quality(
            "test.md",
            content=self.high_quality_content,
            content_type="comprehensive_study_notes",
            week_number=1
        )
        
        # Should have high completeness for comprehensive notes
        self.assertGreater(high_quality_report.completeness_score, 0.8)
        
        # Check for required sections
        required_sections = self.quality_agent.quality_config["required_sections"]["comprehensive_study_notes"]
        missing_sections = high_quality_report.completeness_check.missing_sections
        
        # Should have most required sections
        self.assertLess(len(missing_sections), len(required_sections) * 0.3)

    def test_error_handling_in_integration(self):
        """Test error handling during integration workflows"""
        # Create invalid consolidation result
        invalid_consolidation = Mock()
        invalid_consolidation.success = False
        del invalid_consolidation.processed_files  # Missing required attribute
        
        # Should handle gracefully
        result = self.quality_agent.integrate_with_consolidation(invalid_consolidation)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["quality_assessed_files"], 0)

    def test_performance_monitoring(self):
        """Test performance monitoring during quality assessment"""
        test_files = self.create_test_files()
        
        # Perform batch assessment with timing
        start_time = datetime.now()
        batch_results = self.quality_agent.validate_batch_content(test_files)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance metrics are collected
        for report_dict in batch_results["reports"]:
            self.assertIn("processing_time", report_dict)
            self.assertGreater(report_dict["processing_time"], 0.0)
        
        # Total processing should be reasonable (less than 30 seconds for 3 files)
        self.assertLess(processing_time, 30.0)

    def test_report_generation_and_persistence(self):
        """Test quality report generation and file persistence"""
        test_files = self.create_test_files()
        
        # Perform batch assessment with output directory
        batch_results = self.quality_agent.validate_batch_content(
            test_files, 
            output_dir=self.output_dir
        )
        
        # Check if report files were generated
        report_files = [f for f in os.listdir(self.output_dir) if f.startswith("quality_assessment_report_")]
        self.assertGreater(len(report_files), 0)
        
        # Verify report file content
        report_file = os.path.join(self.output_dir, report_files[0])
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        self.assertIn("total_files", report_data)
        self.assertIn("reports", report_data)
        self.assertEqual(report_data["total_files"], 3)


if __name__ == '__main__':
    unittest.main()