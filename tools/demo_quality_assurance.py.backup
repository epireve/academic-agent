#!/usr/bin/env python
"""
Demo script for Content Quality Assurance Module

This script demonstrates the capabilities of the Content Quality Assurance Agent
including quality assessment, auto-fixing, batch processing, and integration
with the consolidation workflow.

Usage:
    python demo_quality_assurance.py --help
    python demo_quality_assurance.py --demo-single
    python demo_quality_assurance.py --demo-batch
    python demo_quality_assurance.py --demo-integration
    python demo_quality_assurance.py --demo-analytics
"""

import os
import sys
import argparse
import tempfile
import shutil
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.content_quality_agent import ContentQualityAgent
from src.agents.consolidation_agent import ContentConsolidationAgent, FileMapping


class QualityAssuranceDemo:
    """Demonstration of Content Quality Assurance capabilities"""

    def __init__(self):
        self.quality_agent = ContentQualityAgent()
        self.consolidation_agent = ContentConsolidationAgent()
        self.temp_dir = None
        
        # Sample content with varying quality levels
        self.sample_contents = {
            "high_quality": """---
week: 1
content_type: comprehensive_study_notes
---

# Chapter 1: Introduction to Security Risk Assessment

## High-Level Concept Overview

```mermaid
graph TD
    A[Security Manager] --> B{Risk Assessment Process}
    B --> C[Asset Identification]
    B --> D[Threat Analysis]
    B --> E[Vulnerability Assessment]
    C --> F[Risk Calculation]
    D --> F
    E --> F
    F --> G[Risk Mitigation Strategy]
```

## Executive Summary

Security Risk Assessment (SRA) is a systematic process used to identify, analyze, and evaluate 
potential security risks that could impact an organization's information assets. This fundamental 
security practice enables organizations to make informed decisions about resource allocation, 
risk treatment strategies, and security control implementation.

The SRA process involves comprehensive analysis of three key components: assets (what needs protection), 
threats (potential sources of harm), and vulnerabilities (weaknesses that could be exploited). 
By understanding the relationships between these components, organizations can calculate risk levels 
and develop appropriate mitigation strategies.

## Key Concepts

### Security Risk Assessment Fundamentals

#### Definition and Purpose
A Security Risk Assessment is defined as a systematic evaluation process that:
- Identifies information assets and their business value
- Analyzes potential threats to these assets
- Evaluates existing vulnerabilities
- Calculates risk levels based on likelihood and impact
- Recommends appropriate risk treatment strategies

#### Risk Components
The fundamental equation for risk assessment is:
**Risk = Threat × Vulnerability × Asset Value**

**Assets**: Information, systems, processes, and resources that have value to the organization.
Examples include customer databases, intellectual property, and critical business applications.

**Threats**: Potential causes of incidents that could result in harm to assets. These can be:
- Natural disasters (floods, earthquakes)
- Human threats (malicious insiders, cybercriminals)
- Technical failures (hardware malfunctions, software bugs)
- Environmental factors (power outages, facility issues)

**Vulnerabilities**: Weaknesses in security controls that could be exploited by threats.
Common vulnerability categories include:
- Technical vulnerabilities (unpatched software, misconfigurations)
- Administrative vulnerabilities (inadequate policies, poor training)
- Physical vulnerabilities (unsecured facilities, inadequate access controls)

## Detailed Analysis

### Risk Assessment Methodologies

#### Qualitative Assessment
Qualitative risk assessment uses descriptive scales and subjective analysis:
- **Advantages**: Easy to understand, quick to implement, good for communication
- **Disadvantages**: Less precise, potential for bias, difficult to compare across assessments
- **Best Used For**: Initial assessments, high-level overviews, organizations with limited data

#### Quantitative Assessment
Quantitative risk assessment uses numerical values and statistical analysis:
- **Advantages**: Objective, precise, enables cost-benefit analysis
- **Disadvantages**: Requires extensive data, time-consuming, may create false precision
- **Best Used For**: Critical assets, regulatory compliance, detailed business case development

#### Hybrid Approach
Most organizations use a hybrid approach that combines both methods:
- Qualitative assessment for initial screening and communication
- Quantitative assessment for critical risks and detailed analysis
- Balanced approach that maximizes benefits while managing resource constraints

### Implementation Process

#### Phase 1: Planning and Preparation
- Define assessment scope and objectives
- Identify stakeholders and assign responsibilities
- Develop assessment methodology and criteria
- Establish communication protocols

#### Phase 2: Asset Identification and Valuation
- Inventory information assets
- Classify assets by type and criticality
- Determine asset values (replacement cost, business impact)
- Map asset dependencies and relationships

#### Phase 3: Threat and Vulnerability Analysis
- Identify relevant threat sources
- Assess threat capabilities and motivations
- Conduct vulnerability assessments
- Analyze security control effectiveness

#### Phase 4: Risk Analysis and Evaluation
- Calculate risk levels for identified scenarios
- Prioritize risks based on severity and likelihood
- Develop risk treatment recommendations
- Document findings and rationale

## Practical Applications

### Industry-Specific Implementations

#### Financial Services
Banks and financial institutions use SRA to:
- Protect customer financial data
- Comply with regulations (SOX, PCI DSS, GLBA)
- Secure payment processing systems
- Manage operational risk

#### Healthcare Organizations
Healthcare providers implement SRA to:
- Safeguard patient health information (HIPAA compliance)
- Protect medical devices and systems
- Ensure business continuity for critical care
- Manage research data security

#### Manufacturing Companies
Manufacturing organizations use SRA for:
- Protecting intellectual property and trade secrets
- Securing industrial control systems
- Managing supply chain risks
- Ensuring operational continuity

### Case Study: Multi-National Corporation SRA Implementation

**Background**: A global manufacturing company needed to standardize risk assessment across 50+ facilities worldwide.

**Challenges**:
- Diverse regulatory environments
- Varying maturity levels across locations
- Limited security expertise at some sites
- Need for consistent risk reporting

**Solution**:
- Developed standardized SRA methodology
- Created risk assessment templates and tools
- Implemented centralized risk management platform
- Established regional centers of excellence

**Results**:
- 40% reduction in assessment time
- Improved risk visibility across organization
- Enhanced regulatory compliance
- Better resource allocation for security investments

## Exam Focus Areas

### Critical Concepts for Assessment

1. **Risk Assessment Fundamentals**
   - Definition of risk and its components
   - Relationship between assets, threats, and vulnerabilities
   - Purpose and benefits of conducting SRA

2. **Assessment Methodologies**
   - Qualitative vs. quantitative approaches
   - Advantages and limitations of each method
   - When to use hybrid approaches

3. **Implementation Process**
   - Key phases of SRA implementation
   - Stakeholder roles and responsibilities
   - Documentation and reporting requirements

4. **Risk Treatment Strategies**
   - Risk acceptance criteria and processes
   - Risk mitigation techniques and controls
   - Risk transfer mechanisms (insurance, outsourcing)
   - Risk avoidance through process changes

### Common Exam Questions

**Question Types to Expect**:
- Scenario-based questions requiring risk analysis
- Multiple choice on methodology selection
- Essay questions on implementation challenges
- Case study analysis and recommendations

## Review Questions

### Knowledge Check

1. **Fundamental Concepts**
   - What are the three primary components of security risk?
   - How does the risk equation (Risk = Threat × Vulnerability × Asset Value) work in practice?
   - What is the difference between a threat and a vulnerability?

2. **Methodology Selection**
   - When would you choose qualitative over quantitative risk assessment?
   - What are the key advantages of a hybrid assessment approach?
   - How do you determine the appropriate level of assessment detail?

3. **Implementation Planning**
   - What factors should be considered when defining assessment scope?
   - How do you ensure stakeholder buy-in for the SRA process?
   - What documentation is essential for a successful assessment?

4. **Risk Treatment**
   - What criteria should guide risk acceptance decisions?
   - How do you prioritize risk mitigation investments?
   - When is risk transfer an appropriate strategy?

### Practice Scenarios

**Scenario 1: Cloud Migration Assessment**
Your organization is planning to migrate critical business applications to a public cloud environment. 
Design a risk assessment approach that addresses:
- Data security and privacy concerns
- Vendor risk and dependency issues
- Compliance and regulatory requirements
- Business continuity and disaster recovery

**Scenario 2: Merger and Acquisition Due Diligence**
Your company is acquiring a smaller competitor. Develop a security risk assessment framework for:
- Evaluating the target company's security posture
- Identifying integration risks and challenges
- Assessing regulatory and compliance gaps
- Planning post-merger security improvements

**Scenario 3: Incident Response Planning**
Following a significant security breach, your organization needs to:
- Conduct a post-incident risk assessment
- Identify systemic vulnerabilities
- Develop improved security controls
- Create metrics for ongoing risk monitoring

### Additional Resources

- NIST SP 800-30: Guide for Conducting Risk Assessments
- ISO 27005: Information Security Risk Management
- FAIR (Factor Analysis of Information Risk) methodology
- OCTAVE (Operationally Critical Threat, Asset, and Vulnerability Evaluation)
""",
            
            "medium_quality": """# Week 2: Risk Analysis Methods

## Overview
Risk analysis is important for organizations to understand their security posture.

## Types of Risk Analysis

### Qualitative Analysis
- Uses descriptions like High, Medium, Low
- Easier to understand
- Less precise

### Quantitative Analysis
- Uses numbers and calculations
- More precise
- Requires more data

## Process Steps
1. Identify assets
2. Find threats
3. Assess vulnerabilities  
4. Calculate risk
5. Develop mitigation

## Examples
- Banks assess credit card fraud risk
- Hospitals protect patient data
- Companies secure trade secrets

## Benefits
Risk analysis helps organizations:
- Make better decisions
- Allocate resources effectively
- Meet compliance requirements
- Reduce security incidents

## Summary
Effective risk analysis requires both qualitative and quantitative approaches depending on the situation and available resources.
""",
            
            "poor_quality": """#Risk Assessment

no space after hash

this is a very long line that goes on and on without any consideration for readability or proper formatting practices and should be flagged as a quality issue

-poorlist
no space after bullet

Risk assessment is important.

Some threats:
*hackers
*malware
*disasters

__inconsistent__bold**formatting**

![](missing-alt-text.jpg)

Incomplete content with missing sections.
"""
        }

    def setup_demo_environment(self):
        """Set up temporary environment for demonstrations"""
        self.temp_dir = tempfile.mkdtemp(prefix="quality_demo_")
        print(f"Demo environment created at: {self.temp_dir}")
        
        # Create sample files
        self.sample_files = {}
        for quality_level, content in self.sample_contents.items():
            filename = f"{quality_level}_quality_content.md"
            filepath = os.path.join(self.temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.sample_files[quality_level] = filepath
        
        return self.temp_dir

    def cleanup_demo_environment(self):
        """Clean up temporary environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Demo environment cleaned up: {self.temp_dir}")

    def demo_single_file_assessment(self):
        """Demonstrate single file quality assessment"""
        print("\n" + "="*60)
        print("DEMO: Single File Quality Assessment")
        print("="*60)
        
        self.setup_demo_environment()
        
        try:
            for quality_level, filepath in self.sample_files.items():
                print(f"\n--- Assessing {quality_level.replace('_', ' ').title()} Content ---")
                print(f"File: {os.path.basename(filepath)}")
                
                # Perform quality assessment
                report = self.quality_agent.assess_content_quality(filepath)
                
                # Display results
                print(f"\nQuality Scores:")
                print(f"  Overall Quality:    {report.overall_quality_score:.2f}")
                print(f"  Completeness:       {report.completeness_score:.2f}")
                print(f"  Formatting:         {report.formatting_score:.2f}")
                print(f"  Consistency:        {report.consistency_score:.2f}")
                print(f"  Academic Quality:   {report.academic_quality_score:.2f}")
                
                # Show content structure
                structure = report.content_structure
                print(f"\nContent Structure:")
                print(f"  Headers:            {len(structure.headers)}")
                print(f"  Word Count:         {structure.word_count}")
                print(f"  Images:             {structure.images}")
                print(f"  Code Blocks:        {structure.code_blocks}")
                print(f"  Mermaid Diagrams:   {structure.mermaid_diagrams}")
                
                # Show issues if any
                if report.formatting_issues:
                    print(f"\nFormatting Issues ({len(report.formatting_issues)}):")
                    for issue in report.formatting_issues[:3]:  # Show first 3
                        print(f"  - {issue.issue_type}: {issue.description}")
                    if len(report.formatting_issues) > 3:
                        print(f"  ... and {len(report.formatting_issues) - 3} more")
                
                # Show improvement suggestions
                if report.improvement_suggestions:
                    print(f"\nImprovement Suggestions:")
                    for suggestion in report.improvement_suggestions[:3]:
                        print(f"  - {suggestion}")
                
                print("-" * 50)
        
        finally:
            self.cleanup_demo_environment()

    def demo_batch_processing(self):
        """Demonstrate batch quality processing"""
        print("\n" + "="*60)
        print("DEMO: Batch Quality Processing")
        print("="*60)
        
        self.setup_demo_environment()
        
        try:
            # Create additional test files
            additional_files = []
            for i in range(3):
                filename = f"week_{i+4:02d}_notes.md"
                filepath = os.path.join(self.temp_dir, filename)
                
                # Use medium quality content as template
                content = self.sample_contents["medium_quality"].replace("Week 2", f"Week {i+4}")
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                additional_files.append(filepath)
            
            # Combine all files for batch processing
            all_files = list(self.sample_files.values()) + additional_files
            
            print(f"Processing {len(all_files)} files in batch...")
            
            # Perform batch assessment
            batch_results = self.quality_agent.validate_batch_content(
                all_files,
                output_dir=os.path.join(self.temp_dir, "batch_output")
            )
            
            # Display batch results
            print(f"\nBatch Processing Results:")
            print(f"  Total Files:        {batch_results['total_files']}")
            print(f"  Processed:          {batch_results['processed_files']}")
            print(f"  Failed:             {batch_results['failed_files']}")
            
            # Show summary statistics
            if batch_results.get('summary'):
                summary = batch_results['summary']
                print(f"\nQuality Summary:")
                print(f"  Average Quality:    {summary['average_quality']:.2f}")
                print(f"  Pass Rate:          {summary['pass_rate']:.1f}%")
                print(f"  Content Types:      {', '.join(summary['content_types'])}")
                print(f"  Total Issues:       {summary['total_issues']}")
                print(f"  Auto-fixable:       {summary['auto_fixable_issues']}")
            
            # Show quality distribution
            scores = [report["overall_quality_score"] for report in batch_results["reports"]]
            print(f"\nQuality Score Distribution:")
            print(f"  Highest:            {max(scores):.2f}")
            print(f"  Lowest:             {min(scores):.2f}")
            print(f"  Average:            {sum(scores)/len(scores):.2f}")
            
        finally:
            self.cleanup_demo_environment()

    def demo_auto_fix_capabilities(self):
        """Demonstrate auto-fix capabilities"""
        print("\n" + "="*60)
        print("DEMO: Auto-Fix Capabilities")
        print("="*60)
        
        self.setup_demo_environment()
        
        try:
            # Use poor quality content for auto-fix demo
            poor_file = self.sample_files["poor_quality"]
            
            print(f"File: {os.path.basename(poor_file)}")
            
            # Assess quality before fixes
            print("\n--- Before Auto-Fix ---")
            initial_report = self.quality_agent.assess_content_quality(poor_file)
            print(f"Quality Score: {initial_report.overall_quality_score:.2f}")
            print(f"Formatting Issues: {len(initial_report.formatting_issues)}")
            
            # Show auto-fixable issues
            auto_fixable = [issue for issue in initial_report.formatting_issues if issue.auto_fixable]
            print(f"Auto-fixable Issues: {len(auto_fixable)}")
            for issue in auto_fixable:
                print(f"  - {issue.issue_type}: {issue.suggestion}")
            
            # Apply auto-fixes
            print("\n--- Applying Auto-Fixes ---")
            fix_results = self.quality_agent.auto_fix_issues(poor_file, backup=True)
            
            print(f"Auto-fix Results:")
            print(f"  Success: {fix_results['success']}")
            print(f"  Fixes Applied: {fix_results['fixes_applied']}")
            print(f"  Issues Fixed: {', '.join(fix_results['issues_fixed'])}")
            if fix_results.get('backup_path'):
                print(f"  Backup Created: {os.path.basename(fix_results['backup_path'])}")
            
            # Assess quality after fixes
            print("\n--- After Auto-Fix ---")
            final_report = self.quality_agent.assess_content_quality(poor_file)
            print(f"Quality Score: {final_report.overall_quality_score:.2f}")
            print(f"Formatting Issues: {len(final_report.formatting_issues)}")
            
            # Show improvement
            improvement = final_report.overall_quality_score - initial_report.overall_quality_score
            print(f"Quality Improvement: {improvement:+.2f}")
            
        finally:
            self.cleanup_demo_environment()

    def demo_integration_with_consolidation(self):
        """Demonstrate integration with consolidation workflow"""
        print("\n" + "="*60)
        print("DEMO: Integration with Consolidation")
        print("="*60)
        
        self.setup_demo_environment()
        
        try:
            # Create mock consolidation scenario
            input_dir = os.path.join(self.temp_dir, "input")
            output_dir = os.path.join(self.temp_dir, "output")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create source files with different naming patterns
            source_files = []
            content_variations = [
                ("transcript_week1.md", "transcript", 1),
                ("lecture_notes_w02.md", "lecture", 2),
                ("study_guide_week_03.md", "notes", 3)
            ]
            
            for filename, content_type, week in content_variations:
                filepath = os.path.join(input_dir, filename)
                # Use appropriate content based on type
                if content_type == "transcript":
                    content = self.sample_contents["high_quality"]
                else:
                    content = self.sample_contents["medium_quality"]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content.replace("week: 1", f"week: {week}"))
                source_files.append(filepath)
            
            print(f"Created {len(source_files)} source files with inconsistent naming")
            
            # Simulate consolidation process
            print("\n--- Simulating Consolidation ---")
            
            # Create file mappings (simulating consolidation agent output)
            from agents.academic.consolidation_agent import ConsolidationResult
            
            processed_files = []
            for i, (src_file, (_, content_type, week)) in enumerate(zip(source_files, content_variations)):
                target_filename = f"week_{week:02d}_{content_type}.md"
                target_path = os.path.join(output_dir, target_filename)
                
                # Copy and rename file
                shutil.copy2(src_file, target_path)
                
                file_mapping = FileMapping(
                    source_path=src_file,
                    target_path=target_path,
                    confidence=0.8 + i * 0.1,
                    week_number=week,
                    content_type=content_type,
                    metadata={"original_filename": os.path.basename(src_file)}
                )
                processed_files.append(file_mapping)
            
            # Create consolidation result
            consolidation_result = ConsolidationResult(
                success=True,
                processed_files=processed_files,
                skipped_files=[],
                errors=[],
                consolidation_report={
                    "total_files_processed": len(processed_files),
                    "processing_date": datetime.now().isoformat()
                },
                unified_structure={"output": output_dir}
            )
            
            print(f"Consolidation completed: {len(processed_files)} files processed")
            
            # Integrate quality assessment
            print("\n--- Quality Assessment Integration ---")
            integration_result = self.quality_agent.integrate_with_consolidation(consolidation_result)
            
            print(f"Integration Results:")
            print(f"  Consolidation Success: {integration_result['consolidation_success']}")
            print(f"  Files Assessed: {integration_result['quality_assessed_files']}")
            
            # Show integrated metrics
            if integration_result.get('integrated_metrics'):
                metrics = integration_result['integrated_metrics']
                print(f"\nIntegrated Metrics:")
                print(f"  Overall Quality Score: {metrics['overall_quality_score']:.2f}")
                print(f"  Quality Pass Rate: {metrics['quality_pass_rate']:.1f}%")
                print(f"  Consolidation Efficiency: {metrics['consolidation_efficiency']:.2f}")
                
                # Show content type quality
                if metrics.get('content_type_quality_scores'):
                    print(f"  Content Type Quality:")
                    for content_type, score in metrics['content_type_quality_scores'].items():
                        print(f"    {content_type}: {score:.2f}")
            
            # Show recommendations
            if integration_result.get('recommendations'):
                print(f"\nRecommendations:")
                for rec in integration_result['recommendations'][:3]:
                    print(f"  - {rec}")
        
        finally:
            self.cleanup_demo_environment()

    def demo_quality_analytics(self):
        """Demonstrate quality analytics and reporting"""
        print("\n" + "="*60)
        print("DEMO: Quality Analytics and Reporting")
        print("="*60)
        
        self.setup_demo_environment()
        
        try:
            # Generate multiple assessments to build analytics data
            print("Building assessment history...")
            
            # Assess original sample files multiple times
            for i in range(3):
                for quality_level, filepath in self.sample_files.items():
                    self.quality_agent.assess_content_quality(filepath)
            
            # Create additional varied content
            for week in range(4, 8):
                filename = f"week_{week:02d}_notes.md"
                filepath = os.path.join(self.temp_dir, filename)
                
                # Vary quality randomly
                import random
                quality_levels = list(self.sample_contents.keys())
                selected_quality = random.choice(quality_levels)
                content = self.sample_contents[selected_quality]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content.replace("week: 1", f"week: {week}"))
                
                self.quality_agent.assess_content_quality(filepath)
            
            print(f"Generated {len(self.quality_agent.assessment_history)} assessment records")
            
            # Generate analytics
            print("\n--- Generating Quality Analytics ---")
            analytics = self.quality_agent.generate_quality_analytics()
            
            print(f"Analytics Summary:")
            print(f"  Total Assessments: {analytics.total_files_assessed}")
            print(f"  Average Quality: {analytics.average_quality_score:.2f}")
            
            # Show quality distribution
            print(f"\nQuality Distribution:")
            for quality_level, count in analytics.quality_distribution.items():
                percentage = (count / analytics.total_files_assessed) * 100
                print(f"  {quality_level.title()}: {count} files ({percentage:.1f}%)")
            
            # Show content type quality
            print(f"\nContent Type Quality:")
            for content_type, score in analytics.content_type_quality.items():
                print(f"  {content_type}: {score:.2f}")
            
            # Show week quality trends
            if analytics.week_quality_trends:
                print(f"\nWeek Quality Trends:")
                for week, score in sorted(analytics.week_quality_trends.items()):
                    print(f"  Week {week}: {score:.2f}")
            
            # Show common issues
            print(f"\nCommon Issues:")
            for issue in analytics.common_issues[:5]:
                print(f"  {issue['issue_type']}: {issue['count']} occurrences ({issue['percentage']:.1f}%)")
            
            # Show improvement opportunities
            print(f"\nImprovement Opportunities:")
            for opportunity in analytics.improvement_opportunities:
                print(f"  - {opportunity}")
            
            # Show assessment summary
            summary = analytics.assessment_summary
            print(f"\nAssessment Summary:")
            print(f"  Pass Rate: {summary['pass_rate']:.1f}%")
            print(f"  Score Range: {summary['lowest_quality_score']:.2f} - {summary['highest_quality_score']:.2f}")
            print(f"  Standard Deviation: {summary['quality_std_dev']:.2f}")
            
        finally:
            self.cleanup_demo_environment()

    def demo_all(self):
        """Run all demonstration scenarios"""
        print("CONTENT QUALITY ASSURANCE MODULE - COMPREHENSIVE DEMO")
        print("=" * 80)
        print("This demo showcases the complete functionality of the Quality Assurance Module")
        print("=" * 80)
        
        self.demo_single_file_assessment()
        self.demo_batch_processing()
        self.demo_auto_fix_capabilities()
        self.demo_integration_with_consolidation()
        self.demo_quality_analytics()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print("The Content Quality Assurance Module provides comprehensive quality validation")
        print("capabilities for academic content, including:")
        print("  • Detailed quality assessment with multiple scoring dimensions")
        print("  • Automated formatting issue detection and correction")
        print("  • Batch processing for efficient workflow integration")
        print("  • Seamless integration with content consolidation processes")
        print("  • Advanced analytics and trending for continuous improvement")
        print("="*80)


def main():
    """Main entry point for the demo script"""
    parser = argparse.ArgumentParser(
        description="Content Quality Assurance Module Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_quality_assurance.py --demo-single
  python demo_quality_assurance.py --demo-batch
  python demo_quality_assurance.py --demo-all
        """
    )
    
    parser.add_argument("--demo-single", action="store_true",
                       help="Demo single file quality assessment")
    parser.add_argument("--demo-batch", action="store_true",
                       help="Demo batch processing capabilities")
    parser.add_argument("--demo-auto-fix", action="store_true",
                       help="Demo auto-fix functionality")
    parser.add_argument("--demo-integration", action="store_true",
                       help="Demo integration with consolidation")
    parser.add_argument("--demo-analytics", action="store_true",
                       help="Demo quality analytics and reporting")
    parser.add_argument("--demo-all", action="store_true",
                       help="Run all demonstration scenarios")
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = QualityAssuranceDemo()
    
    try:
        if args.demo_single:
            demo.demo_single_file_assessment()
        elif args.demo_batch:
            demo.demo_batch_processing()
        elif args.demo_auto_fix:
            demo.demo_auto_fix_capabilities()
        elif args.demo_integration:
            demo.demo_integration_with_consolidation()
        elif args.demo_analytics:
            demo.demo_quality_analytics()
        elif args.demo_all:
            demo.demo_all()
        else:
            # Default to showing help and running a quick demo
            parser.print_help()
            print("\nRunning quick single file demo...\n")
            demo.demo_single_file_assessment()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()