#!/usr/bin/env python3
"""
Export System Demo for Academic Agent

This script demonstrates the comprehensive export system capabilities including:
- Single file exports in multiple formats
- Batch processing of document collections
- Integration with study notes generator
- Quality validation and reporting
- Image consolidation and optimization
- Template customization
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.processors.export_manager import ExportManager, ExportConfig
# TODO: Map export_integration functionality to new export_manager
# from src.processors.export_manager import (
#     ExportSystemIntegrator, IntegratedExportRequest,
#     export_study_notes, export_processed_pdfs, batch_export_directory
# )

def create_sample_content():
    """Create sample content for demonstration"""
    
    sample_dir = Path("./export_demo_content")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample markdown content with academic structure
    sample_content = """# Introduction to Risk Assessment

## Abstract

This document provides a comprehensive overview of risk assessment methodologies
commonly used in cybersecurity contexts. The content includes theoretical frameworks,
practical applications, and case studies that demonstrate real-world implementation.

**Keywords:** Risk Assessment, Cybersecurity, Threat Analysis, Vulnerability Management

## 1. Overview

Risk assessment is a systematic process used to identify, analyze, and evaluate risks
that could potentially impact an organization's assets, operations, or objectives.

### 1.1 Key Components

- **Asset Identification**: Cataloging valuable resources
- **Threat Analysis**: Identifying potential threats
- **Vulnerability Assessment**: Finding weaknesses
- **Risk Calculation**: Quantifying potential impact

### 1.2 Methodologies

```mermaid
graph TD
    A[Risk Assessment Process] --> B[Asset Identification]
    A --> C[Threat Analysis]
    A --> D[Vulnerability Assessment]
    B --> E[Risk Calculation]
    C --> E
    D --> E
    E --> F[Risk Mitigation]
```

## 2. Implementation Framework

The implementation of risk assessment involves several structured phases:

1. **Planning Phase**
   - Define scope and objectives
   - Identify stakeholders
   - Establish assessment criteria

2. **Analysis Phase**
   - Collect relevant data
   - Perform threat modeling
   - Evaluate existing controls

3. **Evaluation Phase**
   - Calculate risk levels
   - Prioritize findings
   - Develop recommendations

## 3. Case Study Example

Consider a financial institution implementing a new online banking system:

### 3.1 Asset Analysis
- Customer data (PII, financial records)
- Transaction processing systems
- Authentication mechanisms
- Network infrastructure

### 3.2 Threat Landscape
- External attackers (hackers, organized crime)
- Insider threats (employees, contractors)
- System failures (hardware, software bugs)
- Natural disasters (floods, earthquakes)

### 3.3 Risk Calculation

| Asset | Threat | Vulnerability | Likelihood | Impact | Risk Level |
|-------|--------|---------------|------------|--------|------------|
| Customer DB | Data Breach | Weak encryption | Medium | High | High |
| Auth System | Brute Force | No rate limiting | High | Medium | High |
| Network | DDoS Attack | Limited bandwidth | Medium | Medium | Medium |

## 4. Best Practices

### 4.1 Continuous Monitoring
Risk assessment is not a one-time activity. Organizations should:
- Regularly update threat intelligence
- Monitor for new vulnerabilities
- Review and update risk models
- Test incident response procedures

### 4.2 Stakeholder Engagement
Effective risk assessment requires collaboration between:
- **IT Security Teams**: Technical expertise
- **Business Units**: Operational context
- **Management**: Strategic oversight
- **Compliance**: Regulatory requirements

## 5. Conclusion

Risk assessment provides the foundation for informed security decision-making.
By following structured methodologies and maintaining continuous vigilance,
organizations can effectively manage their security posture and protect
critical assets from evolving threats.

## References

1. NIST Special Publication 800-30: Guide for Conducting Risk Assessments
2. ISO 27005: Information Security Risk Management
3. FAIR (Factor Analysis of Information Risk) Framework
4. OCTAVE (Operationally Critical Threat, Asset, and Vulnerability Evaluation)

---

*This document was created for demonstration purposes and represents best practices
in cybersecurity risk assessment.*
"""

    # Write sample content
    sample_file = sample_dir / "risk_assessment_overview.md"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    # Create additional sample files for batch demo
    topics = [
        ("threat_modeling", "Threat Modeling Fundamentals"),
        ("vulnerability_assessment", "Vulnerability Assessment Techniques"),
        ("incident_response", "Incident Response Planning")
    ]
    
    for filename, title in topics:
        content = f"""# {title}

## Introduction

This document covers key concepts and practices related to {title.lower()}.

## Key Concepts

- Concept 1: Definition and importance
- Concept 2: Implementation strategies
- Concept 3: Best practices and standards

## Practical Applications

### Use Case 1
Description of a practical application scenario.

### Use Case 2
Another example of real-world implementation.

## Summary

{title} is a critical component of comprehensive cybersecurity strategy.

## Further Reading

- Industry standards and frameworks
- Academic research papers
- Practical guides and tutorials
"""
        
        topic_file = sample_dir / f"{filename}.md"
        with open(topic_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return sample_dir

def demo_basic_export():
    """Demonstrate basic export functionality"""
    
    print("🚀 DEMO 1: Basic Export Functionality")
    print("=" * 50)
    
    # Create sample content
    content_dir = create_sample_content()
    sample_file = content_dir / "risk_assessment_overview.md"
    
    # Initialize export tool
    export_tool = ExportSystemTool()
    
    # Demo different export formats
    formats = ['pdf', 'html', 'docx']
    
    for format_type in formats:
        print(f"\n📄 Exporting to {format_type.upper()}...")
        
        config = {
            "output_format": format_type,
            "template_name": "academic",
            "image_sizing": "medium",
            "include_diagrams": True,
            "consolidate_images": True,
            "resolve_references": True,
            "metadata": {
                "title": "Risk Assessment Overview",
                "author": "Academic Agent Demo",
                "subject": "Cybersecurity",
                "keywords": "risk assessment, cybersecurity, threat analysis"
            }
        }
        
        start_time = time.time()
        result = export_tool.forward(
            content_paths=[str(sample_file)],
            output_directory="./export_demo_output/basic",
            export_config=config,
            batch_mode=False
        )
        
        processing_time = time.time() - start_time
        
        if result["summary"]["success"]:
            print(f"   ✅ Success! Processing time: {processing_time:.2f}s")
            print(f"   📁 Output files: {len(result['summary']['all_output_files'])}")
            for file_path in result["summary"]["all_output_files"]:
                file_size = Path(file_path).stat().st_size / 1024
                print(f"      • {Path(file_path).name} ({file_size:.1f} KB)")
        else:
            print(f"   ❌ Failed: {result['summary'].get('error', 'Unknown error')}")
    
    print(f"\n📊 Basic Export Demo Summary:")
    print(f"   • Formats tested: {len(formats)}")
    print(f"   • Sample content created in: {content_dir}")
    print(f"   • Output directory: ./export_demo_output/basic")

def demo_batch_processing():
    """Demonstrate batch processing capabilities"""
    
    print("\n\n🚀 DEMO 2: Batch Processing")
    print("=" * 50)
    
    # Use existing sample content
    content_dir = Path("./export_demo_content")
    
    print(f"📚 Processing all markdown files in: {content_dir}")
    
    # Find all markdown files
    md_files = list(content_dir.glob("*.md"))
    print(f"   Found {len(md_files)} files to process")
    
    # Batch export to multiple formats
    export_tool = ExportSystemTool()
    
    config = {
        "output_format": "all",  # Export to all formats
        "template_name": "academic",
        "image_sizing": "medium",
        "include_diagrams": True,
        "consolidate_images": True,
        "resolve_references": True,
        "metadata": {
            "author": "Academic Agent Demo",
            "subject": "Cybersecurity Collection",
            "export_type": "batch_demo"
        }
    }
    
    start_time = time.time()
    result = export_tool.forward(
        content_paths=[str(f) for f in md_files],
        output_directory="./export_demo_output/batch",
        export_config=config,
        batch_mode=True
    )
    
    processing_time = time.time() - start_time
    
    print(f"\n📊 Batch Processing Results:")
    print(f"   ✅ Successful exports: {result['summary']['successful_exports']}")
    print(f"   ❌ Failed exports: {result['summary']['failed_exports']}")
    print(f"   📁 Total output files: {len(result['summary']['all_output_files'])}")
    print(f"   💾 Total output size: {result['summary']['total_output_size_mb']:.2f} MB")
    print(f"   ⏱️  Processing time: {processing_time:.2f}s")
    
    # Show image consolidation results
    if result.get("consolidation_result"):
        consolidation = result["consolidation_result"]
        print(f"   🖼️  Images consolidated: {len(consolidation['consolidated_images'])}")
        print(f"   📉 Optimization ratio: {consolidation['optimization_ratio']*100:.1f}%")

def demo_study_notes_integration():
    """Demonstrate integration with study notes generator"""
    
    print("\n\n🚀 DEMO 3: Study Notes Integration")
    print("=" * 50)
    
    # Use sample content
    content_dir = Path("./export_demo_content")
    sample_file = content_dir / "risk_assessment_overview.md"
    
    print(f"📝 Generating study notes from: {sample_file.name}")
    
    # Generate and export study notes
    start_time = time.time()
    result = export_study_notes(
        content_path=str(sample_file),
        title="Risk Assessment Study Guide",
        subject="Cybersecurity",
        output_dir="./export_demo_output/study_notes",
        formats=['pdf', 'html']
    )
    
    processing_time = time.time() - start_time
    
    print(f"\n📊 Study Notes Integration Results:")
    print(f"   ✅ Success: {result['success']}")
    print(f"   ⏱️  Processing time: {processing_time:.2f}s")
    
    if result['success']:
        print(f"   📄 Export results: {len(result['export_results'])}")
        
        # Show individual results
        for export_result in result['export_results']:
            print(f"      • Format: {export_result['format_type']}")
            print(f"        Size: {export_result['file_size_mb']:.2f} MB")
            print(f"        Files: {len(export_result['output_files'])}")
    else:
        print(f"   ❌ Errors: {result.get('errors', [])}")

def demo_quality_validation():
    """Demonstrate quality validation features"""
    
    print("\n\n🚀 DEMO 4: Quality Validation")
    print("=" * 50)
    
    # Initialize integrator for validation features
    integrator = ExportSystemIntegrator()
    
    # Create a test export request
    content_dir = Path("./export_demo_content")
    sample_files = list(content_dir.glob("*.md"))[:2]  # Use first 2 files
    
    request = IntegratedExportRequest(
        source_type='directory',
        source_paths=[str(f) for f in sample_files],
        output_directory="./export_demo_output/quality_validation",
        export_formats=['pdf'],
        template_name='academic',
        include_quality_validation=True,
        consolidate_images=True,
        generate_index=True
    )
    
    print(f"🔍 Processing {len(sample_files)} files with quality validation...")
    
    start_time = time.time()
    result = integrator.process_integrated_export_request(request)
    processing_time = time.time() - start_time
    
    print(f"\n📊 Quality Validation Results:")
    print(f"   ✅ Success: {result.success}")
    print(f"   📄 Files processed: {len(result.export_results)}")
    print(f"   ⏱️  Processing time: {processing_time:.2f}s")
    print(f"   📋 Index generated: {result.index_generated}")
    
    # Show quality validation details
    if result.quality_validation:
        qa = result.quality_validation
        print(f"   🎯 Overall QA score: {qa.get('overall_score', 0):.2f}")
        print(f"   ✅ Passed validation: {qa.get('passed_validation', 0)}/{qa.get('total_validated', 0)}")
        print(f"   📈 Pass rate: {qa.get('pass_rate', 0)*100:.1f}%")
        
        if qa.get('recommendations'):
            print(f"   💡 Recommendations:")
            for rec in qa['recommendations'][:3]:  # Show first 3
                print(f"      • {rec}")

def demo_system_status():
    """Demonstrate system status and statistics"""
    
    print("\n\n🚀 DEMO 5: System Status & Statistics")
    print("=" * 50)
    
    # Get comprehensive system status
    integrator = ExportSystemIntegrator()
    status = integrator.get_export_system_status()
    
    print("📊 Export System Status:")
    
    # CMS Statistics
    cms_stats = status.get('cms_statistics', {})
    print(f"\n📈 CMS Statistics:")
    print(f"   • Total operations: {cms_stats.get('total_operations', 0)}")
    print(f"   • Successful operations: {cms_stats.get('successful_operations', 0)}")
    print(f"   • Success rate: {cms_stats.get('success_rate', 0)*100:.1f}%")
    print(f"   • Total output size: {cms_stats.get('total_output_size_mb', 0):.2f} MB")
    print(f"   • Average processing time: {cms_stats.get('average_processing_time', 0):.2f}s")
    
    # Format usage
    formats_used = cms_stats.get('formats_used', {})
    if formats_used:
        print(f"   • Format usage:")
        for fmt, count in formats_used.items():
            print(f"     - {fmt.upper()}: {count} exports")
    
    # System Health
    health = status.get('system_health', {})
    print(f"\n🏥 System Health:")
    print(f"   • Export tool available: {health.get('export_tool_available', False)}")
    
    integrations = health.get('integrations_loaded', {})
    print(f"   • Integrations loaded:")
    for integration, loaded in integrations.items():
        status_icon = "✅" if loaded else "❌"
        print(f"     - {integration.replace('_', ' ').title()}: {status_icon}")
    
    # Recent Operations
    recent_exports = status.get('recent_exports', [])
    if recent_exports:
        print(f"\n📋 Recent Operations ({len(recent_exports)}):")
        for i, export in enumerate(recent_exports[:3], 1):
            timestamp = export.get('timestamp', 'Unknown')
            success = export.get('result_summary', {}).get('success', False)
            formats = export.get('result_summary', {}).get('formats', [])
            status_icon = "✅" if success else "❌"
            print(f"   {i}. {status_icon} {timestamp[:19]} - {', '.join(formats)}")

def demo_template_customization():
    """Demonstrate template customization capabilities"""
    
    print("\n\n🚀 DEMO 6: Template Customization")
    print("=" * 50)
    
    print("🎨 Available Templates and Features:")
    
    # Academic template features
    print("\n📚 Academic Template:")
    print("   • Professional typography (Times New Roman)")
    print("   • Academic citation support")
    print("   • Table of contents generation")
    print("   • Figure captions and numbering")
    print("   • Reference list formatting")
    print("   • Page headers and footers")
    print("   • Proper margin settings for academic papers")
    
    # Template configuration options
    print("\n⚙️  Template Configuration Options:")
    print("   • Page size: A4, Letter, Legal")
    print("   • Margin settings: Custom margins")
    print("   • Font family: Times, Arial, Helvetica")
    print("   • Image sizing: Small, Medium, Large, Original")
    print("   • Diagram format: PNG, SVG, Text-based")
    print("   • Quality level: Low, Medium, High")
    
    # Custom CSS example
    print("\n🎨 Custom CSS Styling:")
    print("   • Override default styles")
    print("   • Brand-specific colors")
    print("   • Custom header/footer designs")
    print("   • Responsive design for HTML export")
    
    # Show actual template usage
    content_dir = Path("./export_demo_content")
    sample_file = content_dir / "risk_assessment_overview.md"
    
    # Export with different customizations
    export_tool = ExportSystemTool()
    
    customizations = [
        {
            "name": "Large Images",
            "config": {"image_sizing": "large", "quality_level": "high"}
        },
        {
            "name": "Print Optimized",
            "config": {"optimize_for_print": True, "embed_images": True}
        },
        {
            "name": "Web Optimized",
            "config": {"optimize_for_print": False, "image_sizing": "medium"}
        }
    ]
    
    print(f"\n🔧 Demonstrating Template Customizations:")
    
    for custom in customizations:
        print(f"\n   📝 {custom['name']} Configuration:")
        
        config = {
            "output_format": "pdf",
            "template_name": "academic",
            "include_diagrams": True,
            "consolidate_images": True,
            **custom['config'],
            "metadata": {
                "title": f"Sample Document - {custom['name']}",
                "author": "Template Demo"
            }
        }
        
        result = export_tool.forward(
            content_paths=[str(sample_file)],
            output_directory=f"./export_demo_output/templates/{custom['name'].lower().replace(' ', '_')}",
            export_config=config,
            batch_mode=False
        )
        
        if result["summary"]["success"]:
            file_size = result["summary"]["total_output_size_mb"]
            print(f"      ✅ Generated: {file_size:.2f} MB")
        else:
            print(f"      ❌ Failed")

def run_comprehensive_demo():
    """Run all demonstration modules"""
    
    print("🎯 ACADEMIC AGENT EXPORT SYSTEM DEMONSTRATION")
    print("=" * 60)
    print()
    print("This demo showcases the comprehensive export system capabilities")
    print("including multiple formats, batch processing, quality validation,")
    print("and integration with existing academic agent components.")
    print()
    
    # Run all demos
    demos = [
        ("Basic Export Functionality", demo_basic_export),
        ("Batch Processing", demo_batch_processing),
        ("Study Notes Integration", demo_study_notes_integration),
        ("Quality Validation", demo_quality_validation),
        ("System Status & Statistics", demo_system_status),
        ("Template Customization", demo_template_customization)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*60}")
            print(f"🎬 Starting: {demo_name}")
            print(f"{'='*60}")
            
            start_time = time.time()
            demo_func()
            processing_time = time.time() - start_time
            
            results.append({
                'name': demo_name,
                'success': True,
                'time': processing_time,
                'error': None
            })
            
            print(f"\n✅ {demo_name} completed in {processing_time:.2f}s")
            
        except Exception as e:
            results.append({
                'name': demo_name,
                'success': False,
                'time': 0,
                'error': str(e)
            })
            
            print(f"\n❌ {demo_name} failed: {e}")
    
    # Final summary
    print(f"\n\n{'='*60}")
    print("📊 DEMONSTRATION SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['time'] for r in results)
    
    print(f"✅ Successful demos: {successful}/{len(results)}")
    print(f"⏱️  Total demo time: {total_time:.2f}s")
    print()
    
    print("📋 Individual Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.2f}s" if result['success'] else "N/A"
        print(f"   {status} {result['name']}: {time_str}")
        if result['error']:
            print(f"      Error: {result['error']}")
    
    print(f"\n📁 Demo outputs saved to: ./export_demo_output/")
    print(f"📄 Sample content created in: ./export_demo_content/")
    
    # Cleanup note
    print(f"\n🧹 Note: Demo files can be removed after review:")
    print(f"   rm -rf ./export_demo_content ./export_demo_output")

if __name__ == "__main__":
    # Run the comprehensive demonstration
    run_comprehensive_demo()