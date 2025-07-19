#!/usr/bin/env python3
"""
Study Notes Generator Integration Demo

This script demonstrates the complete study notes generation system integration,
showing how all components work together to create comprehensive study materials.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from .study_notes_generator import StudyNotesGeneratorTool
    from .notes_agent import NotesAgent  
    from .content_templates import ContentTemplateManager, CrossReferenceManager, ContentType
    from .quality_manager import QualityManager
except ImportError:
    print("Warning: Some imports failed. This demo may have limited functionality.")
    # Create mock classes for demonstration
    class StudyNotesGeneratorTool:
        def __init__(self, base_dir):
            self.base_dir = base_dir
        
        def forward(self, **kwargs):
            return {
                "study_notes": {"title": "Mock Study Notes"},
                "output_files": [],
                "processing_stats": {"success": True}
            }
    
    class NotesAgent:
        def __init__(self):
            self.base_dir = Path.cwd()
        
        def process_file(self, file_path, **kwargs):
            return {"success": True, "output_files": []}
    
    class ContentTemplateManager:
        def apply_template(self, content, content_type, metadata=None):
            return f"Template Applied:\n{content}"
    
    class CrossReferenceManager:
        def __init__(self, base_dir):
            pass
        
        def generate_cross_reference_report(self):
            return "Mock cross-reference report"
    
    class QualityManager:
        def assess_notes_quality(self, notes):
            return 0.85


class StudyNotesIntegrationDemo:
    """Demonstrates the complete study notes generation pipeline"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / str(get_output_manager().outputs_dir) / "study_notes_demo"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("🔧 Initializing Study Notes Generation System...")
        
        try:
            self.study_notes_generator = StudyNotesGeneratorTool(self.project_root)
            print("✅ Study Notes Generator initialized")
        except Exception as e:
            print(f"⚠️  Study Notes Generator failed to initialize: {e}")
            self.study_notes_generator = None
        
        try:
            self.notes_agent = NotesAgent()
            print("✅ Notes Agent initialized")
        except Exception as e:
            print(f"⚠️  Notes Agent failed to initialize: {e}")
            self.notes_agent = None
        
        self.template_manager = ContentTemplateManager()
        print("✅ Template Manager initialized")
        
        self.xref_manager = CrossReferenceManager(self.output_dir)
        print("✅ Cross-Reference Manager initialized")
        
        self.quality_manager = QualityManager()
        print("✅ Quality Manager initialized")
        
        # Demo data
        self.demo_results = []
    
    def create_sample_content(self) -> Dict[str, str]:
        """Create sample academic content for demonstration"""
        sample_files = {}
        
        # Sample 1: Lecture Notes
        lecture_content = """
# Introduction to Cybersecurity Risk Management

## Learning Objectives
- Understand the fundamentals of cybersecurity risk management
- Learn to identify and assess security risks
- Develop risk mitigation strategies

## Key Concepts

### Risk Components
Risk in cybersecurity consists of three main components:
1. **Assets** - Information, systems, and resources that have value
2. **Threats** - Potential causes of harm to assets
3. **Vulnerabilities** - Weaknesses that can be exploited by threats

### Risk Assessment Process
The risk assessment process follows these steps:
1. Asset identification and valuation
2. Threat identification and analysis
3. Vulnerability assessment
4. Risk calculation and prioritization
5. Risk treatment planning

### Risk Treatment Options
Organizations have four main options for treating identified risks:
- **Accept** - Acknowledge the risk but take no action
- **Avoid** - Eliminate the risk by removing the asset or activity
- **Mitigate** - Reduce the likelihood or impact of the risk
- **Transfer** - Share or shift the risk to another party

## Risk Metrics and Measurement

### Qualitative vs Quantitative Assessment
- Qualitative assessments use descriptive scales (Low, Medium, High)
- Quantitative assessments use numerical values and calculations
- Hybrid approaches combine both methods for comprehensive analysis

### Common Risk Frameworks
- NIST Risk Management Framework (RMF)
- ISO 27005 Information Security Risk Management
- FAIR (Factor Analysis of Information Risk)
- OCTAVE (Operationally Critical Threat, Asset, and Vulnerability Evaluation)

## Summary
Effective cybersecurity risk management requires systematic identification, assessment, and treatment of risks. Organizations must balance security investments with business objectives while maintaining an acceptable level of risk.
"""
        
        # Sample 2: Textbook Chapter
        textbook_content = """
# Chapter 5: Network Security Fundamentals

## Introduction
Network security forms the backbone of modern cybersecurity infrastructure. This chapter explores the essential concepts, technologies, and practices used to protect network communications and infrastructure.

## Network Security Architecture

### Defense in Depth
A layered security approach that implements multiple security controls:
- **Perimeter Security** - Firewalls, intrusion detection systems
- **Network Segmentation** - VLANs, subnets, access controls
- **Endpoint Security** - Antivirus, host-based firewalls
- **Application Security** - Input validation, secure coding
- **Data Security** - Encryption, access controls

### Network Security Components

#### Firewalls
Firewalls control network traffic based on predetermined rules:
- **Packet Filtering** - Examines individual packets
- **Stateful Inspection** - Tracks connection state
- **Application Layer** - Deep packet inspection
- **Next-Generation** - Advanced threat detection

#### Intrusion Detection and Prevention
- **Network-based IDS/IPS** - Monitors network traffic
- **Host-based IDS/IPS** - Monitors individual systems
- **Signature-based Detection** - Known attack patterns
- **Anomaly-based Detection** - Behavioral analysis

## Network Threats and Attacks

### Common Network Attacks
- **Denial of Service (DoS)** - Overwhelming systems with traffic
- **Man-in-the-Middle** - Intercepting communications
- **Packet Sniffing** - Capturing network traffic
- **Session Hijacking** - Taking over active sessions
- **Network Scanning** - Reconnaissance activities

### Advanced Persistent Threats (APTs)
Sophisticated, long-term attacks that:
- Maintain persistent access to networks
- Use multiple attack vectors
- Employ social engineering techniques
- Focus on data exfiltration

## Secure Network Protocols

### Encryption Protocols
- **SSL/TLS** - Secure web communications
- **IPSec** - Network layer encryption
- **SSH** - Secure remote access
- **HTTPS** - Secure HTTP communications

### Authentication Protocols
- **Kerberos** - Network authentication protocol
- **RADIUS** - Remote authentication service
- **LDAP** - Directory access protocol
- **SAML** - Security assertion markup language

## Best Practices
1. Implement network segmentation
2. Use strong encryption for data in transit
3. Deploy comprehensive monitoring
4. Regularly update security controls
5. Conduct security assessments
6. Train users on security awareness
7. Develop incident response procedures

## Summary
Network security requires a comprehensive approach combining technology, processes, and people. Organizations must implement layered defenses while maintaining usability and performance.
"""
        
        # Sample 3: Study Guide
        study_guide_content = """
# Cybersecurity Fundamentals - Final Exam Study Guide

## Exam Information
- Date: March 15, 2024
- Duration: 3 hours
- Format: Multiple choice and essay questions
- Coverage: Chapters 1-8, all lectures

## Key Topics to Review

### 1. Security Principles
- **CIA Triad**: Confidentiality, Integrity, Availability
- **Authentication vs Authorization**
- **Non-repudiation**
- **Defense in Depth**

### 2. Risk Management
- Risk assessment methodologies
- Qualitative vs quantitative analysis
- Risk treatment strategies
- Business impact analysis

### 3. Cryptography
- Symmetric vs asymmetric encryption
- Hashing algorithms
- Digital signatures
- Public key infrastructure (PKI)

### 4. Network Security
- Firewall types and configurations
- Intrusion detection systems
- VPNs and secure protocols
- Network segmentation

### 5. Access Control
- Authentication methods
- Role-based access control (RBAC)
- Mandatory access control (MAC)
- Discretionary access control (DAC)

## Important Formulas

### Risk Calculation
- **Risk = Threat × Vulnerability × Asset Value**
- **Annual Loss Expectancy (ALE) = SLE × ARO**
- **Single Loss Expectancy (SLE) = Asset Value × Exposure Factor**

### Cryptographic Calculations
- **Key space = 2^n** (where n is key length in bits)
- **Entropy = log₂(possible outcomes)**

## Study Tips
1. Review lecture slides and textbook chapters
2. Practice with sample questions
3. Create concept maps for complex topics
4. Form study groups for discussion
5. Focus on understanding, not memorization

## Common Exam Mistakes to Avoid
- Confusing authentication with authorization
- Mixing up symmetric and asymmetric encryption
- Not understanding the difference between threats and vulnerabilities
- Forgetting to consider business impact in risk scenarios

## Practice Questions
1. What are the three components of the CIA triad?
2. Explain the difference between qualitative and quantitative risk assessment
3. How does public key cryptography work?
4. What is the purpose of network segmentation?
5. Describe the principle of least privilege

## Time Management
- Read all questions first (15 minutes)
- Answer easy questions first (45 minutes)
- Tackle complex problems (90 minutes)
- Review and check answers (30 minutes)
"""
        
        sample_files = {
            "cybersecurity_risk_management_lecture.md": lecture_content,
            "network_security_chapter.md": textbook_content,
            "cybersecurity_fundamentals_study_guide.md": study_guide_content
        }
        
        # Write sample files
        for filename, content in sample_files.items():
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print(f"📝 Created {len(sample_files)} sample content files")
        return sample_files
    
    def demonstrate_template_application(self, content_files: Dict[str, str]) -> None:
        """Demonstrate content template application"""
        print("\n🎨 Demonstrating Content Template Application...")
        
        template_examples = []
        
        for filename, content in content_files.items():
            file_path = self.output_dir / filename
            
            # Determine content type based on filename
            if "lecture" in filename:
                content_type = ContentType.LECTURE_NOTES
                metadata = {
                    "lecture_date": "2024-02-15",
                    "instructor": "Dr. Smith",
                    "course_code": "CS401",
                    "duration": "90 minutes"
                }
            elif "chapter" in filename:
                content_type = ContentType.TEXTBOOK_CHAPTER
                metadata = {
                    "chapter_number": "5",
                    "page_range": "145-180",
                    "author": "Johnson & Brown",
                    "edition": "3rd"
                }
            elif "study_guide" in filename:
                content_type = ContentType.STUDY_GUIDE
                metadata = {
                    "exam_date": "2024-03-15",
                    "difficulty_level": "intermediate",
                    "estimated_study_time": "6 hours"
                }
            else:
                content_type = ContentType.LECTURE_NOTES
                metadata = {}
            
            # Apply template
            try:
                formatted_content = self.template_manager.apply_template(
                    content, content_type, metadata
                )
                
                # Save formatted version
                formatted_path = self.output_dir / f"formatted_{filename}"
                with open(formatted_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
                
                template_examples.append({
                    "original_file": filename,
                    "formatted_file": f"formatted_{filename}",
                    "content_type": content_type.value,
                    "metadata_fields": len(metadata)
                })
                
                print(f"  ✅ Applied {content_type.value} template to {filename}")
                
            except Exception as e:
                print(f"  ❌ Failed to apply template to {filename}: {e}")
        
        self.demo_results.append({
            "step": "template_application",
            "results": template_examples
        })
    
    def demonstrate_study_notes_generation(self, content_files: Dict[str, str]) -> None:
        """Demonstrate comprehensive study notes generation"""
        print("\n📚 Demonstrating Study Notes Generation...")
        
        generation_results = []
        
        for filename, content in content_files.items():
            file_path = self.output_dir / filename
            title = filename.replace('.md', '').replace('_', ' ').title()
            
            try:
                if self.study_notes_generator:
                    # Use comprehensive generator
                    result = self.study_notes_generator.forward(
                        content_path=str(file_path),
                        title=title,
                        subject="Cybersecurity",
                        output_formats=[get_processed_output_path(ContentType.MARKDOWN), "json", "html"],
                        include_diagrams=True
                    )
                    
                    if result["processing_stats"]["success"]:
                        generation_results.append({
                            "source_file": filename,
                            "title": title,
                            "output_files": result["output_files"],
                            "diagrams_generated": result["processing_stats"].get("diagrams_generated", 0),
                            "processing_time": result["processing_stats"].get("processing_time_seconds", 0),
                            "success": True
                        })
                        print(f"  ✅ Generated comprehensive study notes for {filename}")
                    else:
                        print(f"  ❌ Failed to generate study notes for {filename}: {result['processing_stats'].get('error', 'Unknown error')}")
                
                elif self.notes_agent:
                    # Fallback to notes agent
                    result = self.notes_agent.process_file(
                        str(file_path),
                        output_formats=[get_processed_output_path(ContentType.MARKDOWN), "json"],
                        use_comprehensive_generator=False
                    )
                    
                    if result["success"]:
                        generation_results.append({
                            "source_file": filename,
                            "title": title,
                            "output_files": result["output_files"],
                            "quality_score": result.get("quality_score", 0),
                            "processing_time": result.get("processing_time", 0),
                            "success": True
                        })
                        print(f"  ✅ Generated basic study notes for {filename}")
                    else:
                        print(f"  ❌ Failed to generate notes for {filename}: {result.get('error', 'Unknown error')}")
                
                else:
                    print(f"  ⚠️  No study notes generator available for {filename}")
                
            except Exception as e:
                print(f"  ❌ Exception while processing {filename}: {e}")
        
        self.demo_results.append({
            "step": "study_notes_generation",
            "results": generation_results
        })
    
    def demonstrate_cross_reference_management(self, content_files: Dict[str, str]) -> None:
        """Demonstrate cross-reference and topic management"""
        print("\n🔗 Demonstrating Cross-Reference Management...")
        
        # Add topics to the knowledge graph
        topics_added = []
        
        # Add cybersecurity topics
        topics = [
            {
                "id": "risk_management",
                "title": "Cybersecurity Risk Management",
                "description": "Process of identifying, assessing, and treating cybersecurity risks",
                "concepts": ["risk", "threat", "vulnerability", "asset", "mitigation"],
                "difficulty": 3
            },
            {
                "id": "network_security",
                "title": "Network Security",
                "description": "Protection of network infrastructure and communications",
                "concepts": ["firewall", "IDS", "encryption", "VPN", "segmentation"],
                "difficulty": 4
            },
            {
                "id": "access_control",
                "title": "Access Control",
                "description": "Managing who can access what resources",
                "concepts": ["authentication", "authorization", "RBAC", "identity"],
                "difficulty": 2
            },
            {
                "id": "cryptography",
                "title": "Cryptography",
                "description": "Science of protecting information through mathematical algorithms",
                "concepts": ["encryption", "hashing", "digital signatures", "PKI"],
                "difficulty": 5
            }
        ]
        
        for topic_data in topics:
            try:
                topic = self.xref_manager.add_topic(
                    topic_data["id"],
                    topic_data["title"],
                    topic_data["description"],
                    topic_data["concepts"],
                    topic_data["difficulty"]
                )
                topics_added.append(topic_data["id"])
                print(f"  ✅ Added topic: {topic_data['title']}")
            except Exception as e:
                print(f"  ❌ Failed to add topic {topic_data['title']}: {e}")
        
        # Link related topics
        topic_links = [
            ("risk_management", "network_security", 0.8),
            ("network_security", "access_control", 0.7),
            ("access_control", "cryptography", 0.6),
            ("cryptography", "network_security", 0.9)
        ]
        
        links_created = []
        for topic1, topic2, strength in topic_links:
            try:
                self.xref_manager.link_topics(topic1, topic2, strength)
                links_created.append((topic1, topic2, strength))
                print(f"  🔗 Linked {topic1} ↔ {topic2} (strength: {strength})")
            except Exception as e:
                print(f"  ❌ Failed to link {topic1} and {topic2}: {e}")
        
        # Generate cross-reference report
        try:
            report = self.xref_manager.generate_cross_reference_report()
            report_path = self.output_dir / "cross_reference_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"  📄 Generated cross-reference report: {report_path}")
        except Exception as e:
            print(f"  ❌ Failed to generate cross-reference report: {e}")
        
        self.demo_results.append({
            "step": "cross_reference_management",
            "topics_added": len(topics_added),
            "links_created": len(links_created),
            "report_generated": True
        })
    
    def demonstrate_quality_assessment(self) -> None:
        """Demonstrate quality assessment of generated notes"""
        print("\n🎯 Demonstrating Quality Assessment...")
        
        # Find generated notes files
        notes_files = list(self.output_dir.glob("*study_notes*"))
        
        quality_results = []
        
        for notes_file in notes_files:
            try:
                # Mock quality assessment (since we might not have real notes)
                quality_score = self.quality_manager.assess_notes_quality({
                    "title": notes_file.stem,
                    "sections": [{"content": "Sample content"}],
                    "metadata": {"word_count": 1000}
                })
                
                quality_results.append({
                    "file": notes_file.name,
                    "quality_score": quality_score,
                    "assessment": "Good" if quality_score > 0.7 else "Needs Improvement"
                })
                
                print(f"  📊 {notes_file.name}: Quality Score {quality_score:.2f}")
                
            except Exception as e:
                print(f"  ❌ Failed to assess quality of {notes_file.name}: {e}")
        
        self.demo_results.append({
            "step": "quality_assessment",
            "files_assessed": len(quality_results),
            "average_quality": sum(r["quality_score"] for r in quality_results) / len(quality_results) if quality_results else 0,
            "results": quality_results
        })
    
    def generate_demo_summary(self) -> str:
        """Generate a comprehensive summary of the demonstration"""
        summary_lines = []
        summary_lines.append("# Study Notes Generation System Demo Summary")
        summary_lines.append(f"**Generated**: {datetime.now().isoformat()}")
        summary_lines.append(f"**Project Root**: {self.project_root}")
        summary_lines.append("")
        
        # System Components Status
        summary_lines.append("## System Components Status")
        summary_lines.append("")
        components = [
            ("Study Notes Generator", self.study_notes_generator is not None),
            ("Notes Agent", self.notes_agent is not None),
            ("Template Manager", True),
            ("Cross-Reference Manager", True),
            ("Quality Manager", True)
        ]
        
        for component, status in components:
            status_icon = "✅" if status else "❌"
            summary_lines.append(f"- {status_icon} {component}")
        summary_lines.append("")
        
        # Demo Results
        summary_lines.append("## Demo Results")
        summary_lines.append("")
        
        for result in self.demo_results:
            step_name = result["step"].replace("_", " ").title()
            summary_lines.append(f"### {step_name}")
            
            if result["step"] == "template_application":
                templates_applied = len(result["results"])
                summary_lines.append(f"- Templates applied: {templates_applied}")
                for item in result["results"]:
                    summary_lines.append(f"  - {item['content_type']}: {item['original_file']}")
            
            elif result["step"] == "study_notes_generation":
                successful = len([r for r in result["results"] if r["success"]])
                total = len(result["results"])
                summary_lines.append(f"- Files processed: {successful}/{total}")
                total_diagrams = sum(r.get("diagrams_generated", 0) for r in result["results"])
                summary_lines.append(f"- Diagrams generated: {total_diagrams}")
            
            elif result["step"] == "cross_reference_management":
                summary_lines.append(f"- Topics added: {result['topics_added']}")
                summary_lines.append(f"- Links created: {result['links_created']}")
                summary_lines.append(f"- Report generated: {'Yes' if result['report_generated'] else 'No'}")
            
            elif result["step"] == "quality_assessment":
                summary_lines.append(f"- Files assessed: {result['files_assessed']}")
                summary_lines.append(f"- Average quality: {result['average_quality']:.2f}")
            
            summary_lines.append("")
        
        # Output Files
        summary_lines.append("## Generated Files")
        summary_lines.append("")
        output_files = list(self.output_dir.glob("*"))
        for file_path in sorted(output_files):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                summary_lines.append(f"- {file_path.name} ({size_kb:.1f} KB)")
        
        summary_lines.append("")
        summary_lines.append("## Next Steps")
        summary_lines.append("")
        summary_lines.append("1. Review generated study notes and templates")
        summary_lines.append("2. Examine cross-reference relationships")
        summary_lines.append("3. Analyze quality assessment results") 
        summary_lines.append("4. Integrate with existing academic workflows")
        summary_lines.append("5. Customize templates for specific use cases")
        
        return "\n".join(summary_lines)
    
    def run_complete_demo(self) -> str:
        """Run the complete demonstration pipeline"""
        print("🚀 Starting Study Notes Generation System Demo")
        print("=" * 60)
        
        # Step 1: Create sample content
        content_files = self.create_sample_content()
        
        # Step 2: Demonstrate template application
        self.demonstrate_template_application(content_files)
        
        # Step 3: Demonstrate study notes generation
        self.demonstrate_study_notes_generation(content_files)
        
        # Step 4: Demonstrate cross-reference management
        self.demonstrate_cross_reference_management(content_files)
        
        # Step 5: Demonstrate quality assessment
        self.demonstrate_quality_assessment()
        
        # Generate summary
        summary = self.generate_demo_summary()
        summary_path = self.output_dir / "demo_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print("\n" + "=" * 60)
        print("🎉 Study Notes Generation System Demo Complete!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📄 Demo summary: {summary_path}")
        print("=" * 60)
        
        return str(summary_path)


async def main():
    """Main function to run the demonstration"""
    # Get project root (adjust path as needed)
    project_root = Path.cwd()
    
    # Create and run demo
    demo = StudyNotesIntegrationDemo(str(project_root))
    summary_path = demo.run_complete_demo()
    
    # Print summary
    print(f"\nDemo Summary available at: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())