#!/usr/bin/env python3
"""
Content Templates and Cross-Reference Manager for Study Notes Generator

This module provides structured templates for different types of academic content
and manages cross-references between documents and topics.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class ContentType(Enum):
    """Types of academic content"""
    LECTURE_NOTES = "lecture_notes"
    TEXTBOOK_CHAPTER = "textbook_chapter"
    RESEARCH_PAPER = "research_paper"
    STUDY_GUIDE = "study_guide"
    EXAM_PREP = "exam_prep"
    CASE_STUDY = "case_study"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"


class DiagramType(Enum):
    """Types of Mermaid diagrams"""
    FLOWCHART = "flowchart"
    MINDMAP = "mindmap"
    SEQUENCE = "sequenceDiagram"
    CLASS = "classDiagram"
    ENTITY_RELATIONSHIP = "erDiagram"
    GANTT = "gantt"
    PIE = "pie"
    STATE = "stateDiagram"


@dataclass
class ContentTemplate:
    """Template for structuring academic content"""
    content_type: ContentType
    title_format: str
    required_sections: List[str]
    optional_sections: List[str]
    suggested_diagrams: List[DiagramType]
    metadata_fields: List[str]
    formatting_rules: Dict[str, str]


@dataclass
class CrossReference:
    """Represents a cross-reference between content pieces"""
    source_id: str
    target_id: str
    relationship_type: str  # "depends_on", "related_to", "precedes", "includes"
    strength: float  # 0.0 to 1.0
    description: str
    auto_generated: bool = True


@dataclass
class TopicNode:
    """Represents a topic in the knowledge graph"""
    topic_id: str
    title: str
    description: str
    concepts: List[str]
    content_types: List[ContentType]
    difficulty_level: int  # 1-5
    prerequisites: List[str]
    related_topics: List[str]
    created_date: str
    last_updated: str


class ContentTemplateManager:
    """Manages content templates and formatting"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[ContentType, ContentTemplate]:
        """Initialize default content templates"""
        templates = {}
        
        # Lecture Notes Template
        templates[ContentType.LECTURE_NOTES] = ContentTemplate(
            content_type=ContentType.LECTURE_NOTES,
            title_format="Lecture {number}: {topic}",
            required_sections=[
                "Overview",
                "Key Concepts", 
                "Main Content",
                "Summary",
                "Key Takeaways"
            ],
            optional_sections=[
                "Prerequisites",
                "Related Readings",
                "Practice Questions",
                "Additional Resources",
                "Discussion Points"
            ],
            suggested_diagrams=[
                DiagramType.FLOWCHART,
                DiagramType.MINDMAP,
                DiagramType.SEQUENCE
            ],
            metadata_fields=[
                "lecture_date",
                "instructor",
                "course_code",
                "duration",
                "slides_count"
            ],
            formatting_rules={
                "max_section_length": "800",
                "use_bullet_points": "true",
                "include_timestamps": "true",
                "highlight_definitions": "true"
            }
        )
        
        # Textbook Chapter Template
        templates[ContentType.TEXTBOOK_CHAPTER] = ContentTemplate(
            content_type=ContentType.TEXTBOOK_CHAPTER,
            title_format="Chapter {number}: {title}",
            required_sections=[
                "Introduction",
                "Learning Objectives",
                "Main Sections",
                "Chapter Summary",
                "Key Terms",
                "Review Questions"
            ],
            optional_sections=[
                "Case Studies",
                "Examples",
                "Further Reading",
                "Exercises",
                "Chapter Notes"
            ],
            suggested_diagrams=[
                DiagramType.FLOWCHART,
                DiagramType.CLASS,
                DiagramType.ENTITY_RELATIONSHIP
            ],
            metadata_fields=[
                "chapter_number",
                "page_range",
                "author",
                "edition",
                "isbn"
            ],
            formatting_rules={
                "max_section_length": "1200",
                "use_subsections": "true",
                "include_page_numbers": "true",
                "formal_tone": "true"
            }
        )
        
        # Study Guide Template
        templates[ContentType.STUDY_GUIDE] = ContentTemplate(
            content_type=ContentType.STUDY_GUIDE,
            title_format="{topic} - Study Guide",
            required_sections=[
                "Study Overview",
                "Key Concepts Review",
                "Important Formulas/Methods",
                "Study Tips",
                "Practice Materials",
                "Self-Assessment"
            ],
            optional_sections=[
                "Common Mistakes",
                "Memory Aids",
                "Time Management",
                "Additional Practice",
                "Exam Strategy"
            ],
            suggested_diagrams=[
                DiagramType.MINDMAP,
                DiagramType.FLOWCHART,
                DiagramType.PIE
            ],
            metadata_fields=[
                "exam_date",
                "coverage_period",
                "difficulty_level",
                "estimated_study_time"
            ],
            formatting_rules={
                "use_checklists": "true",
                "highlight_priorities": "true",
                "include_difficulty_indicators": "true",
                "concise_format": "true"
            }
        )
        
        # Research Paper Template
        templates[ContentType.RESEARCH_PAPER] = ContentTemplate(
            content_type=ContentType.RESEARCH_PAPER,
            title_format="{title} - Research Summary",
            required_sections=[
                "Abstract/Summary",
                "Research Problem",
                "Methodology",
                "Key Findings",
                "Implications",
                "Critique/Evaluation"
            ],
            optional_sections=[
                "Background",
                "Literature Review",
                "Data Analysis",
                "Limitations",
                "Future Research"
            ],
            suggested_diagrams=[
                DiagramType.FLOWCHART,
                DiagramType.SEQUENCE,
                DiagramType.CLASS
            ],
            metadata_fields=[
                "authors",
                "publication_year",
                "journal",
                "doi",
                "citation_count"
            ],
            formatting_rules={
                "academic_tone": "true",
                "include_citations": "true",
                "critical_analysis": "true",
                "structured_format": "true"
            }
        )
        
        return templates
    
    def get_template(self, content_type: ContentType) -> ContentTemplate:
        """Get template for a specific content type"""
        return self.templates.get(content_type, self.templates[ContentType.LECTURE_NOTES])
    
    def apply_template(self, content: str, content_type: ContentType, 
                      metadata: Dict[str, Any] = None) -> str:
        """Apply template formatting to content"""
        template = self.get_template(content_type)
        metadata = metadata or {}
        
        # Apply formatting rules
        formatted_content = self._apply_formatting_rules(content, template.formatting_rules)
        
        # Add required sections if missing
        formatted_content = self._ensure_required_sections(formatted_content, template)
        
        # Add metadata header
        header = self._generate_metadata_header(template, metadata)
        
        return f"{header}\n\n{formatted_content}"
    
    def _apply_formatting_rules(self, content: str, rules: Dict[str, str]) -> str:
        """Apply formatting rules to content"""
        formatted = content
        
        # Apply various formatting rules
        if rules.get("highlight_definitions") == "true":
            # Bold important terms
            formatted = re.sub(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', r'**\1**', formatted)
        
        if rules.get("use_bullet_points") == "true":
            # Convert numbered lists to bullet points where appropriate
            formatted = re.sub(r'^(\d+\.)\s+', '- ', formatted, flags=re.MULTILINE)
        
        if rules.get("concise_format") == "true":
            # Shorten paragraphs
            paragraphs = formatted.split('\n\n')
            formatted = '\n\n'.join([p[:500] + '...' if len(p) > 500 else p for p in paragraphs])
        
        return formatted
    
    def _ensure_required_sections(self, content: str, template: ContentTemplate) -> str:
        """Ensure all required sections are present"""
        existing_headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        existing_headers = [h.strip() for h in existing_headers]
        
        missing_sections = []
        for required in template.required_sections:
            if not any(required.lower() in h.lower() for h in existing_headers):
                missing_sections.append(required)
        
        # Add missing sections as placeholders
        if missing_sections:
            content += "\n\n"
            for section in missing_sections:
                content += f"\n## {section}\n\n*[Content to be added]*\n"
        
        return content
    
    def _generate_metadata_header(self, template: ContentTemplate, 
                                metadata: Dict[str, Any]) -> str:
        """Generate metadata header for content"""
        header_lines = []
        header_lines.append("---")
        header_lines.append(f"content_type: {template.content_type.value}")
        header_lines.append(f"generated_date: {datetime.now().isoformat()}")
        
        for field in template.metadata_fields:
            if field in metadata:
                header_lines.append(f"{field}: {metadata[field]}")
        
        header_lines.append("---")
        
        return "\n".join(header_lines)


class CrossReferenceManager:
    """Manages cross-references between content pieces"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.references_file = base_dir / "cross_references.json"
        self.topics_file = base_dir / "topics_graph.json"
        
        # Load existing data
        self.cross_references = self._load_cross_references()
        self.topics_graph = self._load_topics_graph()
    
    def _load_cross_references(self) -> List[CrossReference]:
        """Load existing cross-references"""
        if self.references_file.exists():
            try:
                with open(self.references_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return [CrossReference(**ref) for ref in data]
            except Exception:
                return []
        return []
    
    def _load_topics_graph(self) -> Dict[str, TopicNode]:
        """Load existing topics graph"""
        if self.topics_file.exists():
            try:
                with open(self.topics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {tid: TopicNode(**topic) for tid, topic in data.items()}
            except Exception:
                return {}
        return {}
    
    def add_cross_reference(self, source_id: str, target_id: str, 
                          relationship_type: str, strength: float = 0.5,
                          description: str = "") -> None:
        """Add a new cross-reference"""
        ref = CrossReference(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            description=description,
            auto_generated=True
        )
        
        # Remove any existing reference between these items
        self.cross_references = [r for r in self.cross_references 
                               if not (r.source_id == source_id and r.target_id == target_id)]
        
        self.cross_references.append(ref)
        self._save_cross_references()
    
    def find_related_content(self, content_id: str, 
                           relationship_types: List[str] = None,
                           min_strength: float = 0.3) -> List[CrossReference]:
        """Find content related to a given content ID"""
        if relationship_types is None:
            relationship_types = ["related_to", "depends_on", "precedes"]
        
        related = []
        for ref in self.cross_references:
            if (ref.source_id == content_id or ref.target_id == content_id):
                if (ref.relationship_type in relationship_types and 
                    ref.strength >= min_strength):
                    related.append(ref)
        
        return sorted(related, key=lambda x: x.strength, reverse=True)
    
    def analyze_content_relationships(self, content: str, content_id: str,
                                   existing_content_ids: List[str]) -> List[CrossReference]:
        """Analyze content to identify potential cross-references"""
        potential_refs = []
        content_lower = content.lower()
        
        # Simple keyword-based analysis
        for other_id in existing_content_ids:
            if other_id == content_id:
                continue
            
            # Check if other content ID appears in current content
            if other_id.lower().replace('_', ' ') in content_lower:
                strength = self._calculate_relationship_strength(content, other_id)
                if strength > 0.3:
                    potential_refs.append(CrossReference(
                        source_id=content_id,
                        target_id=other_id,
                        relationship_type="related_to",
                        strength=strength,
                        description=f"Content mentions {other_id}",
                        auto_generated=True
                    ))
        
        return potential_refs
    
    def _calculate_relationship_strength(self, content: str, target_id: str) -> float:
        """Calculate the strength of relationship between content and target"""
        content_words = content.lower().split()
        target_words = target_id.lower().replace('_', ' ').split()
        
        # Count occurrences of target words in content
        matches = 0
        for word in target_words:
            matches += content_words.count(word)
        
        # Normalize by content length
        strength = min(matches / max(len(content_words) / 100, 1), 1.0)
        return strength
    
    def add_topic(self, topic_id: str, title: str, description: str,
                 concepts: List[str] = None, difficulty_level: int = 1) -> TopicNode:
        """Add a new topic to the knowledge graph"""
        concepts = concepts or []
        
        topic = TopicNode(
            topic_id=topic_id,
            title=title,
            description=description,
            concepts=concepts,
            content_types=[],
            difficulty_level=difficulty_level,
            prerequisites=[],
            related_topics=[],
            created_date=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        self.topics_graph[topic_id] = topic
        self._save_topics_graph()
        return topic
    
    def link_topics(self, topic1_id: str, topic2_id: str, 
                   relationship_strength: float = 0.5) -> None:
        """Link two topics in the knowledge graph"""
        if topic1_id in self.topics_graph and topic2_id in self.topics_graph:
            if topic2_id not in self.topics_graph[topic1_id].related_topics:
                self.topics_graph[topic1_id].related_topics.append(topic2_id)
            if topic1_id not in self.topics_graph[topic2_id].related_topics:
                self.topics_graph[topic2_id].related_topics.append(topic1_id)
            
            # Also add as cross-reference
            self.add_cross_reference(
                topic1_id, topic2_id, "related_to", 
                relationship_strength, "Topic relationship"
            )
            
            self._save_topics_graph()
    
    def generate_cross_reference_report(self) -> str:
        """Generate a report of all cross-references"""
        report_lines = []
        report_lines.append("# Cross-Reference Report")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # Group by relationship type
        by_type = {}
        for ref in self.cross_references:
            if ref.relationship_type not in by_type:
                by_type[ref.relationship_type] = []
            by_type[ref.relationship_type].append(ref)
        
        for rel_type, refs in by_type.items():
            report_lines.append(f"## {rel_type.replace('_', ' ').title()} Relationships")
            report_lines.append("")
            
            for ref in sorted(refs, key=lambda x: x.strength, reverse=True):
                report_lines.append(f"- **{ref.source_id}** → **{ref.target_id}** "
                                  f"(Strength: {ref.strength:.2f})")
                if ref.description:
                    report_lines.append(f"  - {ref.description}")
            report_lines.append("")
        
        # Topics overview
        if self.topics_graph:
            report_lines.append("## Topics Overview")
            report_lines.append("")
            
            for topic in self.topics_graph.values():
                report_lines.append(f"### {topic.title}")
                report_lines.append(f"- **ID**: {topic.topic_id}")
                report_lines.append(f"- **Difficulty**: {topic.difficulty_level}/5")
                report_lines.append(f"- **Concepts**: {len(topic.concepts)}")
                report_lines.append(f"- **Related Topics**: {len(topic.related_topics)}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _save_cross_references(self) -> None:
        """Save cross-references to file"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.references_file, 'w', encoding='utf-8') as f:
            data = [asdict(ref) for ref in self.cross_references]
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_topics_graph(self) -> None:
        """Save topics graph to file"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.topics_file, 'w', encoding='utf-8') as f:
            data = {tid: asdict(topic) for tid, topic in self.topics_graph.items()}
            json.dump(data, f, indent=2, ensure_ascii=False)


def create_content_from_template(content_type: ContentType, title: str,
                               raw_content: str, metadata: Dict[str, Any] = None) -> str:
    """Create structured content using a template"""
    template_manager = ContentTemplateManager()
    return template_manager.apply_template(raw_content, content_type, metadata)


def main():
    """Example usage of content templates and cross-references"""
    
    # Create template manager
    template_manager = ContentTemplateManager()
    
    # Example content
    raw_content = """
    Security risk assessment is the foundation of information security management.
    It involves identifying assets, threats, and vulnerabilities to calculate risk levels.
    
    The process includes several key steps:
    1. Asset identification and valuation
    2. Threat identification and analysis
    3. Vulnerability assessment
    4. Risk calculation and prioritization
    5. Risk treatment planning
    """
    
    # Apply study guide template
    formatted_content = template_manager.apply_template(
        raw_content, 
        ContentType.STUDY_GUIDE,
        metadata={
            "exam_date": "2024-03-15",
            "difficulty_level": 3,
            "estimated_study_time": "4 hours"
        }
    )
    
    print("Formatted Study Guide:")
    print(formatted_content)
    print("\n" + "="*50 + "\n")
    
    # Cross-reference example
    xref_manager = CrossReferenceManager(Path("./output/study_notes"))
    
    # Add some topics
    xref_manager.add_topic(
        "risk_assessment", 
        "Security Risk Assessment",
        "Process of identifying and evaluating security risks",
        ["risk", "threat", "vulnerability", "asset"],
        difficulty_level=3
    )
    
    xref_manager.add_topic(
        "threat_modeling",
        "Threat Modeling", 
        "Systematic approach to identifying potential threats",
        ["threat", "model", "attack vector"],
        difficulty_level=4
    )
    
    # Link topics
    xref_manager.link_topics("risk_assessment", "threat_modeling", 0.8)
    
    # Generate report
    report = xref_manager.generate_cross_reference_report()
    print("Cross-Reference Report:")
    print(report)


if __name__ == "__main__":
    main()