#!/usr/bin/env python3
"""
Comprehensive Study Notes Generator for Academic Agent

This module creates comprehensive study notes with Mermaid diagrams, content summarization,
cross-referencing, and multiple export formats. It integrates with the existing PDF
processor and academic agent system.
"""

import os
import json
import re
import time
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib

from dotenv import load_dotenv
from smolagents import Tool
import litellm

# Load environment variables
load_dotenv()


@dataclass
class StudySection:
    """Represents a section of study notes"""
    title: str
    content: str
    level: int  # 1, 2, 3 for H1, H2, H3
    key_concepts: List[str]
    summary: str
    diagrams: List[str]
    cross_references: List[str]
    metadata: Dict[str, Any]


@dataclass
class StudyNote:
    """Represents a complete study note document"""
    title: str
    subject: str
    topic: str
    sections: List[StudySection]
    overview_diagram: Optional[str]
    key_takeaways: List[str]
    cross_references: Dict[str, List[str]]
    metadata: Dict[str, Any]
    generated_date: str


@dataclass
class DiagramSpec:
    """Specification for generating a Mermaid diagram"""
    diagram_type: str  # 'flowchart', 'mindmap', 'sequence', 'class', 'er'
    title: str
    concepts: List[str]
    relationships: List[Tuple[str, str, str]]  # (from, to, relationship)
    style_preferences: Dict[str, Any]


class ContentExtractor:
    """Extracts and analyzes content from various sources"""
    
    def __init__(self):
        self.model_config = {
            "model": "groq/llama-3.3-70b-versatile",
            "temperature": 0.3,
            "max_tokens": 4096
        }
        
        # Set up environment for litellm
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    
    def extract_key_concepts(self, content: str, max_concepts: int = 15) -> List[str]:
        """Extract key concepts from content using AI"""
        prompt = f"""Analyze the following academic content and extract the {max_concepts} most important key concepts.
        
Return only a JSON array of concept strings, no additional text.

Content:
{content[:3000]}...

Format: ["concept1", "concept2", ...]"""

        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **self.model_config
            )
            
            # Parse the JSON response
            concepts_text = response.choices[0].message.content.strip()
            concepts = json.loads(concepts_text)
            return concepts[:max_concepts]
            
        except Exception as e:
            print(f"Warning: Error extracting key concepts: {e}")
            # Fallback to simple extraction
            return self._extract_concepts_fallback(content, max_concepts)
    
    def _extract_concepts_fallback(self, content: str, max_concepts: int) -> List[str]:
        """Fallback method for concept extraction"""
        # Simple extraction based on capitalized phrases and technical terms
        patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Two-word capitalized phrases
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\*\*([^*]+)\*\*',  # Bold text
            r'\*([^*]+)\*'  # Italic text
        ]
        
        concepts = set()
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    concepts.add(match[0])
                else:
                    concepts.add(match)
        
        # Filter and clean concepts
        cleaned_concepts = []
        for concept in concepts:
            if len(concept) > 3 and len(concept) < 50:
                cleaned_concepts.append(concept.strip())
        
        return sorted(list(set(cleaned_concepts)))[:max_concepts]
    
    def generate_summary(self, content: str, target_length: int = 200) -> str:
        """Generate a concise summary of content"""
        prompt = f"""Create a concise {target_length}-word summary of the following academic content.
        Focus on the main concepts and key points.

Content:
{content[:4000]}...

Provide only the summary text, no additional formatting."""

        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **self.model_config
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Warning: Error generating summary: {e}")
            # Fallback to first sentences
            sentences = content.split('. ')
            return '. '.join(sentences[:3]) + '.'
    
    def identify_cross_references(self, content: str, existing_topics: List[str]) -> List[str]:
        """Identify potential cross-references to other topics"""
        references = []
        content_lower = content.lower()
        
        for topic in existing_topics:
            topic_lower = topic.lower()
            if topic_lower in content_lower and len(topic) > 3:
                references.append(topic)
        
        return list(set(references))


class MermaidDiagramGenerator:
    """Generates Mermaid diagrams for study notes"""
    
    def __init__(self):
        self.model_config = {
            "model": "groq/llama-3.3-70b-versatile",
            "temperature": 0.4,
            "max_tokens": 2048
        }
        
        # Set up environment for litellm
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
    
    def generate_overview_diagram(self, concepts: List[str], title: str) -> str:
        """Generate a high-level overview diagram"""
        concepts_text = ", ".join(concepts[:12])  # Limit to avoid overwhelming
        
        prompt = f"""Create a Mermaid.js flowchart diagram that shows the relationships between these key concepts for "{title}".

Key concepts: {concepts_text}

Requirements:
- Use flowchart TD (top-down) format
- Show logical relationships between concepts
- Use subgraphs to group related concepts
- Include appropriate styling
- Keep it clean and readable
- Maximum 15 nodes

Return only the Mermaid code starting with ```mermaid and ending with ```"""

        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **self.model_config
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract mermaid code from response
            if "```mermaid" in content:
                mermaid_code = content.split("```mermaid")[1].split("```")[0].strip()
                return mermaid_code
            else:
                return content.strip()
                
        except Exception as e:
            print(f"Warning: Error generating overview diagram: {e}")
            return self._generate_fallback_diagram(concepts, title)
    
    def generate_concept_diagram(self, section_title: str, concepts: List[str], 
                               diagram_type: str = "flowchart") -> str:
        """Generate a diagram for a specific section"""
        concepts_text = ", ".join(concepts[:8])  # Limit for section diagrams
        
        prompt = f"""Create a Mermaid.js {diagram_type} diagram for the section "{section_title}".

Concepts to include: {concepts_text}

Requirements:
- Show relationships between these concepts
- Use appropriate Mermaid syntax for {diagram_type}
- Keep it focused and readable
- Maximum 10 nodes
- Include basic styling

Return only the Mermaid code starting with ```mermaid and ending with ```"""

        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **self.model_config
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract mermaid code from response
            if "```mermaid" in content:
                mermaid_code = content.split("```mermaid")[1].split("```")[0].strip()
                return mermaid_code
            else:
                return content.strip()
                
        except Exception as e:
            print(f"Warning: Error generating concept diagram: {e}")
            return self._generate_fallback_section_diagram(section_title, concepts)
    
    def _generate_fallback_diagram(self, concepts: List[str], title: str) -> str:
        """Generate a simple fallback diagram"""
        diagram = "graph TD\n"
        diagram += f"    A[{title}]\n"
        
        for i, concept in enumerate(concepts[:6], 1):
            clean_concept = concept.replace('"', '').replace('[', '').replace(']', '')
            diagram += f"    A --> B{i}[{clean_concept}]\n"
        
        return diagram
    
    def _generate_fallback_section_diagram(self, title: str, concepts: List[str]) -> str:
        """Generate a simple fallback section diagram"""
        diagram = "graph LR\n"
        
        for i, concept in enumerate(concepts[:4], 1):
            clean_concept = concept.replace('"', '').replace('[', '').replace(']', '')
            diagram += f"    A{i}[{clean_concept}]\n"
            if i > 1:
                diagram += f"    A{i-1} --> A{i}\n"
        
        return diagram
    
    def convert_to_png(self, mermaid_code: str, output_path: Path) -> bool:
        """Convert Mermaid diagram to PNG using mermaid-cli"""
        try:
            # Create temporary file for mermaid code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as mmd_file:
                mmd_file.write(mermaid_code)
                mmd_file_path = mmd_file.name
            
            # Run mermaid-cli to convert to PNG
            cmd = [
                'npx', '@mermaid-js/mermaid-cli',
                '-i', mmd_file_path,
                '-o', str(output_path),
                '--theme', 'default',
                '--width', '1200',
                '--height', '800',
                '--backgroundColor', 'white',
                '--scale', '2'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Clean up temporary file
            os.unlink(mmd_file_path)
            
            if result.returncode == 0 and output_path.exists():
                return True
            else:
                print(f"Warning: Mermaid conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Warning: Error converting Mermaid to PNG: {e}")
            return False


class StudyNotesFormatter:
    """Formats study notes for different output formats"""
    
    def format_markdown(self, study_note: StudyNote) -> str:
        """Format study note as markdown"""
        md_content = []
        
        # Header
        md_content.append(f"# {study_note.title}")
        md_content.append("")
        md_content.append(f"**Subject:** {study_note.subject}")
        md_content.append(f"**Topic:** {study_note.topic}")
        md_content.append(f"**Generated:** {study_note.generated_date}")
        md_content.append("")
        
        # Overview diagram
        if study_note.overview_diagram:
            md_content.append("## Overview")
            md_content.append("")
            md_content.append("```mermaid")
            md_content.append(study_note.overview_diagram)
            md_content.append("```")
            md_content.append("")
        
        # Sections
        for section in study_note.sections:
            # Section header
            header_prefix = "#" * (section.level + 1)
            md_content.append(f"{header_prefix} {section.title}")
            md_content.append("")
            
            # Summary
            if section.summary:
                md_content.append("### Summary")
                md_content.append(section.summary)
                md_content.append("")
            
            # Key concepts
            if section.key_concepts:
                md_content.append("### Key Concepts")
                for concept in section.key_concepts:
                    md_content.append(f"- **{concept}**")
                md_content.append("")
            
            # Content
            md_content.append("### Content")
            md_content.append(section.content)
            md_content.append("")
            
            # Diagrams
            for diagram in section.diagrams:
                md_content.append("```mermaid")
                md_content.append(diagram)
                md_content.append("```")
                md_content.append("")
            
            # Cross-references
            if section.cross_references:
                md_content.append("### Related Topics")
                for ref in section.cross_references:
                    md_content.append(f"- {ref}")
                md_content.append("")
        
        # Key takeaways
        if study_note.key_takeaways:
            md_content.append("## Key Takeaways")
            md_content.append("")
            for takeaway in study_note.key_takeaways:
                md_content.append(f"- {takeaway}")
            md_content.append("")
        
        # Cross-references index
        if study_note.cross_references:
            md_content.append("## Cross-References")
            md_content.append("")
            for topic, refs in study_note.cross_references.items():
                md_content.append(f"### {topic}")
                for ref in refs:
                    md_content.append(f"- {ref}")
                md_content.append("")
        
        return "\n".join(md_content)
    
    def format_json(self, study_note: StudyNote) -> str:
        """Format study note as JSON"""
        return json.dumps(asdict(study_note), indent=2, ensure_ascii=False)
    
    def format_html(self, study_note: StudyNote) -> str:
        """Format study note as HTML with embedded diagrams"""
        # This would include HTML template with CSS styling
        # For now, return a basic HTML structure
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{study_note.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .mermaid {{ text-align: center; margin: 20px 0; }}
        .key-concepts {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .cross-references {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{study_note.title}</h1>
    <p><strong>Subject:</strong> {study_note.subject}</p>
    <p><strong>Generated:</strong> {study_note.generated_date}</p>
    
    {self._sections_to_html(study_note.sections)}
    
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
"""
        return html_content
    
    def _sections_to_html(self, sections: List[StudySection]) -> str:
        """Convert sections to HTML"""
        html_parts = []
        for section in sections:
            html_parts.append(f"<h{section.level + 1}>{section.title}</h{section.level + 1}>")
            if section.summary:
                html_parts.append(f"<p><strong>Summary:</strong> {section.summary}</p>")
            html_parts.append(f"<div>{section.content}</div>")
        return "\n".join(html_parts)


class StudyNotesGeneratorTool(Tool):
    """Main tool for generating comprehensive study notes"""
    
    name = "study_notes_generator_tool"
    description = "Generate comprehensive study notes with Mermaid diagrams and cross-references"
    inputs = {
        "content_path": {
            "type": "string",
            "description": "Path to the content file (markdown, JSON, or directory)",
        },
        "title": {
            "type": "string", 
            "description": "Title for the study notes",
        },
        "subject": {
            "type": "string",
            "description": "Subject area (e.g., 'Cybersecurity', 'Risk Assessment')",
        },
        "output_formats": {
            "type": "array",
            "description": "Output formats: markdown, json, html, pdf",
            "nullable": True,
        },
        "include_diagrams": {
            "type": "boolean",
            "description": "Whether to generate Mermaid diagrams",
            "nullable": True,
        },
    }
    outputs = {
        "study_notes": {
            "type": "object",
            "description": "Generated study notes with metadata",
        },
        "output_files": {
            "type": "array",
            "description": "List of generated output files",
        },
        "processing_stats": {
            "type": "object",
            "description": "Processing statistics and metrics",
        },
    }
    output_type = "object"
    
    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__()
        self.base_dir = base_dir or Path.cwd()
        self.output_dir = self.base_dir / str(get_output_manager().outputs_dir) / "study_notes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.content_extractor = ContentExtractor()
        self.diagram_generator = MermaidDiagramGenerator()
        self.formatter = StudyNotesFormatter()
        
        # Processing state
        self.existing_topics = []
        self.processing_stats = {
            "start_time": None,
            "end_time": None,
            "sections_processed": 0,
            "diagrams_generated": 0,
            "cross_references_found": 0,
            "tokens_used": 0,
        }
    
    def _load_content(self, content_path: str) -> Tuple[str, Dict[str, Any]]:
        """Load content from file or directory"""
        path = Path(content_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Content path not found: {content_path}")
        
        metadata = {"source": str(path), "type": "unknown"}
        
        if path.is_file():
            if path.suffix.lower() == '.md':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata["type"] = get_processed_output_path(ContentType.MARKDOWN)
            elif path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    content = json.dumps(data, indent=2)
                metadata["type"] = "json"
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                metadata["type"] = "text"
        else:
            # Directory - combine markdown files
            md_files = list(path.glob("**/*.md"))
            content_parts = []
            for md_file in md_files:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content_parts.append(f"# {md_file.stem}\n\n{f.read()}")
            content = "\n\n---\n\n".join(content_parts)
            metadata["type"] = "directory"
            metadata["files_processed"] = len(md_files)
        
        return content, metadata
    
    def _parse_markdown_sections(self, content: str) -> List[StudySection]:
        """Parse markdown content into structured sections"""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for headers
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = StudySection(
                    title=title,
                    content="",
                    level=level,
                    key_concepts=[],
                    summary="",
                    diagrams=[],
                    cross_references=[],
                    metadata={}
                )
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def _enhance_sections(self, sections: List[StudySection]) -> List[StudySection]:
        """Enhance sections with AI-generated content"""
        enhanced_sections = []
        
        for section in sections:
            if not section.content.strip():
                continue
            
            # Extract key concepts
            section.key_concepts = self.content_extractor.extract_key_concepts(
                section.content, max_concepts=8
            )
            
            # Generate summary
            section.summary = self.content_extractor.generate_summary(
                section.content, target_length=150
            )
            
            # Identify cross-references
            section.cross_references = self.content_extractor.identify_cross_references(
                section.content, self.existing_topics
            )
            
            # Generate diagram for important sections
            if len(section.content) > 500 and section.key_concepts:
                try:
                    diagram_code = self.diagram_generator.generate_concept_diagram(
                        section.title, section.key_concepts
                    )
                    section.diagrams.append(diagram_code)
                    self.processing_stats["diagrams_generated"] += 1
                except Exception as e:
                    print(f"Warning: Could not generate diagram for section '{section.title}': {e}")
            
            # Update metadata
            section.metadata = {
                "word_count": len(section.content.split()),
                "concept_count": len(section.key_concepts),
                "has_diagram": len(section.diagrams) > 0,
                "cross_ref_count": len(section.cross_references),
            }
            
            enhanced_sections.append(section)
            self.processing_stats["sections_processed"] += 1
            self.processing_stats["cross_references_found"] += len(section.cross_references)
        
        return enhanced_sections
    
    def _generate_key_takeaways(self, sections: List[StudySection]) -> List[str]:
        """Generate key takeaways from all sections"""
        all_concepts = []
        for section in sections:
            all_concepts.extend(section.key_concepts)
        
        # Get most common concepts
        concept_counts = {}
        for concept in all_concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Sort by frequency and take top concepts
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        top_concepts = [concept for concept, count in sorted_concepts[:10]]
        
        # Generate takeaways using AI
        prompt = f"""Based on these key concepts from study material, generate 5-7 concise key takeaways:

Key concepts: {', '.join(top_concepts)}

Format as a JSON array of strings. Each takeaway should be one clear, actionable insight."""

        try:
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                model="groq/llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )
            
            takeaways_text = response.choices[0].message.content.strip()
            takeaways = json.loads(takeaways_text)
            return takeaways
            
        except Exception as e:
            print(f"Warning: Error generating takeaways: {e}")
            # Fallback to concept-based takeaways
            return [f"Understanding {concept} is crucial for this topic" for concept in top_concepts[:5]]
    
    def _build_cross_reference_index(self, sections: List[StudySection]) -> Dict[str, List[str]]:
        """Build a cross-reference index"""
        cross_ref_index = {}
        
        for section in sections:
            if section.cross_references:
                cross_ref_index[section.title] = section.cross_references
        
        return cross_ref_index
    
    def forward(self, content_path: str, title: str, subject: str,
                output_formats: Optional[List[str]] = None,
                include_diagrams: bool = True) -> Dict[str, Any]:
        """Generate comprehensive study notes"""
        
        self.processing_stats["start_time"] = datetime.now()
        
        try:
            # Set defaults
            if output_formats is None:
                output_formats = [get_processed_output_path(ContentType.MARKDOWN), "json"]
            
            # Load content
            content, content_metadata = self._load_content(content_path)
            
            # Parse into sections
            sections = self._parse_markdown_sections(content)
            
            # Enhance sections with AI
            enhanced_sections = self._enhance_sections(sections)
            
            # Generate overview diagram
            overview_diagram = None
            if include_diagrams and enhanced_sections:
                all_concepts = []
                for section in enhanced_sections:
                    all_concepts.extend(section.key_concepts)
                
                unique_concepts = list(set(all_concepts))[:15]  # Limit for overview
                
                try:
                    overview_diagram = self.diagram_generator.generate_overview_diagram(
                        unique_concepts, title
                    )
                    self.processing_stats["diagrams_generated"] += 1
                except Exception as e:
                    print(f"Warning: Could not generate overview diagram: {e}")
            
            # Generate key takeaways
            key_takeaways = self._generate_key_takeaways(enhanced_sections)
            
            # Build cross-reference index
            cross_references = self._build_cross_reference_index(enhanced_sections)
            
            # Create study note object
            study_note = StudyNote(
                title=title,
                subject=subject,
                topic=Path(content_path).stem,
                sections=enhanced_sections,
                overview_diagram=overview_diagram,
                key_takeaways=key_takeaways,
                cross_references=cross_references,
                metadata={
                    "source": content_metadata,
                    "processing_stats": self.processing_stats.copy(),
                    "ai_enhanced": True,
                    "diagram_count": sum(len(s.diagrams) for s in enhanced_sections),
                    "total_concepts": len(set(c for s in enhanced_sections for c in s.key_concepts)),
                },
                generated_date=datetime.now().isoformat()
            )
            
            # Generate output files
            output_files = []
            safe_title = re.sub(r'[^\w\-_]', '_', title.lower())
            
            for format_type in output_formats:
                if format_type == get_processed_output_path(ContentType.MARKDOWN):
                    md_content = self.formatter.format_markdown(study_note)
                    md_path = self.output_dir / f"{safe_title}_study_notes.md"
                    with open(md_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    output_files.append(str(md_path))
                
                elif format_type == "json":
                    json_content = self.formatter.format_json(study_note)
                    json_path = self.output_dir / f"{safe_title}_study_notes.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        f.write(json_content)
                    output_files.append(str(json_path))
                
                elif format_type == "html":
                    html_content = self.formatter.format_html(study_note)
                    html_path = self.output_dir / f"{safe_title}_study_notes.html"
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    output_files.append(str(html_path))
            
            # Generate PNG diagrams if requested
            if include_diagrams:
                diagrams_dir = self.output_dir / f"{safe_title}_diagrams"
                diagrams_dir.mkdir(exist_ok=True)
                
                # Overview diagram
                if overview_diagram:
                    overview_png = diagrams_dir / "overview_diagram.png"
                    if self.diagram_generator.convert_to_png(overview_diagram, overview_png):
                        output_files.append(str(overview_png))
                
                # Section diagrams
                for i, section in enumerate(enhanced_sections):
                    for j, diagram in enumerate(section.diagrams):
                        diagram_png = diagrams_dir / f"section_{i+1}_{j+1}_diagram.png"
                        if self.diagram_generator.convert_to_png(diagram, diagram_png):
                            output_files.append(str(diagram_png))
            
            # Update processing stats
            self.processing_stats["end_time"] = datetime.now()
            processing_time = (self.processing_stats["end_time"] - 
                             self.processing_stats["start_time"]).total_seconds()
            
            return {
                "study_notes": asdict(study_note),
                "output_files": output_files,
                "processing_stats": {
                    **self.processing_stats,
                    "processing_time_seconds": processing_time,
                    "output_formats": output_formats,
                    "files_generated": len(output_files),
                    "success": True,
                }
            }
            
        except Exception as e:
            self.processing_stats["end_time"] = datetime.now()
            error_msg = f"Error generating study notes: {str(e)}"
            print(error_msg)
            
            return {
                "study_notes": None,
                "output_files": [],
                "processing_stats": {
                    **self.processing_stats,
                    "error": error_msg,
                    "success": False,
                }
            }


# Integration with existing academic agent system
def setup_study_notes_generator(base_dir: Optional[Path] = None) -> StudyNotesGeneratorTool:
    """Set up the study notes generator tool"""
    return StudyNotesGeneratorTool(base_dir)


if __name__ == "__main__":
    # Example usage
    generator = StudyNotesGeneratorTool()
    
    # Test with a sample content path
    content_path = "sample_content.md"
    result = generator.forward(
        content_path=content_path,
        title="Security Risk Assessment Fundamentals",
        subject="Cybersecurity",
        output_formats=[get_processed_output_path(ContentType.MARKDOWN), "json", "html"],
        include_diagrams=True
    )
    
    print("Study Notes Generation Results:")
    print(f"Success: {result['processing_stats']['success']}")
    print(f"Files generated: {len(result['output_files'])}")
    print(f"Processing time: {result['processing_stats'].get('processing_time_seconds', 0):.2f}s")