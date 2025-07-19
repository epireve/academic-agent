#!/usr/bin/env python
"""
Enhanced Notes Agent for Academic Processing System

This agent handles comprehensive note generation with AI-powered enhancements,
quality control, multi-format export capabilities, and integration with the
study notes generator.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Import from unified architecture - no more path manipulation!
# Use unified BaseAgent for standardized interface  
from ...src.agents.base_agent import BaseAgent
from ...src.agents.quality_manager import QualityManager
from .study_notes_generator import StudyNotesGeneratorTool, StudyNote, StudySection
from ...src.core.output_manager import get_output_manager, OutputCategory, ContentType


@dataclass
class NotesMetadata:
    """Metadata for generated notes"""
    source_file: str
    generated_date: str
    word_count: int
    sections_count: int
    concepts_count: int
    quality_score: float
    processing_time: float
    ai_enhanced: bool = True
    diagrams_generated: int = 0


@dataclass 
class NotesSection:
    """Represents a section in the notes"""
    title: str
    content: str
    level: int
    key_points: List[str]
    summary: str
    concepts: List[str]
    diagrams: List[str] = None
    cross_references: List[str] = None
    
    def __post_init__(self):
        if self.diagrams is None:
            self.diagrams = []
        if self.cross_references is None:
            self.cross_references = []


@dataclass
class GeneratedNotes:
    """Complete notes structure"""
    title: str
    sections: List[NotesSection]
    metadata: NotesMetadata
    key_takeaways: List[str]
    cross_references: List[str]
    overview_diagram: Optional[str] = None


class NotesAgent(BaseAgent):
    """Enhanced notes generation agent with AI integration and study notes generation"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__("notes_agent", config_path)
        
        # Initialize components
        self.output_manager = None
        self.quality_manager = QualityManager()
        
        try:
            from .study_notes_generator import StudyNotesGeneratorTool
            self.study_notes_generator = StudyNotesGeneratorTool(self.base_dir)
        except ImportError:
            self.study_notes_generator = None
            self.logger.warning("Study notes generator not available")
        
        # AI configuration
        self.ai_config = {
            "model": "groq/llama-3.3-70b-versatile",
            "temperature": 0.3,
            "max_tokens": 4096
        }
        
        # Note: Output directory setup moved to initialize method
        self.notes_output_dir = None
        
        self.logger.info("NotesAgent initialized successfully with study notes generation")
    
    async def initialize(self):
        """Initialize agent-specific resources."""
        try:
            # Initialize output manager
            self.output_manager = get_output_manager()
            
            # Setup output directories for notes
            self.notes_output_dir = self.output_manager.get_output_path(
                OutputCategory.PROCESSED, 
                ContentType.MARKDOWN, 
                subdirectory="notes"
            )
            self.notes_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup study notes directory
            study_notes_dir = self.output_manager.get_output_path(
                OutputCategory.PROCESSED,
                ContentType.MARKDOWN,
                subdirectory="study_notes"
            )
            study_notes_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize quality manager if needed
            if hasattr(self.quality_manager, 'initialize'):
                await self.quality_manager.initialize()
                
            # Initialize study notes generator if available
            if self.study_notes_generator and hasattr(self.study_notes_generator, 'initialize'):
                await self.study_notes_generator.initialize()
            
            self.logger.info(f"{self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.agent_id}: {e}")
            raise

    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            # Cleanup quality manager
            if hasattr(self.quality_manager, 'cleanup'):
                await self.quality_manager.cleanup()
                
            # Cleanup study notes generator
            if self.study_notes_generator and hasattr(self.study_notes_generator, 'cleanup'):
                await self.study_notes_generator.cleanup()
            
            self.logger.info(f"{self.agent_id} cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during {self.agent_id} cleanup: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for notes generation."""
        if isinstance(input_data, str):
            # String input (file path or content)
            return len(input_data.strip()) > 0
        elif isinstance(input_data, dict):
            # Dict input should have required fields
            required_fields = ["content", "output_path"]
            return all(field in input_data for field in required_fields)
        return False

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data from notes generation."""
        if isinstance(output_data, dict):
            required_fields = ["success", "notes_path", "quality_score"]
            return all(field in output_data for field in required_fields)
        return False

    def check_quality(self, content: Any) -> float:
        """Check quality of generated notes."""
        if isinstance(content, dict) and "notes_content" in content:
            notes_content = content["notes_content"]
        elif isinstance(content, str):
            notes_content = content
        else:
            return 0.0
        
        # Basic quality checks
        quality_score = 1.0
        
        # Check content length
        if len(notes_content) < 100:
            quality_score -= 0.3
        
        # Check for proper markdown structure
        if not any(line.startswith("#") for line in notes_content.split("\n")):
            quality_score -= 0.2
            
        # Check for empty content
        if not notes_content.strip():
            quality_score = 0.0
            
        # Use quality manager if available
        if hasattr(self, 'quality_manager') and self.quality_manager:
            try:
                manager_score = self.quality_manager.assess_notes_quality(notes_content)
                quality_score = (quality_score + manager_score) / 2
            except:
                pass
        
        return max(0.0, min(1.0, quality_score))
    
    def extract_sections_from_markdown(self, content: str) -> List[NotesSection]:
        """Extract structured sections from markdown content"""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for headers
            if line.startswith('#'):
                # Save previous section
                if current_section and current_content:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = NotesSection(
                    title=title,
                    content="",
                    level=level,
                    key_points=[],
                    summary="",
                    concepts=[],
                    diagrams=[],
                    cross_references=[]
                )
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        return sections
    
    def enhance_section_with_ai(self, section: NotesSection) -> NotesSection:
        """Enhance a section with AI-generated insights"""
        if not section.content.strip():
            return section
        
        try:
            # Generate key points
            key_points = self._extract_key_points(section.content)
            section.key_points = key_points
            
            # Generate summary
            summary = self._generate_summary(section.content)
            section.summary = summary
            
            # Extract concepts
            concepts = self._extract_concepts(section.content)
            section.concepts = concepts
            
            # Generate diagram if content is substantial and study notes generator is available
            if (len(section.content) > 300 and len(concepts) >= 3 and 
                self.study_notes_generator is not None):
                try:
                    diagram = self._generate_section_diagram(section.title, concepts)
                    if diagram:
                        section.diagrams.append(diagram)
                except Exception as e:
                    self.logger.warning(f"Failed to generate diagram for section {section.title}: {e}")
            
            self.logger.debug(f"Enhanced section: {section.title}")
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance section {section.title}: {str(e)}")
        
        return section
    
    def _generate_section_diagram(self, title: str, concepts: List[str]) -> str:
        """Generate a Mermaid diagram for a section"""
        try:
            # Use the study notes generator's diagram component
            if self.study_notes_generator:
                diagram_code = self.study_notes_generator.diagram_generator.generate_concept_diagram(
                    title, concepts, "flowchart"
                )
                return diagram_code
            return ""
        except Exception as e:
            self.logger.warning(f"Error generating section diagram: {e}")
            return ""
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content using AI"""
        prompt = f"""Extract 3-5 key points from the following academic content.
        Return as a JSON array of strings.
        
        Content:
        {content[:2000]}...
        
        Format: ["point1", "point2", ...]"""
        
        try:
            response = self.ai_request(prompt)
            points = json.loads(response)
            return points[:5] if isinstance(points, list) else []
        except Exception as e:
            self.logger.warning(f"Error extracting key points: {e}")
            # Fallback to simple extraction
            sentences = content.split('. ')
            return [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
    
    def _generate_summary(self, content: str) -> str:
        """Generate summary using AI"""
        prompt = f"""Create a concise 2-3 sentence summary of this academic content:
        
        {content[:2000]}...
        
        Provide only the summary, no additional text."""
        
        try:
            return self.ai_request(prompt).strip()
        except Exception as e:
            self.logger.warning(f"Error generating summary: {e}")
            # Fallback to first sentences
            sentences = content.split('. ')
            return '. '.join(sentences[:2]) + '.'
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        prompt = f"""Extract 5-8 key concepts/terms from this academic content.
        Return as a JSON array of strings.
        
        Content:
        {content[:2000]}...
        
        Format: ["concept1", "concept2", ...]"""
        
        try:
            response = self.ai_request(prompt)
            concepts = json.loads(response)
            return concepts[:8] if isinstance(concepts, list) else []
        except Exception as e:
            self.logger.warning(f"Error extracting concepts: {e}")
            # Fallback to capitalized words
            import re
            concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            return list(set(concepts))[:8]
    
    def generate_notes_from_content(self, content: str, title: str) -> GeneratedNotes:
        """Generate comprehensive notes from content"""
        start_time = time.time()
        
        # Extract sections
        sections = self.extract_sections_from_markdown(content)
        
        # Enhance each section with AI
        enhanced_sections = []
        diagrams_count = 0
        
        for section in sections:
            enhanced_section = self.enhance_section_with_ai(section)
            enhanced_sections.append(enhanced_section)
            diagrams_count += len(enhanced_section.diagrams)
        
        # Generate overview diagram
        overview_diagram = None
        try:
            if self.study_notes_generator:
                all_concepts = []
                for section in enhanced_sections:
                    all_concepts.extend(section.concepts)
                
                unique_concepts = list(set(all_concepts))[:12]
                if unique_concepts:
                    overview_diagram = self.study_notes_generator.diagram_generator.generate_overview_diagram(
                        unique_concepts, title
                    )
                    if overview_diagram:
                        diagrams_count += 1
        except Exception as e:
            self.logger.warning(f"Failed to generate overview diagram: {e}")
        
        # Generate key takeaways
        key_takeaways = self._generate_key_takeaways(enhanced_sections)
        
        # Generate cross-references
        cross_references = self._generate_cross_references(enhanced_sections)
        
        # Create metadata
        processing_time = time.time() - start_time
        word_count = len(content.split())
        concepts_count = sum(len(s.concepts) for s in enhanced_sections)
        
        metadata = NotesMetadata(
            source_file=title,
            generated_date=datetime.now().isoformat(),
            word_count=word_count,
            sections_count=len(enhanced_sections),
            concepts_count=concepts_count,
            quality_score=0.0,  # Will be calculated later
            processing_time=processing_time,
            ai_enhanced=True,
            diagrams_generated=diagrams_count
        )
        
        # Create notes object
        notes = GeneratedNotes(
            title=title,
            sections=enhanced_sections,
            metadata=metadata,
            key_takeaways=key_takeaways,
            cross_references=cross_references,
            overview_diagram=overview_diagram
        )
        
        return notes
    
    def _generate_key_takeaways(self, sections: List[NotesSection]) -> List[str]:
        """Generate key takeaways from all sections"""
        all_points = []
        for section in sections:
            all_points.extend(section.key_points)
        
        if not all_points:
            return []
        
        points_text = '\n'.join(all_points)
        prompt = f"""Based on these key points, generate 3-5 overarching takeaways:
        
        {points_text}
        
        Return as JSON array: ["takeaway1", "takeaway2", ...]"""
        
        try:
            response = self.ai_request(prompt)
            takeaways = json.loads(response)
            return takeaways[:5] if isinstance(takeaways, list) else []
        except Exception as e:
            self.logger.warning(f"Error generating takeaways: {e}")
            return all_points[:3]  # Fallback to first key points
    
    def _generate_cross_references(self, sections: List[NotesSection]) -> List[str]:
        """Generate cross-references between sections"""
        cross_refs = []
        
        for i, section in enumerate(sections):
            for j, other_section in enumerate(sections):
                if i != j:
                    # Check for concept overlap
                    overlap = set(section.concepts) & set(other_section.concepts)
                    if len(overlap) >= 2:  # Require at least 2 shared concepts
                        cross_refs.append(f"{section.title} → {other_section.title}")
        
        return list(set(cross_refs))[:10]  # Limit to avoid overwhelming
    
    def generate_comprehensive_study_notes(self, content_path: str, title: str, 
                                         subject: str = "Academic Content",
                                         output_formats: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive study notes using the advanced study notes generator"""
        if output_formats is None:
            output_formats = ["markdown", "json", "html"]
        
        if not self.study_notes_generator:
            return {
                "study_notes": None,
                "output_files": [],
                "processing_stats": {
                    "success": False,
                    "error": "Study notes generator not available"
                }
            }
        
        try:
            # Use the study notes generator for comprehensive processing
            result = self.study_notes_generator.forward(
                content_path=content_path,
                title=title,
                subject=subject,
                output_formats=output_formats,
                include_diagrams=True
            )
            
            if result["processing_stats"]["success"]:
                self.logger.info(f"Successfully generated comprehensive study notes for {title}")
                return result
            else:
                self.logger.error(f"Failed to generate study notes: {result['processing_stats'].get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            error_msg = f"Error generating comprehensive study notes: {str(e)}"
            self.logger.error(error_msg)
            return {
                "study_notes": None,
                "output_files": [],
                "processing_stats": {
                    "success": False,
                    "error": error_msg
                }
            }
    
    def format_notes_as_markdown(self, notes: GeneratedNotes) -> str:
        """Format notes as markdown with diagrams"""
        md_parts = []
        
        # Header
        md_parts.append(f"# {notes.title}")
        md_parts.append("")
        md_parts.append(f"**Generated:** {notes.metadata.generated_date}")
        md_parts.append(f"**Word Count:** {notes.metadata.word_count}")
        md_parts.append(f"**Sections:** {notes.metadata.sections_count}")
        md_parts.append(f"**Processing Time:** {notes.metadata.processing_time:.2f}s")
        md_parts.append(f"**AI Enhanced:** {'Yes' if notes.metadata.ai_enhanced else 'No'}")
        md_parts.append(f"**Diagrams Generated:** {notes.metadata.diagrams_generated}")
        md_parts.append("")
        
        # Overview diagram
        if notes.overview_diagram:
            md_parts.append("## Overview")
            md_parts.append("")
            md_parts.append("```mermaid")
            md_parts.append(notes.overview_diagram)
            md_parts.append("```")
            md_parts.append("")
        
        # Sections
        for section in notes.sections:
            # Section title
            header_prefix = "#" * (section.level + 1)
            md_parts.append(f"{header_prefix} {section.title}")
            md_parts.append("")
            
            # Summary
            if section.summary:
                md_parts.append("### Summary")
                md_parts.append(section.summary)
                md_parts.append("")
            
            # Key points
            if section.key_points:
                md_parts.append("### Key Points")
                for point in section.key_points:
                    md_parts.append(f"- {point}")
                md_parts.append("")
            
            # Concepts
            if section.concepts:
                md_parts.append("### Key Concepts")
                for concept in section.concepts:
                    md_parts.append(f"- **{concept}**")
                md_parts.append("")
            
            # Section diagrams
            for diagram in section.diagrams:
                md_parts.append("### Concept Diagram")
                md_parts.append("")
                md_parts.append("```mermaid")
                md_parts.append(diagram)
                md_parts.append("```")
                md_parts.append("")
            
            # Content
            md_parts.append("### Content")
            md_parts.append(section.content)
            md_parts.append("")
            
            # Cross-references
            if section.cross_references:
                md_parts.append("### Related Sections")
                for ref in section.cross_references:
                    md_parts.append(f"- {ref}")
                md_parts.append("")
        
        # Key takeaways
        if notes.key_takeaways:
            md_parts.append("## Key Takeaways")
            md_parts.append("")
            for takeaway in notes.key_takeaways:
                md_parts.append(f"- {takeaway}")
            md_parts.append("")
        
        # Cross-references
        if notes.cross_references:
            md_parts.append("## Cross-References")
            md_parts.append("")
            for ref in notes.cross_references:
                md_parts.append(f"- {ref}")
            md_parts.append("")
        
        return "\n".join(md_parts)
    
    def save_notes(self, notes: GeneratedNotes, output_format: str = "markdown") -> str:
        """Save notes to file"""
        safe_title = self.sanitize_filename(notes.title)
        
        if output_format == "markdown":
            output_path = self.notes_output_dir / f"{safe_title}_enhanced_notes.md"
            content = self.format_notes_as_markdown(notes)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif output_format == "json":
            output_path = self.notes_output_dir / f"{safe_title}_enhanced_notes.json"
            content = json.dumps(asdict(notes), indent=2, ensure_ascii=False)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        return str(output_path)
    
    def process_file(self, file_path: str, output_formats: List[str] = None,
                    use_comprehensive_generator: bool = True) -> Dict[str, Any]:
        """Process a single file and generate enhanced notes"""
        if output_formats is None:
            output_formats = ["markdown", "json"]
        
        try:
            # Extract title from filename
            title = Path(file_path).stem
            
            if use_comprehensive_generator and self.study_notes_generator:
                # Use comprehensive study notes generator
                result = self.generate_comprehensive_study_notes(
                    content_path=file_path,
                    title=title,
                    subject="Academic Content",
                    output_formats=output_formats
                )
                
                if result["processing_stats"]["success"]:
                    return {
                        "success": True,
                        "comprehensive_notes": result["study_notes"],
                        "output_files": result["output_files"],
                        "processing_stats": result["processing_stats"]
                    }
                else:
                    # Fallback to basic processing
                    self.logger.warning("Comprehensive generator failed, falling back to basic processing")
            
            # Basic notes generation
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            notes = self.generate_notes_from_content(content, title)
            
            # Quality assessment
            quality_score = self.quality_manager.assess_notes_quality(notes)
            notes.metadata.quality_score = quality_score
            
            # Save in requested formats
            output_files = []
            for format_type in output_formats:
                try:
                    output_path = self.save_notes(notes, format_type)
                    output_files.append(output_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save notes in {format_type} format: {e}")
            
            result = {
                "success": True,
                "notes": asdict(notes),
                "output_files": output_files,
                "quality_score": quality_score,
                "processing_time": notes.metadata.processing_time
            }
            
            self.logger.info(f"Successfully processed {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.logger.error(error_msg)
            
            return {
                "success": False,
                "error": error_msg,
                "notes": None,
                "output_files": [],
                "quality_score": 0.0,
                "processing_time": 0.0
            }
    
    def process_directory(self, directory_path: str, 
                         output_formats: List[str] = None,
                         file_pattern: str = "*.md",
                         use_comprehensive_generator: bool = True) -> Dict[str, Any]:
        """Process all files in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Find files to process
        files_to_process = list(directory.glob(file_pattern))
        
        if not files_to_process:
            self.logger.warning(f"No files found matching pattern {file_pattern} in {directory_path}")
            return {
                "success": True,
                "processed_files": [],
                "failed_files": [],
                "total_files": 0,
                "processing_summary": {}
            }
        
        # Process each file
        processed_files = []
        failed_files = []
        total_processing_time = 0
        total_quality_score = 0
        
        for file_path in files_to_process:
            self.logger.info(f"Processing: {file_path}")
            
            result = self.process_file(str(file_path), output_formats, use_comprehensive_generator)
            
            if result["success"]:
                processed_files.append({
                    "file": str(file_path),
                    "output_files": result["output_files"],
                    "quality_score": result.get("quality_score", 0.0),
                    "processing_time": result.get("processing_time", 0.0)
                })
                total_processing_time += result.get("processing_time", 0.0)
                total_quality_score += result.get("quality_score", 0.0)
            else:
                failed_files.append({
                    "file": str(file_path),
                    "error": result["error"]
                })
        
        # Calculate summary statistics
        successful_count = len(processed_files)
        average_quality = total_quality_score / successful_count if successful_count > 0 else 0
        
        summary = {
            "total_files": len(files_to_process),
            "successful": successful_count,
            "failed": len(failed_files),
            "total_processing_time": total_processing_time,
            "average_quality_score": average_quality,
            "average_processing_time": total_processing_time / successful_count if successful_count > 0 else 0
        }
        
        self.logger.info(f"Directory processing complete: {summary}")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "total_files": len(files_to_process),
            "processing_summary": summary
        }


def main():
    """Main function for testing"""
    notes_agent = NotesAgent()
    
    # Example usage
    test_content = """
    # Introduction to Security Risk Assessment
    
    Security risk assessment is a fundamental process in cybersecurity that involves identifying, analyzing, and evaluating potential threats to an organization's information assets.
    
    ## Key Components
    
    The main components include asset identification, threat analysis, vulnerability assessment, and risk calculation.
    
    ### Asset Identification
    
    This involves cataloging all information assets within the organization, including hardware, software, data, and personnel.
    
    ### Threat Analysis
    
    Identifying potential threats that could exploit vulnerabilities in the organization's assets.
    """
    
    notes = notes_agent.generate_notes_from_content(test_content, "Test Security Risk Assessment Notes")
    markdown_output = notes_agent.format_notes_as_markdown(notes)
    
    print("Generated Enhanced Notes with Diagrams:")
    print(markdown_output)


if __name__ == "__main__":
    main()