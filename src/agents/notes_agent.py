#!/usr/bin/env python
"""
Enhanced Notes Agent for Academic Processing System

This agent handles comprehensive note generation with AI-powered enhancements,
quality control, multi-format export capabilities, and integration with the
study notes generator.

Migrated to use unified base agent architecture.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

# Use unified imports
from .base_agent import BaseAgent
from .quality_manager import QualityManager

@dataclass
class NotesMetadata:
    """Metadata for notes generation"""
    source_file: str
    generated_date: str
    processing_version: str
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


@dataclass
class NotesData:
    """Complete notes data structure"""
    sections: List[NotesSection]
    metadata: NotesMetadata
    outline: List[Dict[str, Any]]
    ai_enhancements: Dict[str, Any]


class NotesAgent(BaseAgent):
    """Enhanced notes generation agent with AI integration and study notes generation"""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__("notes_agent", config_path)
        
        # Initialize components
        self.quality_manager = QualityManager()
        
        # TODO: Re-implement study notes generator integration with unified architecture
        self.study_notes_generator = None
        self.logger.warning("Study notes generator integration pending migration")
        
        # Output directory setup
        self.notes_output_dir = self.base_dir / "processed" / "notes"
        self.notes_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhancement settings
        self.ai_enhancement_enabled = True
        self.quality_threshold = 0.7
        
        self.logger.info("NotesAgent initialized with unified architecture")

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file to generate notes"""
        try:
            self.logger.info(f"Processing file for notes: {file_path}")
            
            # Use unified architecture capabilities
            if not await self.validate_input(file_path):
                raise ValueError(f"Invalid input file: {file_path}")
            
            start_time = time.time()
            
            # Read and analyze content
            content = await self._read_content(file_path)
            sections = await self._generate_sections(content)
            
            # Create notes data
            metadata = NotesMetadata(
                source_file=file_path,
                generated_date=datetime.now().isoformat(),
                processing_version="unified-v2.0",
                quality_score=0.0,
                processing_time=time.time() - start_time
            )
            
            notes_data = NotesData(
                sections=sections,
                metadata=metadata,
                outline=[],
                ai_enhancements={}
            )
            
            # Quality evaluation
            quality_result = self.quality_manager.evaluate_content(
                asdict(notes_data), "notes"
            )
            notes_data.metadata.quality_score = quality_result.quality_score
            
            # Save results
            output_path = await self._save_notes(notes_data, file_path)
            
            result = {
                "success": True,
                "output_path": str(output_path),
                "quality_score": quality_result.quality_score,
                "processing_time": notes_data.metadata.processing_time,
                "sections_count": len(sections)
            }
            
            if await self.validate_output(result):
                self.logger.info(f"Notes generation completed: {output_path}")
                return result
            else:
                raise ValueError("Output validation failed")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    async def _read_content(self, file_path: str) -> str:
        """Read and prepare content for processing"""
        path = Path(file_path)
        
        if path.suffix.lower() == '.md':
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        elif path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        else:
            # Try to read as text
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()

    async def _generate_sections(self, content: str) -> List[NotesSection]:
        """Generate notes sections from content"""
        # Basic section generation - can be enhanced with AI
        sections = []
        
        # Split content by headers
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections.append(self._create_section(
                        current_section, '\n'.join(current_content)
                    ))
                
                # Start new section
                current_section = line.strip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_section:
            sections.append(self._create_section(
                current_section, '\n'.join(current_content)
            ))
        
        return sections

    def _create_section(self, title: str, content: str) -> NotesSection:
        """Create a notes section with analysis"""
        # Basic analysis - can be enhanced with AI
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Extract key points (lines that start with bullets or numbers)
        key_points = [
            line for line in lines 
            if line.startswith(('-', '*', '+')) or 
            (len(line) > 2 and line[0].isdigit() and line[1] == '.')
        ]
        
        # Generate summary (first 2 sentences)
        sentences = content.split('. ')
        summary = '. '.join(sentences[:2]) + '.' if sentences else content[:200]
        
        # Extract concepts (simple keyword extraction)
        concepts = self._extract_concepts(content)
        
        return NotesSection(
            title=title,
            content=content,
            level=title.count('#') if title.startswith('#') else 1,
            key_points=key_points,
            summary=summary,
            concepts=concepts
        )

    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Simple concept extraction - can be enhanced with NLP
        words = content.lower().split()
        
        # Filter for potential concepts (longer words, capitalized terms)
        concepts = []
        for word in words:
            if len(word) > 5 and word.isalpha():
                concepts.append(word.title())
        
        # Return unique concepts, limited to top 10
        return list(set(concepts))[:10]

    async def _save_notes(self, notes_data: NotesData, source_file: str) -> Path:
        """Save notes data to output directory"""
        source_path = Path(source_file)
        output_name = f"{source_path.stem}_notes.json"
        output_path = self.notes_output_dir / output_name
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(notes_data), f, indent=2, ensure_ascii=False)
        
        # Also save as markdown for human readability
        md_path = output_path.with_suffix('.md')
        await self._save_as_markdown(notes_data, md_path)
        
        return output_path

    async def _save_as_markdown(self, notes_data: NotesData, output_path: Path):
        """Save notes as markdown file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Notes: {notes_data.metadata.source_file}\n\n")
            f.write(f"*Generated: {notes_data.metadata.generated_date}*\n")
            f.write(f"*Quality Score: {notes_data.metadata.quality_score:.2f}*\n\n")
            
            for section in notes_data.sections:
                f.write(f"{'#' * (section.level + 1)} {section.title}\n\n")
                f.write(f"{section.content}\n\n")
                
                if section.key_points:
                    f.write("**Key Points:**\n")
                    for point in section.key_points:
                        f.write(f"- {point}\n")
                    f.write("\n")
                
                if section.concepts:
                    f.write("**Key Concepts:** ")
                    f.write(", ".join(section.concepts))
                    f.write("\n\n")

    async def batch_process_directory(self, directory_path: str) -> Dict[str, Any]:
        """Process all files in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            return {"success": False, "error": "Directory not found"}
        
        results = []
        supported_extensions = ['.md', '.json', '.txt']
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = await self.process_file(str(file_path))
                results.append(result)
        
        successful = sum(1 for r in results if r.get("success"))
        
        return {
            "success": True,
            "total_processed": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "results": results
        }

    async def validate_input(self, input_data: Any) -> bool:
        """Validate input file path"""
        if isinstance(input_data, str):
            path = Path(input_data)
            return path.exists() and path.is_file()
        return False

    async def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if isinstance(output_data, dict):
            return output_data.get("success", False) and "output_path" in output_data
        return False