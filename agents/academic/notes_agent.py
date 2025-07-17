#!/usr/bin/env python
"""
Notes Agent - Specialized agent for generating comprehensive academic notes
from source materials and outlines.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import re
from dotenv import load_dotenv
from groq import Groq
from .base_agent import BaseAgent

try:
    from smolagents import CodeAgent
    from smolagents import HfApiModel
except ImportError:
    print("Installing smolagents...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents import HfApiModel

# Load environment variables
load_dotenv()

# Define base paths
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
OUTPUT_DIR = BASE_DIR / "output"
PROCESSED_DIR = BASE_DIR / "processed"
OUTLINES_DIR = PROCESSED_DIR / "outlines"
NOTES_DIR = PROCESSED_DIR / "notes"


class NotesAgent(BaseAgent):
    """Agent responsible for expanding outline into comprehensive notes"""

    def __init__(self, groq_api_key: str):
        super().__init__("notes_agent")
        self.groq = Groq(api_key=groq_api_key)
        self.min_section_words = 300
        self.max_section_words = 1000

    def expand_outline(
        self, outline_path: str, source_files: List[str]
    ) -> Dict[str, Any]:
        """Expand outline into comprehensive notes"""
        try:
            start_time = datetime.now()

            # Load outline
            with open(outline_path, "r", encoding="utf-8") as f:
                outline_content = f.read()

            # Load source materials
            source_contents = self._load_source_files(source_files)

            # Process outline sections
            sections = self._process_outline_sections(outline_content, source_contents)

            # Generate expanded notes
            notes = self._generate_expanded_notes(sections, source_contents)

            # Save notes
            output = self._save_notes(notes)

            # Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()

            # Check quality
            quality_score = self.check_quality(
                {"notes": notes, "sections": sections, "sources": source_contents}
            )

            return {
                "success": True,
                "notes_path": output["notes_path"],
                "quality_score": quality_score,
                "processing_metrics": {
                    "processing_time": processing_time,
                    "total_sections": len(sections),
                    "total_words": sum(
                        len(section["content"].split()) for section in notes["sections"]
                    ),
                },
            }

        except Exception as e:
            self.handle_error(e, {"operation": "notes_expansion"})
            return {"success": False, "error": str(e)}

    def _load_source_files(self, source_files: List[str]) -> Dict[str, str]:
        """Load content from source markdown files"""
        source_contents = {}
        for file_path in source_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_contents[file_path] = f.read()
            except Exception as e:
                self.logger.error(f"Error loading source file {file_path}: {str(e)}")
        return source_contents

    def _process_outline_sections(
        self, outline_content: str, source_contents: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Process outline sections and prepare for expansion"""
        sections = []

        # Extract sections from outline
        import re

        section_pattern = r"## Section \d+: (.*?)\n(.*?)(?=## Section \d+:|$)"
        matches = re.finditer(section_pattern, outline_content, re.DOTALL)

        for match in matches:
            title = match.group(1).strip()
            content = match.group(2).strip()

            # Find relevant source content
            relevant_sources = self._find_relevant_sources(
                title, content, source_contents
            )

            sections.append(
                {
                    "title": title,
                    "outline_content": content,
                    "relevant_sources": relevant_sources,
                }
            )

        return sections

    def _find_relevant_sources(
        self, title: str, content: str, source_contents: Dict[str, str]
    ) -> Dict[str, str]:
        """Find relevant source content for a section"""
        relevant_sources = {}

        for source_path, source_content in source_contents.items():
            # Simple relevance check based on keyword matching
            # In a production system, this would use more sophisticated NLP
            keywords = set(title.lower().split() + content.lower().split())
            source_words = set(source_content.lower().split())

            relevance = len(keywords.intersection(source_words)) / len(keywords)

            if relevance > 0.3:  # Arbitrary threshold
                relevant_sources[source_path] = source_content

        return relevant_sources

    def _generate_expanded_notes(
        self, sections: List[Dict[str, Any]], source_contents: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate expanded notes from outline sections"""
        notes = {
            "title": "Comprehensive Academic Notes",
            "created_at": datetime.now().isoformat(),
            "sections": [],
        }

        for section in sections:
            expanded_section = self._expand_section(section)
            notes["sections"].append(expanded_section)

        return notes

    def _expand_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Expand a single section using LLM"""
        prompt = f"""
        Create comprehensive academic notes for the following section:
        
        Title: {section['title']}
        Outline Content: {section['outline_content']}
        
        Using these source materials:
        {self._format_sources(section['relevant_sources'])}
        
        Generate detailed notes that:
        1. Thoroughly explain the main concept
        2. Include relevant examples and applications
        3. Cite source materials appropriately
        4. Maintain academic rigor and clarity
        5. Include 300-1000 words
        
        Return the content in markdown format with appropriate headings,
        lists, and emphasis where needed.
        """

        response = self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )

        content = response.choices[0].message.content

        # Validate content length
        word_count = len(content.split())
        if word_count < self.min_section_words:
            self.logger.warning(
                f"Section {section['title']} content too short: {word_count} words"
            )
        elif word_count > self.max_section_words:
            self.logger.warning(
                f"Section {section['title']} content too long: {word_count} words"
            )

        return {
            "title": section["title"],
            "content": content,
            "word_count": word_count,
            "sources": list(section["relevant_sources"].keys()),
        }

    def _format_sources(self, sources: Dict[str, str]) -> str:
        """Format source content for LLM prompt"""
        formatted = []
        for path, content in sources.items():
            # Take first 1000 characters of each source
            excerpt = content[:1000] + "..." if len(content) > 1000 else content
            formatted.append(f"Source ({os.path.basename(path)}):\n{excerpt}\n")
        return "\n".join(formatted)

    def _save_notes(self, notes: Dict[str, Any]) -> Dict[str, str]:
        """Save expanded notes to file"""
        notes_dir = os.path.join("processed", "notes")
        os.makedirs(notes_dir, exist_ok=True)

        # Convert notes to markdown
        markdown_content = self._convert_notes_to_markdown(notes)

        # Save main notes file
        notes_path = os.path.join(notes_dir, "comprehensive_notes.md")
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        # Save individual chapter files
        for i, section in enumerate(notes["sections"], 1):
            chapter_path = os.path.join(notes_dir, f"chapter_{i:02d}.md")
            chapter_content = self._format_chapter(section, i)
            with open(chapter_path, "w", encoding="utf-8") as f:
                f.write(chapter_content)

        return {"notes_path": notes_path}

    def _convert_notes_to_markdown(self, notes: Dict[str, Any]) -> str:
        """Convert notes dictionary to markdown format"""
        markdown = [
            f"# {notes['title']}\n",
            f"*Generated on: {notes['created_at']}*\n\n",
            "## Table of Contents\n",
        ]

        # Add TOC
        for i, section in enumerate(notes["sections"], 1):
            markdown.append(f"{i}. [{section['title']}](#chapter-{i})\n")

        markdown.append("\n---\n")

        # Add sections
        for i, section in enumerate(notes["sections"], 1):
            markdown.extend(
                [
                    f"## Chapter {i}: {section['title']}\n",
                    f"{section['content']}\n\n",
                    "### Sources\n",
                ]
            )

            for source in section["sources"]:
                markdown.append(f"- {os.path.basename(source)}\n")

            markdown.append("\n")

        return "".join(markdown)

    def _format_chapter(self, section: Dict[str, Any], chapter_num: int) -> str:
        """Format a single chapter for individual file"""
        return f"""# Chapter {chapter_num}: {section['title']}

{section['content']}

## Sources
{chr(10).join(f'- {os.path.basename(source)}' for source in section['sources'])}
"""

    def check_quality(self, content: Dict[str, Any]) -> float:
        """Check quality of generated notes"""
        quality_score = 1.0
        deductions = []

        # Check section length
        for section in content["notes"]["sections"]:
            word_count = section["word_count"]
            if word_count < self.min_section_words:
                deductions.append(0.2)
            elif word_count > self.max_section_words:
                deductions.append(0.1)

        # Check source usage
        if not all(section["sources"] for section in content["notes"]["sections"]):
            deductions.append(0.3)

        # Check content structure
        for section in content["notes"]["sections"]:
            if not all(marker in section["content"] for marker in ["#", "-", "*"]):
                deductions.append(0.1)

        # Apply deductions
        for deduction in deductions:
            quality_score -= deduction

        return max(0.0, min(1.0, quality_score))

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if not isinstance(input_data, dict):
            return False

        required_fields = ["outline_path", "source_files"]
        return all(field in input_data for field in required_fields)

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if not isinstance(output_data, dict):
            return False

        required_fields = ["success", "notes_path", "quality_score"]
        return all(field in output_data for field in required_fields)


def main():
    """Main entry point for the notes agent"""
    parser = argparse.ArgumentParser(
        description="Notes Agent for generating academic notes"
    )

    # Define command line arguments
    parser.add_argument("--sources", nargs="+", help="Paths to source markdown files")
    parser.add_argument("--outline", help="Path to outline JSON file")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--update", help="Path to existing notes file to update")
    parser.add_argument(
        "--new-sources", nargs="+", help="Paths to new source files for update"
    )
    parser.add_argument("--output", help="Custom output path for updated notes")
    parser.add_argument("--api-key", help="Groq API key")

    args = parser.parse_args()

    # Get API key from environment or command line
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    # Create the notes agent
    agent = NotesAgent(api_key)

    # Handle generate or update
    if args.sources and args.outline:
        # Generate new notes
        notes_path = agent.expand_outline(args.outline, args.sources)
        if notes_path:
            print(f"Notes generated and saved to: {notes_path}")

    elif args.update and args.new_sources:
        # Update existing notes
        updated_path = agent.update_notes(args.update, args.new_sources, args.output)
        if updated_path:
            print(f"Notes updated and saved to: {updated_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
