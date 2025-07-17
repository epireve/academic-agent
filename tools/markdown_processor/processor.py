"""
Advanced markdown processing tools using markdownify
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import json
from markdownify import markdownify as md
import re
from bs4 import BeautifulSoup
import html2text


class MarkdownProcessor:
    """
    Advanced markdown processing tool with support for HTML conversion,
    content analysis, and structured document generation.
    """

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the markdown processor

        Args:
            base_dir: Base directory for saving markdown files
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.h2t = html2text.HTML2Text()
        self.h2t.body_width = 0  # Don't wrap text
        self.ensure_directories()

    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        dirs = ["analysis", "notes", "outlines", "raw"]
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def html_to_markdown(self, html_content: str, **kwargs) -> str:
        """
        Convert HTML content to markdown using markdownify

        Args:
            html_content: HTML string to convert
            **kwargs: Additional arguments for markdownify

        Returns:
            Markdown formatted string
        """
        return md(html_content, **kwargs)

    def clean_markdown(self, content: str) -> str:
        """
        Clean and format markdown content

        Args:
            content: Markdown content to clean

        Returns:
            Cleaned markdown content
        """
        # Remove multiple blank lines
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)

        # Fix heading spacing
        content = re.sub(
            r"(#{1,6})\s*(.+?)\s*#*$", r"\1 \2", content, flags=re.MULTILINE
        )

        # Fix list item spacing
        content = re.sub(r"^\s*[-*+]\s+", "- ", content, flags=re.MULTILINE)

        return content.strip()

    def extract_sections(self, content: str) -> List[Dict[str, str]]:
        """
        Extract sections from markdown content

        Args:
            content: Markdown content to process

        Returns:
            List of sections with titles and content
        """
        sections = []
        current_section = None
        current_content = []

        for line in content.split("\n"):
            if re.match(r"^#{1,6}\s", line):
                if current_section:
                    sections.append(
                        {
                            "title": current_section,
                            "content": "\n".join(current_content).strip(),
                        }
                    )
                current_section = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            sections.append(
                {
                    "title": current_section,
                    "content": "\n".join(current_content).strip(),
                }
            )

        return sections

    def save_analysis(
        self, analysis: Dict[str, Any], base_filename: str
    ) -> Dict[str, str]:
        """
        Save analysis results in both markdown and JSON formats

        Args:
            analysis: Analysis dictionary containing results
            base_filename: Base name for the output files

        Returns:
            Dictionary with paths to saved files
        """
        # Prepare paths
        analysis_dir = self.base_dir / "analysis"
        json_path = analysis_dir / f"{base_filename}_analysis.json"
        md_path = analysis_dir / f"{base_filename}_analysis.md"

        # Save JSON
        json_path.write_text(json.dumps(analysis, indent=2))

        # Generate markdown content
        md_content = [
            f"# Content Analysis: {base_filename}",
            "",
            "## Analysis Information",
            f"- Source File: {analysis.get('source_file', 'Unknown')}",
            f"- Analysis Date: {analysis.get('analysis_date', datetime.now().isoformat())}",
            "",
            "## Main Topics",
            *[f"- {topic}" for topic in analysis.get("main_topics", [])],
            "",
            "## Key Concepts",
            *[f"- {concept}" for concept in analysis.get("key_concepts", [])],
            "",
            "## Document Structure",
        ]

        # Add structure sections
        for section in analysis.get("structure", []):
            md_content.extend(
                [
                    f"### {section.get('title', 'Untitled Section')}",
                    section.get("summary", "No summary available"),
                    "",
                ]
            )

        # Add summary if available
        if "summary" in analysis:
            md_content.extend(["## Overall Summary", analysis["summary"]])

        # Save markdown
        md_path.write_text("\n".join(md_content))

        return {"json_path": str(json_path), "markdown_path": str(md_path)}

    def save_notes(self, notes: Dict[str, Any], filename: str) -> str:
        """
        Save generated notes in markdown format

        Args:
            notes: Notes dictionary
            filename: Output filename

        Returns:
            Path to saved markdown file
        """
        notes_dir = self.base_dir / "notes"
        output_path = notes_dir / f"{filename}.md"

        # Generate markdown content
        md_content = [
            f"# Notes: {filename}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Content",
            *notes.get("content", ["No content available"]),
            "",
            "## References",
            f"Source: {notes.get('source_file', 'Unknown')}",
        ]

        output_path.write_text("\n".join(md_content))
        return str(output_path)

    def save_outline(self, outline: Dict[str, Any], filename: str) -> str:
        """
        Save generated outline in markdown format

        Args:
            outline: Outline dictionary
            filename: Output filename

        Returns:
            Path to saved markdown file
        """
        outlines_dir = self.base_dir / "outlines"
        output_path = outlines_dir / f"{filename}.md"

        # Generate markdown content
        md_content = [
            f"# Outline: {filename}",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Sections",
        ]

        # Add sections
        for section in outline.get("sections", []):
            md_content.extend(
                [
                    f"### {section.get('title', 'Untitled Section')}",
                    section.get("content", "No content available"),
                    "",
                ]
            )

        output_path.write_text("\n".join(md_content))
        return str(output_path)

    def merge_markdown_files(self, files: List[str], output_filename: str) -> str:
        """
        Merge multiple markdown files into one

        Args:
            files: List of markdown file paths
            output_filename: Name for the merged file

        Returns:
            Path to the merged file
        """
        merged_content = []

        for file_path in files:
            with open(file_path, "r") as f:
                content = f.read().strip()
                merged_content.append(content)
                merged_content.append("\n---\n")  # Add separator

        output_path = self.base_dir / f"{output_filename}.md"
        output_path.write_text("\n".join(merged_content))

        return str(output_path)
