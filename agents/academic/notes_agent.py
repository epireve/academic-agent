#!/usr/bin/env python
"""
Notes Agent - Specialized agent for generating comprehensive academic notes
Part of the Academic Agent system
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from smolagents import CodeAgent
    from smolagents import HfApiModel
except ImportError:
    print("Installing smolagents...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents import HfApiModel

# Define base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = BASE_DIR / "output"
PROCESSED_DIR = BASE_DIR / "processed"
OUTLINES_DIR = PROCESSED_DIR / "outlines"
NOTES_DIR = PROCESSED_DIR / "notes"


class NotesAgent:
    """
    Agent specialized in generating comprehensive academic notes from source 
    materials and outlines
    """
    
    def __init__(self, api_key: str):
        # Configure Groq model
        self.model = HfApiModel(
            model_id="llama3-70b-8192",
            provider="groq",
            api_key=api_key
        )
        
        # Create the agent
        self.agent = CodeAgent(
            tools=[],  # No special tools needed for this agent
            model=self.model
        )
        
        # Ensure directories exist
        os.makedirs(NOTES_DIR, exist_ok=True)
    
    def _load_sources(self, markdown_paths: List[str]) -> Dict[str, str]:
        """
        Load source content from markdown files
        
        Args:
            markdown_paths: List of paths to markdown files
            
        Returns:
            Dictionary mapping file paths to their content
        """
        sources = {}
        
        for path in markdown_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    sources[path] = f.read()
            else:
                print(f"Warning: Source file not found: {path}")
                
        return sources
    
    def _load_outline(self, outline_path: str) -> Dict[str, Any]:
        """
        Load an outline structure from JSON file
        
        Args:
            outline_path: Path to the outline JSON file
            
        Returns:
            Dictionary representing the outline structure
        """
        if not os.path.exists(outline_path):
            print(f"Warning: Outline file not found: {outline_path}")
            return {}
            
        try:
            with open(outline_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not parse outline file: {outline_path}")
            return {}
    
    def generate_notes(self, markdown_paths: List[str], outline_path: str, 
                      output_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive notes based on source materials and outline
        
        Args:
            markdown_paths: List of paths to source markdown files
            outline_path: Path to the outline JSON file
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to the generated notes file
        """
        if not output_dir:
            output_dir = str(NOTES_DIR)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the sources and outline
        sources = self._load_sources(markdown_paths)
        outline = self._load_outline(outline_path)
        
        if not sources:
            print("Error: No valid source files found")
            return ""
            
        if not outline:
            print("Error: No valid outline found")
            return ""
        
        # Generate the notes
        notes = self._generate_notes_from_outline(sources, outline)
        
        # Determine output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outline_title = outline.get('title', 'Academic Notes')
        sanitized_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in outline_title)
        sanitized_title = sanitized_title.replace(' ', '_')
        notes_filename = f"{sanitized_title}_{timestamp}.md"
        notes_path = Path(output_dir) / notes_filename
        
        # Save the notes
        with open(notes_path, 'w') as f:
            f.write(notes)
            
        return str(notes_path)
    
    def _generate_notes_from_outline(self, sources: Dict[str, str], 
                                    outline: Dict[str, Any]) -> str:
        """
        Generate comprehensive notes following the outline structure
        
        Args:
            sources: Dictionary mapping file paths to their content
            outline: Dictionary representing the outline structure
            
        Returns:
            Generated notes as a string
        """
        # Extract outline details
        title = outline.get('title', 'Academic Notes')
        subject = outline.get('subject', 'Academic Subject')
        created = outline.get('created', datetime.now().isoformat())
        sections = outline.get('sections', [])
        
        # Create notes header
        notes = f"# {title}\n\n"
        notes += f"**Subject**: {subject}  \n"
        notes += f"**Date**: {created}  \n\n"
        
        # Add source references
        notes += "## Sources\n\n"
        for path in sources.keys():
            notes += f"- {os.path.basename(path)}\n"
        notes += "\n"
        
        # Process sections recursively
        
        def process_section(section, level=2):
            nonlocal notes
            
            # Add section title
            section_title = section.get('title', 'Section')
            notes += f"{'#' * level} {section_title}\n\n"
            
            # Find relevant content from sources for this section
            # This is a placeholder for the actual content generation
            # In a full implementation, this would use the LLM to synthesize 
            # content from sources based on the section title and key points
            
            # Add placeholder content
            notes += "This section covers the following key points:\n\n"
            
            key_points = section.get('key_points', [])
            for point in key_points:
                notes += f"- **{point}**: Content related to this key point would be generated here.\n"
            notes += "\n"
            
            # Process subsections
            subsections = section.get('subsections', [])
            for subsection in subsections:
                process_section(subsection, level + 1)
        
        # Process all top-level sections
        for section in sections:
            process_section(section)
            
        # In a full implementation, we would use the LLM to generate
        # content for each section based on the sources and outline
        # For now, we'll return the placeholder notes structure
            
        return notes
    
    def update_notes(self, existing_notes_path: str, new_markdown_paths: List[str],
                    output_path: Optional[str] = None) -> str:
        """
        Update existing notes with new information from additional sources
        
        Args:
            existing_notes_path: Path to existing notes file
            new_markdown_paths: List of paths to new source markdown files
            output_path: Where to save the updated notes (optional)
            
        Returns:
            Path to the updated notes file
        """
        if not output_path:
            # Generate new filename for the updated notes
            dirname = os.path.dirname(existing_notes_path)
            basename = os.path.basename(existing_notes_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(dirname, f"updated_{basename.split('.')[0]}_{timestamp}.md")
        
        # Load existing notes
        with open(existing_notes_path, 'r') as f:
            existing_notes = f.read()
            
        # Load new sources
        new_sources = self._load_sources(new_markdown_paths)
        
        if not new_sources:
            print("Error: No valid new source files found")
            return ""
            
        # Create the update
        updated_notes = self._update_notes_with_new_sources(existing_notes, new_sources)
        
        # Save the updated notes
        with open(output_path, 'w') as f:
            f.write(updated_notes)
            
        return output_path
    
    def _update_notes_with_new_sources(self, existing_notes: str, 
                                     new_sources: Dict[str, str]) -> str:
        """
        Update existing notes with information from new sources
        
        Args:
            existing_notes: Existing notes content
            new_sources: Dictionary mapping file paths to their content
            
        Returns:
            Updated notes as a string
        """
        # This is a placeholder for the actual update process
        # In a full implementation, this would use the LLM to intelligently
        # integrate new information with the existing notes
        
        # Add update header
        updated_notes = existing_notes
        updated_notes += "\n\n## Updates\n\n"
        updated_notes += f"_Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        
        # Add references to new sources
        updated_notes += "### New Sources\n\n"
        for path in new_sources.keys():
            updated_notes += f"- {os.path.basename(path)}\n"
        updated_notes += "\n"
        
        # Add placeholder for new content
        updated_notes += "### New Information\n\n"
        updated_notes += "Additional content from new sources would be integrated here.\n\n"
            
        return updated_notes


def main():
    """Main entry point for the notes agent"""
    parser = argparse.ArgumentParser(description="Notes Agent for generating academic notes")
    
    # Define command line arguments
    parser.add_argument("--sources", nargs='+', help="Paths to source markdown files")
    parser.add_argument("--outline", help="Path to outline JSON file")
    parser.add_argument("--output-dir", help="Custom output directory")
    parser.add_argument("--update", help="Path to existing notes file to update")
    parser.add_argument("--new-sources", nargs='+', help="Paths to new source files for update")
    parser.add_argument("--output", help="Custom output path for updated notes")
    parser.add_argument("--api-key", help="Groq API key")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create the notes agent
    agent = NotesAgent(api_key)
    
    # Handle generate or update
    if args.sources and args.outline:
        # Generate new notes
        notes_path = agent.generate_notes(args.sources, args.outline, args.output_dir)
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
