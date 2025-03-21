#!/usr/bin/env python
"""
Update Agent - Specialized agent for updating existing academic notes with new information
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
PROCESSED_DIR = BASE_DIR / "processed"
NOTES_DIR = PROCESSED_DIR / "notes"


class UpdateAgent:
    """
    Agent specialized in updating existing academic notes with new information
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
    
    def update_notes(self, 
                    notes_path: str, 
                    new_sources: List[str], 
                    output_path: Optional[str] = None,
                    merge_style: str = "append") -> str:
        """
        Update existing notes with new information
        
        Args:
            notes_path: Path to existing notes file
            new_sources: List of paths to new source markdown files
            output_path: Path to save updated notes (optional)
            merge_style: How to merge new information ("append", "inline", "restructure")
            
        Returns:
            Path to the updated notes file
        """
        if not os.path.exists(notes_path):
            print(f"Error: Notes file not found: {notes_path}")
            return ""
            
        # Load existing notes
        with open(notes_path, 'r') as f:
            existing_notes = f.read()
            
        # Load new sources
        source_contents = {}
        for path in new_sources:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    source_contents[path] = f.read()
            else:
                print(f"Warning: Source file not found: {path}")
                
        if not source_contents:
            print("Error: No valid new source files found")
            return ""
            
        # Determine output path
        if not output_path:
            dirname = os.path.dirname(notes_path)
            basename = os.path.basename(notes_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(dirname, f"updated_{basename.split('.')[0]}_{timestamp}.md")
            
        # Update the notes based on the merge style
        if merge_style == "append":
            updated_notes = self._append_updates(existing_notes, source_contents)
        elif merge_style == "inline":
            updated_notes = self._inline_updates(existing_notes, source_contents)
        elif merge_style == "restructure":
            updated_notes = self._restructure_notes(existing_notes, source_contents)
        else:
            print(f"Warning: Unknown merge style '{merge_style}'. Using 'append' style.")
            updated_notes = self._append_updates(existing_notes, source_contents)
            
        # Save updated notes
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(updated_notes)
            
        return output_path
    
    def _append_updates(self, existing_notes: str, new_sources: Dict[str, str]) -> str:
        """
        Append new information to the end of existing notes
        
        Args:
            existing_notes: Existing notes content
            new_sources: Dictionary mapping file paths to their content
            
        Returns:
            Updated notes with appended information
        """
        # This is a placeholder for the actual update process
        # In a full implementation, this would use the LLM to intelligently
        # extract and append relevant new information
        
        updated_notes = existing_notes
        
        # Add updates header
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
    
    def _inline_updates(self, existing_notes: str, new_sources: Dict[str, str]) -> str:
        """
        Integrate new information inline with the existing structure
        
        Args:
            existing_notes: Existing notes content
            new_sources: Dictionary mapping file paths to their content
            
        Returns:
            Updated notes with inline information
        """
        # This is a placeholder for the actual update process
        # In a full implementation, this would use the LLM to intelligently
        # integrate new information within the existing structure
        
        # Parse the existing notes structure
        lines = existing_notes.split('\n')
        sections = []
        current_section = {"level": 0, "title": "", "content": [], "line_start": 0, "line_end": 0}
        
        # Very simple section detection
        for i, line in enumerate(lines):
            if line.startswith('#'):
                # New section found
                if current_section["title"]:
                    current_section["line_end"] = i - 1
                    sections.append(current_section)
                    
                level = len(line.split(' ')[0])
                title = line[level+1:]
                current_section = {
                    "level": level,
                    "title": title,
                    "content": [],
                    "line_start": i,
                    "line_end": len(lines) - 1
                }
            else:
                current_section["content"].append(line)
                
        # Add the last section
        if current_section["title"]:
            sections.append(current_section)
            
        # Placeholder for inline updates
        updated_lines = lines.copy()
        
        # Add an update marker to each section
        for section in sections:
            if section["level"] <= 2:  # Only add to main sections
                update_line = f"\n\n_[Updated: {datetime.now().strftime('%Y-%m-%d')}]_\n"
                insert_position = section["line_start"] + 1
                updated_lines.insert(insert_position, update_line)
                
                # Adjust positions for subsequent sections
                for s in sections:
                    if s["line_start"] > section["line_start"]:
                        s["line_start"] += 1
                        s["line_end"] += 1
        
        # Add attribution at the end
        updated_lines.append("\n\n---\n")
        updated_lines.append(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        updated_lines.append("\nNew information from:")
        for path in new_sources.keys():
            updated_lines.append(f"- {os.path.basename(path)}")
            
        return '\n'.join(updated_lines)
    
    def _restructure_notes(self, existing_notes: str, new_sources: Dict[str, str]) -> str:
        """
        Completely restructure the notes to incorporate new information
        
        Args:
            existing_notes: Existing notes content
            new_sources: Dictionary mapping file paths to their content
            
        Returns:
            Restructured notes
        """
        # This is a placeholder for the actual restructuring process
        # In a full implementation, this would use the LLM to completely
        # reorganize the notes based on all available information
        
        # Create a header for the restructured notes
        restructured = "# Restructured Academic Notes\n\n"
        restructured += f"_Regenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
        
        # Add sources section
        restructured += "## Sources\n\n"
        restructured += "### Original Notes\n"
        restructured += "- Original notes document\n\n"
        
        restructured += "### New Sources\n"
        for path in new_sources.keys():
            restructured += f"- {os.path.basename(path)}\n"
        restructured += "\n"
        
        # Add placeholder content sections
        restructured += "## Introduction\n\n"
        restructured += "This document has been restructured to incorporate new information.\n\n"
        
        restructured += "## Main Content\n\n"
        restructured += "The main content would be completely reorganized based on all available information.\n\n"
        
        restructured += "## Conclusion\n\n"
        restructured += "Updated conclusions based on all information sources.\n\n"
        
        return restructured
    
    def identify_changes(self, original_notes_path: str, updated_notes_path: str) -> Dict[str, Any]:
        """
        Identify and summarize the changes between original and updated notes
        
        Args:
            original_notes_path: Path to original notes file
            updated_notes_path: Path to updated notes file
            
        Returns:
            Dictionary summarizing the changes
        """
        if not os.path.exists(original_notes_path):
            return {"error": f"Original notes file not found: {original_notes_path}"}
            
        if not os.path.exists(updated_notes_path):
            return {"error": f"Updated notes file not found: {updated_notes_path}"}
            
        # Load both files
        with open(original_notes_path, 'r') as f:
            original = f.read()
            
        with open(updated_notes_path, 'r') as f:
            updated = f.read()
            
        # This is a placeholder for actual change detection
        # In a full implementation, this would use the LLM to identify
        # and summarize meaningful changes
        
        changes = {
            "original_file": original_notes_path,
            "updated_file": updated_notes_path,
            "timestamp": datetime.now().isoformat(),
            "summary": "Changes would be identified and summarized here.",
            "sections_added": [],
            "sections_modified": [],
            "sections_removed": []
        }
        
        return changes


def main():
    """Main entry point for the update agent"""
    parser = argparse.ArgumentParser(description="Update Agent for academic notes")
    
    # Define command line arguments
    parser.add_argument("--notes", help="Path to existing notes file to update")
    parser.add_argument("--sources", nargs='+', help="Paths to new source markdown files")
    parser.add_argument("--output", help="Custom output path for updated notes")
    parser.add_argument("--style", choices=["append", "inline", "restructure"], 
                      default="append", help="How to merge new information")
    parser.add_argument("--api-key", help="Groq API key")
    parser.add_argument("--compare", nargs=2, help="Compare original and updated notes")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create the update agent
    agent = UpdateAgent(api_key)
    
    # Process according to the provided arguments
    if args.compare:
        # Compare original and updated notes
        changes = agent.identify_changes(args.compare[0], args.compare[1])
        print("Changes identified:")
        print(json.dumps(changes, indent=2))
        
    elif args.notes and args.sources:
        # Update notes with new sources
        updated_path = agent.update_notes(
            args.notes, 
            args.sources, 
            args.output,
            args.style
        )
        
        if updated_path:
            print(f"Notes updated using '{args.style}' style")
            print(f"Updated notes saved to: {updated_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
