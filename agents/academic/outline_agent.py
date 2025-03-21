#!/usr/bin/env python
"""
Outline Agent - Specialized agent for creating structured outlines from academic content
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
ANALYSIS_DIR = PROCESSED_DIR / "analysis"


class OutlineAgent:
    """
    Agent specialized in creating structured outlines from academic content
    based on content analysis
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
        os.makedirs(OUTLINES_DIR, exist_ok=True)
        
    def create_outline(self, markdown_paths: List[str], 
                      analysis_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive outline from one or more academic documents
        
        Args:
            markdown_paths: Paths to markdown files
            analysis_path: Path to an existing analysis file (optional)
            
        Returns:
            Dictionary representing the outline structure
        """
        # Concatenate key parts of the content to analyze
        combined_content = ""
        for path in markdown_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    # Extract first 2000 chars per file to stay within token limits
                    content = f.read()
                    combined_content += content[:2000] + "\n\n---\n\n"
            
        # Load analysis if available
        analysis_data = {}
        if analysis_path and os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                try:
                    analysis_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse analysis file: {analysis_path}")
        
        # Prepare the prompt for the LLM
        prompt = f"""
        Create a comprehensive academic outline from the following content:
        
        ```
        {combined_content}
        ```
        
        {"Additional analysis information:" if analysis_data else ""}
        {json.dumps(analysis_data, indent=2) if analysis_data else ""}
        
        Please generate a detailed hierarchical outline that:
        1. Covers all main topics and subtopics
        2. Uses logical grouping and sequencing
        3. Follows academic conventions for outline structure
        4. Includes appropriate depth (typically 3-4 levels)
        5. Maintains consistent formatting
        
        Return the outline as a structured JSON object with these fields:
        - title: A descriptive title for the outline
        - subject: Main academic subject
        - sections: Array of top-level sections, each with:
          - title: Section title
          - key_points: Array of key points to cover
          - subsections: Array of subsections with the same structure
        """
        
        response = self.agent.run(prompt)
        
        # In a full implementation, we would parse the JSON from the response
        # For now, we'll create a placeholder outline structure
        
        # Generate a filename for the outline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(markdown_paths) == 1:
            base_name = os.path.basename(markdown_paths[0]).replace('.md', '')
            outline_filename = f"{base_name}_outline_{timestamp}.json"
        else:
            outline_filename = f"combined_outline_{timestamp}.json"
            
        output_path = OUTLINES_DIR / outline_filename
        
        # Create a placeholder outline structure
        outline = {
            "title": "Academic Outline",
            "subject": "Subject Area",
            "created": datetime.now().isoformat(),
            "sources": markdown_paths,
            "sections": [
                {
                    "title": "Introduction",
                    "key_points": ["Background", "Context", "Importance"],
                    "subsections": []
                },
                {
                    "title": "Core Concepts",
                    "key_points": ["Fundamental principles", "Key theories"],
                    "subsections": [
                        {
                            "title": "Concept 1",
                            "key_points": ["Definition", "Applications"],
                            "subsections": []
                        },
                        {
                            "title": "Concept 2",
                            "key_points": ["Definition", "Applications"],
                            "subsections": []
                        }
                    ]
                },
                {
                    "title": "Applications",
                    "key_points": ["Real-world examples", "Case studies"],
                    "subsections": []
                },
                {
                    "title": "Conclusion",
                    "key_points": ["Summary", "Implications"],
                    "subsections": []
                }
            ]
        }
        
        # Save the outline
        with open(output_path, 'w') as f:
            json.dump(outline, f, indent=2)
            
        return outline
    
    def generate_markdown_outline(self, outline_data: Dict[str, Any], output_path: str) -> str:
        """
        Convert a JSON outline structure to a formatted Markdown document
        
        Args:
            outline_data: The outline structure as a dictionary
            output_path: Where to save the Markdown outline
            
        Returns:
            Path to the generated Markdown file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write the title
            f.write(f"# {outline_data.get('title', 'Academic Outline')}\n\n")
            
            # Write metadata
            f.write(f"Subject: {outline_data.get('subject', 'Academic Subject')}\n")
            f.write(f"Created: {outline_data.get('created', datetime.now().isoformat())}\n\n")
            
            # Write sources if available
            sources = outline_data.get('sources', [])
            if sources:
                f.write("## Sources\n\n")
                for source in sources:
                    f.write(f"- {source}\n")
                f.write("\n")
            
            # Write the sections recursively
            
            def write_section(section, level=2):
                # Write section title
                f.write(f"{'#' * level} {section.get('title', 'Section')}\n\n")
                
                # Write key points if available
                key_points = section.get('key_points', [])
                if key_points:
                    for point in key_points:
                        f.write(f"- {point}\n")
                    f.write("\n")
                
                # Write subsections recursively
                subsections = section.get('subsections', [])
                for subsection in subsections:
                    write_section(subsection, level + 1)
            
            # Process each top-level section
            for section in outline_data.get('sections', []):
                write_section(section)
        
        return output_path


def main():
    """Main entry point for the outline agent"""
    parser = argparse.ArgumentParser(description="Outline Agent for creating academic outlines")
    
    # Define command line arguments
    parser.add_argument("--files", nargs='+', help="Paths to markdown files to outline")
    parser.add_argument("--analysis", help="Path to existing analysis file (optional)")
    parser.add_argument("--output", help="Custom output path for the outline")
    parser.add_argument("--markdown", action="store_true", help="Also generate a markdown version of the outline")
    parser.add_argument("--api-key", help="Groq API key")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Check for required arguments
    if not args.files:
        print("Error: At least one markdown file must be specified with --files")
        parser.print_help()
        sys.exit(1)
    
    # Create the outline agent
    agent = OutlineAgent(api_key)
    
    # Create the outline
    outline = agent.create_outline(args.files, args.analysis)
    
    # Determine output path for the JSON outline
    if args.output:
        output_path = args.output
    else:
        # Generate a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(args.files) == 1:
            base_name = os.path.basename(args.files[0]).replace('.md', '')
            outline_filename = f"{base_name}_outline_{timestamp}.json"
        else:
            outline_filename = f"combined_outline_{timestamp}.json"
        output_path = str(OUTLINES_DIR / outline_filename)
    
    # Save the outline in JSON format
    with open(output_path, 'w') as f:
        json.dump(outline, f, indent=2)
    
    print(f"Outline created and saved to: {output_path}")
    
    # Generate markdown version if requested
    if args.markdown:
        markdown_path = output_path.replace('.json', '.md')
        agent.generate_markdown_outline(outline, markdown_path)
        print(f"Markdown outline saved to: {markdown_path}")


if __name__ == "__main__":
    main()
