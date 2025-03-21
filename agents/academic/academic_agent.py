#!/usr/bin/env python
"""
Academic Agent - An intelligent system for processing and synthesizing academic materials
using smolagents framework and Groq LLM.

This agent:
1. Ingests PDF data from various sources (lectures, transcripts, notes)
2. Analyzes and organizes content
3. Creates comprehensive outlines and notes
4. Updates materials as new information becomes available
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
    from smolagents import Tool
except ImportError:
    print("Installing smolagents...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents import HfApiModel
    from smolagents import Tool

# Add parent directory to path for importing the pdf_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from docling_processor.pdf_processor_agent import DoclingProcessor

# Define base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = BASE_DIR / "output"
PROCESSED_DIR = BASE_DIR / "processed"


class PDFIngestionTool(Tool):
    """Tool for ingesting PDF files using docling processor"""
    
    name = "pdf_ingestion_tool"
    description = "Ingest PDF files and convert to markdown for further processing"
    
    def __init__(self):
        self.processor = DoclingProcessor()
    
    def __call__(self, pdf_path: str, output_dir: Optional[str] = None) -> str:
        """
        Process a PDF file and convert it to markdown
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to the generated markdown file
        """
        if not output_dir:
            output_dir = str(OUTPUT_DIR)
            
        return self.processor.process_pdf(pdf_path, output_dir)


class ContentAnalysisTool(Tool):
    """Tool for analyzing academic content and identifying key concepts"""
    
    name = "content_analysis_tool"
    description = "Analyze academic content to identify topics, structure, and key concepts"
    
    def __call__(self, markdown_path: str) -> Dict[str, Any]:
        """
        Analyze a markdown file and extract key information
        
        Args:
            markdown_path: Path to the markdown file
            
        Returns:
            Dictionary of extracted information
        """
        if not os.path.exists(markdown_path):
            return {"error": f"File not found: {markdown_path}"}
            
        with open(markdown_path, 'r') as f:
            content = f.read()
            
        # This is a placeholder. In a complete implementation, the agent would
        # perform the actual analysis of the content
        return {
            "file_path": markdown_path,
            "file_size": os.path.getsize(markdown_path),
            "word_count": len(content.split()),
            "line_count": len(content.split('\n')),
            "has_images": "![" in content,
            "timestamp": datetime.now().isoformat()
        }


class OutlineGenerationTool(Tool):
    """Tool for generating academic outlines from content"""
    
    name = "outline_generation_tool"
    description = "Generate structured outlines from academic content"
    
    def __call__(self, markdown_paths: List[str], output_path: str) -> str:
        """
        Generate an outline from multiple markdown files
        
        Args:
            markdown_paths: List of paths to markdown files
            output_path: Where to save the generated outline
            
        Returns:
            Path to the generated outline
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # This is a placeholder. In a complete implementation, the agent would
        # generate the actual outline based on the content
        outline = {
            "title": "Generated Outline",
            "sources": markdown_paths,
            "sections": [
                {
                    "title": "Introduction",
                    "subsections": []
                },
                {
                    "title": "Main Concepts",
                    "subsections": []
                },
                {
                    "title": "Detailed Analysis",
                    "subsections": []
                },
                {
                    "title": "Conclusion",
                    "subsections": []
                }
            ],
            "generated": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(outline, f, indent=2)
            
        return output_path


class NotesGenerationTool(Tool):
    """Tool for generating comprehensive notes from outlines and content"""
    
    name = "notes_generation_tool"
    description = "Generate comprehensive notes from outlines and academic content"
    
    def __call__(self, outline_path: str, markdown_paths: List[str], output_path: str) -> str:
        """
        Generate comprehensive notes based on an outline and source content
        
        Args:
            outline_path: Path to the outline JSON file
            markdown_paths: List of paths to source markdown files
            output_path: Where to save the generated notes
            
        Returns:
            Path to the generated notes
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load the outline
        with open(outline_path, 'r') as f:
            outline = json.load(f)
        
        # This is a placeholder. In a complete implementation, the agent would
        # generate the actual notes based on the outline and content
        with open(output_path, 'w') as f:
            f.write(f"# {outline['title']}\n\n")
            f.write("## Generated Notes\n\n")
            f.write(f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
            
            for section in outline["sections"]:
                f.write(f"## {section['title']}\n\n")
                f.write("Content would be generated here based on the source materials.\n\n")
        
        return output_path


class ContentUpdateTool(Tool):
    """Tool for updating notes with new information"""
    
    name = "content_update_tool"
    description = "Update existing notes with new information from additional sources"
    
    def __call__(self, 
                notes_path: str, 
                new_markdown_paths: List[str], 
                output_path: Optional[str] = None) -> str:
        """
        Update existing notes with new information
        
        Args:
            notes_path: Path to existing notes file
            new_markdown_paths: List of paths to new source markdown files
            output_path: Where to save the updated notes (defaults to overwriting existing)
            
        Returns:
            Path to the updated notes
        """
        if not output_path:
            output_path = notes_path
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load existing notes
        with open(notes_path, 'r') as f:
            existing_notes = f.read()
        
        # This is a placeholder. In a complete implementation, the agent would
        # update the notes with new information
        with open(output_path, 'w') as f:
            f.write(existing_notes)
            f.write("\n\n## Updates\n\n")
            f.write(f"_Updated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
            f.write("New information would be integrated here.\n\n")
            f.write(f"Sources: {', '.join(new_markdown_paths)}\n")
        
        return output_path


def setup_agent(api_key: str):
    """
    Set up and configure the academic agent with all necessary tools
    
    Args:
        api_key: API key for Groq
        
    Returns:
        Configured CodeAgent
    """
    # Set up the model (using Groq)
    model = HfApiModel(
        model_id="llama3-70b-8192", 
        provider="groq",
        api_key=api_key
    )
    
    # Set up tools
    tools = [
        PDFIngestionTool(),
        ContentAnalysisTool(),
        OutlineGenerationTool(),
        NotesGenerationTool(),
        ContentUpdateTool()
    ]
    
    # Create the agent
    agent = CodeAgent(
        tools=tools,
        model=model,
    )
    
    return agent


def main():
    """Main entry point for the academic agent"""
    parser = argparse.ArgumentParser(description="Academic Agent for processing academic materials")
    
    # Define command line arguments
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze processed content")
    parser.add_argument("--generate-outline", action="store_true", help="Generate an outline")
    parser.add_argument("--generate-notes", action="store_true", help="Generate comprehensive notes")
    parser.add_argument("--update", help="Update existing notes with new content")
    parser.add_argument("--api-key", help="Groq API key")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Set up the agent
    agent = setup_agent(api_key)
    
    # Interactive mode
    if args.interactive:
        print("Starting Academic Agent in interactive mode.")
        print("Enter 'exit' to quit.")
        
        while True:
            prompt = input("\nEnter your query: ")
            if prompt.lower() == 'exit':
                break
                
            response = agent.run(prompt)
            print(f"\nAgent Response:\n{response}")
        
        return
    
    # Process PDF
    if args.pdf or args.dir:
        prompt = f"Process the following academic material"
        if args.pdf:
            prompt += f" from PDF at {args.pdf}"
        else:
            prompt += f" from PDFs in directory {args.dir}"
            
        prompt += f" and save to {args.output}."
        
        if args.analyze:
            prompt += " After processing, analyze the content to identify key concepts and structure."
        
        if args.generate_outline:
            prompt += " Then generate a detailed outline of the material."
        
        if args.generate_notes:
            prompt += " Finally, create comprehensive notes based on the outline and content."
        
        response = agent.run(prompt)
        print(response)
    
    # Update existing notes
    elif args.update:
        prompt = f"Update the existing notes at {args.update} with new information."
        response = agent.run(prompt)
        print(response)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
