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
from typing import List, Dict, Any, Optional, Tuple
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
from tools.pdf_processor import DoclingProcessor

# Define base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROCESSED_DIR = BASE_DIR / "processed"
INGESTION_DIR = PROCESSED_DIR / "ingestion"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
OUTLINES_DIR = PROCESSED_DIR / "outlines"
NOTES_DIR = PROCESSED_DIR / "notes"


class PDFIngestionTool(Tool):
    """Tool for ingesting PDF files using docling processor"""
    
    name = "pdf_ingestion_tool"
    description = "Ingest PDF files and convert to markdown for further processing"
    
    def __init__(self, device: str = "mps"):
        self.processor = DoclingProcessor(device=device)
        
        # Ensure output directories exist
        os.makedirs(INGESTION_DIR, exist_ok=True)
        
    def __call__(self, 
                pdf_path: str, 
                output_dir: Optional[str] = None,
                rename_smartly: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Process a PDF file and extract its content
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Custom output directory (optional)
            rename_smartly: Whether to rename files based on content analysis
            
        Returns:
            Tuple of (path to the generated markdown file, metadata)
        """
        if not output_dir:
            output_dir = str(INGESTION_DIR)
            
        result, metadata = self.processor.process_pdf(pdf_path, output_dir, rename_smartly)
        return result, metadata


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
        # Ensure output directory exists
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        
        # Read the markdown file
        with open(markdown_path, 'r') as f:
            content = f.read()
            
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(markdown_path))[0]
        
        # Get info from metadata if it exists (from smart naming)
        metadata_file = os.path.join(ANALYSIS_DIR, f"{filename}_metadata.json")
        existing_metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            except Exception as e:
                print(f"Error loading existing metadata: {e}")
        
        # Combine existing metadata with basic analysis
        analysis = {
            "source_file": markdown_path,
            "analyzed_date": datetime.now().isoformat(),
            "existing_metadata": existing_metadata,
            # This would be expanded with actual analysis in a production system
            "analysis_result": {
                "word_count": len(content.split()),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
                "sections": self._extract_sections(content)
            }
        }
        
        # Save analysis result
        output_path = os.path.join(ANALYSIS_DIR, f"{filename}_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
        
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract section headings and their content"""
        sections = []
        
        # Simple regex to find markdown headings
        import re
        headings = re.findall(r'(#+)\s+(.*?)$', content, re.MULTILINE)
        
        for level, heading in headings:
            sections.append({
                "level": len(level),
                "heading": heading.strip(),
                "position": content.find(f"{level} {heading}")
            })
            
        return sections


class OutlineGenerationTool(Tool):
    """Tool for generating academic outlines from content"""
    
    name = "outline_generation_tool"
    description = "Generate structured outlines from academic content"
    
    def __call__(self, 
                markdown_paths: List[str], 
                output_path: str) -> str:
        """
        Generate an outline from multiple markdown files
        
        Args:
            markdown_paths: List of paths to markdown files
            output_path: Where to save the generated outline
            
        Returns:
            Path to the generated outline
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(OUTLINES_DIR, exist_ok=True)
        
        # Combine content from multiple files
        combined_content = ""
        files_info = []
        
        for path in markdown_paths:
            # Check for metadata from smart naming 
            filename = os.path.splitext(os.path.basename(path))[0]
            metadata_file = os.path.join(ANALYSIS_DIR, f"{filename}_metadata.json")
            file_info = {"path": path, "sequence_info": {}}
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        file_info["metadata"] = metadata
                        if "file_info" in metadata and "sequence_number" in metadata["file_info"]:
                            file_info["sequence_info"] = metadata["file_info"]
                except Exception as e:
                    print(f"Error loading metadata: {e}")
            
            # Read content
            with open(path, 'r') as f:
                content = f.read()
                file_info["content"] = content
                file_info["word_count"] = len(content.split())
                
            files_info.append(file_info)
            
        # Sort files by sequence number if available
        files_info.sort(key=lambda x: (
            x.get("sequence_info", {}).get("sequence_type", ""),
            x.get("sequence_info", {}).get("sequence_number", 999)
        ))
        
        # Create outline structure
        outline = {
            "title": "Generated Academic Outline",
            "date_generated": datetime.now().isoformat(),
            "source_files": [f["path"] for f in files_info],
            "sections": []
        }
        
        # Extract sections from each file and compile into outline
        for file_info in files_info:
            # Try to get sequence info
            seq_type = file_info.get("sequence_info", {}).get("sequence_type", "")
            seq_num = file_info.get("sequence_info", {}).get("sequence_number", "")
            
            # Try to get title from metadata or default to filename
            title = file_info.get("metadata", {}).get("file_info", {}).get("title", "")
            if not title:
                title = file_info.get("metadata", {}).get("llm_analysis", {}).get("Title", "")
            if not title:
                title = os.path.basename(file_info["path"])
                
            # Create section for this file
            section = {
                "title": title,
                "source_file": file_info["path"],
                "sequence_type": seq_type,
                "sequence_number": seq_num,
                "topics": []
            }
            
            # Extract headings to use as topics
            import re
            content = file_info["content"]
            headings = re.findall(r'(#+)\s+(.*?)$', content, re.MULTILINE)
            
            for level, heading in headings:
                if len(level) < 3:  # Only include top-level headings
                    section["topics"].append({
                        "title": heading.strip(),
                        "level": len(level)
                    })
            
            outline["sections"].append(section)
            
        # Save outline
        with open(output_path, 'w') as f:
            json.dump(outline, f, indent=2)
            
        print(f"Generated outline saved to {output_path}")
        return output_path


class NotesGenerationTool(Tool):
    """Tool for generating comprehensive notes from outlines and content"""
    
    name = "notes_generation_tool"
    description = "Generate comprehensive notes from outlines and academic content"
    
    def __call__(self, 
                outline_path: str, 
                markdown_paths: List[str], 
                output_path: str) -> str:
        """
        Generate comprehensive notes based on an outline and source content
        
        Args:
            outline_path: Path to the outline JSON file
            markdown_paths: List of paths to source markdown files
            output_path: Where to save the generated notes
            
        Returns:
            Path to the generated notes
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(NOTES_DIR, exist_ok=True)
        
        # Load the outline
        with open(outline_path, 'r') as f:
            outline = json.load(f)
            
        # Initialize notes with header
        notes = f"# {outline.get('title', 'Comprehensive Academic Notes')}\n\n"
        notes += f"*Generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        notes += "## Table of Contents\n\n"
        
        # Create table of contents
        for i, section in enumerate(outline.get("sections", [])):
            section_title = section.get("title", f"Section {i+1}")
            seq_type = section.get("sequence_type", "")
            seq_num = section.get("sequence_number", "")
            
            if seq_type and seq_num:
                section_title = f"{seq_type.capitalize()} {seq_num}: {section_title}"
                
            notes += f"- [{section_title}](#section-{i+1})\n"
            for j, topic in enumerate(section.get("topics", [])):
                notes += f"  - [{topic.get('title', f'Topic {j+1}')}](#section-{i+1}-topic-{j+1})\n"
        
        notes += "\n---\n\n"
        
        # Generate content for each section
        for i, section in enumerate(outline.get("sections", [])):
            section_title = section.get("title", f"Section {i+1}")
            seq_type = section.get("sequence_type", "")
            seq_num = section.get("sequence_number", "")
            
            if seq_type and seq_num:
                section_title = f"{seq_type.capitalize()} {seq_num}: {section_title}"
                
            notes += f"## <a id='section-{i+1}'></a>{section_title}\n\n"
            
            # Find the corresponding source file content
            source_file = section.get("source_file", "")
            source_content = ""
            
            for path in markdown_paths:
                if path == source_file or os.path.basename(path) == os.path.basename(source_file):
                    with open(path, 'r') as f:
                        source_content = f.read()
                    break
            
            # Add content for each topic
            for j, topic in enumerate(section.get("topics", [])):
                topic_title = topic.get("title", f"Topic {j+1}")
                notes += f"### <a id='section-{i+1}-topic-{j+1}'></a>{topic_title}\n\n"
                
                # Find the corresponding content for this topic in the source
                # This is a simplified approach - a more sophisticated version would
                # extract the actual content between headings
                import re
                topic_pattern = re.escape(topic_title)
                match = re.search(
                    f"#{{{topic.get('level', 1)}}}\\s+{topic_pattern}\\s*\n(.*?)(?=#{{{topic.get('level', 1)}}})|\Z",
                    source_content,
                    re.DOTALL
                )
                
                if match:
                    notes += match.group(1).strip() + "\n\n"
                else:
                    notes += "*No content found for this topic.*\n\n"
            
            notes += "---\n\n"
        
        # Save notes
        with open(output_path, 'w') as f:
            f.write(notes)
            
        print(f"Generated notes saved to {output_path}")
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
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read existing notes
        with open(notes_path, 'r') as f:
            notes_content = f.read()
            
        # Read new content from markdown files
        new_content = ""
        for path in new_markdown_paths:
            with open(path, 'r') as f:
                new_content += f"\n\n## New Content from {os.path.basename(path)}\n\n"
                new_content += f.read()
        
        # Append new content with a separator
        updated_notes = notes_content + "\n\n## Updates\n\n"
        updated_notes += f"*Updated on {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        updated_notes += new_content
        
        # Save updated notes
        with open(output_path, 'w') as f:
            f.write(updated_notes)
            
        print(f"Updated notes saved to {output_path}")
        return output_path


def setup_agent(api_key: str, device: str = "mps"):
    """
    Set up and configure the academic agent with all necessary tools
    
    Args:
        api_key: API key for Groq
        device: Device to use for PDF processing
        
    Returns:
        Configured CodeAgent
    """
    # Configure Groq model
    model = HfApiModel(
        model_id="llama3-70b-8192",
        provider="groq",
        api_key=api_key
    )
    
    # Create tools
    tools = [
        PDFIngestionTool(device=device),
        ContentAnalysisTool(),
        OutlineGenerationTool(),
        NotesGenerationTool(),
        ContentUpdateTool()
    ]
    
    # Create agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        verbose=True
    )
    
    return agent


def main():
    """Main entry point for the academic agent"""
    parser = argparse.ArgumentParser(
        description="Academic Agent for processing and analyzing academic materials"
    )
    
    # Define command line arguments
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument("--output", help="Base output directory", default=str(PROCESSED_DIR))
    parser.add_argument("--analyze", action="store_true", help="Analyze content after ingestion")
    parser.add_argument("--generate-outline", action="store_true", help="Generate outline from content")
    parser.add_argument("--generate-notes", action="store_true", help="Generate notes from outline and content")
    parser.add_argument("--update-notes", help="Path to existing notes to update")
    parser.add_argument("--api-key", help="Groq API key")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"], 
                       help="Device to use for PDF processing")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
        
    # Set up base directories
    ingestion_dir = os.path.join(args.output, "ingestion")
    analysis_dir = os.path.join(args.output, "analysis")
    outlines_dir = os.path.join(args.output, "outlines") 
    notes_dir = os.path.join(args.output, "notes")
    
    os.makedirs(ingestion_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(outlines_dir, exist_ok=True)
    os.makedirs(notes_dir, exist_ok=True)
    
    # Set up the agent
    agent = setup_agent(api_key, device=args.device)
    
    if args.interactive:
        # Simple interactive loop
        print("Academic Agent Interactive Mode")
        print("Type 'exit' or 'quit' to exit")
        
        while True:
            command = input("\nEnter command: ")
            
            if command.lower() in ["exit", "quit"]:
                break
                
            response = agent.run(command)
            print(f"\nResponse: {response}")
            
    else:
        # Process files based on command line arguments
        markdown_paths = []
        
        # Ingest PDFs if specified
        if args.pdf:
            print(f"Processing PDF: {args.pdf}")
            result, metadata = agent.tools["pdf_ingestion_tool"](args.pdf, ingestion_dir)
            if result:
                markdown_paths.append(result)
                print(f"Processed PDF: {result}")
                if metadata and metadata.get("sequence_number"):
                    seq_type = metadata.get("sequence_type", "")
                    seq_num = metadata.get("sequence_number", "")
                    print(f"Detected as {seq_type} {seq_num}")
                
        elif args.dir:
            print(f"Processing directory: {args.dir}")
            for root, _, files in os.walk(args.dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        result, metadata = agent.tools["pdf_ingestion_tool"](pdf_path, ingestion_dir)
                        if result:
                            markdown_paths.append(result)
                            print(f"Processed PDF: {result}")
        
        # Analyze content if requested
        if args.analyze and markdown_paths:
            print("Analyzing content...")
            for path in markdown_paths:
                analysis = agent.tools["content_analysis_tool"](path)
                print(f"Analyzed: {path}")
        
        # Generate outline if requested
        outline_path = None
        if args.generate_outline and markdown_paths:
            print("Generating outline...")
            outline_path = os.path.join(outlines_dir, "generated_outline.json")
            outline_path = agent.tools["outline_generation_tool"](markdown_paths, outline_path)
            print(f"Outline generated: {outline_path}")
        
        # Generate notes if requested
        if args.generate_notes and outline_path and markdown_paths:
            print("Generating notes...")
            notes_path = os.path.join(notes_dir, "comprehensive_notes.md")
            notes_path = agent.tools["notes_generation_tool"](outline_path, markdown_paths, notes_path)
            print(f"Notes generated: {notes_path}")
        
        # Update existing notes if requested
        if args.update_notes and markdown_paths:
            print(f"Updating notes: {args.update_notes}")
            updated_path = agent.tools["content_update_tool"](args.update_notes, markdown_paths)
            print(f"Notes updated: {updated_path}")


if __name__ == "__main__":
    main()
