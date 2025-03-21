#!/usr/bin/env python
"""
Ingestion Agent - Specialized agent for ingesting and processing PDF files
Part of the Academic Agent system
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
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

# Add parent directory to path for importing the pdf_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from docling_processor.pdf_processor_agent import DoclingProcessor

# Define base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = BASE_DIR / "output"


class IngestionAgent:
    """
    Agent specifically designed for ingesting PDFs from various sources
    and preparing them for further analysis
    """
    
    def __init__(self, api_key: str):
        self.processor = DoclingProcessor()
        
        # Configure Groq model
        self.model = HfApiModel(
            model_id="llama3-70b-8192",
            provider="groq",
            api_key=api_key
        )
        
        # Create an agent for metadata extraction and classification
        self.agent = CodeAgent(
            tools=[],  # No special tools needed for this agent
            model=self.model
        )
        
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> str:
        """
        Process a single PDF file and extract its content
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Custom output directory (optional)
            
        Returns:
            Path to the generated markdown file
        """
        if not output_dir:
            output_dir = str(OUTPUT_DIR)
            
        # Use the docling processor to convert PDF to markdown
        markdown_path = self.processor.process_pdf(pdf_path, output_dir)
        
        # Extract additional metadata using the LLM
        if markdown_path:
            self.extract_metadata(markdown_path)
            
        return markdown_path
    
    def process_directory(self, dir_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Process all PDF files in a directory
        
        Args:
            dir_path: Path to directory containing PDFs
            output_dir: Custom output directory (optional)
            
        Returns:
            List of paths to generated markdown files
        """
        if not output_dir:
            output_dir = str(OUTPUT_DIR)
            
        results = []
        
        # Process each PDF in the directory
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    result = self.process_pdf(pdf_path, output_dir)
                    if result:
                        results.append(result)
                        
        return results
    
    def extract_metadata(self, markdown_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a processed markdown file using the LLM
        
        Args:
            markdown_path: Path to the markdown file
            
        Returns:
            Dictionary of extracted metadata
        """
        # Read the markdown file
        with open(markdown_path, 'r') as f:
            content = f.read()
            
        # Use a sample of the content (first 3000 chars) to avoid token limits
        content_sample = content[:3000]
        
        # Ask the LLM to extract metadata
        prompt = f"""
        Extract key metadata from this academic document. Analyze the following content sample:
        
        ```
        {content_sample}
        ```
        
        Extract and return the following as a JSON structure:
        1. Title of the document
        2. Subject/course area (e.g., "Security Risk Analysis", "Machine Learning")
        3. Content type (lecture, note, transcript)
        4. Key topics covered (list)
        5. Estimated academic level (undergraduate, graduate, etc.)
        """
        
        response = self.agent.run(prompt)
        
        # In a full implementation, we would parse the JSON response
        # For now, just log that we've extracted metadata
        print(f"Extracted metadata for {markdown_path}")
        
        # This is a placeholder for the metadata dict
        metadata = {
            "file": markdown_path,
            "extraction_successful": True
        }
        
        return metadata


def main():
    """Main entry point for the ingestion agent"""
    parser = argparse.ArgumentParser(description="Ingestion Agent for processing PDF files")
    
    # Define command line arguments
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--api-key", help="Groq API key")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create the ingestion agent
    agent = IngestionAgent(api_key)
    
    # Process PDF or directory
    if args.pdf:
        result = agent.process_pdf(args.pdf, args.output)
        print(f"Processed PDF: {result}")
    elif args.dir:
        results = agent.process_directory(args.dir, args.output)
        print(f"Processed {len(results)} PDF files")
        for path in results:
            print(f"- {path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
