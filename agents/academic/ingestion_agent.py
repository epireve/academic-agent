#!/usr/bin/env python
"""
Ingestion Agent - Specialized agent for ingesting and processing PDF files
Part of the Academic Agent system
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
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
from tools.pdf_processor import DoclingProcessor

# Define base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = BASE_DIR / "processed" / "ingestion"
ANALYSIS_DIR = BASE_DIR / "processed" / "analysis"


class IngestionAgent:
    """
    Agent specifically designed for ingesting PDFs from various sources
    and preparing them for further analysis
    """
    
    def __init__(self, api_key: str, device: str = "mps"):
        """
        Initialize the ingestion agent
        
        Args:
            api_key: Groq API key for LLM access
            device: Device to use for PDF processing (cpu, cuda, mps)
        """
        self.processor = DoclingProcessor(device=device)
        
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
        
        # Ensure output directories exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process a single PDF file and extract its content
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Custom output directory (optional)
            
        Returns:
            Tuple of (path to the generated markdown file, extracted metadata)
        """
        if not output_dir:
            output_dir = str(OUTPUT_DIR)
            
        # Use the docling processor to convert PDF to markdown with smart naming
        markdown_path, auto_metadata = self.processor.process_pdf(pdf_path, output_dir)
        
        # Extract additional metadata using the LLM if conversion was successful
        if markdown_path:
            llm_metadata = self.extract_metadata(markdown_path)
            
            # Combine automatically extracted metadata with LLM-extracted metadata
            combined_metadata = {
                "file_info": auto_metadata,
                "llm_analysis": llm_metadata,
                "markdown_path": markdown_path
            }
            
            # Save combined metadata
            metadata_file = os.path.join(
                ANALYSIS_DIR, 
                f"{os.path.splitext(os.path.basename(markdown_path))[0]}_metadata.json"
            )
            
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            
            with open(metadata_file, 'w') as f:
                json.dump(combined_metadata, f, indent=2)
                
            print(f"Saved combined metadata to {metadata_file}")
            
            return markdown_path, combined_metadata
        
        return "", {}
    
    def process_directory(self, dir_path: str, output_dir: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Process all PDF files in a directory
        
        Args:
            dir_path: Path to directory containing PDFs
            output_dir: Custom output directory (optional)
            
        Returns:
            List of tuples (path to generated markdown file, metadata)
        """
        if not output_dir:
            output_dir = str(OUTPUT_DIR)
            
        # Use the processor's directory processing functionality directly
        # as it now handles smart naming and metadata extraction
        results = self.processor.process_directory(dir_path, output_dir)
        
        # Process each result with our LLM to enhance metadata
        enhanced_results = []
        
        for markdown_path, auto_metadata in results:
            llm_metadata = self.extract_metadata(markdown_path)
            
            # Combine automatically extracted metadata with LLM-extracted metadata
            combined_metadata = {
                "file_info": auto_metadata,
                "llm_analysis": llm_metadata,
                "markdown_path": markdown_path
            }
            
            # Save combined metadata
            metadata_file = os.path.join(
                ANALYSIS_DIR, 
                f"{os.path.splitext(os.path.basename(markdown_path))[0]}_metadata.json"
            )
            
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            
            with open(metadata_file, 'w') as f:
                json.dump(combined_metadata, f, indent=2)
                
            enhanced_results.append((markdown_path, combined_metadata))
        
        return enhanced_results
    
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
            
        # Use a sample of the content (first 4000 chars) to avoid token limits
        content_sample = content[:4000]
        
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
        4. Main course or module number if available
        5. Key topics covered (list of 3-5 topics)
        6. Estimated academic level (undergraduate, graduate, etc.)
        7. Lecture number or week number if mentioned
        8. Key concepts or terminology introduced
        
        Return a clean, well-formatted JSON object.
        """
        
        response = self.agent.run(prompt)
        
        # Attempt to extract JSON from the response
        try:
            # Find JSON in the response - look for content between ```json and ``` or just extract anything that looks like JSON
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            
            if json_match:
                metadata_str = json_match.group(1)
            else:
                # Try to find anything that looks like a JSON object
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    metadata_str = json_match.group(1)
                else:
                    metadata_str = "{}"
            
            metadata = json.loads(metadata_str)
            
            # Add the extraction status
            metadata["extraction_successful"] = True
            
        except Exception as e:
            print(f"Error parsing metadata JSON: {e}")
            # Fallback metadata
            metadata = {
                "extraction_successful": False,
                "raw_response": response,
                "error": str(e)
            }
        
        return metadata


def main():
    """Main entry point for the ingestion agent"""
    parser = argparse.ArgumentParser(description="Ingestion Agent for processing PDF files")
    
    # Define command line arguments
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument("--output", help="Output directory (defaults to processed/ingestion)")
    parser.add_argument("--api-key", help="Groq API key")
    parser.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"], 
                       help="Device to use for PDF processing")
    parser.add_argument("--no-smart-rename", action="store_true",
                       help="Disable smart file renaming")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create the ingestion agent with specified device
    agent = IngestionAgent(api_key, device=args.device)
    
    # Process PDF or directory
    if args.pdf:
        output_dir = args.output or str(OUTPUT_DIR)
        result, metadata = agent.process_pdf(args.pdf, output_dir)
        if result:
            print(f"Processed PDF: {result}")
            if metadata.get("file_info", {}).get("sequence_number"):
                seq_type = metadata.get("file_info", {}).get("sequence_type", "")
                seq_num = metadata.get("file_info", {}).get("sequence_number", "")
                print(f"Detected as {seq_type} {seq_num}")
        else:
            print("Processing failed")
    elif args.dir:
        output_dir = args.output or str(OUTPUT_DIR)
        results = agent.process_directory(args.dir, output_dir)
        print(f"Processed {len(results)} PDF files")
        for path, metadata in results:
            seq_info = ""
            if metadata.get("file_info", {}).get("sequence_number"):
                seq_type = metadata.get("file_info", {}).get("sequence_type", "")
                seq_num = metadata.get("file_info", {}).get("sequence_number", "")
                seq_info = f" ({seq_type} {seq_num})"
            print(f"- {path}{seq_info}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
