#!/usr/bin/env python
"""
PDF Processor Agent using smolagents framework
This agent uses docling to convert PDF files to Markdown while maintaining rich content
"""

import os
import sys
import subprocess
from typing import List, Dict, Any, Optional
import argparse

try:
    from smolagents import CodeAgent
    from smolagents.models import OpenAIModel
except ImportError:
    print("smolagents not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents.models import OpenAIModel

# Define tools for our PDF processing agent
class DoclingProcessor:
    """Tool for processing PDFs with docling"""
    
    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str, 
                               image_mode: str = "referenced",
                               enrich_pictures: bool = True,
                               enrich_formulas: bool = True) -> str:
        """
        Convert a PDF to Markdown using docling
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory where to save the markdown file
            image_mode: How to handle images (referenced, embedded, placeholder)
            enrich_pictures: Whether to enable picture classification
            enrich_formulas: Whether to enable formula enrichment
            
        Returns:
            Path to the generated markdown file
        """
        if not os.path.exists(pdf_path):
            return f"Error: PDF file {pdf_path} does not exist"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Build the docling command
        cmd = ["docling", "--to", "md", "--output", output_dir, "--device", "mps", 
              f"--image-export-mode={image_mode}"]
        
        if enrich_pictures:
            cmd.append("--enrich-picture-classes")
        
        if enrich_formulas:
            cmd.append("--enrich-formula")
        
        cmd.append(pdf_path)
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            filename = os.path.basename(pdf_path)
            filename_no_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{filename_no_ext}.md")
            return f"Converted {pdf_path} to {output_path}"
        except subprocess.CalledProcessError as e:
            return f"Error converting PDF: {e.stderr}"

    def process_directory(self, source_dir: str, output_dir: str) -> List[str]:
        """
        Process all PDFs in a directory
        
        Args:
            source_dir: Directory containing PDF files
            output_dir: Directory where to save markdown files
            
        Returns:
            List of results
        """
        if not os.path.exists(source_dir):
            return [f"Error: Source directory {source_dir} does not exist"]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        results = []
        
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    result = self.convert_pdf_to_markdown(pdf_path, output_dir)
                    results.append(result)
        
        return results

def main():
    """Main function to set up and run the agent"""
    parser = argparse.ArgumentParser(description="PDF Processor Agent")
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument("--output", required=True, help="Output directory for markdown files")
    parser.add_argument("--api-key", help="OpenAI API key (if using OpenAI)")
    
    args = parser.parse_args()
    
    # Define our tools
    tools = [DoclingProcessor()]
    
    # For now, let's just use the tools directly without an agent
    processor = DoclingProcessor()
    
    if args.pdf:
        result = processor.convert_pdf_to_markdown(args.pdf, args.output)
        print(result)
    elif args.dir:
        results = processor.process_directory(args.dir, args.output)
        for result in results:
            print(result)
    else:
        print("Error: Please provide either --pdf or --dir argument")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
