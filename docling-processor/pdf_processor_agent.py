#!/usr/bin/env python
"""
PDF Processing Tool using docling
Converts PDF files to Markdown with enhanced image handling
"""

import os
import sys
import subprocess
import argparse
from typing import List

class DoclingProcessor:
    """Simple wrapper for docling PDF processing"""
    
    def process_pdf(self, pdf_path: str, output_dir: str) -> str:
        """
        Convert a PDF to Markdown using docling with enhanced image handling
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory for markdown and images
            
        Returns:
            Path to the generated markdown file
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file {pdf_path} does not exist")
            return ""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Build the docling command with default enriched settings
        cmd = [
            "docling",
            "--to", "md",
            "--output", output_dir,
            "--device", "mps",
            "--image-export-mode", "referenced",
            "--enrich-picture-classes",
            "--enrich-formula",
            pdf_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            output_file = os.path.join(output_dir, f"{base_filename}.md")
            print(f"Successfully converted {pdf_path} to {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error converting PDF: {str(e)}")
            return ""
    
    def process_directory(self, dir_path: str, output_dir: str) -> List[str]:
        """
        Process all PDF files in a directory
        
        Args:
            dir_path: Path to directory containing PDFs
            output_dir: Output directory for markdown and images
            
        Returns:
            List of paths to generated markdown files
        """
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist")
            return []
        
        results = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    result = self.process_pdf(pdf_path, output_dir)
                    if result:
                        results.append(result)
        
        return results

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown with enhanced image handling"
    )
    parser.add_argument(
        "--pdf",
        help="Path to a single PDF file to convert"
    )
    parser.add_argument(
        "--dir",
        help="Path to directory containing PDFs to convert"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for markdown files and images"
    )
    
    args = parser.parse_args()
    
    if not args.pdf and not args.dir:
        parser.error("Either --pdf or --dir must be specified")
    
    processor = DoclingProcessor()
    
    if args.pdf:
        processor.process_pdf(args.pdf, args.output)
    elif args.dir:
        processor.process_directory(args.dir, args.output)

if __name__ == "__main__":
    main()
