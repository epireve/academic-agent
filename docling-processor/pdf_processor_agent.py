#!/usr/bin/env python3
"""
PDF Processor Agent

This script provides functionality to process PDF files and convert them to markdown
with enhanced metadata handling and image extraction.
"""

import os
import argparse
import subprocess
import json
import re
from typing import Dict, List, Optional

# Base output directory for all converted files
OUTPUT_DIR = "/Users/invoture/dev.local/academic-agent/data/output"

class PDFProcessor:
    """
    Simple agent for processing PDF files and converting them to markdown
    with metadata handling and image extraction.
    """
    
    def process_pdf(self, 
                    pdf_path: str, 
                    output_dir: str = OUTPUT_DIR, 
                    course: str = "sra", 
                    category: str = "lectures", 
                    use_mps: bool = True) -> str:
        """
        Process a single PDF file, converting it to markdown with images

        Args:
            pdf_path: Path to the PDF file
            output_dir: Base directory for output
            course: Course category
            category: Content category (lectures, notes, transcripts)
            use_mps: Whether to use MPS (Metal Performance Shaders) for acceleration

        Returns:
            Path to the generated markdown file
        """
        full_output_dir = os.path.join(output_dir, course, category, "markdown")
        images_dir = os.path.join(output_dir, course, category, "images")
        
        # Create directories if they don't exist
        os.makedirs(full_output_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        
        # Get the base filename without extension
        filename = os.path.basename(pdf_path)
        base_filename = os.path.splitext(filename)[0]
        
        # Prepare the docling command
        cmd = [
            "docling",
            "--to", "md",
            "--output", full_output_dir,
            "--image-export-mode", "referenced",
            "--enrich-picture-classes",
            "--enrich-formula"
        ]
        
        # Add MPS device if requested
        if use_mps:
            cmd.extend(["--device", "mps"])
        
        # Add the PDF path
        cmd.append(pdf_path)
        
        # Execute the command
        subprocess.run(cmd, check=True)
        
        # Return the path to the markdown file
        return os.path.join(full_output_dir, f"{base_filename}.md")
    
    def process_directory(self, 
                         dir_path: str, 
                         output_dir: str = OUTPUT_DIR, 
                         course: str = "sra", 
                         category: str = "lectures", 
                         use_mps: bool = True) -> List[str]:
        """
        Process all PDF files in a directory

        Args:
            dir_path: Path to directory containing PDFs
            output_dir: Base directory for output
            course: Course category
            category: Content category (lectures, notes, transcripts)
            use_mps: Whether to use MPS for acceleration

        Returns:
            List of paths to generated markdown files
        """
        results = []
        
        # Process all PDF files in the directory
        for filename in os.listdir(dir_path):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(dir_path, filename)
                result_path = self.process_pdf(
                    pdf_path,
                    output_dir,
                    course,
                    category,
                    use_mps
                )
                results.append(result_path)
                print(f"Processed: {filename} -> {result_path}")
        
        return results
    
    def extract_metadata_from_path(self, file_path: str) -> Dict[str, str]:
        """
        Try to extract course, and category information from a filepath
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Dictionary with course, category if they could be extracted
        """
        path_lower = file_path.lower()
        
        # Default values
        metadata = {
            "course": "sra",    # Default
            "category": "lectures"  # Default
        }
        
        # Try to detect course from path
        if "security" in path_lower or "risk" in path_lower or "sra" in path_lower:
            metadata["course"] = "sra"
        elif "cyber" in path_lower:
            metadata["course"] = "cyber"
        elif "data" in path_lower and "science" in path_lower:
            metadata["course"] = "ds"
        elif "ai" in path_lower or "artificial intelligence" in path_lower:
            metadata["course"] = "ai"
        
        # Try to detect category from path
        if any(term in path_lower for term in ["lecture", "slide", "presentation"]):
            metadata["category"] = "lectures"
        elif any(term in path_lower for term in ["note", "summary"]):
            metadata["category"] = "notes"
        elif any(term in path_lower for term in ["transcript", "recording"]):
            metadata["category"] = "transcripts"
            
        return metadata

def main():
    parser = argparse.ArgumentParser(description="Process PDF files and convert to markdown with metadata")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to a PDF file")
    group.add_argument("--dir", help="Path to a directory of PDF files")
    
    parser.add_argument("--course", default="sra", help="Course category")
    parser.add_argument("--category", default="lectures", help="Content category (lectures, notes, transcripts)")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Base output directory")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of MPS")
    parser.add_argument("--auto-metadata", action="store_true", help="Auto-detect metadata from filepath")
    
    args = parser.parse_args()
    
    processor = PDFProcessor()
    
    # Auto-detect metadata if requested
    if args.auto_metadata:
        file_path = args.pdf if args.pdf else args.dir
        metadata = processor.extract_metadata_from_path(file_path)
        course = metadata["course"]
        category = metadata["category"]
        print(f"Auto-detected metadata: course={course}, category={category}")
    else:
        course = args.course
        category = args.category
    
    use_mps = not args.cpu
    
    if args.pdf:
        output_path = processor.process_pdf(
            args.pdf,
            args.output,
            course,
            category,
            use_mps
        )
        print(f"Processed PDF: {args.pdf}")
        print(f"Output: {output_path}")
    else:
        output_paths = processor.process_directory(
            args.dir,
            args.output,
            course,
            category,
            use_mps
        )
        print(f"Processed {len(output_paths)} PDF files from {args.dir}")
        for path in output_paths:
            print(f"- {path}")

if __name__ == "__main__":
    main()
