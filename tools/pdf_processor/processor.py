#!/usr/bin/env python
"""
PDF Processing Tool using docling
Converts PDF files to Markdown with enhanced image handling and smart content analysis
"""

import os
import subprocess
import re
import json
from typing import List, Dict, Tuple, Optional


class DoclingProcessor:
    """
    Advanced wrapper for docling PDF processing with content analysis capabilities
    for smart file naming and organization
    """
    
    def __init__(self, device: str = "mps"):
        """
        Initialize the processor with configuration
        
        Args:
            device: Device to use for PDF processing (cpu, cuda, mps)
        """
        self.device = device
        
    def _extract_sequence_info(self, pdf_path: str, content: str) -> Dict[str, any]:
        """
        Extract sequence information from PDF filename and content
        
        Args:
            pdf_path: Path to the PDF file
            content: Content of the generated markdown file
            
        Returns:
            Dictionary with sequence information
        """
        filename = os.path.basename(pdf_path)
        info = {
            "original_filename": filename,
            "sequence_number": None,
            "sequence_type": None,
            "title": None,
            "keywords": []
        }
        
        # Check filename for patterns like "Lecture 1", "Week 2", etc.
        filename_patterns = [
            (r'lecture[_\s-]*(\d+)', 'lecture'),
            (r'week[_\s-]*(\d+)', 'week'),
            (r'chapter[_\s-]*(\d+)', 'chapter'),
            (r'session[_\s-]*(\d+)', 'session'),
            (r'module[_\s-]*(\d+)', 'module'),
            (r'part[_\s-]*(\d+)', 'part'),
            (r'unit[_\s-]*(\d+)', 'unit'),
            (r'(\d+)[_\s-]', None)  # Generic number pattern
        ]
        
        # Check content for sequence patterns
        content_patterns = [
            (r'lecture\s*(\d+)', 'lecture'),
            (r'week\s*(\d+)', 'week'),
            (r'chapter\s*(\d+)', 'chapter'),
            (r'session\s*(\d+)', 'session'),
            (r'module\s*(\d+)', 'module')
        ]
        
        # First try to extract from filename
        for pattern, seq_type in filename_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                info["sequence_number"] = int(match.group(1))
                info["sequence_type"] = seq_type or 'part'
                break
        
        # If not found in filename, try content
        if not info["sequence_number"] and content:
            for pattern, seq_type in content_patterns:
                match = re.search(pattern, content.lower())
                if match:
                    info["sequence_number"] = int(match.group(1))
                    info["sequence_type"] = seq_type
                    break
        
        # Extract potential title (first header)
        title_match = re.search(r'# (.*?)$', content, re.MULTILINE)
        if title_match:
            info["title"] = title_match.group(1).strip()
        
        # Extract keywords
        keywords = []
        # Common academic keywords to look for
        keyword_patterns = [
            r'key\s+concepts?:?\s+(.*?)(?:\n\n|\Z)',
            r'learning\s+outcomes?:?\s+(.*?)(?:\n\n|\Z)',
            r'objectives?:?\s+(.*?)(?:\n\n|\Z)',
            r'topic:?\s+(.*?)(?:\n\n|\Z)'
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                # Extract bullet points or comma-separated items
                items = match.group(1)
                # Handle bullet lists
                bullet_items = re.findall(r'[-*]\s+(.*?)(?=\n[-*]|\n\n|\Z)', items)
                if bullet_items:
                    keywords.extend(bullet_items)
                else:
                    # Handle comma-separated lists
                    comma_items = [item.strip() for item in items.split(',')]
                    keywords.extend(comma_items)
        
        # If no structured keywords found, extract nouns from first few paragraphs
        if not keywords and content:
            # Get first 3 paragraphs
            paragraphs = content.split('\n\n')[:3]
            # Extract potential topic nouns (capitalized words)
            for para in paragraphs:
                capitalized_words = re.findall(r'\b([A-Z][a-z]{2,})\b', para)
                keywords.extend(capitalized_words[:5])  # Limit to first 5
        
        info["keywords"] = list(set(keywords))[:10]  # Deduplicate and limit to top 10
                
        return info
    
    def _generate_smart_filename(self, info: Dict[str, any]) -> str:
        """
        Generate a smart filename based on sequence information
        
        Args:
            info: Sequence information dictionary
            
        Returns:
            Smart filename (without extension)
        """
        # Default to original filename without extension
        if not info or not info.get("original_filename"):
            return "document"
            
        original_name = os.path.splitext(info["original_filename"])[0]
        
        if info.get("sequence_number") is None:
            return original_name
            
        # Format: {sequence_type}{sequence_number}_{title}
        # Example: lecture1_introduction_to_security
        
        seq_type = info.get("sequence_type", "part")
        seq_num = info.get("sequence_number", 0)
        
        # Use title if available, otherwise use keywords or original name
        if info.get("title"):
            # Clean and truncate title
            title = info["title"]
            # Remove special characters and replace spaces with underscores
            title = re.sub(r'[^\w\s-]', '', title).strip().lower()
            title = re.sub(r'[\s-]+', '_', title)
            # Truncate to reasonable length
            if len(title) > 40:
                title = title[:40].rstrip('_')
        elif info.get("keywords"):
            # Use first 3 keywords
            keywords = info["keywords"][:3]
            title = "_".join([k.lower() for k in keywords])
            title = re.sub(r'[^\w\s-]', '', title).strip()
            title = re.sub(r'[\s-]+', '_', title)
        else:
            # Use cleaned original name
            title = re.sub(r'[^\w\s-]', '', original_name).strip().lower()
            title = re.sub(r'[\s-]+', '_', title)
        
        return f"{seq_type}{seq_num}_{title}"
    
    def process_pdf(self, pdf_path: str, output_dir: str, rename_smartly: bool = True) -> Tuple[str, Optional[Dict]]:
        """
        Convert a PDF to Markdown using docling with enhanced image handling
        and smart file naming
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory for markdown and images
            rename_smartly: Whether to rename files based on content analysis
            
        Returns:
            Tuple of (path to the generated markdown file, metadata dictionary)
        """
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file {pdf_path} does not exist")
            return "", None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Build the docling command with default enriched settings
        cmd = [
            "docling",
            "--to", "md",
            "--output", output_dir,
            "--device", self.device,
            "--image-export-mode", "referenced",
            "--enrich-picture-classes",
            "--enrich-formula",
            pdf_path
        ]
        
        try:
            print(f"Converting {pdf_path}...")
            subprocess.run(cmd, check=True)
            
            # Initial output file from docling
            initial_output_file = os.path.join(output_dir, f"{base_filename}.md")
            
            if not os.path.exists(initial_output_file):
                print(f"Error: Expected output file {initial_output_file} not found")
                return "", None
                
            # Read the content for analysis
            with open(initial_output_file, 'r') as f:
                content = f.read()
                
            # Extract sequence info
            metadata = self._extract_sequence_info(pdf_path, content)
            
            # Rename the file if requested
            if rename_smartly and metadata.get("sequence_number") is not None:
                smart_filename = self._generate_smart_filename(metadata)
                smart_output_file = os.path.join(output_dir, f"{smart_filename}.md")
                
                # Rename file
                os.rename(initial_output_file, smart_output_file)
                
                # Update image references in the file
                if os.path.exists(smart_output_file):
                    with open(smart_output_file, 'r') as f:
                        updated_content = f.read()
                    
                    # Replace image references if needed
                    if base_filename != smart_filename:
                        img_dir = os.path.join(output_dir, f"{base_filename}_images")
                        if os.path.exists(img_dir):
                            # Rename image directory
                            new_img_dir = os.path.join(output_dir, f"{smart_filename}_images")
                            os.rename(img_dir, new_img_dir)
                            
                            # Update references in markdown
                            updated_content = updated_content.replace(
                                f"{base_filename}_images/", 
                                f"{smart_filename}_images/"
                            )
                            
                            with open(smart_output_file, 'w') as f:
                                f.write(updated_content)
                
                print(f"Successfully converted and renamed to {smart_output_file}")
                return smart_output_file, metadata
            else:
                print(f"Successfully converted to {initial_output_file}")
                return initial_output_file, metadata
                
        except subprocess.CalledProcessError as e:
            print(f"Error converting PDF: {str(e)}")
            return "", None
    
    def process_directory(self, dir_path: str, output_dir: str, rename_smartly: bool = True) -> List[Tuple[str, Dict]]:
        """
        Process all PDF files in a directory
        
        Args:
            dir_path: Path to directory containing PDFs
            output_dir: Output directory for markdown and images
            rename_smartly: Whether to rename files based on content analysis
            
        Returns:
            List of tuples (path to generated markdown file, metadata dictionary)
        """
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist")
            return []
        
        results = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    result, metadata = self.process_pdf(pdf_path, output_dir, rename_smartly)
                    if result:
                        results.append((result, metadata))
        
        # After processing all files, generate a metadata summary file
        if results:
            metadata_summary = {
                "processed_count": len(results),
                "source_directory": dir_path,
                "output_directory": output_dir,
                "files": []
            }
            
            # Sort by sequence number if available
            results.sort(key=lambda x: (
                x[1].get("sequence_type", ""),
                x[1].get("sequence_number", 999)
            ) if x[1] else (None, 999))
            
            for result, metadata in results:
                if metadata:
                    metadata_summary["files"].append({
                        "output_file": result,
                        "sequence_type": metadata.get("sequence_type"),
                        "sequence_number": metadata.get("sequence_number"),
                        "title": metadata.get("title"),
                        "keywords": metadata.get("keywords", [])
                    })
            
            # Save metadata summary
            metadata_path = os.path.join(output_dir, "processing_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata_summary, f, indent=2)
                
            print(f"Generated metadata summary at {metadata_path}")
        
        return results
