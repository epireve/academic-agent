#!/usr/bin/env python
"""
Command-line interface for PDF processing tool
"""

import os
import argparse
from .processor import DoclingProcessor

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown with enhanced image handling and smart file naming"
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
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device to use for PDF processing"
    )
    parser.add_argument(
        "--no-smart-rename",
        action="store_true",
        help="Disable smart file renaming based on content analysis"
    )
    
    args = parser.parse_args()
    
    if not args.pdf and not args.dir:
        parser.error("Either --pdf or --dir must be specified")
    
    processor = DoclingProcessor(device=args.device)
    
    if args.pdf:
        processor.process_pdf(args.pdf, args.output, not args.no_smart_rename)
    elif args.dir:
        processor.process_directory(args.dir, args.output, not args.no_smart_rename)

if __name__ == "__main__":
    main()
