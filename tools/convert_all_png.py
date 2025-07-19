#!/usr/bin/env python3
"""
Convert all files using PNG diagrams
"""

from pathlib import Path
from png_based_pdf_converter import PNGBasedPDFConverter

def main():
    project_root = Path.cwd()
    input_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    output_dir = input_dir
    
    converter = PNGBasedPDFConverter(input_dir, output_dir)
    
    print("=" * 60)
    print("CONVERTING ALL FILES WITH PNG DIAGRAMS")
    print("=" * 60)
    
    converter.convert_all_files()

if __name__ == "__main__":
    main()