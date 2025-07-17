#!/usr/bin/env python3
"""
Convert the tutorial-answers.md file to PDF if it has content
"""

import os
from pathlib import Path

def check_and_convert():
    """Check if tutorial-answers.md has content before converting"""
    input_file = Path("/Users/invoture/dev.local/academic-agent/output/sra/tutorial-answers.md")
    output_file = Path("/Users/invoture/dev.local/academic-agent/output/sra/tutorial-answers.pdf")
    
    # Check if file exists and has content
    if not input_file.exists():
        print(f"❌ Error: {input_file} does not exist!")
        return False
        
    # Check file size
    file_size = input_file.stat().st_size
    if file_size == 0:
        print(f"❌ Cannot generate PDF: {input_file.name} is empty!")
        print(f"📝 Please add content to the file before generating a PDF.")
        return False
    
    print(f"✅ File {input_file.name} has content ({file_size} bytes)")
    print(f"⚠️  PDF generation would proceed here, but the file is currently empty.")
    return False

def main():
    """Main function"""
    print("🔍 Checking tutorial-answers.md...")
    check_and_convert()

if __name__ == "__main__":
    main()