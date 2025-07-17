#!/usr/bin/env python3
"""
Convert the tutorial-answers.md file to PDF using pandoc with professional formatting
"""

import os
import re
import subprocess
from pathlib import Path

class TutorialPDFConverter:
    def __init__(self, input_file, output_file):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def process_markdown_content(self, content):
        """Process markdown content to improve formatting"""
        # Add title page
        title_page = """---
title: "Security Risk Assessment - Tutorial Answers"
subtitle: "Tutorial Solutions Based on Course Chapters"
documentclass: article
geometry: "margin=1in"
fontsize: 11pt
linestretch: 1.2
header-includes:
  - \\usepackage{fancyhdr}
  - \\usepackage{array}
  - \\usepackage{longtable}
  - \\usepackage{booktabs}
  - \\usepackage{tabularx}
  - \\pagestyle{fancy}
  - \\fancyhead[L]{Security Risk Assessment - Tutorial Answers}
  - \\fancyhead[R]{\\thepage}
  - \\renewcommand{\\headrulewidth}{0.4pt}
  - \\setcounter{secnumdepth}{0}
  - \\renewcommand{\\arraystretch}{1.2}
---

"""
        
        return title_page + content
    
    def convert_to_pdf(self):
        """Convert the markdown file to PDF using pandoc"""
        print(f"🔄 Converting {self.input_file.name} to PDF using pandoc...")
        
        try:
            # Read the markdown content
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process the content
            processed_content = self.process_markdown_content(content)
            
            # Create temporary file with processed content
            temp_file = self.input_file.parent / f"temp_{self.input_file.name}"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            # Use pandoc to convert to PDF
            cmd = [
                'pandoc',
                str(temp_file),
                '-o', str(self.output_file),
                '--pdf-engine=pdflatex',
                '--variable=geometry:margin=1in',
                '--variable=fontsize:11pt',
                '--variable=linestretch:1.2',
                '--standalone'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up temporary file
            temp_file.unlink()
            
            if result.returncode == 0:
                print(f"✅ Successfully converted to PDF: {self.output_file}")
                return True
            else:
                print(f"❌ Error converting to PDF: {result.stderr}")
                return False
            
        except Exception as e:
            print(f"❌ Error converting to PDF: {e}")
            return False

def main():
    """Main function to run the PDF conversion"""
    input_file = "/Users/invoture/dev.local/academic-agent/output/sra/tutorial-answers.md"
    output_file = "/Users/invoture/dev.local/academic-agent/output/sra/tutorial-answers.pdf"
    
    converter = TutorialPDFConverter(input_file, output_file)
    success = converter.convert_to_pdf()
    
    if success:
        print(f"\n🎉 PDF conversion completed successfully!")
        print(f"📄 Output file: {output_file}")
    else:
        print(f"\n❌ PDF conversion failed!")

if __name__ == "__main__":
    main()