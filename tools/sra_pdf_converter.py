#!/usr/bin/env python3
"""
Convert the SRA Gemini Summarise markdown file to PDF with chapter breaks and proper formatting
"""

import os
import re
from pathlib import Path
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import markdown

class SRAPDFConverter:
    def __init__(self, input_file, output_file):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
    def create_css_styles(self):
        """Create CSS styles for PDF with proper chapter breaks and formatting"""
        return """
        @page {
            size: A4;
            margin: 2cm 1.5cm 2cm 1.5cm;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #2c3e50;
            margin: 0;
            padding: 0;
        }
        
        .chapter-break {
            page-break-before: always;
        }
        
        h1, h2, h3 {
            page-break-after: avoid;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 24pt;
            font-weight: bold;
            margin-top: 0;
            margin-bottom: 20pt;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10pt;
        }
        
        h2 {
            color: #34495e;
            font-size: 18pt;
            font-weight: bold;
            margin-top: 20pt;
            margin-bottom: 14pt;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 6pt;
        }
        
        h3 {
            color: #7f8c8d;
            font-size: 16pt;
            font-weight: bold;
            margin-top: 18pt;
            margin-bottom: 12pt;
        }
        
        h4 {
            color: #95a5a6;
            font-size: 14pt;
            font-weight: bold;
            margin-top: 16pt;
            margin-bottom: 10pt;
        }
        
        p {
            margin-bottom: 12pt;
            text-align: justify;
            orphans: 2;
            widows: 2;
        }
        
        strong {
            font-weight: bold;
            color: #2c3e50;
        }
        
        em {
            font-style: italic;
            color: #7f8c8d;
        }
        
        /* Bullet points and lists */
        ul, ol {
            margin-bottom: 12pt;
            padding-left: 24pt;
        }
        
        ul ul, ol ol {
            margin-top: 6pt;
            margin-bottom: 6pt;
        }
        
        li {
            margin-bottom: 6pt;
            line-height: 1.4;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 16pt 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }
        
        th, td {
            border: 1px solid #bdc3c7;
            padding: 8pt;
            text-align: left;
            vertical-align: top;
        }
        
        th {
            background-color: #ecf0f1;
            font-weight: bold;
            color: #2c3e50;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        /* Code blocks */
        code {
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 2px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }
        
        pre {
            background-color: #f4f4f4;
            padding: 12pt;
            border-radius: 4px;
            overflow: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.3;
            page-break-inside: avoid;
            margin: 12pt 0;
        }
        
        /* Blockquotes */
        blockquote {
            margin: 16pt 0;
            padding: 12pt 16pt;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
        /* Horizontal rules */
        hr {
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 20pt 0;
        }
        
        /* Chapter sections */
        .chapter-section {
            margin-bottom: 30pt;
        }
        
        /* Key concepts highlighting */
        .key-concept {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12pt;
            margin: 16pt 0;
        }
        
        /* Avoid orphans and widows */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }
        
        /* Ensure tables don't break awkwardly */
        table {
            page-break-inside: avoid;
        }
        
        /* Special styling for chapter titles */
        .chapter-title {
            text-align: center;
            margin-bottom: 30pt;
            padding: 20pt;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        """
    
    def process_markdown_content(self, content):
        """Process markdown content to add chapter breaks and improve formatting"""
        # Split content by chapter separators (---)
        chapters = re.split(r'\n---\n', content)
        
        processed_chapters = []
        for i, chapter in enumerate(chapters):
            if i > 0:  # Add page break before each chapter (except the first)
                chapter = '<div class="chapter-break"></div>\n' + chapter
            
            # Wrap chapter in a section
            chapter = f'<div class="chapter-section">\n{chapter}\n</div>'
            processed_chapters.append(chapter)
        
        # Join chapters back together
        processed_content = '\n'.join(processed_chapters)
        
        # Highlight key concepts
        processed_content = re.sub(
            r'\*\*Key Concepts Introduced:\*\*',
            '<div class="key-concept">**Key Concepts Introduced:**</div>',
            processed_content
        )
        
        processed_content = re.sub(
            r'\*\*Core Argument:\*\*',
            '<div class="key-concept">**Core Argument:**</div>',
            processed_content
        )
        
        return processed_content
    
    def convert_to_pdf(self):
        """Convert the markdown file to PDF"""
        print(f"🔄 Converting {self.input_file.name} to PDF...")
        
        try:
            # Read the markdown content
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process the content for chapter breaks
            processed_content = self.process_markdown_content(content)
            
            # Convert markdown to HTML
            html_content = markdown.markdown(
                processed_content, 
                extensions=['tables', 'fenced_code', 'nl2br']
            )
            
            # Create complete HTML document
            full_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Security Risk Assessment Summary</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Create font configuration
            font_config = FontConfiguration()
            
            # Create CSS
            css = CSS(string=self.create_css_styles(), font_config=font_config)
            
            # Generate PDF
            HTML(string=full_html, base_url=str(self.input_file.parent)).write_pdf(
                self.output_file,
                stylesheets=[css],
                font_config=font_config,
                optimize_size=('fonts', 'images')
            )
            
            print(f"✅ Successfully converted to PDF: {self.output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error converting to PDF: {e}")
            return False

def main():
    """Main function to run the PDF conversion"""
    input_file = "/Users/invoture/dev.local/academic-agent/output/sra/gemini-summarise.md"
    output_file = "/Users/invoture/dev.local/academic-agent/output/sra/gemini-summarise.pdf"
    
    converter = SRAPDFConverter(input_file, output_file)
    success = converter.convert_to_pdf()
    
    if success:
        print(f"\n🎉 PDF conversion completed successfully!")
        print(f"📄 Output file: {output_file}")
    else:
        print(f"\n❌ PDF conversion failed!")

if __name__ == "__main__":
    main()