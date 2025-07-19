#!/usr/bin/env python3
"""
Final comprehensive markdown to PDF converter with all fixes
"""

import os
import re
import base64
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
from improved_mermaid_converter import ImprovedMermaidConverter

class FinalMarkdownToPDFConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mermaid_converter = ImprovedMermaidConverter()
        
    def create_css_styles(self):
        """Create comprehensive CSS styles for PDF"""
        return """
        @page {
            size: A4;
            margin: 1.5cm 1.2cm 1.5cm 1.2cm;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }
        
        h1 {
            color: #2c3e50;
            font-size: 20pt;
            margin-top: 0;
            margin-bottom: 16pt;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8pt;
            page-break-before: auto;
        }
        
        h2 {
            color: #34495e;
            font-size: 16pt;
            margin-top: 20pt;
            margin-bottom: 12pt;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 4pt;
        }
        
        h3 {
            color: #7f8c8d;
            font-size: 14pt;
            margin-top: 16pt;
            margin-bottom: 10pt;
            font-weight: 600;
        }
        
        h4 {
            color: #95a5a6;
            font-size: 12pt;
            margin-top: 14pt;
            margin-bottom: 8pt;
            font-weight: 600;
        }
        
        h5, h6 {
            color: #95a5a6;
            font-size: 11pt;
            margin-top: 12pt;
            margin-bottom: 6pt;
            font-weight: 600;
        }
        
        p {
            margin-bottom: 10pt;
            margin-top: 0;
            text-align: justify;
            orphans: 2;
            widows: 2;
            line-height: 1.6;
        }
        
        /* Mermaid diagram styling - 50% size */
        .mermaid-container {
            text-align: center;
            margin: 20pt 0;
            page-break-inside: avoid;
            max-width: 100%;
            overflow: hidden;
        }
        
        .mermaid-container svg {
            max-width: 50% !important;
            width: 50% !important;
            height: auto !important;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            background: white;
        }
        
        .mermaid-container svg text {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 11px !important;
            fill: #2c3e50 !important;
            font-weight: 400 !important;
        }
        
        .mermaid-container svg tspan {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 11px !important;
            fill: #2c3e50 !important;
            font-weight: 400 !important;
        }
        
        .mermaid-placeholder {
            background-color: #f8f9fa;
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 16pt;
            text-align: center;
            margin: 16pt 0;
            page-break-inside: avoid;
            color: #2c3e50;
            font-family: monospace;
            font-size: 10pt;
        }
        
        /* Image styling */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 12pt auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Improved bullet point formatting */
        ul {
            margin-bottom: 12pt;
            margin-top: 6pt;
            padding-left: 18pt;
            list-style-type: disc;
        }
        
        ol {
            margin-bottom: 12pt;
            margin-top: 6pt;
            padding-left: 18pt;
            list-style-type: decimal;
        }
        
        li {
            margin-bottom: 6pt;
            margin-top: 0;
            line-height: 1.6;
            text-align: left;
            display: list-item;
        }
        
        li p {
            margin-bottom: 4pt;
            margin-top: 0;
            text-align: left;
            line-height: 1.6;
        }
        
        /* Nested lists with proper indentation */
        ul ul, ol ol, ul ol, ol ul {
            margin-top: 4pt;
            margin-bottom: 4pt;
            padding-left: 16pt;
        }
        
        li ul, li ol {
            margin-top: 4pt;
            margin-bottom: 4pt;
        }
        
        /* Strong emphasis in lists */
        li strong, li b {
            color: #2c3e50;
            font-weight: 600;
        }
        
        /* Italic emphasis in lists */
        li em, li i {
            color: #34495e;
            font-style: italic;
        }
        
        /* Table styling */
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
            line-height: 1.4;
        }
        
        th {
            background-color: #ecf0f1;
            font-weight: 600;
            color: #2c3e50;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        /* Code styling */
        code {
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 2px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            color: #d63384;
        }
        
        pre {
            background-color: #f4f4f4;
            padding: 12pt;
            border-radius: 4px;
            overflow: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
            margin: 12pt 0;
            border: 1px solid #e0e0e0;
        }
        
        pre code {
            background-color: transparent;
            padding: 0;
            color: #333;
        }
        
        /* Blockquote styling */
        blockquote {
            margin: 12pt 0;
            padding: 12pt 16pt;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            font-style: italic;
            color: #555;
        }
        
        /* Page break controls */
        .page-break {
            page-break-before: always;
        }
        
        .no-break {
            page-break-inside: avoid;
        }
        
        /* Caption styling */
        .caption {
            font-size: 9pt;
            color: #666;
            font-style: italic;
            text-align: center;
            margin-top: 4pt;
            margin-bottom: 8pt;
        }
        
        /* Executive summary styling */
        .executive-summary {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 16pt;
            margin: 16pt 0;
            border-radius: 4px;
        }
        
        .executive-summary p {
            margin-bottom: 8pt;
            margin-top: 0;
        }
        
        .executive-summary ul, .executive-summary ol {
            margin-top: 8pt;
        }
        
        /* Key takeaways styling */
        .key-takeaways {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 16pt;
            margin: 16pt 0;
            border-radius: 4px;
        }
        
        .key-takeaways p {
            margin-bottom: 8pt;
            margin-top: 0;
        }
        
        .key-takeaways ul, .key-takeaways ol {
            margin-top: 8pt;
        }
        
        .key-takeaways li {
            margin-bottom: 4pt;
        }
        
        /* Avoid orphans and widows */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }
        
        /* Horizontal rule styling */
        hr {
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 20pt 0;
        }
        
        /* Bold and italic text */
        strong, b {
            font-weight: 600;
            color: #2c3e50;
        }
        
        em, i {
            font-style: italic;
            color: #34495e;
        }
        
        /* Link styling */
        a {
            color: #3498db;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        """
    
    def process_images(self, content, md_file_path):
        """Process images to ensure they work in PDF"""
        def replace_image(match):
            alt_text = match.group(1)
            img_path = match.group(2)
            
            # Handle relative paths
            if not img_path.startswith(('http://', 'https://', 'data:')):
                # Convert relative path to absolute
                abs_path = (md_file_path.parent / img_path).resolve()
                if abs_path.exists():
                    img_path = str(abs_path)
                else:
                    return f'<div class="mermaid-placeholder">Image not found: {img_path}</div>'
            
            return f'<img src="{img_path}" alt="{alt_text}" />'
        
        # Replace markdown images with HTML img tags
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
        return content
    
    def convert_markdown_to_html(self, md_file_path):
        """Convert markdown file to HTML with properly rendered Mermaid diagrams"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process images first
        content = self.process_images(content, md_file_path)
        
        # Find and replace Mermaid diagrams
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        
        def replace_mermaid(match):
            mermaid_code = match.group(1)
            svg_content = self.mermaid_converter.convert_mermaid_to_svg(mermaid_code)
            
            if svg_content:
                # Embed SVG directly
                return f'<div class="mermaid-container">{svg_content}</div>'
            else:
                # Fall back to placeholder
                return f'<div class="mermaid-placeholder">📊 Mermaid Diagram:<br/><pre>{mermaid_code}</pre></div>'
        
        # Process all Mermaid diagrams
        mermaid_matches = list(re.finditer(mermaid_pattern, content, re.DOTALL))
        
        for match in reversed(mermaid_matches):  # Process in reverse to maintain positions
            replacement = replace_mermaid(match)
            content = content[:match.start()] + replacement + content[match.end():]
        
        # Convert remaining markdown to HTML
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        
        # Add special styling classes
        html_content = html_content.replace('<h3>Executive Summary</h3>', '<h3>Executive Summary</h3><div class="executive-summary">')
        html_content = html_content.replace('<h3>Key Takeaways</h3>', '</div><h3>Key Takeaways</h3><div class="key-takeaways">')
        html_content += '</div>'  # Close the last div
        
        return html_content
    
    def convert_single_file(self, md_file_path):
        """Convert a single markdown file to PDF"""
        print(f"🔄 Converting {md_file_path.name} to PDF with final fixes...")
        
        try:
            # Convert markdown to HTML
            html_content = self.convert_markdown_to_html(md_file_path)
            
            # Create complete HTML document
            full_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{md_file_path.stem}</title>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # Generate PDF
            output_path = self.output_dir / f"{md_file_path.stem}.pdf"
            
            # Create font configuration
            font_config = FontConfiguration()
            
            # Create CSS
            css = CSS(string=self.create_css_styles(), font_config=font_config)
            
            # Generate PDF
            HTML(string=full_html, base_url=str(md_file_path.parent)).write_pdf(
                output_path,
                stylesheets=[css],
                font_config=font_config,
                optimize_size=('fonts', 'images')
            )
            
            print(f"✅ Successfully converted {md_file_path.name} → {output_path.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error converting {md_file_path.name}: {e}")
            return False
    
    def convert_specific_files(self, file_patterns):
        """Convert specific markdown files for testing"""
        md_files = []
        
        for pattern in file_patterns:
            matched_files = list(self.input_dir.glob(pattern))
            md_files.extend(matched_files)
        
        if not md_files:
            print("❌ No matching markdown files found")
            return
        
        print(f"🚀 Starting final conversion of {len(md_files)} files...")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for md_file in sorted(md_files):
            success = self.convert_single_file(md_file)
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        print(f"\n📊 Final Conversion Summary:")
        print(f"✅ Successful: {successful_conversions}")
        print(f"❌ Failed: {failed_conversions}")
        print(f"📁 Output directory: {self.output_dir}")
        
        return successful_conversions, failed_conversions
    
    def convert_all_files(self):
        """Convert all markdown files in the input directory to PDF"""
        md_files = list(self.input_dir.glob("*.md"))
        
        if not md_files:
            print("❌ No markdown files found in input directory")
            return
        
        print(f"🚀 Starting final conversion of {len(md_files)} markdown files...")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for md_file in sorted(md_files):
            success = self.convert_single_file(md_file)
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        print(f"\n📊 Final Conversion Summary:")
        print(f"✅ Successful: {successful_conversions}")
        print(f"❌ Failed: {failed_conversions}")
        print(f"📁 Output directory: {self.output_dir}")

def main():
    """Main function to run the final PDF conversion"""
    project_root = Path.cwd()
    input_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    output_dir = input_dir  # Same folder as markdown files
    
    converter = FinalMarkdownToPDFConverter(input_dir, output_dir)
    
    # Test with just a couple of files first
    test_files = ["week_01_comprehensive_study_notes.md", "week_05_comprehensive_study_notes.md"]
    successful, failed = converter.convert_specific_files(test_files)
    
    if successful > 0:
        print(f"\n🎉 Test conversion successful! {successful} files converted.")
        print("📋 Check the generated PDFs to verify:")
        print("   • Mermaid diagrams are 50% smaller with visible text")
        print("   • Bullet points have proper line breaks")
        print("   • Images display correctly")
        print("   • Overall formatting is clean and professional")
    else:
        print("❌ Test conversion failed. Please check the errors above.")

if __name__ == "__main__":
    main()