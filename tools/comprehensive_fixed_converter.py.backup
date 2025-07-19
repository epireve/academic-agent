#!/usr/bin/env python3
"""
Comprehensive fixed converter addressing all image sizing and Mermaid text issues
"""

import os
import re
import base64
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
from mermaid_to_svg import MermaidToSVGConverter
from better_mermaid_renderer import BetterMermaidRenderer

class ComprehensiveFixedConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mermaid_converter = MermaidToSVGConverter()
        self.html_renderer = BetterMermaidRenderer()
        
    def create_css_styles(self):
        """Create CSS styles with all fixes applied"""
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
        
        h4, h5, h6 {
            color: #95a5a6;
            font-size: 12pt;
            margin-top: 12pt;
            margin-bottom: 8pt;
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
        
        /* ALL IMAGES - 50% size including textbook figures */
        img {
            max-width: 50% !important;
            width: 50% !important;
            height: auto !important;
            display: block;
            margin: 12pt auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Mermaid SVG diagrams - 50% size */
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
        
        /* Force text visibility in SVG */
        .mermaid-container svg text {
            fill: #2c3e50 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            opacity: 1 !important;
            stroke: none !important;
        }
        
        .mermaid-container svg tspan {
            fill: #2c3e50 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            opacity: 1 !important;
            stroke: none !important;
        }
        
        /* HTML fallback for Mermaid when SVG fails */
        .mermaid-html-container {
            background-color: #f8f9fa;
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 16pt;
            margin: 16pt auto;
            max-width: 70%;
            page-break-inside: avoid;
            text-align: center;
        }
        
        .mermaid-title {
            font-size: 14pt;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 12pt;
        }
        
        .mermaid-content {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 12pt;
            margin: 12pt 0;
        }
        
        .mermaid-code {
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
            color: #333;
            background-color: transparent;
            border: none;
            margin: 0;
            padding: 0;
        }
        
        .mermaid-note {
            font-size: 9pt;
            color: #666;
            font-style: italic;
            margin-top: 8pt;
        }
        
        .mermaid-placeholder {
            background-color: #f8f9fa;
            border: 2px dashed #3498db;
            border-radius: 8px;
            padding: 16pt;
            text-align: center;
            margin: 16pt auto;
            max-width: 70%;
            page-break-inside: avoid;
            color: #2c3e50;
            font-family: monospace;
            font-size: 10pt;
        }
        
        /* Fixed bullet point formatting */
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
        
        /* Ensure proper paragraph spacing in list items */
        li p {
            margin-bottom: 4pt;
            margin-top: 0;
            text-align: left;
            line-height: 1.6;
        }
        
        /* Better nested list formatting */
        ul ul, ol ol, ul ol, ol ul {
            margin-top: 4pt;
            margin-bottom: 4pt;
            padding-left: 16pt;
        }
        
        /* Emphasis in lists */
        li strong, li b {
            color: #2c3e50;
            font-weight: 600;
        }
        
        li em, li i {
            color: #34495e;
            font-style: italic;
        }
        
        /* Table formatting */
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
        
        /* Blockquote styling */
        blockquote {
            margin: 12pt 0;
            padding: 12pt 16pt;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
        /* Special sections */
        .executive-summary {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 16pt;
            margin: 16pt 0;
            border-radius: 4px;
        }
        
        .key-takeaways {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 16pt;
            margin: 16pt 0;
            border-radius: 4px;
        }
        
        .executive-summary p, .key-takeaways p {
            margin-bottom: 8pt;
            margin-top: 0;
        }
        
        .executive-summary ul, .key-takeaways ul {
            margin-top: 8pt;
        }
        
        /* Page break controls */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }
        
        hr {
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 20pt 0;
        }
        
        /* Text emphasis */
        strong, b {
            font-weight: 600;
            color: #2c3e50;
        }
        
        em, i {
            font-style: italic;
            color: #34495e;
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
        """
    
    def process_images(self, content, md_file_path):
        """Process images to ensure they work in PDF with 50% sizing"""
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
            
            # All images will be 50% size due to CSS
            return f'<img src="{img_path}" alt="{alt_text}" />'
        
        # Replace markdown images with HTML img tags
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
        return content
    
    def enhance_svg_text(self, svg_content):
        """Aggressively enhance SVG text visibility"""
        if not svg_content:
            return svg_content
        
        # Remove any problematic styling that might hide text
        svg_content = re.sub(r'fill="transparent"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="white"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="#ffffff"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="#fff"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="none"', r'fill="#2c3e50"', svg_content)
        
        # Force stroke to none to avoid text outline issues
        svg_content = re.sub(r'stroke="[^"]*"', r'stroke="none"', svg_content)
        
        # Add explicit text styling with !important equivalent
        svg_content = re.sub(
            r'<text([^>]*)>',
            r'<text\1 style="fill: #2c3e50 !important; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif !important; font-size: 11px !important; font-weight: 500 !important; opacity: 1 !important; stroke: none !important;">',
            svg_content
        )
        
        # Also fix tspan elements
        svg_content = re.sub(
            r'<tspan([^>]*)>',
            r'<tspan\1 style="fill: #2c3e50 !important; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif !important; font-size: 11px !important; font-weight: 500 !important; opacity: 1 !important; stroke: none !important;">',
            svg_content
        )
        
        return svg_content
    
    def convert_markdown_to_html(self, md_file_path):
        """Convert markdown file to HTML with all fixes"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process images first (all images will be 50% size)
        content = self.process_images(content, md_file_path)
        
        # Find and replace Mermaid diagrams
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        
        def replace_mermaid(match):
            mermaid_code = match.group(1)
            
            # Try SVG conversion first
            svg_content = self.mermaid_converter.convert_mermaid_to_svg(mermaid_code)
            
            if svg_content:
                # Enhance SVG text visibility
                svg_content = self.enhance_svg_text(svg_content)
                return f'<div class="mermaid-container">{svg_content}</div>'
            else:
                # Fall back to HTML representation
                html_content = self.html_renderer.convert_mermaid_to_html(mermaid_code)
                return html_content
        
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
        print(f"🔄 Converting {md_file_path.name} to PDF with comprehensive fixes...")
        
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
    
    def convert_test_files(self):
        """Convert test files to verify fixes"""
        test_files = ["week_01_comprehensive_study_notes.md", "week_08_comprehensive_study_notes.md"]
        
        print("🧪 Testing comprehensive fixes...")
        print("📋 Fixes applied:")
        print("   • ALL images (including textbook figures) → 50% size")
        print("   • Mermaid diagrams → 50% size with enhanced text visibility")
        print("   • Bullet points → proper line breaks and spacing")
        print("   • HTML fallback for Mermaid when SVG fails\n")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for test_file in test_files:
            file_path = self.input_dir / test_file
            if file_path.exists():
                success = self.convert_single_file(file_path)
                if success:
                    successful_conversions += 1
                else:
                    failed_conversions += 1
            else:
                print(f"❌ File not found: {test_file}")
                failed_conversions += 1
        
        print(f"\n📊 Test Results:")
        print(f"✅ Successful: {successful_conversions}")
        print(f"❌ Failed: {failed_conversions}")
        
        if successful_conversions > 0:
            print(f"\n🎉 Test successful! Check these PDFs:")
            for test_file in test_files:
                pdf_name = test_file.replace('.md', '.pdf')
                print(f"   📄 {pdf_name}")
            print(f"\n📁 Location: {self.output_dir}")
        
        return successful_conversions > 0

def main():
    """Main function to test the comprehensive fixes"""
    project_root = Path.cwd()
    input_dir = project_root / "output" / "sra" / "ai_enhanced_study_notes"
    output_dir = input_dir  # Same folder as markdown files
    
    converter = ComprehensiveFixedConverter(input_dir, output_dir)
    success = converter.convert_test_files()
    
    if success:
        print("\n✅ Test conversion completed successfully!")
        print("📋 Please check the generated PDFs to verify all fixes are working correctly.")
    else:
        print("\n❌ Test conversion failed. Please check the errors above.")

if __name__ == "__main__":
    main()