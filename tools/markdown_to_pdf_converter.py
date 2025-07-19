#!/usr/bin/env python3
"""
Convert markdown study notes to PDF with proper formatting for Mermaid diagrams and images
"""

import os
import re
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration
import base64
from urllib.parse import quote
import shutil

class MarkdownToPDFConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_css_styles(self):
        """Create CSS styles for PDF with reduced margins and proper formatting"""
        return """
        @page {
            size: A4;
            margin: 1.5cm 1.2cm 1.5cm 1.2cm;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 11pt;
            line-height: 1.4;
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
            margin-top: 16pt;
            margin-bottom: 12pt;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 4pt;
        }
        
        h3 {
            color: #7f8c8d;
            font-size: 14pt;
            margin-top: 14pt;
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
            text-align: justify;
            orphans: 2;
            widows: 2;
        }
        
        .mermaid-container {
            text-align: center;
            margin: 16pt 0;
            page-break-inside: avoid;
            max-width: 100%;
            overflow: hidden;
        }
        
        .mermaid-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 12pt auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        ul, ol {
            margin-bottom: 10pt;
            padding-left: 20pt;
        }
        
        li {
            margin-bottom: 4pt;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 12pt 0;
            font-size: 10pt;
            page-break-inside: avoid;
        }
        
        th, td {
            border: 1px solid #bdc3c7;
            padding: 6pt;
            text-align: left;
            vertical-align: top;
        }
        
        th {
            background-color: #ecf0f1;
            font-weight: 600;
            color: #2c3e50;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 2pt 4pt;
            border-radius: 2px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }
        
        pre {
            background-color: #f4f4f4;
            padding: 10pt;
            border-radius: 4px;
            overflow: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.3;
            page-break-inside: avoid;
        }
        
        blockquote {
            margin: 12pt 0;
            padding: 8pt 12pt;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
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
        }
        
        /* Executive summary styling */
        .executive-summary {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 12pt;
            margin: 16pt 0;
        }
        
        /* Key takeaways styling */
        .key-takeaways {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12pt;
            margin: 16pt 0;
        }
        
        /* Avoid orphans and widows */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }
        
        /* Ensure proper spacing around sections */
        hr {
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 16pt 0;
        }
        """
    
    async def render_mermaid_diagram(self, mermaid_code):
        """Convert Mermaid diagram to SVG using Playwright"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Create HTML with Mermaid
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
                    <style>
                        body {{
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background: white;
                        }}
                        .mermaid {{
                            max-width: 100%;
                            height: auto;
                        }}
                    </style>
                </head>
                <body>
                    <div class="mermaid">
                        {mermaid_code}
                    </div>
                    <script>
                        mermaid.initialize({{
                            startOnLoad: true,
                            theme: 'default',
                            flowchart: {{
                                htmlLabels: true,
                                curve: 'basis'
                            }},
                            themeVariables: {{
                                primaryColor: '#3498db',
                                primaryTextColor: '#2c3e50',
                                primaryBorderColor: '#3498db',
                                lineColor: '#34495e',
                                secondaryColor: '#ecf0f1',
                                tertiaryColor: '#f8f9fa'
                            }}
                        }});
                    </script>
                </body>
                </html>
                """
                
                await page.set_content(html_content)
                await page.wait_for_load_state('networkidle')
                
                # Wait for Mermaid to render
                await page.wait_for_selector('.mermaid svg', timeout=10000)
                
                # Get the SVG element
                svg_element = await page.query_selector('.mermaid svg')
                if svg_element:
                    # Get SVG content
                    svg_content = await svg_element.inner_html()
                    svg_attributes = await svg_element.get_attribute('viewBox')
                    width = await svg_element.get_attribute('width')
                    height = await svg_element.get_attribute('height')
                    
                    # Create complete SVG
                    full_svg = f'<svg viewBox="{svg_attributes}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{svg_content}</svg>'
                    
                    await browser.close()
                    return full_svg
                else:
                    await browser.close()
                    return None
                    
        except Exception as e:
            print(f"Error rendering Mermaid diagram: {e}")
            return None
    
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
                    return f'<!-- Image not found: {img_path} -->'
            
            return f'<img src="{img_path}" alt="{alt_text}" />'
        
        # Replace markdown images with HTML img tags
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
        return content
    
    async def convert_markdown_to_html(self, md_file_path):
        """Convert markdown file to HTML with Mermaid diagrams rendered"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process images first
        content = self.process_images(content, md_file_path)
        
        # Find and replace Mermaid diagrams
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        
        async def replace_mermaid(match):
            mermaid_code = match.group(1)
            svg_content = await self.render_mermaid_diagram(mermaid_code)
            
            if svg_content:
                # Encode SVG for data URI
                svg_base64 = base64.b64encode(svg_content.encode('utf-8')).decode('utf-8')
                return f'<div class="mermaid-container"><img src="data:image/svg+xml;base64,{svg_base64}" alt="Mermaid Diagram" /></div>'
            else:
                return f'<div class="mermaid-container"><pre>{mermaid_code}</pre></div>'
        
        # Process all Mermaid diagrams
        mermaid_matches = list(re.finditer(mermaid_pattern, content, re.DOTALL))
        
        for match in reversed(mermaid_matches):  # Process in reverse to maintain positions
            replacement = await replace_mermaid(match)
            content = content[:match.start()] + replacement + content[match.end():]
        
        # Convert remaining markdown to HTML
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        
        # Add special styling classes
        html_content = html_content.replace('<h3>Executive Summary</h3>', '<h3>Executive Summary</h3><div class="executive-summary">')
        html_content = html_content.replace('<h3>Key Takeaways</h3>', '</div><h3>Key Takeaways</h3><div class="key-takeaways">')
        html_content += '</div>'  # Close the last div
        
        return html_content
    
    async def convert_single_file(self, md_file_path):
        """Convert a single markdown file to PDF"""
        print(f"🔄 Converting {md_file_path.name} to PDF...")
        
        try:
            # Convert markdown to HTML
            html_content = await self.convert_markdown_to_html(md_file_path)
            
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
    
    async def convert_all_files(self):
        """Convert all markdown files in the input directory to PDF"""
        md_files = list(self.input_dir.glob("*.md"))
        
        if not md_files:
            print("❌ No markdown files found in input directory")
            return
        
        print(f"🚀 Starting conversion of {len(md_files)} markdown files...")
        
        successful_conversions = 0
        failed_conversions = 0
        
        for md_file in sorted(md_files):
            success = await self.convert_single_file(md_file)
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
        
        print(f"\n📊 Conversion Summary:")
        print(f"✅ Successful: {successful_conversions}")
        print(f"❌ Failed: {failed_conversions}")
        print(f"📁 Output directory: {self.output_dir}")

async def main():
    """Main function to run the PDF conversion"""
    project_root = Path.cwd()
    input_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    output_dir = input_dir  # Same folder as markdown files
    
    converter = MarkdownToPDFConverter(input_dir, output_dir)
    await converter.convert_all_files()

if __name__ == "__main__":
    asyncio.run(main())