#!/usr/bin/env python3
"""
Simple converter that shows Mermaid as formatted code blocks
"""

import os
import re
from pathlib import Path
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

class SimpleCodeBlockConverter:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_css_styles(self):
        """Create CSS styles with all fixes"""
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
        
        /* ALL IMAGES - 50% size */
        img {
            max-width: 50% !important;
            width: auto !important;
            height: auto !important;
            display: block;
            margin: 12pt auto;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Mermaid diagram code block styling */
        .mermaid-block {
            background-color: #f8f9fa;
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 16pt;
            margin: 16pt auto;
            max-width: 80%;
            page-break-inside: avoid;
        }
        
        .mermaid-header {
            font-size: 12pt;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8pt;
            text-align: center;
        }
        
        .mermaid-code {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 12pt;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            line-height: 1.4;
            color: #333;
            white-space: pre-wrap;
            overflow-wrap: break-word;
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
        
        li p {
            margin-bottom: 4pt;
            margin-top: 0;
            text-align: left;
            line-height: 1.6;
        }
        
        ul ul, ol ol, ul ol, ol ul {
            margin-top: 4pt;
            margin-bottom: 4pt;
            padding-left: 16pt;
        }
        
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
        
        blockquote {
            margin: 12pt 0;
            padding: 12pt 16pt;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
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
        
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
        }
        
        hr {
            border: none;
            border-top: 1px solid #bdc3c7;
            margin: 20pt 0;
        }
        
        strong, b {
            font-weight: 600;
            color: #2c3e50;
        }
        
        em, i {
            font-style: italic;
            color: #34495e;
        }
        
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
        """Process images with 50% sizing"""
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
                    return f'<p style="color: red; text-align: center;">[Image not found: {img_path}]</p>'
            
            # Return image tag - CSS will handle 50% sizing
            return f'<img src="{img_path}" alt="{alt_text}" />'
        
        # Replace markdown images with HTML img tags
        content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image, content)
        return content
    
    def convert_markdown_to_html(self, md_file_path):
        """Convert markdown to HTML with Mermaid as code blocks"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Process images first (all will be 50% size)
        content = self.process_images(content, md_file_path)
        
        # Find and replace Mermaid diagrams with formatted code blocks
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        
        def replace_mermaid(match):
            mermaid_code = match.group(1).strip()
            
            # Create a simple code block representation
            html_block = f'''
            <div class="mermaid-block">
                <div class="mermaid-header">📊 Concept Diagram (Mermaid)</div>
                <div class="mermaid-code">{mermaid_code}</div>
            </div>
            '''
            
            return html_block
        
        # Process all Mermaid diagrams
        content = re.sub(mermaid_pattern, replace_mermaid, content, flags=re.DOTALL)
        
        # Convert remaining markdown to HTML
        html_content = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        
        # Add special styling classes
        html_content = html_content.replace('<h3>Executive Summary</h3>', '<h3>Executive Summary</h3><div class="executive-summary">')
        html_content = html_content.replace('<h3>Key Takeaways</h3>', '</div><h3>Key Takeaways</h3><div class="key-takeaways">')
        html_content += '</div>'  # Close the last div
        
        return html_content
    
    def convert_single_file(self, md_file_path):
        """Convert a single markdown file to PDF"""
        print(f"🔄 Converting {md_file_path.name} to PDF...")
        
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
    
    def convert_all_files(self):
        """Convert all markdown files"""
        md_files = list(self.input_dir.glob("*.md"))
        
        if not md_files:
            print("❌ No markdown files found")
            return
        
        print(f"🚀 Converting all {len(md_files)} markdown files...")
        print("📋 Features:")
        print("   • Mermaid diagrams → Formatted code blocks")
        print("   • All images → 50% size")
        print("   • Bullet points → Proper formatting")
        print("   • Clean, professional layout\n")
        
        successful = 0
        failed = 0
        
        for md_file in sorted(md_files):
            if self.convert_single_file(md_file):
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 Final Results:")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📁 All PDFs saved in: {self.output_dir}")

def main():
    """Main function"""
    project_root = Path.cwd()
    input_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    output_dir = input_dir
    
    converter = SimpleCodeBlockConverter(input_dir, output_dir)
    
    print("=" * 60)
    print("SIMPLE SOLUTION: Mermaid as Code Blocks + 50% Images")
    print("=" * 60)
    
    # Convert all files directly
    converter.convert_all_files()

if __name__ == "__main__":
    main()