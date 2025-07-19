#!/usr/bin/env python3
"""
Convert all markdown files to PDF with properly rendered Mermaid diagrams and images
"""

import re
import base64
from pathlib import Path
import subprocess
import shutil
import tempfile
from urllib.parse import quote

def ensure_dependencies():
    """Ensure all required dependencies are installed"""
    try:
        import markdown
        import weasyprint
    except ImportError:
        print("Installing required dependencies...")
        subprocess.run(['.venv/bin/python', '-m', 'pip', 'install', 'markdown', 'weasyprint'], check=True)
        print("Dependencies installed successfully!")

def convert_mermaid_to_png(mermaid_code, output_path, diagram_num):
    """Convert Mermaid code to PNG using mermaid-cli"""
    
    # Create temp file for mermaid code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
        f.write(mermaid_code)
        temp_mmd = f.name
    
    # Create optimized config
    config_content = {
        "theme": "default",
        "themeVariables": {
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": "16px",
            "primaryColor": "#ffffff",
            "primaryTextColor": "#000000",
            "primaryBorderColor": "#7C0000",
            "lineColor": "#333333",
            "background": "#ffffff"
        },
        "flowchart": {
            "htmlLabels": True,
            "curve": "basis",
            "diagramPadding": 8,
            "nodeSpacing": 50,
            "rankSpacing": 50
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        json.dump(config_content, f)
        config_file = f.name
    
    # Convert to PNG
    png_file = output_path / f"diagram_{diagram_num}.png"
    
    cmd = [
        'npx', '@mermaid-js/mermaid-cli',
        '-i', temp_mmd,
        '-o', str(png_file),
        '-c', config_file,
        '--width', '1200',
        '--height', '900',
        '--scale', '2',
        '--backgroundColor', 'white'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and png_file.exists():
            # Clean up temp files
            Path(temp_mmd).unlink()
            Path(config_file).unlink()
            return png_file
        else:
            print(f"Warning: Failed to convert diagram: {result.stderr}")
            return None
    except Exception as e:
        print(f"Warning: Error converting diagram: {e}")
        return None

def process_markdown_content(content, md_file_path, temp_dir):
    """Process markdown content to handle images and Mermaid diagrams"""
    
    # Counter for diagrams
    diagram_counter = 0
    
    # Process Mermaid diagrams
    def replace_mermaid(match):
        nonlocal diagram_counter
        diagram_counter += 1
        mermaid_code = match.group(1).strip()
        
        # Convert to PNG
        png_path = convert_mermaid_to_png(mermaid_code, temp_dir, diagram_counter)
        
        if png_path and png_path.exists():
            # Return as image markdown
            return f"\n![Mermaid Diagram {diagram_counter}]({png_path})\n"
        else:
            # Fallback to code block
            return f"\n```\n{mermaid_code}\n```\n"
    
    # Replace Mermaid blocks with PNG images
    content = re.sub(r'```mermaid\n(.*?)\n```', replace_mermaid, content, flags=re.DOTALL)
    
    # Process image paths
    def fix_image_path(match):
        img_alt = match.group(1)
        img_path = match.group(2)
        
        # Handle relative paths
        if not img_path.startswith(('http://', 'https://', '/')):
            # Assume it's relative to the markdown file
            full_path = md_file_path.parent / img_path
            if full_path.exists():
                return f"![{img_alt}]({full_path})"
        
        return match.group(0)
    
    # Fix image paths
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_image_path, content)
    
    return content

def markdown_to_pdf(md_file_path, output_path):
    """Convert a single markdown file to PDF"""
    
    # Read markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create temp directory for diagrams
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Process content
        processed_content = process_markdown_content(content, md_file_path, temp_dir_path)
        
        # Convert markdown to HTML
        import markdown
        md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
        html_content = md.convert(processed_content)
        
        # Create full HTML document with styling
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 15mm 20mm 15mm 20mm;
        }}
        
        body {{
            font-family: Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        
        h1 {{
            color: #1a1a1a;
            border-bottom: 3px solid #1a1a1a;
            padding-bottom: 10px;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #2c3e50;
            margin-top: 24px;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #34495e;
            margin-top: 20px;
            page-break-after: avoid;
        }}
        
        img {{
            max-width: 70%;
            height: auto;
            display: block;
            margin: 20px auto;
            page-break-inside: avoid;
        }}
        
        pre {{
            background-color: #f5f5f5;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            page-break-inside: avoid;
            font-size: 10pt;
        }}
        
        code {{
            background-color: #f0f0f0;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }}
        
        blockquote {{
            border-left: 4px solid #ddd;
            margin: 0;
            padding-left: 16px;
            color: #666;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            page-break-inside: avoid;
        }}
        
        table th, table td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        
        table th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        
        ul, ol {{
            margin: 10px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 5px 0;
        }}
        
        .page-break {{
            page-break-after: always;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
        
        # Convert HTML to PDF
        from weasyprint import HTML, CSS
        HTML(string=html_template).write_pdf(output_path)

def convert_all_to_pdf():
    """Convert all markdown files to PDF"""
    
    # Ensure dependencies
    ensure_dependencies()
    
    # Paths
    input_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes")
    output_dir = input_dir  # Same directory as requested
    
    # Find all markdown files
    md_files = sorted(input_dir.glob("*.md"))
    
    print(f"📚 Converting {len(md_files)} markdown files to PDF...\n")
    
    successful = 0
    failed = 0
    
    for md_file in md_files:
        pdf_file = output_dir / f"{md_file.stem}.pdf"
        
        print(f"📄 Converting: {md_file.name}")
        
        try:
            markdown_to_pdf(md_file, pdf_file)
            
            if pdf_file.exists():
                file_size_kb = pdf_file.stat().st_size / 1024
                print(f"   ✅ Success! Size: {file_size_kb:.1f} KB")
                successful += 1
            else:
                print(f"   ❌ Failed: PDF file not created")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 PDF files saved to: {output_dir}")

if __name__ == "__main__":
    convert_all_to_pdf()