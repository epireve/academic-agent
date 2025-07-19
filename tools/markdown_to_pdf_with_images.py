#!/usr/bin/env python3
"""
Convert markdown files to PDF using pre-generated PNG images for Mermaid diagrams
"""

import re
import base64
from pathlib import Path
import subprocess
from urllib.parse import quote

from src.core.output_manager import get_output_manager, get_final_output_path, get_processed_output_path, get_analysis_output_path
from src.core.output_manager import OutputCategory, ContentType


def ensure_dependencies():
    """Ensure all required dependencies are installed"""
    try:
        import markdown
        import weasyprint
    except ImportError:
        print("Installing required dependencies...")
        subprocess.run(['.venv/bin/python', '-m', 'pip', 'install', get_processed_output_path(ContentType.MARKDOWN), 'weasyprint'], check=True)
        print("Dependencies installed successfully!")

def image_to_base64(image_path):
    """Convert image file to base64 data URI"""
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
            
        # Determine MIME type
        ext = image_path.suffix.lower()
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml'
        }
        mime_type = mime_types.get(ext, 'image/png')
        
        # Create base64 data URI
        b64_data = base64.b64encode(data).decode('utf-8')
        return f"data:{mime_type};base64,{b64_data}"
    except Exception as e:
        print(f"Warning: Could not convert {image_path} to base64: {e}")
        return None

def process_markdown_content(content, md_file_path):
    """Process markdown content to handle images and Mermaid diagrams"""
    
    # Path to pre-generated PNG diagrams
    png_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/mermaid_diagrams/png_output")
    
    # Get the week number from the filename
    week_match = re.search(r'week_(\d+)', md_file_path.name)
    week_num = week_match.group(1) if week_match else None
    
    # Counter for diagrams in this file
    diagram_counter = 0
    
    # Process Mermaid diagrams - replace with pre-generated PNGs
    def replace_mermaid(match):
        nonlocal diagram_counter
        diagram_counter += 1
        
        if week_num:
            # Construct the PNG filename
            png_filename = f"week_{week_num}_comprehensive_study_notes_diagram_{diagram_counter}.png"
            png_path = png_dir / png_filename
            
            if png_path.exists():
                # Convert to base64 for reliable embedding
                base64_data = image_to_base64(png_path)
                if base64_data:
                    return f"\n![Mermaid Diagram {diagram_counter}]({base64_data})\n"
                else:
                    # Fallback to file path
                    return f"\n![Mermaid Diagram {diagram_counter}]({png_path})\n"
            else:
                print(f"Warning: PNG not found: {png_path}")
                # Return the original Mermaid code as a code block
                return f"\n```\n{match.group(1).strip()}\n```\n"
        else:
            # Can't determine week number, return as code block
            return f"\n```\n{match.group(1).strip()}\n```\n"
    
    # Replace Mermaid blocks with PNG images
    content = re.sub(r'```mermaid\n(.*?)\n```', replace_mermaid, content, flags=re.DOTALL)
    
    # Process regular image paths (textbook images)
    def fix_image_path(match):
        img_alt = match.group(1)
        img_path = match.group(2)
        
        # Skip if already a data URI
        if img_path.startswith('data:'):
            return match.group(0)
        
        # Handle relative paths
        if not img_path.startswith(('http://', 'https://', '/')):
            # Resolve relative to the markdown file
            full_path = md_file_path.parent / img_path
            
            # Try to resolve the path
            if full_path.exists():
                # Convert to base64 for reliable embedding
                base64_data = image_to_base64(full_path)
                if base64_data:
                    return f"![{img_alt}]({base64_data})"
                else:
                    # Use absolute path as fallback
                    return f"![{img_alt}]({full_path.absolute()})"
            else:
                print(f"Warning: Image not found: {full_path}")
                # Try alternative resolution (go up one more level for textbook images)
                alt_path = md_file_path.parent.parent / img_path.lstrip('../')
                if alt_path.exists():
                    base64_data = image_to_base64(alt_path)
                    if base64_data:
                        return f"![{img_alt}]({base64_data})"
                    else:
                        return f"![{img_alt}]({alt_path.absolute()})"
        
        return match.group(0)
    
    # Fix image paths
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_image_path, content)
    
    return content

def markdown_to_pdf(md_file_path, output_path):
    """Convert a single markdown file to PDF"""
    
    # Read markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Process content to handle images
    processed_content = process_markdown_content(content, md_file_path)
    
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
    
    # Save HTML for debugging if needed
    debug_html = output_path.with_suffix('.debug.html')
    with open(debug_html, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    try:
        HTML(string=html_template).write_pdf(output_path)
        # Remove debug HTML if successful
        debug_html.unlink()
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print(f"Debug HTML saved to: {debug_html}")
        raise

def convert_all_to_pdf():
    """Convert all markdown files to PDF"""
    
    # Ensure dependencies
    ensure_dependencies()
    
    # Paths
    input_dir = Path(get_final_output_path(ContentType.STUDY_NOTES))
    output_dir = input_dir  # Same directory as requested
    
    # Find all markdown files
    md_files = sorted(input_dir.glob("*.md"))
    
    print(f"📚 Converting {len(md_files)} markdown files to PDF...")
    print("   Using pre-generated PNG images for Mermaid diagrams")
    print("   All images will be displayed at 70% width\n")
    
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