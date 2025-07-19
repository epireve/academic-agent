#!/usr/bin/env python3
"""
Simplified PDF converter that uses local images in the same folder as markdown files.
All images are now consolidated in the study notes directory.
"""

import re
import base64
from pathlib import Path
import subprocess
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

def get_diagram_size_for_week(week_num):
    """Get the appropriate diagram size for each week based on user requirements"""
    week_sizes = {
        '01': '100%',  # Chapter 1
        '03': '100%',  # Chapter 3
        '04': '30%',   # Chapter 4
        '05': '100%',  # Chapter 5
        '06': '100%',  # Chapter 6
        '07': '70%',   # Chapter 7 - default
        '08': '100%',  # Chapter 8
        '09': '100%',  # Chapter 9
        '10': '100%',  # Chapter 10
        '11': '70%',   # Chapter 11 - default
        '12': '30%',   # Chapter 12
        '13': '100%',  # Chapter 13
    }
    return week_sizes.get(week_num, '70%')  # Default to 70%

def fix_bullet_formatting(content):
    """Fix bullet point formatting issues in markdown"""
    
    # Fix lines that have bullet points followed by text that should be on separate lines
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a bullet point line
        if line.strip().startswith('*'):
            # Check if the next line exists and doesn't start with * or is not empty
            if i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].strip().startswith('*'):
                # Check if the next line looks like it should be a new bullet point
                next_line = lines[i + 1].strip()
                if next_line.startswith('**') and next_line.endswith(':**'):
                    # This should be a new bullet point
                    fixed_lines.append(line)
                    fixed_lines.append('')  # Add blank line
                    fixed_lines.append('*   ' + lines[i + 1].strip())
                    i += 2
                    continue
            
            # Also check for numbered lists that follow bullet points
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if re.match(r'^\d+\.', next_line):
                    # Add a blank line before numbered list
                    fixed_lines.append(line)
                    fixed_lines.append('')
                    i += 1
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    # Fix multi-paragraph bullet points
    content = '\n'.join(fixed_lines)
    
    # Ensure consistent bullet formatting (asterisk + 3 spaces)
    content = re.sub(r'^\*\s+', '*   ', content, flags=re.MULTILINE)
    
    # Ensure blank lines between different bullet points at the same level
    content = re.sub(r'(\*   .+)\n(\*   )', r'\1\n\n\2', content)
    
    return content

def process_markdown_content(content, md_file_path):
    """Process markdown content to handle images and Mermaid diagrams"""
    
    # Fix bullet formatting first
    content = fix_bullet_formatting(content)
    
    # Get the week number from the filename
    week_match = re.search(r'week_(\d+)', md_file_path.name)
    week_num = week_match.group(1) if week_match else None
    
    # Get the appropriate diagram size for this week
    diagram_size = get_diagram_size_for_week(week_num) if week_num else '70%'
    
    # Counter for diagrams in this file
    diagram_counter = 0
    
    # Process Mermaid diagrams - replace with local PNG files
    def replace_mermaid(match):
        nonlocal diagram_counter
        diagram_counter += 1
        
        if week_num:
            # Construct the PNG filename (now in same directory)
            png_filename = f"week_{week_num}_comprehensive_study_notes_diagram_{diagram_counter}.png"
            png_path = md_file_path.parent / png_filename
            
            if png_path.exists():
                # Convert to base64 for reliable embedding
                base64_data = image_to_base64(png_path)
                if base64_data:
                    # Add custom size class
                    size_class = diagram_size.replace('%', '').replace('-', '_')
                    return f"\n![Mermaid Diagram {diagram_counter}]({base64_data}){{.diagram-size-{size_class}}}\n"
                else:
                    # Fallback to file path
                    size_class = diagram_size.replace('%', '').replace('-', '_')
                    return f"\n![Mermaid Diagram {diagram_counter}]({png_path}){{.diagram-size-{size_class}}}\n"
            else:
                print(f"Warning: PNG not found: {png_path}")
                # Return the original Mermaid code as a code block
                return f"\n```\n{match.group(1).strip()}\n```\n"
        else:
            # Can't determine week number, return as code block
            return f"\n```\n{match.group(1).strip()}\n```\n"
    
    # Replace Mermaid blocks with PNG images
    content = re.sub(r'```mermaid\n(.*?)\n```', replace_mermaid, content, flags=re.DOTALL)
    
    # Process regular image paths (now all in same directory)
    def fix_image_path(match):
        img_alt = match.group(1)
        img_path = match.group(2)
        
        # Skip if already a data URI
        if img_path.startswith('data:'):
            return match.group(0)
        
        # All images are now local - just use the filename
        if '/' in img_path or '\\' in img_path:
            # Extract just the filename
            img_filename = Path(img_path).name
        else:
            img_filename = img_path
        
        # Construct full path (images are in same directory as markdown)
        full_path = md_file_path.parent / img_filename
        
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
            return match.group(0)
    
    # Fix image paths
    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_image_path, content)
    
    return content

def markdown_to_pdf(md_file_path, output_path):
    """Convert a single markdown file to PDF"""
    
    # Read markdown content
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Process content to handle images and fix formatting
    processed_content = process_markdown_content(content, md_file_path)
    
    # Convert markdown to HTML with proper extensions
    import markdown
    
    md = markdown.Markdown(extensions=[
        'tables', 
        'fenced_code', 
        'codehilite',
        'attr_list'  # Support for image attributes
    ])
    html_content = md.convert(processed_content)
    
    # Get week number for this file
    week_match = re.search(r'week_(\d+)', md_file_path.name)
    week_num = week_match.group(1) if week_match else None
    diagram_size = get_diagram_size_for_week(week_num) if week_num else '70%'
    
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
        
        /* Default image sizing */
        img {{
            max-width: 70%;
            height: auto;
            display: block;
            margin: 20px auto;
            page-break-inside: avoid;
        }}
        
        /* Custom sizing for specific diagrams */
        img.diagram-size-30 {{
            max-width: 30% !important;
        }}
        
        img.diagram-size-50 {{
            max-width: 50% !important;
        }}
        
        img.diagram-size-70 {{
            max-width: 70% !important;
        }}
        
        img.diagram-size-100 {{
            max-width: 100% !important;
        }}
        
        img.diagram-size-full_height {{
            max-width: 100% !important;
            max-height: 80vh !important;
            height: auto !important;
        }}
        
        pre {{
            background-color: #f5f5f5;
            padding: 12px;
            border-radius: 4px;
            overflow-x: auto;
            page-break-inside: avoid;
            font-size: 10pt;
            white-space: pre-wrap;
        }}
        
        code {{
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
            color: #333 !important;
            display: inline-block;
            border: 1px solid #ddd;
        }}
        
        /* Ensure inline code in paragraphs is visible */
        p code, li code, td code {{
            background-color: #e8e8e8;
            color: #000 !important;
            font-weight: 600;
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
            font-size: 9pt;  /* 2 points smaller than body text (11pt) */
        }}
        
        table th {{
            background-color: #f5f5f5;
            font-weight: bold;
            font-size: 10pt;  /* 1 point smaller than body text for headers */
        }}
        
        ul, ol {{
            margin: 12px 0;
            padding-left: 30px;
        }}
        
        li {{
            margin: 8px 0;
            line-height: 1.6;
        }}
        
        /* Ensure proper spacing for nested lists */
        li > ul, li > ol {{
            margin-top: 8px;
            margin-bottom: 8px;
        }}
        
        /* Ensure list items have proper paragraph spacing */
        li > p {{
            margin: 5px 0;
        }}
        
        /* Fix for inline lists in paragraphs */
        p + ul, p + ol {{
            margin-top: 5px;
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
        return True
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print(f"Debug HTML saved to: {debug_html}")
        return False

def convert_all_to_pdf():
    """Convert all markdown files to PDF using local images"""
    
    # Ensure dependencies
    ensure_dependencies()
    
    # Paths
    input_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes")
    output_dir = input_dir  # Same directory as requested
    
    # Find all markdown files
    md_files = sorted(input_dir.glob("*.md"))
    
    print(f"📚 Converting {len(md_files)} markdown files to PDF with local images...")
    print("   All images are now in the same folder as markdown files")
    print("   Using custom diagram sizing per chapter")
    print("   Fixed bullet point formatting")
    print("   Simplified image path resolution\n")
    
    successful = 0
    failed = 0
    
    for md_file in md_files:
        pdf_file = output_dir / f"{md_file.stem}.pdf"
        
        # Get week number for custom sizing info
        week_match = re.search(r'week_(\d+)', md_file.name)
        if week_match:
            week_num = week_match.group(1)
            size = get_diagram_size_for_week(week_num)
            print(f"📄 Converting: {md_file.name} (diagram size: {size})")
        else:
            print(f"📄 Converting: {md_file.name}")
        
        if markdown_to_pdf(md_file, pdf_file):
            if pdf_file.exists():
                file_size_kb = pdf_file.stat().st_size / 1024
                print(f"   ✅ Success! Size: {file_size_kb:.1f} KB")
                successful += 1
            else:
                print(f"   ❌ Failed: PDF file not created")
                failed += 1
        else:
            failed += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 PDF files saved to: {output_dir}")
    
    # Count total files in the directory
    all_files = list(input_dir.iterdir())
    md_count = len([f for f in all_files if f.suffix == '.md'])
    pdf_count = len([f for f in all_files if f.suffix == '.pdf'])
    jpeg_count = len([f for f in all_files if f.suffix == '.jpeg'])
    png_count = len([f for f in all_files if f.suffix == '.png'])
    
    print(f"\n📁 Directory Contents Summary:")
    print(f"   Markdown files: {md_count}")
    print(f"   PDF files: {pdf_count}")
    print(f"   JPEG images: {jpeg_count}")
    print(f"   PNG images: {png_count}")
    print(f"   Total files: {len(all_files)}")

if __name__ == "__main__":
    convert_all_to_pdf()