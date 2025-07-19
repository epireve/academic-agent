#!/usr/bin/env python3
"""
Extract all Mermaid diagrams from markdown files and save them separately
"""

import re
from pathlib import Path

def extract_mermaid_diagrams():
    """Extract all Mermaid diagrams from study notes"""
    
    # Paths
    input_dir = Path(get_final_output_path(ContentType.STUDY_NOTES))
    output_dir = Path(get_output_manager().get_output_path(OutputCategory.ASSETS, ContentType.DIAGRAMS, subdirectory="mermaid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pattern to find Mermaid diagrams
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    
    # Track all diagrams found
    all_diagrams = []
    
    # Process each markdown file
    md_files = sorted(input_dir.glob("week_*.md"))
    
    print(f"🔍 Scanning {len(md_files)} markdown files for Mermaid diagrams...\n")
    
    for md_file in md_files:
        print(f"📄 Processing: {md_file.name}")
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all Mermaid diagrams in this file
        matches = list(re.finditer(mermaid_pattern, content, re.DOTALL))
        
        if matches:
            print(f"   ✅ Found {len(matches)} diagram(s)")
            
            for i, match in enumerate(matches):
                mermaid_code = match.group(1).strip()
                
                # Save each diagram
                diagram_filename = f"{md_file.stem}_diagram_{i+1}.mmd"
                diagram_path = output_dir / diagram_filename
                
                with open(diagram_path, 'w', encoding='utf-8') as f:
                    f.write(mermaid_code)
                
                # Also create a more readable version with comments
                readable_filename = f"{md_file.stem}_diagram_{i+1}_readable.txt"
                readable_path = output_dir / readable_filename
                
                with open(readable_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Mermaid Diagram from {md_file.name}\n")
                    f.write(f"# Diagram {i+1} of {len(matches)}\n")
                    f.write("# " + "="*50 + "\n\n")
                    f.write(mermaid_code)
                    f.write("\n\n# " + "="*50 + "\n")
                    f.write("# To render this diagram:\n")
                    f.write("# 1. Copy the code above (without these comments)\n")
                    f.write("# 2. Paste into https://mermaid.live/\n")
                    f.write("# 3. Or use any Mermaid-compatible tool\n")
                
                all_diagrams.append({
                    'file': md_file.name,
                    'index': i+1,
                    'code': mermaid_code,
                    'output_file': diagram_filename
                })
        else:
            print(f"   ⚠️  No diagrams found")
    
    # Create a master index file
    index_path = output_dir / "README.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# Mermaid Diagrams Index\n\n")
        f.write(f"Total diagrams extracted: {len(all_diagrams)}\n\n")
        
        current_file = None
        for diagram in all_diagrams:
            if diagram['file'] != current_file:
                current_file = diagram['file']
                f.write(f"\n## {current_file}\n\n")
            
            f.write(f"- Diagram {diagram['index']}: `{diagram['output_file']}`\n")
            
            # Add first few lines of the diagram as preview
            lines = diagram['code'].split('\n')[:3]
            preview = '\n  '.join(lines)
            f.write(f"  ```mermaid\n  {preview}\n  ...\n  ```\n\n")
    
    print(f"\n✅ Extraction complete!")
    print(f"📁 {len(all_diagrams)} diagrams saved to: {output_dir}")
    print(f"📋 Index created: {index_path}")
    
    # Also create a test HTML file to view diagrams
    test_html_path = output_dir / "test_diagrams.html"
    with open(test_html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Mermaid Diagram Test</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                htmlLabels: true,
                curve: 'basis'
            }
        });
    </script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        .diagram-container {
            margin: 20px 0;
            padding: 20px;
            border: 1px solid #ccc;
            background: #f5f5f5;
        }
        h2 {
            color: #333;
        }
        .mermaid {
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Mermaid Diagrams Test Page</h1>
    <p>This page tests rendering of all extracted Mermaid diagrams.</p>
""")
        
        for diagram in all_diagrams:
            f.write(f"""
    <div class="diagram-container">
        <h2>{diagram['file']} - Diagram {diagram['index']}</h2>
        <div class="mermaid">
{diagram['code']}
        </div>
    </div>
""")
        
        f.write("""
</body>
</html>
""")
    
    print(f"🌐 Test HTML created: {test_html_path}")
    print("   Open this file in a browser to see all diagrams rendered")

if __name__ == "__main__":
    extract_mermaid_diagrams()