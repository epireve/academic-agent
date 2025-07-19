#!/usr/bin/env python3
"""
Convert all Mermaid diagrams to high-quality PNG images
"""

import subprocess
from pathlib import Path
import json

def convert_all_to_png():
    """Convert all Mermaid diagrams to PNG with optimal settings"""
    
    diagrams_dir = Path(get_output_manager().get_output_path(OutputCategory.ASSETS, ContentType.DIAGRAMS, subdirectory="mermaid"))
    output_dir = diagrams_dir / "png_output"
    output_dir.mkdir(exist_ok=True)
    
    # Create optimized config for better text rendering
    config_file = diagrams_dir / "optimal_config.json"
    config_content = {
        "theme": "default",
        "themeVariables": {
            "fontFamily": "Arial, Helvetica, sans-serif",
            "fontSize": "16px",
            "primaryColor": "#ffffff",
            "primaryTextColor": "#000000",
            "primaryBorderColor": "#7C0000",
            "lineColor": "#333333",
            "secondaryColor": "#006100",
            "tertiaryColor": "#fff",
            "background": "#ffffff",
            "mainBkg": "#ffffff",
            "secondBkg": "#f4f4f4",
            "tertiaryBkg": "#fff"
        },
        "flowchart": {
            "htmlLabels": True,
            "curve": "basis",
            "diagramPadding": 8,
            "nodeSpacing": 50,
            "rankSpacing": 50
        }
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_content, f, indent=2)
    
    # Find all .mmd files
    mmd_files = sorted(diagrams_dir.glob("week_*_diagram_*.mmd"))
    
    print(f"🔄 Converting {len(mmd_files)} Mermaid diagrams to PNG...\n")
    
    successful = 0
    failed = 0
    
    for mmd_file in mmd_files:
        output_file = output_dir / f"{mmd_file.stem}.png"
        
        print(f"Converting: {mmd_file.name}")
        
        cmd = [
            'npx', '@mermaid-js/mermaid-cli',
            '-i', str(mmd_file),
            '-o', str(output_file),
            '-c', str(config_file),
            '--width', '2400',  # High resolution
            '--height', '1800',
            '--scale', '3',     # 3x scale for crisp text
            '--backgroundColor', 'white'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_file.exists():
                file_size_kb = output_file.stat().st_size / 1024
                print(f"  ✅ Success! Size: {file_size_kb:.1f} KB")
                successful += 1
            else:
                print(f"  ❌ Failed: {result.stderr}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    # Clean up config
    if config_file.exists():
        config_file.unlink()
    
    print(f"\n📊 Conversion Summary:")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 PNG files saved to: {output_dir}")
    
    # Create an HTML gallery to view all PNGs
    gallery_html = output_dir / "gallery.html"
    with open(gallery_html, 'w') as f:
        f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Mermaid Diagrams Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .diagram-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .diagram-card h2 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 18px;
        }
        .diagram-card img {
            width: 100%;
            height: auto;
            border: 1px solid #eee;
            border-radius: 4px;
        }
        .note {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Mermaid Diagrams Gallery - PNG Format</h1>
    <div class="note">
        These are high-resolution PNG images (2400x1800 @ 3x scale) for optimal text clarity.
    </div>
    <div class="gallery">
""")
        
        for png_file in sorted(output_dir.glob("*.png")):
            week_num = png_file.stem.split('_')[1]
            f.write(f"""
        <div class="diagram-card">
            <h2>Week {week_num} - Concept Diagram</h2>
            <img src="{png_file.name}" alt="Week {week_num} Diagram">
        </div>
""")
        
        f.write("""
    </div>
</body>
</html>
""")
    
    print(f"\n🌐 Gallery created: {gallery_html}")
    print("   Open this file to view all PNG diagrams")

if __name__ == "__main__":
    convert_all_to_png()