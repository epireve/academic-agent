#!/usr/bin/env python3
"""
Test different methods to render Mermaid diagrams
"""

import subprocess
import os
from pathlib import Path
import tempfile

def test_mermaid_cli_rendering():
    """Test rendering with different mermaid-cli options"""
    
    diagrams_dir = Path(get_output_manager().get_output_path(OutputCategory.ASSETS, ContentType.DIAGRAMS, subdirectory="mermaid"))
    test_diagram = diagrams_dir / "week_01_comprehensive_study_notes_diagram_1.mmd"
    
    if not test_diagram.exists():
        print("❌ Test diagram not found!")
        return
    
    print("🧪 Testing Mermaid rendering methods...\n")
    
    # Test 1: Basic PNG render
    print("Test 1: Basic PNG render")
    try:
        output_file = diagrams_dir / "test_basic.png"
        cmd = ['npx', '@mermaid-js/mermaid-cli', '-i', str(test_diagram), '-o', str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_file.exists():
            print(f"✅ Success! File size: {output_file.stat().st_size} bytes")
        else:
            print(f"❌ Failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: PNG with explicit config
    print("\nTest 2: PNG with config")
    try:
        output_file = diagrams_dir / "test_config.png"
        config_file = diagrams_dir / "test_config.json"
        
        # Create config
        config_content = """{
  "theme": "default",
  "themeVariables": {
    "fontFamily": "Arial, sans-serif",
    "fontSize": "16px",
    "primaryColor": "#fff",
    "primaryTextColor": "#000",
    "primaryBorderColor": "#7C0000",
    "lineColor": "#333",
    "background": "#fff"
  }
}"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        cmd = [
            'npx', '@mermaid-js/mermaid-cli', 
            '-i', str(test_diagram), 
            '-o', str(output_file),
            '-c', str(config_file),
            '--width', '1200',
            '--height', '800',
            '--backgroundColor', 'white'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_file.exists():
            print(f"✅ Success! File size: {output_file.stat().st_size} bytes")
        else:
            print(f"❌ Failed: {result.stderr}")
            
        # Clean up
        if config_file.exists():
            os.unlink(config_file)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: SVG render
    print("\nTest 3: SVG render")
    try:
        output_file = diagrams_dir / "test_basic.svg"
        cmd = ['npx', '@mermaid-js/mermaid-cli', '-i', str(test_diagram), '-o', str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_file.exists():
            print(f"✅ Success! File size: {output_file.stat().st_size} bytes")
            
            # Check if SVG contains text
            with open(output_file, 'r') as f:
                svg_content = f.read()
                if 'Security Manager' in svg_content:
                    print("✅ Text found in SVG!")
                else:
                    print("⚠️  Text might be missing in SVG")
        else:
            print(f"❌ Failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: PDF render
    print("\nTest 4: PDF render")
    try:
        output_file = diagrams_dir / "test_basic.pdf"
        cmd = ['npx', '@mermaid-js/mermaid-cli', '-i', str(test_diagram), '-o', str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_file.exists():
            print(f"✅ Success! File size: {output_file.stat().st_size} bytes")
        else:
            print(f"❌ Failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 5: Using different puppeteer config
    print("\nTest 5: PNG with puppeteer config")
    try:
        output_file = diagrams_dir / "test_puppeteer.png"
        puppeteer_config = diagrams_dir / "puppeteer-config.json"
        
        # Create puppeteer config
        puppeteer_content = """{
  "args": ["--no-sandbox", "--disable-setuid-sandbox"]
}"""
        
        with open(puppeteer_config, 'w') as f:
            f.write(puppeteer_content)
        
        cmd = [
            'npx', '@mermaid-js/mermaid-cli', 
            '-i', str(test_diagram), 
            '-o', str(output_file),
            '-p', str(puppeteer_config),
            '--width', '1600',
            '--height', '1200',
            '--scale', '2'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_file.exists():
            print(f"✅ Success! File size: {output_file.stat().st_size} bytes")
        else:
            print(f"❌ Failed: {result.stderr}")
            
        # Clean up
        if puppeteer_config.exists():
            os.unlink(puppeteer_config)
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n📁 Check the outputs in:", diagrams_dir)
    print("🔍 Look for:")
    print("   - test_basic.png")
    print("   - test_config.png")
    print("   - test_basic.svg")
    print("   - test_basic.pdf")
    print("   - test_puppeteer.png")

if __name__ == "__main__":
    test_mermaid_cli_rendering()