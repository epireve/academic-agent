#!/usr/bin/env python3
"""
Convert Mermaid diagrams to SVG using Node.js mermaid-cli
"""

import os
import subprocess
import tempfile
from pathlib import Path
import json

class MermaidToSVGConverter:
    def __init__(self):
        self.check_dependencies()
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            # Check if node is available
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Node.js is not installed. Please install Node.js first.")
                return False
            
            # Check if mermaid-cli is available
            result = subprocess.run(['npx', '@mermaid-js/mermaid-cli', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️  mermaid-cli not found. Installing...")
                self.install_mermaid_cli()
            
            return True
            
        except FileNotFoundError:
            print("❌ Node.js is not installed. Please install Node.js first.")
            return False
    
    def install_mermaid_cli(self):
        """Install mermaid-cli using npm"""
        try:
            result = subprocess.run(['npm', 'install', '-g', '@mermaid-js/mermaid-cli'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ mermaid-cli installed successfully")
            else:
                print(f"❌ Failed to install mermaid-cli: {result.stderr}")
        except Exception as e:
            print(f"❌ Error installing mermaid-cli: {e}")
    
    def convert_mermaid_to_svg(self, mermaid_code):
        """Convert Mermaid diagram code to SVG"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as mmd_file:
                mmd_file.write(mermaid_code)
                mmd_file_path = mmd_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as svg_file:
                svg_file_path = svg_file.name
            
            # Run mermaid-cli to convert to SVG
            cmd = ['npx', '@mermaid-js/mermaid-cli', '-i', mmd_file_path, '-o', svg_file_path, 
                   '--theme', 'default', '--width', '800', '--height', '600']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read the generated SVG
                with open(svg_file_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                
                # Clean up temporary files
                os.unlink(mmd_file_path)
                os.unlink(svg_file_path)
                
                return svg_content
            else:
                print(f"❌ Error converting Mermaid diagram: {result.stderr}")
                # Clean up temporary files
                os.unlink(mmd_file_path)
                if os.path.exists(svg_file_path):
                    os.unlink(svg_file_path)
                return None
                
        except Exception as e:
            print(f"❌ Error in Mermaid conversion: {e}")
            return None

def test_converter():
    """Test the Mermaid converter"""
    converter = MermaidToSVGConverter()
    
    test_mermaid = """
    graph TD
        A[Start] --> B{Decision}
        B -->|Yes| C[Action 1]
        B -->|No| D[Action 2]
        C --> E[End]
        D --> E
    """
    
    svg_result = converter.convert_mermaid_to_svg(test_mermaid)
    
    if svg_result:
        print("✅ Mermaid conversion test successful!")
        print(f"SVG length: {len(svg_result)} characters")
        return True
    else:
        print("❌ Mermaid conversion test failed!")
        return False

if __name__ == "__main__":
    test_converter()