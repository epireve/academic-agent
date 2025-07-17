#!/usr/bin/env python3
"""
Convert Mermaid diagrams to PNG images
"""

import os
import subprocess
import tempfile
from pathlib import Path
import base64

class MermaidToPNGConverter:
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
    
    def convert_mermaid_to_png(self, mermaid_code):
        """Convert Mermaid diagram code to PNG"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as mmd_file:
                mmd_file.write(mermaid_code)
                mmd_file_path = mmd_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as png_file:
                png_file_path = png_file.name
            
            # Run mermaid-cli to convert to PNG with high quality settings
            cmd = [
                'npx', '@mermaid-js/mermaid-cli', 
                '-i', mmd_file_path, 
                '-o', png_file_path, 
                '--theme', 'default',
                '--width', '1600',  # High resolution for better quality
                '--height', '1200',
                '--backgroundColor', 'white',
                '--scale', '2'  # 2x scale for better quality
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read the generated PNG
                with open(png_file_path, 'rb') as f:
                    png_data = f.read()
                
                # Convert to base64 for embedding
                png_base64 = base64.b64encode(png_data).decode('utf-8')
                
                # Clean up temporary files
                os.unlink(mmd_file_path)
                os.unlink(png_file_path)
                
                return png_base64
            else:
                print(f"❌ Error converting Mermaid diagram: {result.stderr}")
                # Clean up temporary files
                os.unlink(mmd_file_path)
                if os.path.exists(png_file_path):
                    os.unlink(png_file_path)
                return None
                
        except Exception as e:
            print(f"❌ Error in Mermaid conversion: {e}")
            return None

def test_png_converter():
    """Test the PNG converter"""
    converter = MermaidToPNGConverter()
    
    test_mermaid = """
    graph TD
        A[Start] --> B{Decision}
        B -->|Yes| C[Action 1]
        B -->|No| D[Action 2]
        C --> E[End]
        D --> E
    """
    
    png_result = converter.convert_mermaid_to_png(test_mermaid)
    
    if png_result:
        print("✅ Mermaid to PNG conversion test successful!")
        print(f"PNG base64 length: {len(png_result)} characters")
        return True
    else:
        print("❌ Mermaid to PNG conversion test failed!")
        return False

if __name__ == "__main__":
    test_png_converter()