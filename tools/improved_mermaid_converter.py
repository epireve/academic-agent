#!/usr/bin/env python3
"""
Improved Mermaid to SVG converter with better text rendering
"""

import os
import subprocess
import tempfile
from pathlib import Path
import json
import re

class ImprovedMermaidConverter:
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
    
    def create_mermaid_config(self):
        """Create a configuration file for better text rendering"""
        config = {
            "theme": "default",
            "width": 800,
            "height": 600,
            "backgroundColor": "white",
            "configFile": None,
            "cssFile": None,
            "scale": 1,
            "themeVariables": {
                "primaryColor": "#3498db",
                "primaryTextColor": "#2c3e50",
                "primaryBorderColor": "#3498db",
                "lineColor": "#34495e",
                "secondaryColor": "#ecf0f1",
                "tertiaryColor": "#f8f9fa",
                "background": "#ffffff",
                "mainBkg": "#ffffff",
                "secondBkg": "#f8f9fa",
                "tertiaryBkg": "#ecf0f1"
            },
            "flowchart": {
                "htmlLabels": True,
                "curve": "basis"
            },
            "sequence": {
                "diagramMarginX": 50,
                "diagramMarginY": 10,
                "actorMargin": 50,
                "width": 150,
                "height": 65,
                "boxMargin": 10,
                "boxTextMargin": 5,
                "noteMargin": 10,
                "messageMargin": 35,
                "mirrorActors": True,
                "bottomMarginAdj": 1,
                "useMaxWidth": True
            },
            "gantt": {
                "leftPadding": 75,
                "gridLineStartPadding": 35,
                "fontSize": 11,
                "fontFamily": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif"
            }
        }
        
        return json.dumps(config, indent=2)
    
    def convert_mermaid_to_svg(self, mermaid_code):
        """Convert Mermaid diagram code to SVG with improved text rendering"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as mmd_file:
                mmd_file.write(mermaid_code)
                mmd_file_path = mmd_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as svg_file:
                svg_file_path = svg_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
                config_file.write(self.create_mermaid_config())
                config_file_path = config_file.name
            
            # Run mermaid-cli to convert to SVG with config
            cmd = [
                'npx', '@mermaid-js/mermaid-cli', 
                '-i', mmd_file_path, 
                '-o', svg_file_path,
                '-c', config_file_path,
                '--theme', 'default',
                '--width', '800',
                '--height', '600',
                '--backgroundColor', 'white'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read the generated SVG
                with open(svg_file_path, 'r', encoding='utf-8') as f:
                    svg_content = f.read()
                
                # Post-process SVG to ensure proper text rendering
                svg_content = self.post_process_svg(svg_content)
                
                # Clean up temporary files
                os.unlink(mmd_file_path)
                os.unlink(svg_file_path)
                os.unlink(config_file_path)
                
                return svg_content
            else:
                print(f"❌ Error converting Mermaid diagram: {result.stderr}")
                # Clean up temporary files
                os.unlink(mmd_file_path)
                if os.path.exists(svg_file_path):
                    os.unlink(svg_file_path)
                if os.path.exists(config_file_path):
                    os.unlink(config_file_path)
                return None
                
        except Exception as e:
            print(f"❌ Error in Mermaid conversion: {e}")
            return None
    
    def post_process_svg(self, svg_content):
        """Post-process SVG to ensure proper text rendering"""
        if not svg_content:
            return svg_content
        
        # Add proper font family and text styling
        svg_content = re.sub(
            r'<svg([^>]*)>',
            r'<svg\1 style="font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;">',
            svg_content
        )
        
        # Ensure all text elements have proper styling
        svg_content = re.sub(
            r'<text([^>]*?)(?:style="[^"]*")?([^>]*?)>',
            r'<text\1 style="fill: #2c3e50; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; font-size: 12px; font-weight: 400;"\2>',
            svg_content
        )
        
        # Fix any transparent or white text
        svg_content = re.sub(r'fill="transparent"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="white"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="#ffffff"', r'fill="#2c3e50"', svg_content)
        svg_content = re.sub(r'fill="#fff"', r'fill="#2c3e50"', svg_content)
        
        # Ensure proper stroke colors for visibility
        svg_content = re.sub(r'stroke="transparent"', r'stroke="#34495e"', svg_content)
        
        # Add explicit text styling for better rendering
        svg_content = re.sub(
            r'(<text[^>]*>)([^<]+)(</text>)',
            r'\1<tspan style="fill: #2c3e50; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif; font-size: 12px;">\2</tspan>\3',
            svg_content
        )
        
        return svg_content

def test_improved_converter():
    """Test the improved Mermaid converter"""
    converter = ImprovedMermaidConverter()
    
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
        print("✅ Improved Mermaid conversion test successful!")
        print(f"SVG length: {len(svg_result)} characters")
        # Check if text styling is present
        if 'font-family' in svg_result and 'fill=' in svg_result:
            print("✅ Text styling detected in SVG")
        else:
            print("⚠️  Text styling may be missing")
        return True
    else:
        print("❌ Improved Mermaid conversion test failed!")
        return False

if __name__ == "__main__":
    test_improved_converter()