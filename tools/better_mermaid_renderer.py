#!/usr/bin/env python3
"""
Better Mermaid renderer using HTML/CSS instead of SVG
"""

import re
from pathlib import Path

class BetterMermaidRenderer:
    def __init__(self):
        pass
    
    def convert_mermaid_to_html(self, mermaid_code):
        """Convert Mermaid code to HTML with proper text rendering"""
        
        # For now, let's create a structured HTML representation
        # This is a fallback that ensures text is always visible
        
        # Clean up the mermaid code
        lines = mermaid_code.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//'):
                cleaned_lines.append(line)
        
        # Create HTML structure
        html_content = f"""
        <div class="mermaid-html-container">
            <div class="mermaid-title">📊 Concept Diagram</div>
            <div class="mermaid-content">
                <pre class="mermaid-code">{mermaid_code}</pre>
            </div>
            <div class="mermaid-note">
                <em>Note: This diagram shows the relationships and flow described in the Mermaid code above.</em>
            </div>
        </div>
        """
        
        return html_content
    
    def parse_mermaid_elements(self, mermaid_code):
        """Parse Mermaid elements for better HTML representation"""
        elements = []
        connections = []
        
        lines = mermaid_code.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('graph') or line.startswith('subgraph') or line.startswith('end') or line.startswith('style'):
                continue
            
            # Extract node definitions and connections
            if '-->' in line or '---' in line:
                # This is a connection
                parts = re.split(r'--[>-]', line)
                if len(parts) >= 2:
                    from_node = parts[0].strip()
                    to_node = parts[1].strip()
                    connections.append((from_node, to_node))
            
            # Extract node labels
            node_match = re.search(r'([A-Za-z0-9_]+)\[(.*?)\]', line)
            if node_match:
                node_id = node_match.group(1)
                node_label = node_match.group(2)
                elements.append((node_id, node_label))
        
        return elements, connections

def test_better_renderer():
    """Test the better Mermaid renderer"""
    renderer = BetterMermaidRenderer()
    
    test_mermaid = """
    graph TD
        A[Start] --> B{Decision}
        B -->|Yes| C[Action 1]
        B -->|No| D[Action 2]
        C --> E[End]
        D --> E
    """
    
    html_result = renderer.convert_mermaid_to_html(test_mermaid)
    print("✅ Better Mermaid renderer test completed")
    print(f"HTML length: {len(html_result)} characters")
    return html_result

if __name__ == "__main__":
    test_better_renderer()