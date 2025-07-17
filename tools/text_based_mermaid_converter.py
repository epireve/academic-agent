#!/usr/bin/env python3
"""
Convert Mermaid diagrams to a text-based representation for PDF
"""

import re
from textwrap import dedent

class TextBasedMermaidConverter:
    def __init__(self):
        pass
    
    def parse_mermaid_to_text(self, mermaid_code):
        """Convert Mermaid code to a structured text representation"""
        lines = mermaid_code.strip().split('\n')
        
        # Detect diagram type
        diagram_type = "Flow Diagram"
        for line in lines:
            if 'graph TD' in line:
                diagram_type = "Top-Down Flow Diagram"
            elif 'graph LR' in line:
                diagram_type = "Left-Right Flow Diagram"
            elif 'graph BT' in line:
                diagram_type = "Bottom-Top Flow Diagram"
        
        # Parse nodes and connections
        nodes = {}
        connections = []
        subgraphs = []
        current_subgraph = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and graph declarations
            if not line or line.startswith('graph') or line == 'end':
                continue
            
            # Handle subgraphs
            if line.startswith('subgraph'):
                match = re.search(r'subgraph\s*"?([^"]*)"?', line)
                if match:
                    current_subgraph = match.group(1).strip()
                    subgraphs.append({'name': current_subgraph, 'nodes': []})
                continue
            
            # Parse node definitions
            node_match = re.search(r'([A-Za-z0-9_]+)\[(.*?)\]', line)
            if node_match:
                node_id = node_match.group(1)
                node_label = node_match.group(2)
                nodes[node_id] = {
                    'label': node_label,
                    'subgraph': current_subgraph
                }
                if current_subgraph and subgraphs:
                    subgraphs[-1]['nodes'].append(node_id)
            
            # Parse diamond/decision nodes
            decision_match = re.search(r'([A-Za-z0-9_]+)\{(.*?)\}', line)
            if decision_match:
                node_id = decision_match.group(1)
                node_label = decision_match.group(2)
                nodes[node_id] = {
                    'label': f"[Decision] {node_label}",
                    'subgraph': current_subgraph,
                    'type': 'decision'
                }
                if current_subgraph and subgraphs:
                    subgraphs[-1]['nodes'].append(node_id)
            
            # Parse connections
            if '-->' in line or '---' in line or '-.->':
                # Extract connection with optional label
                conn_match = re.search(r'([A-Za-z0-9_]+)\s*--[->.-]*\s*(?:\|([^|]*)\|)?\s*([A-Za-z0-9_]+)', line)
                if conn_match:
                    from_node = conn_match.group(1)
                    label = conn_match.group(2) or ""
                    to_node = conn_match.group(3)
                    
                    # Determine connection type
                    if '-.->':
                        conn_type = "dashed"
                    else:
                        conn_type = "solid"
                    
                    connections.append({
                        'from': from_node,
                        'to': to_node,
                        'label': label.strip() if label else "",
                        'type': conn_type
                    })
        
        # Generate text representation
        return self.format_as_text(diagram_type, nodes, connections, subgraphs)
    
    def format_as_text(self, diagram_type, nodes, connections, subgraphs):
        """Format the parsed data as structured text"""
        text_parts = []
        
        # Header
        text_parts.append(f"📊 {diagram_type}")
        text_parts.append("=" * 50)
        text_parts.append("")
        
        # Nodes section
        if nodes:
            text_parts.append("📦 Components/Nodes:")
            text_parts.append("-" * 30)
            
            # Group by subgraph
            ungrouped_nodes = []
            for node_id, node_info in nodes.items():
                if not node_info.get('subgraph'):
                    ungrouped_nodes.append((node_id, node_info))
            
            # Print ungrouped nodes first
            if ungrouped_nodes:
                for node_id, node_info in sorted(ungrouped_nodes):
                    node_type = node_info.get('type', 'process')
                    if node_type == 'decision':
                        text_parts.append(f"  ◆ {node_info['label']}")
                    else:
                        text_parts.append(f"  ▪ {node_info['label']}")
                text_parts.append("")
            
            # Print subgraphs
            for subgraph in subgraphs:
                if subgraph['nodes']:
                    text_parts.append(f"  📁 {subgraph['name']}:")
                    for node_id in subgraph['nodes']:
                        if node_id in nodes:
                            node_info = nodes[node_id]
                            node_type = node_info.get('type', 'process')
                            if node_type == 'decision':
                                text_parts.append(f"    ◆ {node_info['label']}")
                            else:
                                text_parts.append(f"    ▪ {node_info['label']}")
                    text_parts.append("")
        
        # Connections section
        if connections:
            text_parts.append("🔗 Relationships/Flow:")
            text_parts.append("-" * 30)
            
            for conn in connections:
                from_label = nodes.get(conn['from'], {}).get('label', conn['from'])
                to_label = nodes.get(conn['to'], {}).get('label', conn['to'])
                
                if conn['label']:
                    if conn['type'] == 'dashed':
                        text_parts.append(f"  • {from_label} ··→ {to_label}")
                        text_parts.append(f"    ({conn['label']})")
                    else:
                        text_parts.append(f"  • {from_label} → {to_label}")
                        text_parts.append(f"    ({conn['label']})")
                else:
                    if conn['type'] == 'dashed':
                        text_parts.append(f"  • {from_label} ··→ {to_label}")
                    else:
                        text_parts.append(f"  • {from_label} → {to_label}")
        
        text_parts.append("")
        text_parts.append("=" * 50)
        
        return '\n'.join(text_parts)
    
    def convert_to_html(self, mermaid_code):
        """Convert Mermaid code to HTML text representation"""
        text_representation = self.parse_mermaid_to_text(mermaid_code)
        
        # Convert to HTML with proper formatting
        html_lines = []
        for line in text_representation.split('\n'):
            if line.startswith('📊'):
                html_lines.append(f'<div class="diagram-title">{line}</div>')
            elif line.startswith('='):
                html_lines.append('<hr class="diagram-separator">')
            elif line.startswith('📦') or line.startswith('🔗'):
                html_lines.append(f'<div class="diagram-section-title">{line}</div>')
            elif line.startswith('-'):
                html_lines.append('<div class="diagram-subseparator"></div>')
            elif line.strip():
                # Preserve indentation
                indent_level = len(line) - len(line.lstrip())
                indent_class = f"indent-{indent_level // 2}"
                html_lines.append(f'<div class="diagram-item {indent_class}">{line.strip()}</div>')
            else:
                html_lines.append('<div class="diagram-spacer"></div>')
        
        html_content = f"""
        <div class="text-diagram-container">
            {''.join(html_lines)}
        </div>
        """
        
        return html_content

def test_text_converter():
    """Test the text-based converter"""
    converter = TextBasedMermaidConverter()
    
    test_mermaid = """
    graph TD
        A[Security Manager] --> B{Prioritize Initiatives}
        C[Audit Findings] -.-> D{Security Strategy}
        E[New Technology] -.-> D
        F[Compliance Mandates] -.-> D
        G[Security Risk Assessment] --> D
    """
    
    result = converter.parse_mermaid_to_text(test_mermaid)
    print(result)
    
    html_result = converter.convert_to_html(test_mermaid)
    print("\nHTML length:", len(html_result))

if __name__ == "__main__":
    test_text_converter()