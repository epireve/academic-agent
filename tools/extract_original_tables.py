#!/usr/bin/env python3
"""
Extract the original tables from Chapter 13 textbook and convert them to images
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap
import re

from src.core.output_manager import get_output_manager, get_final_output_path, get_processed_output_path, get_analysis_output_path
from src.core.output_manager import OutputCategory, ContentType


def create_table_image_from_markdown(table_markdown, title, filename, width=900, height=700):
    """Create a table image from markdown table format"""
    
    # Create image with white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use system fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        cell_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        cell_font = ImageFont.load_default()
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 15), title, fill='black', font=title_font)
    
    # Parse markdown table
    lines = table_markdown.strip().split('\n')
    table_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('|---')]
    
    # Parse table data
    table_data = []
    for line in table_lines:
        if line.startswith('|') and line.endswith('|'):
            cells = [cell.strip() for cell in line[1:-1].split('|')]
            table_data.append(cells)
    
    if not table_data:
        return None
    
    # Calculate cell dimensions
    num_cols = len(table_data[0])
    num_rows = len(table_data)
    
    cell_width = (width - 60) // num_cols
    cell_height = 30
    
    # Starting position
    start_x = 30
    start_y = 60
    
    # Draw table
    for row_idx, row in enumerate(table_data):
        for col_idx, cell in enumerate(row):
            # Calculate cell position
            x = start_x + (col_idx * cell_width)
            y = start_y + (row_idx * cell_height)
            
            # Draw cell border
            draw.rectangle([x, y, x + cell_width, y + cell_height], outline='black', width=1)
            
            # Fill header cells
            if row_idx == 0:
                draw.rectangle([x+1, y+1, x + cell_width-1, y + cell_height-1], fill='#f0f0f0')
                font = header_font
            else:
                font = cell_font
            
            # Clean and wrap text
            clean_text = cell.replace('<br>', ' ').replace('\\$', '$').strip()
            
            # Handle multi-line cells
            if len(clean_text) > 25:
                wrapped_lines = textwrap.wrap(clean_text, width=20)
                for i, line in enumerate(wrapped_lines[:3]):  # Max 3 lines per cell
                    draw.text((x + 5, y + 5 + i * 10), line, fill='black', font=font)
            else:
                draw.text((x + 5, y + 8), clean_text, fill='black', font=font)
    
    # Save image
    output_path = Path(get_final_output_path(ContentType.STUDY_NOTES)) / filename
    img.save(output_path)
    print(f"Created: {output_path}")
    return output_path

def extract_original_tables():
    """Extract the original tables from Chapter 13 textbook"""
    
    # Table 13.1: Quantitative Measurements
    table_13_1 = """| Asset Valuation Components | Value | Justification |
|-------------------------------|-----------|----------------------------------------------------------------------------------------|
| Direct costs |  |  |
| Building | $100,000 | Cost to rebuild |
| Inventory | $50,000 | Cost to organization |
| Equipment | $48,000 | Replacement cost |
| Indirect costs |  |  |
| Lost business | $24,000 | 4 weeks to return to normal operations; loss of $6,000 profit from orders per week |
| Lost reputation | $31,200* | Expected loss of business—10% of one year's business |
| Employee endangerment | $90,000 | Risk of life is 3%; value of life = $3 million |"""
    
    create_table_image_from_markdown(
        table_13_1,
        "Table 13.1: Quantitative Measurements",
        "Table_13.1_Asset_Valuation_Example.png",
        width=1000,
        height=400
    )
    
    # Table 13.2: Example Qualitative Risk Determination
    table_13_2 = """| Impact Severity Level | Vulnerability Likelihood of Occurrence |  |  |  |  |
|-----------------------------|----------------------------------------|------------|---------------|--------------|----------|
|  | A-Frequent | B-Probable | C-Conceivable | D-Improbable | E-Remote |
| 1 | Risk I | Risk I | Risk I | Risk II | Risk III |
| 2 | Risk I | Risk I | Risk II | Risk II | Risk III |
| 3 | Risk I | Risk II | Risk II | Risk III | Risk III |
| 4 | Risk III | Risk III | Risk IV | Risk IV | Risk IV |"""
    
    create_table_image_from_markdown(
        table_13_2,
        "Table 13.2: Example Qualitative Risk Determination",
        "Table_13.2_Qualitative_Risk_Matrix.png",
        width=800,
        height=300
    )
    
    # Table 13.3: Qualitative Values
    table_13_3 = """| Level | Attempt | Exploit | Impact |
|-------|-------------|---------------------|------------------------------------------------|
| 1 | Likely | Easy | Exposure or loss of proprietary information |
|  |  |  | Loss of integrity of critical information |
|  |  |  | System disruption |
|  |  |  | Major structural damage |
|  |  |  | Loss of physical access control |
|  |  |  | Exposure or loss of sensitive information |
|  |  |  | Grave danger to building occupants |
| 2 | Conceivable | Moderate | Major system damage |
|  |  |  | Significant structural damage |
|  |  |  | Risks to access controls |
|  |  |  | Potential exposure to sensitive information |
|  |  |  | Serious danger to building occupants |
| 3 | Improbable | Difficult | Minor system damage or exposure |
|  |  |  | Some structural damage |
|  |  |  | Reduced access control effectiveness |
|  |  |  | Moderate exposure to sensitive information |
|  |  |  | Moderate danger to building occupants |
| 4 | Remote | Extremely difficult | Less than minor system damage or exposure |
|  |  |  | Extremely limited structural damage |
|  |  |  | Potential effect on access controls |
|  |  |  | Control of sensitive information |
|  |  |  | Safety of building occupants |"""
    
    create_table_image_from_markdown(
        table_13_3,
        "Table 13.3: Qualitative Values",
        "Table_13.3_Qualitative_Ranking_Scales.png",
        width=1000,
        height=600
    )
    
    # Table 13.4: Security Risk Assessment Methods - simplified version
    table_13_4 = """| Security Risk Assessment Approach | Type | Key Phases | Resources Required | Application |
|------------------------------------------------|--------------------------------|----------------------------------------|----------------------------------|----------------------------------------|
| FAA SRM | Open qualitative method | Asset identification, Threat identification, Risk determination | Program managers, Security representatives | FAA projects, General-purpose method |
| OCTAVE | Open qualitative method | Profile threats and assets, Identify vulnerabilities, Develop security strategy | Internal, nonexperts | Large corporations with ability to run own tools |
| FRAP | Open quantitative method | Pre-FRAP meeting, FRAP session, Post-FRAP process | Facilitator and internal manager | Gap and initial assessment where time is essential |
| CRAMM | Commercial qualitative tool | Asset identification, Threat assessment, Countermeasure recommendation | Qualified and experienced practitioners | Demonstrating BS 7799 compliance |
| NSA IAM | Open qualitative method | Pre-assessment, On-site visit, Post-assessment | IAM-trained providers | Government agencies and critical infrastructure |"""
    
    create_table_image_from_markdown(
        table_13_4,
        "Table 13.4: Security Risk Assessment Methods",
        "Table_13.4_Methodology_Comparison.png",
        width=1200,
        height=350
    )

if __name__ == "__main__":
    print("Creating original table images from Chapter 13 textbook...")
    extract_original_tables()
    print("\nAll original table images have been created from the textbook source!")