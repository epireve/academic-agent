#!/usr/bin/env python3
"""
Create placeholder images for missing table references in Chapter 13
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import textwrap

def create_table_image(title, content, filename, width=800, height=600):
    """Create a simple table image with text content"""
    
    # Create image with white background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a system font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        content_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        # Fallback to default font
        title_font = ImageFont.load_default()
        content_font = ImageFont.load_default()
    
    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 20), title, fill='black', font=title_font)
    
    # Draw border
    draw.rectangle([20, 70, width-20, height-20], outline='black', width=2)
    
    # Draw content
    y_offset = 90
    for line in content:
        # Wrap text if too long
        wrapped_lines = textwrap.wrap(line, width=80)
        for wrapped_line in wrapped_lines:
            if y_offset < height - 40:
                draw.text((30, y_offset), wrapped_line, fill='black', font=content_font)
                y_offset += 20
    
    # Save image
    output_path = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes") / filename
    img.save(output_path)
    print(f"Created: {output_path}")
    return output_path

def create_missing_tables():
    """Create all missing table images"""
    
    # Table 13.1: Asset Valuation Example
    table_13_1_content = [
        "Asset Valuation Example",
        "",
        "Asset: Database Server",
        "Direct Costs:",
        "• Hardware replacement: $15,000",
        "• Software licensing: $5,000",
        "• Installation & configuration: $3,000",
        "",
        "Indirect Costs:",
        "• Business disruption (24 hours): $25,000",
        "• Lost productivity: $8,000",
        "• Reputation damage: $10,000",
        "",
        "Total Asset Value (AV): $66,000",
        "",
        "Exposure Factor (EF) examples:",
        "• Complete destruction: 100%",
        "• Partial damage: 50%",
        "• Minor corruption: 10%"
    ]
    
    create_table_image(
        "Table 13.1: Asset Valuation Example",
        table_13_1_content,
        "Table_13.1_Asset_Valuation_Example.png",
        width=600,
        height=500
    )
    
    # Table 13.2: Qualitative Risk Matrix
    table_13_2_content = [
        "Qualitative Risk Matrix",
        "",
        "           | Low Impact | Medium Impact | High Impact",
        "-----------|------------|---------------|------------",
        "High Likely|    Medium  |     High      |    High",
        "Med Likely |    Low     |     Medium    |    High",
        "Low Likely |    Low     |     Low       |    Medium",
        "",
        "Risk Levels:",
        "• High (Red): Immediate action required",
        "• Medium (Yellow): Action needed within 6 months",
        "• Low (Green): Monitor and review annually",
        "",
        "This matrix helps determine risk priority",
        "based on likelihood and impact assessment."
    ]
    
    create_table_image(
        "Table 13.2: Qualitative Risk Matrix",
        table_13_2_content,
        "Table_13.2_Qualitative_Risk_Matrix.png",
        width=700,
        height=500
    )
    
    # Table 13.3: Qualitative Ranking Scales
    table_13_3_content = [
        "Qualitative Ranking Scales",
        "",
        "Threat Likelihood:",
        "• High: Very likely to occur (>70% chance)",
        "• Medium: Moderately likely (30-70% chance)",
        "• Low: Unlikely to occur (<30% chance)",
        "",
        "Impact Severity:",
        "• Critical: Severe disruption, major financial loss",
        "• Major: Significant disruption, moderate loss",
        "• Minor: Limited disruption, minimal loss",
        "",
        "Vulnerability Assessment:",
        "• High: Multiple weaknesses, easy to exploit",
        "• Medium: Some weaknesses, moderate difficulty",
        "• Low: Few weaknesses, difficult to exploit"
    ]
    
    create_table_image(
        "Table 13.3: Qualitative Ranking Scales",
        table_13_3_content,
        "Table_13.3_Qualitative_Ranking_Scales.png",
        width=700,
        height=450
    )
    
    # Table 13.4: Methodology Comparison
    table_13_4_content = [
        "Risk Assessment Methodology Comparison",
        "",
        "Quantitative Methods:",
        "• ALE (Annualized Loss Expectancy)",
        "• FAIR (Factor Analysis of Information Risk)",
        "• Pros: Precise, mathematical, cost-benefit analysis",
        "• Cons: Complex, time-consuming, requires expertise",
        "",
        "Qualitative Methods:",
        "• OCTAVE (Operationally Critical Threat Assessment)",
        "• NIST SP 800-30",
        "• Pros: Faster, easier to understand, less data needed",
        "• Cons: Subjective, less precise, harder to justify",
        "",
        "Hybrid Methods:",
        "• Combines both approaches",
        "• Balances accuracy with practicality"
    ]
    
    create_table_image(
        "Table 13.4: Methodology Comparison",
        table_13_4_content,
        "Table_13.4_Methodology_Comparison.png",
        width=700,
        height=500
    )

if __name__ == "__main__":
    print("Creating missing table images for Chapter 13...")
    create_missing_tables()
    print("\nAll missing table images have been created!")