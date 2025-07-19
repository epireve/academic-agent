#!/usr/bin/env python3
"""
Consolidate all images (textbook, tables, and Mermaid diagrams) into the study notes folder
"""

import shutil
from pathlib import Path
import re

def copy_textbook_images():
    """Copy all textbook images with new naming convention"""
    
    # Source and destination directories
    textbook_base = Path("/Users/invoture/dev.local/academic-agent/output/sra/textbook/markdown")
    dest_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes")
    
    # Mapping of original paths to new names
    image_mappings = {
        "Chapter_1_Introduction/_page_3_Figure_1.jpeg": "Chapter_01_Page03_Figure01.jpeg",
        "Chapter_1_Introduction/_page_4_Figure_1.jpeg": "Chapter_01_Page04_Figure01.jpeg",
        "Chapter_1_Introduction/_page_8_Figure_1.jpeg": "Chapter_01_Page08_Figure01.jpeg",
        "Chapter_2_Information_Security_Risk_Assessment_Basics/_page_1_Figure_1.jpeg": "Chapter_02_Page01_Figure01.jpeg",
        "Chapter_3_Project_Definition/_page_18_Figure_12.jpeg": "Chapter_03_Page18_Figure12.jpeg",
        "Chapter_3_Project_Definition/_page_19_Figure_5.jpeg": "Chapter_03_Page19_Figure05.jpeg",
        "Chapter_4_Security_Risk_Assessment_Preparation/_page_28_Figure_8.jpeg": "Chapter_04_Page28_Figure08.jpeg",
        "Chapter_5_Data_Gathering/_page_11_Figure_4.jpeg": "Chapter_05_Page11_Figure04.jpeg",
        "Chapter_5_Data_Gathering/_page_14_Figure_1.jpeg": "Chapter_05_Page14_Figure01.jpeg",
        "Chapter_6_Administrative_Data_Gathering/_page_8_Figure_1.jpeg": "Chapter_06_Page08_Figure01.jpeg",
        "Chapter_7_Technical_Data_Gathering/_page_11_Figure_5.jpeg": "Chapter_07_Page11_Figure05.jpeg",
        "Chapter_7_Technical_Data_Gathering/_page_13_Figure_7.jpeg": "Chapter_07_Page13_Figure07.jpeg",
        "Chapter_7_Technical_Data_Gathering/_page_14_Figure_7.jpeg": "Chapter_07_Page14_Figure07.jpeg",
        "Chapter_7_Technical_Data_Gathering/_page_28_Figure_7.jpeg": "Chapter_07_Page28_Figure07.jpeg",
        "Chapter_8_Physical_Data_Gathering/_page_9_Figure_1.jpeg": "Chapter_08_Page09_Figure01.jpeg",
        "Chapter_8_Physical_Data_Gathering/_page_11_Figure_1.jpeg": "Chapter_08_Page11_Figure01.jpeg",
        "Chapter_8_Physical_Data_Gathering/_page_19_Figure_6.jpeg": "Chapter_08_Page19_Figure06.jpeg",
        "Chapter_8_Physical_Data_Gathering/_page_21_Figure_6.jpeg": "Chapter_08_Page21_Figure06.jpeg",
        "Chapter_8_Physical_Data_Gathering/_page_22_Figure_1.jpeg": "Chapter_08_Page22_Figure01.jpeg",
        "Chapter_8_Physical_Data_Gathering/_page_22_Picture_3.jpeg": "Chapter_08_Page22_Picture03.jpeg",
        "Chapter_9_Security_Risk_Analysis/_page_1_Figure_8.jpeg": "Chapter_09_Page01_Figure08.jpeg",
        "Chapter_9_Security_Risk_Analysis/_page_2_Figure_7.jpeg": "Chapter_09_Page02_Figure07.jpeg",
        "Chapter_10_Security_Risk_Mitigation/_page_4_Figure_1.jpeg": "Chapter_10_Page04_Figure01.jpeg",
        "Chapter_12_Security_Risk_Assessment_Project_Management/_page_3_Figure_1.jpeg": "Chapter_12_Page03_Figure01.jpeg",
        "Chapter_12_Security_Risk_Assessment_Project_Management/_page_18_Figure_5.jpeg": "Chapter_12_Page18_Figure05.jpeg"
    }
    
    copied_count = 0
    for original_path, new_name in image_mappings.items():
        source_file = textbook_base / original_path
        dest_file = dest_dir / new_name
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"✅ Copied: {new_name}")
            copied_count += 1
        else:
            print(f"❌ Not found: {source_file}")
    
    print(f"\n📚 Copied {copied_count} textbook images")
    return copied_count

def copy_mermaid_diagrams():
    """Copy all Mermaid PNG diagrams to study notes folder"""
    
    source_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/mermaid_diagrams/png_output")
    dest_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes")
    
    # Copy all week_*.png files
    copied_count = 0
    for png_file in source_dir.glob("week_*.png"):
        dest_file = dest_dir / png_file.name
        shutil.copy2(png_file, dest_file)
        print(f"✅ Copied: {png_file.name}")
        copied_count += 1
    
    print(f"\n🎨 Copied {copied_count} Mermaid diagrams")
    return copied_count

def update_markdown_references():
    """Update all image references in markdown files to use local paths"""
    
    md_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes")
    
    # Mapping for textbook image replacements
    replacements = {
        # Chapter 1
        "../textbook/markdown/Chapter_1_Introduction/_page_3_Figure_1.jpeg": "Chapter_01_Page03_Figure01.jpeg",
        "../textbook/markdown/Chapter_1_Introduction/_page_4_Figure_1.jpeg": "Chapter_01_Page04_Figure01.jpeg",
        "../textbook/markdown/Chapter_1_Introduction/_page_8_Figure_1.jpeg": "Chapter_01_Page08_Figure01.jpeg",
        # Chapter 2
        "../textbook/markdown/Chapter_2_Information_Security_Risk_Assessment_Basics/_page_1_Figure_1.jpeg": "Chapter_02_Page01_Figure01.jpeg",
        # Chapter 3
        "../textbook/markdown/Chapter_3_Project_Definition/_page_18_Figure_12.jpeg": "Chapter_03_Page18_Figure12.jpeg",
        "../textbook/markdown/Chapter_3_Project_Definition/_page_19_Figure_5.jpeg": "Chapter_03_Page19_Figure05.jpeg",
        # Chapter 4
        "../textbook/markdown/Chapter_4_Security_Risk_Assessment_Preparation/_page_28_Figure_8.jpeg": "Chapter_04_Page28_Figure08.jpeg",
        # Chapter 5
        "../textbook/markdown/Chapter_5_Data_Gathering/_page_11_Figure_4.jpeg": "Chapter_05_Page11_Figure04.jpeg",
        "../textbook/markdown/Chapter_5_Data_Gathering/_page_14_Figure_1.jpeg": "Chapter_05_Page14_Figure01.jpeg",
        # Chapter 6
        "../textbook/markdown/Chapter_6_Administrative_Data_Gathering/_page_8_Figure_1.jpeg": "Chapter_06_Page08_Figure01.jpeg",
        # Chapter 7
        "../textbook/markdown/Chapter_7_Technical_Data_Gathering/_page_11_Figure_5.jpeg": "Chapter_07_Page11_Figure05.jpeg",
        "../textbook/markdown/Chapter_7_Technical_Data_Gathering/_page_13_Figure_7.jpeg": "Chapter_07_Page13_Figure07.jpeg",
        "../textbook/markdown/Chapter_7_Technical_Data_Gathering/_page_14_Figure_7.jpeg": "Chapter_07_Page14_Figure07.jpeg",
        "../textbook/markdown/Chapter_7_Technical_Data_Gathering/_page_28_Figure_7.jpeg": "Chapter_07_Page28_Figure07.jpeg",
        # Chapter 8
        "../textbook/markdown/Chapter_8_Physical_Data_Gathering/_page_9_Figure_1.jpeg": "Chapter_08_Page09_Figure01.jpeg",
        "../textbook/markdown/Chapter_8_Physical_Data_Gathering/_page_11_Figure_1.jpeg": "Chapter_08_Page11_Figure01.jpeg",
        "../textbook/markdown/Chapter_8_Physical_Data_Gathering/_page_19_Figure_6.jpeg": "Chapter_08_Page19_Figure06.jpeg",
        "../textbook/markdown/Chapter_8_Physical_Data_Gathering/_page_21_Figure_6.jpeg": "Chapter_08_Page21_Figure06.jpeg",
        "../textbook/markdown/Chapter_8_Physical_Data_Gathering/_page_22_Figure_1.jpeg": "Chapter_08_Page22_Figure01.jpeg",
        "../textbook/markdown/Chapter_8_Physical_Data_Gathering/_page_22_Picture_3.jpeg": "Chapter_08_Page22_Picture03.jpeg",
        # Chapter 9
        "../textbook/markdown/Chapter_9_Security_Risk_Analysis/_page_1_Figure_8.jpeg": "Chapter_09_Page01_Figure08.jpeg",
        "../textbook/markdown/Chapter_9_Security_Risk_Analysis/_page_2_Figure_7.jpeg": "Chapter_09_Page02_Figure07.jpeg",
        # Chapter 10
        "../textbook/markdown/Chapter_10_Security_Risk_Mitigation/_page_4_Figure_1.jpeg": "Chapter_10_Page04_Figure01.jpeg",
        # Chapter 12
        "../textbook/markdown/Chapter_12_Security_Risk_Assessment_Project_Management/_page_3_Figure_1.jpeg": "Chapter_12_Page03_Figure01.jpeg",
        "../textbook/markdown/Chapter_12_Security_Risk_Assessment_Project_Management/_page_18_Figure_5.jpeg": "Chapter_12_Page18_Figure05.jpeg"
    }
    
    # Process each markdown file
    updated_count = 0
    for md_file in md_dir.glob("week_*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace all image references
        for old_path, new_path in replacements.items():
            if old_path in content:
                content = content.replace(old_path, new_path)
                print(f"📝 Updated {md_file.name}: {old_path} → {new_path}")
        
        # Save if changes were made
        if content != original_content:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(content)
            updated_count += 1
    
    print(f"\n📝 Updated {updated_count} markdown files")
    return updated_count

def main():
    """Main function to consolidate all images"""
    
    print("🚀 Starting image consolidation process...\n")
    
    # Step 1: Copy textbook images
    textbook_count = copy_textbook_images()
    
    # Step 2: Copy Mermaid diagrams
    mermaid_count = copy_mermaid_diagrams()
    
    # Step 3: Update markdown references
    updated_files = update_markdown_references()
    
    # Summary
    print("\n" + "="*50)
    print("✨ Image Consolidation Complete!")
    print(f"📚 Textbook images copied: {textbook_count}")
    print(f"🎨 Mermaid diagrams copied: {mermaid_count}")
    print(f"📝 Markdown files updated: {updated_files}")
    print(f"📁 All images now in: /Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes/")
    
    # List final contents
    dest_dir = Path("/Users/invoture/dev.local/academic-agent/output/sra/ai_enhanced_study_notes")
    total_files = len(list(dest_dir.glob("*")))
    md_files = len(list(dest_dir.glob("*.md")))
    pdf_files = len(list(dest_dir.glob("*.pdf")))
    jpeg_files = len(list(dest_dir.glob("*.jpeg")))
    png_files = len(list(dest_dir.glob("*.png")))
    
    print(f"\n📊 Final folder contents:")
    print(f"   - Markdown files: {md_files}")
    print(f"   - PDF files: {pdf_files}")
    print(f"   - JPEG images: {jpeg_files}")
    print(f"   - PNG images: {png_files}")
    print(f"   - Total files: {total_files}")

if __name__ == "__main__":
    main()