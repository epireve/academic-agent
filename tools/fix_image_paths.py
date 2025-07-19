#!/usr/bin/env python3
"""
Fix image paths in study notes files to point to correct locations
"""

import os
import re
from pathlib import Path

def fix_image_paths():
    """Fix image paths in all study notes files"""
    
    project_root = Path.cwd()
    notes_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    textbook_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "textbook" / get_processed_output_path(ContentType.MARKDOWN)
    
    # Chapter titles mapping
    chapters = {
        1: "Introduction",
        2: "Information_Security_Risk_Assessment_Basics",
        3: "Project_Definition",
        4: "Security_Risk_Assessment_Preparation",
        5: "Data_Gathering",
        6: "Administrative_Data_Gathering",
        7: "Technical_Data_Gathering",
        8: "Physical_Data_Gathering",
        9: "Security_Risk_Analysis",
        10: "Security_Risk_Mitigation",
        11: "Security_Risk_Assessment_Reporting",
        12: "Security_Risk_Assessment_Project_Management",
        13: "Security_Risk_Assessment_Approaches",
    }
    
    print("🔄 Fixing image paths in study notes...")
    
    for week_num, chapter_title in chapters.items():
        filename = f"week_{week_num:02d}_comprehensive_study_notes.md"
        file_path = notes_dir / filename
        
        if not file_path.exists():
            print(f"⚠️  File not found: {filename}")
            continue
        
        try:
            # Read existing content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find the corresponding textbook chapter directory
            chapter_dir = textbook_dir / f"Chapter_{week_num}_{chapter_title}"
            
            if not chapter_dir.exists():
                print(f"⚠️  Chapter directory not found: {chapter_dir}")
                continue
            
            # Get available images in the chapter directory
            available_images = list(chapter_dir.glob("*.jpeg"))
            available_images.extend(list(chapter_dir.glob("*.jpg")))
            available_images.extend(list(chapter_dir.glob("*.png")))
            
            if not available_images:
                print(f"⚠️  No images found in: {chapter_dir}")
                continue
            
            # Pattern to find image references
            image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            
            def replace_image_path(match):
                alt_text = match.group(1)
                old_path = match.group(2)
                
                # Extract just the filename from the old path
                old_filename = Path(old_path).name
                
                # Find matching image in available images
                for img_path in available_images:
                    if img_path.name == old_filename:
                        # Create relative path from notes directory to image
                        rel_path = os.path.relpath(img_path, notes_dir)
                        return f'![{alt_text}]({rel_path})'
                
                # If no exact match found, try to find similar image
                for img_path in available_images:
                    # Check if any part of the filename matches
                    if any(part in img_path.name for part in old_filename.split('_')):
                        rel_path = os.path.relpath(img_path, notes_dir)
                        return f'![{alt_text}]({rel_path})'
                
                # If still no match, comment out the image reference
                return f'<!-- Image not found: {old_path} -->'
            
            # Replace all image references
            original_content = content
            content = re.sub(image_pattern, replace_image_path, content)
            
            # Write back to file if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Count how many images were updated
                original_images = len(re.findall(image_pattern, original_content))
                updated_images = len(re.findall(image_pattern, content))
                commented_images = len(re.findall(r'<!-- Image not found:', content))
                
                print(f"✅ Week {week_num}: Updated {original_images} → {updated_images} images ({commented_images} not found)")
            else:
                print(f"✅ Week {week_num}: No image updates needed")
                
        except Exception as e:
            print(f"❌ Error processing week {week_num}: {e}")
    
    print("\n🎉 Image paths fixed successfully!")

if __name__ == "__main__":
    fix_image_paths()