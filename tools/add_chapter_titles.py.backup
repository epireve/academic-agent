#!/usr/bin/env python3
"""
Add chapter titles to the top of each study notes file
"""

import os
from pathlib import Path

def add_chapter_titles():
    """Add chapter titles to all study notes files"""
    
    project_root = Path.cwd()
    notes_dir = project_root / "output" / "sra" / "ai_enhanced_study_notes"
    
    # Chapter titles mapping
    chapters = {
        1: "Introduction",
        2: "Information Security Risk Assessment Basics",
        3: "Project Definition",
        4: "Security Risk Assessment Preparation",
        5: "Data Gathering",
        6: "Administrative Data Gathering",
        7: "Technical Data Gathering",
        8: "Physical Data Gathering",
        9: "Security Risk Analysis",
        10: "Security Risk Mitigation",
        11: "Security Risk Assessment Reporting",
        12: "Security Risk Assessment Project Management",
        13: "Security Risk Assessment Approaches",
    }
    
    print("🔄 Adding chapter titles to study notes...")
    
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
            
            # Create new header
            new_header = f"# Chapter {week_num}: {chapter_title}\n\n"
            
            # Check if header already exists
            if content.startswith(f"# Chapter {week_num}:"):
                print(f"✅ Week {week_num}: Header already exists")
                continue
            
            # Remove existing first line if it's a generic header
            lines = content.split('\n')
            if lines[0].startswith("Here are the comprehensive study notes for Week"):
                # Remove first line and any empty lines that follow
                while lines and (lines[0].startswith("Here are the comprehensive") or lines[0].strip() == ""):
                    lines.pop(0)
                content = '\n'.join(lines)
            
            # Add new header
            new_content = new_header + content
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✅ Week {week_num}: Added chapter title - {chapter_title}")
            
        except Exception as e:
            print(f"❌ Error processing week {week_num}: {e}")
    
    print("\n🎉 Chapter titles added successfully!")

if __name__ == "__main__":
    add_chapter_titles()