#!/usr/bin/env python3
"""
Complete remaining weeks for AI-enhanced study notes generation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_enhanced_study_notes_generator import AIEnhancedStudyNotesGenerator
from pathlib import Path

def complete_remaining_weeks():
    """Complete weeks 12 and 13"""
    project_root = Path.cwd()
    generator = AIEnhancedStudyNotesGenerator(str(project_root))
    
    print("🔄 Completing remaining weeks...")
    
    # Process weeks 12 and 13
    for week in [12, 13]:
        print(f"\n📚 Processing Week {week}...")
        weekly_content = generator.process_week(week)
        
        if weekly_content.success:
            print(f"✅ Week {week} completed successfully!")
        else:
            print(f"❌ Week {week} failed")
    
    # Generate master index for all weeks
    print("\n📊 Generating final master index...")
    
    # Load all weeks
    all_weekly_content = []
    for week in range(1, 14):
        weekly_content = generator.process_week(week)  # This will load existing or create new
        all_weekly_content.append(weekly_content)
    
    # Generate master index
    master_index = generator.generate_master_index(all_weekly_content)
    
    # Save master index
    with open(generator.output_path / "master_index.md", 'w', encoding='utf-8') as f:
        f.write(master_index)
    
    # Save processing summary
    generator.save_processing_summary(all_weekly_content)
    
    print("\n🎉 All weeks completed!")
    print(f"📁 Output directory: {generator.output_path}")

if __name__ == "__main__":
    complete_remaining_weeks()