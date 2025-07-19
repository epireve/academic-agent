#!/usr/bin/env python3
"""
Three Source Processor for WOC7017 Security Risk Analysis and Evaluation

This framework processes and integrates content from three sources:
1. Textbook chapters (PDF)
2. Lecture slides (Markdown)
3. Weekly transcripts (to be added)

The goal is to create comprehensive, integrated course notes.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class WeeklyContent:
    """Represents all content for a specific week"""
    week_number: int
    textbook_chapter: str
    textbook_pages: str
    lecture_title: str
    lecture_content: str
    transcript_content: Optional[str] = None
    integrated_notes: Optional[str] = None
    key_concepts: List[str] = None
    learning_objectives: List[str] = None
    gaps_identified: List[str] = None
    
    def __post_init__(self):
        if self.key_concepts is None:
            self.key_concepts = []
        if self.learning_objectives is None:
            self.learning_objectives = []
        if self.gaps_identified is None:
            self.gaps_identified = []

class ThreeSourceProcessor:
    """Main processor for integrating three content sources"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Define textbook chapters structure
        self.textbook_chapters = [
            {"number": 1, "title": "Introduction", "pages": "22-43"},
            {"number": 2, "title": "Information Security Risk Assessment Basics", "pages": "44-58"},
            {"number": 3, "title": "Project Definition", "pages": "60-93"},
            {"number": 4, "title": "Security Risk Assessment Preparation", "pages": "94-131"},
            {"number": 5, "title": "Data Gathering", "pages": "132-164"},
            {"number": 6, "title": "Administrative Data Gathering", "pages": "166-235"},
            {"number": 7, "title": "Technical Data Gathering", "pages": "236-307"},
            {"number": 8, "title": "Physical Data Gathering", "pages": "308-384"},
            {"number": 9, "title": "Security Risk Analysis", "pages": "386-400"},
            {"number": 10, "title": "Security Risk Mitigation", "pages": "402-414"},
            {"number": 11, "title": "Security Risk Assessment Reporting", "pages": "416-428"},
            {"number": 12, "title": "Security Risk Assessment Project Management", "pages": "430-454"},
            {"number": 13, "title": "Security Risk Assessment Approaches", "pages": "456-474"},
        ]
        
        # Define paths
        self.lecture_notes_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "lectures" / get_processed_output_path(ContentType.MARKDOWN)
        self.textbook_chapters_path = self.project_root / "Split_Chapters"
        self.transcripts_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "transcripts"  # To be created
        self.output_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "integrated_notes"
        
        # Create directories
        self.transcripts_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing status
        self.processing_status = {
            "initialized": datetime.now().isoformat(),
            "sources_available": {
                "textbook": True,
                "lectures": True,
                "transcripts": False
            },
            "weeks_processed": 0,
            "total_weeks": 13
        }

    def load_lecture_content(self, week: int) -> Tuple[str, str]:
        """Load lecture content for a specific week"""
        note_file = self.lecture_notes_path / f"Note{week}.md"
        
        if not note_file.exists():
            return f"Note {week} (Missing)", ""
        
        with open(note_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title from first line
        lines = content.split('\n')
        if lines:
            title_line = lines[0].strip('#').strip()
            if 'WOA7107' in title_line:
                parts = title_line.split('-')
                if len(parts) > 1:
                    title = parts[-1].strip()
                else:
                    title = f"Note {week}"
            else:
                title = title_line
        else:
            title = f"Note {week}"
        
        return title, content

    def load_transcript_content(self, week: int) -> Optional[str]:
        """Load transcript content for a specific week (placeholder)"""
        transcript_file = self.transcripts_path / f"week_{week}_transcript.md"
        
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def extract_key_concepts(self, lecture_content: str) -> List[str]:
        """Extract key concepts from lecture content"""
        concepts = []
        
        # Simple keyword extraction based on markdown headers and bullet points
        lines = lecture_content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract from headers
            if line.startswith('##') and not line.startswith('###'):
                concept = line.strip('#').strip()
                if concept and len(concept) > 3:
                    concepts.append(concept)
            
            # Extract from bullet points that seem like key concepts
            elif line.startswith('-') and len(line) > 10:
                concept = line.strip('-').strip()
                if any(keyword in concept.lower() for keyword in ['risk', 'security', 'assessment', 'analysis', 'management']):
                    concepts.append(concept)
        
        # Remove duplicates and return top concepts
        unique_concepts = list(dict.fromkeys(concepts))
        return unique_concepts[:10]  # Limit to top 10 concepts

    def extract_learning_objectives(self, lecture_content: str) -> List[str]:
        """Extract learning objectives from lecture content"""
        objectives = []
        
        # Look for agenda items or structured content
        lines = lecture_content.split('\n')
        in_agenda = False
        
        for line in lines:
            line = line.strip()
            
            if 'agenda' in line.lower():
                in_agenda = True
                continue
            
            if in_agenda:
                if line.startswith('-'):
                    objective = line.strip('-').strip()
                    if objective and len(objective) > 5:
                        objectives.append(f"Understand {objective}")
                elif line.startswith('#'):
                    in_agenda = False
        
        return objectives[:8]  # Limit to top 8 objectives

    def identify_content_gaps(self, textbook_title: str, lecture_title: str, lecture_content: str) -> List[str]:
        """Identify potential content gaps"""
        gaps = []
        
        # Check for missing introduction content
        if "introduction" in textbook_title.lower() and len(lecture_content) < 1000:
            gaps.append("Lecture content seems brief for introductory material")
        
        # Check for missing practical examples
        if "practical" not in lecture_content.lower() and "example" not in lecture_content.lower():
            gaps.append("May need more practical examples")
        
        # Check for missing assessment criteria
        if "assessment" in textbook_title.lower() and "evaluate" not in lecture_content.lower():
            gaps.append("Could benefit from evaluation criteria")
        
        # Check for missing current trends/updates
        if "2024" not in lecture_content and "current" not in lecture_content.lower():
            gaps.append("May need updates on current industry trends")
        
        return gaps

    def process_week(self, week: int) -> WeeklyContent:
        """Process all content for a specific week"""
        # Get textbook chapter info
        textbook_chapter = self.textbook_chapters[week - 1]
        
        # Load lecture content
        lecture_title, lecture_content = self.load_lecture_content(week)
        
        # Load transcript content (if available)
        transcript_content = self.load_transcript_content(week)
        
        # Extract key information
        key_concepts = self.extract_key_concepts(lecture_content)
        learning_objectives = self.extract_learning_objectives(lecture_content)
        gaps_identified = self.identify_content_gaps(
            textbook_chapter["title"], 
            lecture_title, 
            lecture_content
        )
        
        # Create weekly content object
        weekly_content = WeeklyContent(
            week_number=week,
            textbook_chapter=textbook_chapter["title"],
            textbook_pages=textbook_chapter["pages"],
            lecture_title=lecture_title,
            lecture_content=lecture_content,
            transcript_content=transcript_content,
            key_concepts=key_concepts,
            learning_objectives=learning_objectives,
            gaps_identified=gaps_identified
        )
        
        return weekly_content

    def generate_integrated_notes(self, weekly_content: WeeklyContent) -> str:
        """Generate integrated notes combining all sources"""
        notes = []
        
        # Header
        notes.append(f"# Week {weekly_content.week_number}: {weekly_content.textbook_chapter}")
        notes.append(f"**Course**: WOC7017 Security Risk Analysis and Evaluation")
        notes.append(f"**Textbook Pages**: {weekly_content.textbook_pages}")
        notes.append(f"**Lecture Topic**: {weekly_content.lecture_title}")
        notes.append("")
        
        # Learning Objectives
        if weekly_content.learning_objectives:
            notes.append("## Learning Objectives")
            for objective in weekly_content.learning_objectives:
                notes.append(f"- {objective}")
            notes.append("")
        
        # Key Concepts
        if weekly_content.key_concepts:
            notes.append("## Key Concepts")
            for concept in weekly_content.key_concepts:
                notes.append(f"- {concept}")
            notes.append("")
        
        # Lecture Content
        notes.append("## Lecture Content")
        notes.append(weekly_content.lecture_content)
        notes.append("")
        
        # Transcript Integration (if available)
        if weekly_content.transcript_content:
            notes.append("## Class Discussion & Additional Context")
            notes.append(weekly_content.transcript_content)
            notes.append("")
        
        # Textbook Integration Placeholder
        notes.append("## Textbook Integration")
        notes.append(f"**Chapter**: {weekly_content.textbook_chapter}")
        notes.append(f"**Pages**: {weekly_content.textbook_pages}")
        notes.append("*Note: Detailed textbook content integration pending PDF processing*")
        notes.append("")
        
        # Gaps and Recommendations
        if weekly_content.gaps_identified:
            notes.append("## Identified Gaps & Recommendations")
            for gap in weekly_content.gaps_identified:
                notes.append(f"- {gap}")
            notes.append("")
        
        # Study Questions (placeholder)
        notes.append("## Study Questions")
        notes.append("*To be populated based on integrated content analysis*")
        notes.append("")
        
        # Assessment Preparation
        notes.append("## Assessment Preparation")
        notes.append("*Key points for exams and assignments to be added*")
        notes.append("")
        
        return "\n".join(notes)

    def process_all_weeks(self) -> List[WeeklyContent]:
        """Process all weeks and generate integrated content"""
        all_weekly_content = []
        
        for week in range(1, 14):
            print(f"Processing Week {week}...")
            
            # Process week content
            weekly_content = self.process_week(week)
            
            # Generate integrated notes
            integrated_notes = self.generate_integrated_notes(weekly_content)
            weekly_content.integrated_notes = integrated_notes
            
            # Save individual week notes
            week_file = self.output_path / f"week_{week:02d}_integrated_notes.md"
            with open(week_file, 'w', encoding='utf-8') as f:
                f.write(integrated_notes)
            
            all_weekly_content.append(weekly_content)
            self.processing_status["weeks_processed"] += 1
        
        return all_weekly_content

    def generate_master_index(self, all_weekly_content: List[WeeklyContent]) -> str:
        """Generate a master index of all content"""
        index = []
        
        index.append("# WOC7017 Security Risk Analysis and Evaluation")
        index.append("## Master Course Index")
        index.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        index.append("")
        
        # Course overview
        index.append("## Course Overview")
        index.append("This course covers comprehensive security risk analysis and evaluation methodologies.")
        index.append("The content is integrated from three primary sources:")
        index.append("1. **Textbook**: The Security Risk Assessment Handbook (2nd Edition)")
        index.append("2. **Lecture Slides**: Professor's presentation materials")
        index.append("3. **Class Transcripts**: Live class discussions and Q&A sessions")
        index.append("")
        
        # Weekly breakdown
        index.append("## Weekly Content Index")
        index.append("")
        
        for week_content in all_weekly_content:
            index.append(f"### Week {week_content.week_number}: {week_content.textbook_chapter}")
            index.append(f"- **Textbook Pages**: {week_content.textbook_pages}")
            index.append(f"- **Lecture Topic**: {week_content.lecture_title}")
            index.append(f"- **Key Concepts**: {len(week_content.key_concepts)} identified")
            index.append(f"- **Learning Objectives**: {len(week_content.learning_objectives)} defined")
            index.append(f"- **Notes File**: `week_{week_content.week_number:02d}_integrated_notes.md`")
            index.append("")
        
        # Processing status
        index.append("## Processing Status")
        index.append(f"- **Textbook Chapters**: ✅ {len(self.textbook_chapters)} available")
        index.append(f"- **Lecture Slides**: ✅ {self.processing_status['weeks_processed']} processed")
        index.append(f"- **Transcripts**: ⏳ Pending integration")
        index.append("")
        
        # Usage instructions
        index.append("## Usage Instructions")
        index.append("1. **Individual Week Study**: Open the specific week's integrated notes file")
        index.append("2. **Comprehensive Review**: Use this index to navigate between topics")
        index.append("3. **Exam Preparation**: Focus on key concepts and learning objectives")
        index.append("4. **Transcript Integration**: Add weekly transcripts to enhance understanding")
        index.append("")
        
        return "\n".join(index)

    def save_processing_summary(self, all_weekly_content: List[WeeklyContent]):
        """Save processing summary and statistics"""
        summary_data = {
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "total_weeks": len(all_weekly_content),
                "sources_processed": {
                    "textbook_chapters": len(self.textbook_chapters),
                    "lecture_slides": len([w for w in all_weekly_content if w.lecture_content]),
                    "transcripts": len([w for w in all_weekly_content if w.transcript_content])
                }
            },
            "content_statistics": {
                "total_key_concepts": sum(len(w.key_concepts) for w in all_weekly_content),
                "total_learning_objectives": sum(len(w.learning_objectives) for w in all_weekly_content),
                "total_gaps_identified": sum(len(w.gaps_identified) for w in all_weekly_content)
            },
            "weekly_summary": [
                {
                    "week": w.week_number,
                    "textbook_chapter": w.textbook_chapter,
                    "lecture_title": w.lecture_title,
                    "key_concepts_count": len(w.key_concepts),
                    "learning_objectives_count": len(w.learning_objectives),
                    "gaps_count": len(w.gaps_identified),
                    "has_transcript": w.transcript_content is not None
                }
                for w in all_weekly_content
            ]
        }
        
        with open(self.output_path / "processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    def run_complete_processing(self):
        """Run the complete three-source processing pipeline"""
        print("Starting three-source content processing...")
        print(f"Output directory: {self.output_path}")
        
        # Process all weeks
        all_weekly_content = self.process_all_weeks()
        
        # Generate master index
        print("Generating master index...")
        master_index = self.generate_master_index(all_weekly_content)
        
        # Save master index
        with open(self.output_path / "master_index.md", 'w', encoding='utf-8') as f:
            f.write(master_index)
        
        # Save processing summary
        self.save_processing_summary(all_weekly_content)
        
        print(f"\nProcessing complete!")
        print(f"- {len(all_weekly_content)} weeks processed")
        print(f"- Files saved to: {self.output_path}")
        print(f"- Master index: master_index.md")
        print(f"- Individual notes: week_XX_integrated_notes.md")
        
        return all_weekly_content

def main():
    """Main function"""
    project_root = Path.cwd()
    
    processor = ThreeSourceProcessor(str(project_root))
    processor.run_complete_processing()

if __name__ == "__main__":
    main()