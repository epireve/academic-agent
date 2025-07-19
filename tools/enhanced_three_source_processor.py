#!/usr/bin/env python3
"""
Enhanced Three Source Processor for WOC7017 Security Risk Analysis and Evaluation

This enhanced version processes and integrates content from three sources:
1. Textbook chapters (Markdown from marker processing)
2. Lecture slides (Markdown)
3. Weekly transcripts (Multiple directories)

Enhanced features:
- Supports multiple transcript directories
- Handles marker-processed textbook content
- Improved content integration
- Better error handling
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import glob

@dataclass
class WeeklyContent:
    """Represents all content for a specific week"""
    week_number: int
    textbook_chapter: str
    textbook_pages: str
    textbook_markdown_path: Optional[str] = None
    textbook_content: Optional[str] = None
    lecture_title: str = ""
    lecture_content: str = ""
    transcript_sources: List[str] = None
    transcript_content: Optional[str] = None
    integrated_notes: Optional[str] = None
    key_concepts: List[str] = None
    learning_objectives: List[str] = None
    gaps_identified: List[str] = None
    
    def __post_init__(self):
        if self.transcript_sources is None:
            self.transcript_sources = []
        if self.key_concepts is None:
            self.key_concepts = []
        if self.learning_objectives is None:
            self.learning_objectives = []
        if self.gaps_identified is None:
            self.gaps_identified = []

class EnhancedThreeSourceProcessor:
    """Enhanced processor for integrating three content sources"""
    
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
        self.textbook_markdown_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "textbook" / get_processed_output_path(ContentType.MARKDOWN)
        
        # Multiple transcript directories as mentioned by user
        self.transcript_directories = [
            self.project_root / str(get_output_manager().outputs_dir) / "sra" / "transcripts" / get_processed_output_path(ContentType.MARKDOWN),
            Path("/Users/invoture/dev.local/mse-st/sra")
        ]
        
        self.output_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "enhanced_integrated_notes"
        
        # Create directories
        for transcript_dir in self.transcript_directories:
            transcript_dir.mkdir(parents=True, exist_ok=True)
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
            "total_weeks": 13,
            "textbook_markdown_available": False
        }

    def check_textbook_markdown_availability(self) -> bool:
        """Check if textbook markdown files are available"""
        if not self.textbook_markdown_path.exists():
            return False
        
        # Check if at least some chapters are processed
        chapter_dirs = list(self.textbook_markdown_path.glob("chapter_*"))
        return len(chapter_dirs) > 0

    def load_textbook_content(self, week: int) -> Tuple[Optional[str], Optional[str]]:
        """Load textbook content for a specific week"""
        chapter_dir = self.textbook_markdown_path / f"chapter_{week:02d}"
        
        if not chapter_dir.exists():
            return None, None
        
        # Find markdown files in the chapter directory
        md_files = list(chapter_dir.glob("**/*.md"))
        
        if not md_files:
            return None, None
        
        # Use the first markdown file found
        md_file = md_files[0]
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return str(md_file), content
        except Exception as e:
            print(f"Error reading textbook content for week {week}: {e}")
            return str(md_file), None

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

    def find_transcript_files(self, week: int) -> List[str]:
        """Find transcript files for a specific week from multiple directories"""
        transcript_files = []
        
        # Possible naming patterns
        patterns = [
            f"week_{week}_transcript.md",
            f"week{week}_transcript.md",
            f"transcript_week_{week}.md",
            f"Week{week}.md",
            f"week_{week}.md",
            f"w{week}.md",
            f"lecture_{week}.md",
            f"class_{week}.md"
        ]
        
        for transcript_dir in self.transcript_directories:
            if not transcript_dir.exists():
                continue
            
            # Try exact patterns first
            for pattern in patterns:
                transcript_file = transcript_dir / pattern
                if transcript_file.exists():
                    transcript_files.append(str(transcript_file))
            
            # Try glob patterns for more flexible matching
            for pattern in patterns:
                glob_pattern = pattern.replace('.md', '*.md')
                matches = list(transcript_dir.glob(glob_pattern))
                for match in matches:
                    if str(match) not in transcript_files:
                        transcript_files.append(str(match))
        
        return transcript_files

    def load_transcript_content(self, week: int) -> Tuple[List[str], Optional[str]]:
        """Load transcript content for a specific week"""
        transcript_files = self.find_transcript_files(week)
        
        if not transcript_files:
            return [], None
        
        combined_content = []
        
        for transcript_file in transcript_files:
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    combined_content.append(f"## From: {Path(transcript_file).name}\n\n{content}")
            except Exception as e:
                print(f"Error reading transcript {transcript_file}: {e}")
                continue
        
        if combined_content:
            return transcript_files, "\n\n---\n\n".join(combined_content)
        else:
            return transcript_files, None

    def extract_key_concepts(self, textbook_content: str, lecture_content: str, transcript_content: str) -> List[str]:
        """Extract key concepts from all sources"""
        concepts = []
        
        # From textbook (if available)
        if textbook_content:
            # Extract from headers and emphasized text
            lines = textbook_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('#') and len(line) > 3:
                    concept = line.strip('#').strip()
                    if concept and 'Chapter' not in concept:
                        concepts.append(f"Textbook: {concept}")
        
        # From lecture content
        if lecture_content:
            lines = lecture_content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('##') and not line.startswith('###'):
                    concept = line.strip('#').strip()
                    if concept and len(concept) > 3:
                        concepts.append(f"Lecture: {concept}")
        
        # From transcript content
        if transcript_content:
            # Look for key phrases that indicate important concepts
            key_phrases = ['important', 'key point', 'remember', 'note that', 'crucial', 'essential']
            lines = transcript_content.split('\n')
            for line in lines:
                line = line.strip().lower()
                if any(phrase in line for phrase in key_phrases):
                    if len(line) > 10 and len(line) < 100:
                        concepts.append(f"Discussion: {line}")
        
        # Remove duplicates and return top concepts
        unique_concepts = list(dict.fromkeys(concepts))
        return unique_concepts[:15]  # Limit to top 15 concepts

    def extract_learning_objectives(self, textbook_content: str, lecture_content: str) -> List[str]:
        """Extract learning objectives from textbook and lecture content"""
        objectives = []
        
        # From textbook
        if textbook_content:
            # Look for objective-like content
            lines = textbook_content.split('\n')
            for line in lines:
                line = line.strip()
                if any(word in line.lower() for word in ['objective', 'goal', 'learn', 'understand', 'analyze']):
                    if len(line) > 10 and len(line) < 200:
                        objectives.append(f"Textbook: {line}")
        
        # From lecture content
        if lecture_content:
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
                            objectives.append(f"Lecture: Understand {objective}")
                    elif line.startswith('#'):
                        in_agenda = False
        
        return objectives[:12]  # Limit to top 12 objectives

    def identify_content_gaps(self, textbook_content: str, lecture_content: str, transcript_content: str) -> List[str]:
        """Identify potential content gaps across all sources"""
        gaps = []
        
        # Check textbook availability
        if not textbook_content:
            gaps.append("Textbook content not available - PDF processing may be incomplete")
        
        # Check lecture completeness
        if lecture_content and len(lecture_content) < 500:
            gaps.append("Lecture content seems brief - may need more detailed notes")
        
        # Check transcript availability
        if not transcript_content:
            gaps.append("Class transcript not available - missing live discussion context")
        
        # Check for practical examples
        all_content = f"{textbook_content or ''} {lecture_content} {transcript_content or ''}"
        if "example" not in all_content.lower() and "case study" not in all_content.lower():
            gaps.append("May need more practical examples and case studies")
        
        # Check for current trends
        if "2024" not in all_content and "current" not in all_content.lower():
            gaps.append("May need updates on current industry trends and practices")
        
        return gaps

    def process_week(self, week: int) -> WeeklyContent:
        """Process all content for a specific week"""
        # Get textbook chapter info
        textbook_chapter = self.textbook_chapters[week - 1]
        
        # Load textbook content (from marker processing)
        textbook_path, textbook_content = self.load_textbook_content(week)
        
        # Load lecture content
        lecture_title, lecture_content = self.load_lecture_content(week)
        
        # Load transcript content
        transcript_sources, transcript_content = self.load_transcript_content(week)
        
        # Extract key information
        key_concepts = self.extract_key_concepts(textbook_content or "", lecture_content, transcript_content or "")
        learning_objectives = self.extract_learning_objectives(textbook_content or "", lecture_content)
        gaps_identified = self.identify_content_gaps(textbook_content or "", lecture_content, transcript_content or "")
        
        # Create weekly content object
        weekly_content = WeeklyContent(
            week_number=week,
            textbook_chapter=textbook_chapter["title"],
            textbook_pages=textbook_chapter["pages"],
            textbook_markdown_path=textbook_path,
            textbook_content=textbook_content,
            lecture_title=lecture_title,
            lecture_content=lecture_content,
            transcript_sources=transcript_sources,
            transcript_content=transcript_content,
            key_concepts=key_concepts,
            learning_objectives=learning_objectives,
            gaps_identified=gaps_identified
        )
        
        return weekly_content

    def generate_enhanced_integrated_notes(self, weekly_content: WeeklyContent) -> str:
        """Generate enhanced integrated notes combining all sources"""
        notes = []
        
        # Header
        notes.append(f"# Week {weekly_content.week_number}: {weekly_content.textbook_chapter}")
        notes.append(f"**Course**: WOC7017 Security Risk Analysis and Evaluation")
        notes.append(f"**Textbook Pages**: {weekly_content.textbook_pages}")
        notes.append(f"**Lecture Topic**: {weekly_content.lecture_title}")
        notes.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        notes.append("")
        
        # Content Sources Summary
        notes.append("## Content Sources")
        notes.append(f"- **Textbook**: {'✅ Available' if weekly_content.textbook_content else '❌ Not Available'}")
        notes.append(f"- **Lecture Slides**: {'✅ Available' if weekly_content.lecture_content else '❌ Not Available'}")
        notes.append(f"- **Class Transcripts**: {'✅ Available' if weekly_content.transcript_content else '❌ Not Available'}")
        if weekly_content.transcript_sources:
            notes.append("  - Transcript Sources:")
            for source in weekly_content.transcript_sources:
                notes.append(f"    - `{Path(source).name}`")
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
        
        # Textbook Content
        if weekly_content.textbook_content:
            notes.append("## Textbook Content")
            notes.append(f"**Source**: `{Path(weekly_content.textbook_markdown_path).name if weekly_content.textbook_markdown_path else 'N/A'}`")
            notes.append("")
            # Include a summary or first part of textbook content
            textbook_lines = weekly_content.textbook_content.split('\n')
            if len(textbook_lines) > 50:
                notes.append("### Summary (First 50 lines)")
                notes.extend(textbook_lines[:50])
                notes.append("\n*[Full textbook content available in source file]*")
            else:
                notes.extend(textbook_lines)
            notes.append("")
        
        # Lecture Content
        if weekly_content.lecture_content:
            notes.append("## Lecture Content")
            notes.append(weekly_content.lecture_content)
            notes.append("")
        
        # Transcript Content
        if weekly_content.transcript_content:
            notes.append("## Class Discussion & Transcripts")
            notes.append(weekly_content.transcript_content)
            notes.append("")
        
        # Gaps and Recommendations
        if weekly_content.gaps_identified:
            notes.append("## Identified Gaps & Recommendations")
            for gap in weekly_content.gaps_identified:
                notes.append(f"- {gap}")
            notes.append("")
        
        # Study Questions (enhanced)
        notes.append("## Study Questions")
        notes.append("### Textbook-Based Questions")
        notes.append("*To be populated based on textbook content analysis*")
        notes.append("")
        notes.append("### Lecture-Based Questions")
        notes.append("*To be populated based on lecture content analysis*")
        notes.append("")
        notes.append("### Discussion Questions")
        notes.append("*To be populated based on class transcript analysis*")
        notes.append("")
        
        # Assessment Preparation
        notes.append("## Assessment Preparation")
        notes.append("### Key Points for Exams")
        notes.append("*Combined key points from all sources*")
        notes.append("")
        notes.append("### Assignment Relevance")
        notes.append("*How this week's content relates to course assignments*")
        notes.append("")
        
        return "\n".join(notes)

    def process_all_weeks(self) -> List[WeeklyContent]:
        """Process all weeks and generate enhanced integrated content"""
        all_weekly_content = []
        
        # Check textbook markdown availability
        textbook_available = self.check_textbook_markdown_availability()
        self.processing_status["textbook_markdown_available"] = textbook_available
        
        print(f"Textbook markdown available: {textbook_available}")
        
        for week in range(1, 14):
            print(f"Processing Week {week}...")
            
            # Process week content
            weekly_content = self.process_week(week)
            
            # Generate enhanced integrated notes
            integrated_notes = self.generate_enhanced_integrated_notes(weekly_content)
            weekly_content.integrated_notes = integrated_notes
            
            # Save individual week notes
            week_file = self.output_path / f"week_{week:02d}_enhanced_integrated_notes.md"
            with open(week_file, 'w', encoding='utf-8') as f:
                f.write(integrated_notes)
            
            all_weekly_content.append(weekly_content)
            self.processing_status["weeks_processed"] += 1
        
        return all_weekly_content

    def generate_enhanced_master_index(self, all_weekly_content: List[WeeklyContent]) -> str:
        """Generate an enhanced master index of all content"""
        index = []
        
        index.append("# WOC7017 Security Risk Analysis and Evaluation")
        index.append("## Enhanced Master Course Index")
        index.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        index.append("")
        
        # Course overview
        index.append("## Course Overview")
        index.append("This course covers comprehensive security risk analysis and evaluation methodologies.")
        index.append("The content is integrated from three primary sources:")
        index.append("1. **Textbook**: The Security Risk Assessment Handbook (2nd Edition) - *Marker-processed markdown*")
        index.append("2. **Lecture Slides**: Professor's presentation materials")
        index.append("3. **Class Transcripts**: Live class discussions and Q&A sessions")
        index.append("")
        
        # Processing status
        index.append("## Processing Status")
        textbook_available = self.processing_status["textbook_markdown_available"]
        index.append(f"- **Textbook Processing**: {'✅ Markdown available' if textbook_available else '⏳ Processing in progress'}")
        index.append(f"- **Lecture Slides**: ✅ {len(all_weekly_content)} processed")
        
        # Count available transcripts
        transcript_count = len([w for w in all_weekly_content if w.transcript_content])
        index.append(f"- **Transcripts**: {'✅' if transcript_count > 0 else '⏳'} {transcript_count} available")
        
        # Transcript directories
        index.append(f"- **Transcript Directories**:")
        for transcript_dir in self.transcript_directories:
            exists = transcript_dir.exists()
            index.append(f"  - {'✅' if exists else '❌'} `{transcript_dir}`")
        index.append("")
        
        # Weekly breakdown
        index.append("## Weekly Content Index")
        index.append("")
        
        for week_content in all_weekly_content:
            index.append(f"### Week {week_content.week_number}: {week_content.textbook_chapter}")
            index.append(f"- **Textbook Pages**: {week_content.textbook_pages}")
            index.append(f"- **Lecture Topic**: {week_content.lecture_title}")
            index.append(f"- **Textbook Content**: {'✅ Available' if week_content.textbook_content else '❌ Not Available'}")
            index.append(f"- **Transcript Sources**: {len(week_content.transcript_sources)} found")
            index.append(f"- **Key Concepts**: {len(week_content.key_concepts)} identified")
            index.append(f"- **Learning Objectives**: {len(week_content.learning_objectives)} defined")
            index.append(f"- **Notes File**: `week_{week_content.week_number:02d}_enhanced_integrated_notes.md`")
            index.append("")
        
        # Statistics
        total_concepts = sum(len(w.key_concepts) for w in all_weekly_content)
        total_objectives = sum(len(w.learning_objectives) for w in all_weekly_content)
        total_gaps = sum(len(w.gaps_identified) for w in all_weekly_content)
        
        index.append("## Content Statistics")
        index.append(f"- **Total Key Concepts**: {total_concepts}")
        index.append(f"- **Total Learning Objectives**: {total_objectives}")
        index.append(f"- **Total Gaps Identified**: {total_gaps}")
        index.append(f"- **Average Concepts per Week**: {total_concepts / len(all_weekly_content):.1f}")
        index.append("")
        
        # Usage instructions
        index.append("## Usage Instructions")
        index.append("1. **Individual Week Study**: Open the specific week's enhanced integrated notes file")
        index.append("2. **Comprehensive Review**: Use this index to navigate between topics")
        index.append("3. **Multi-Source Learning**: Compare textbook, lecture, and transcript content")
        index.append("4. **Exam Preparation**: Focus on key concepts and learning objectives")
        index.append("5. **Gap Analysis**: Review identified gaps for additional study")
        index.append("")
        
        return "\n".join(index)

    def run_enhanced_processing(self):
        """Run the enhanced three-source processing pipeline"""
        print("Starting enhanced three-source content processing...")
        print(f"Output directory: {self.output_path}")
        print(f"Transcript directories: {[str(d) for d in self.transcript_directories]}")
        
        # Process all weeks
        all_weekly_content = self.process_all_weeks()
        
        # Generate enhanced master index
        print("Generating enhanced master index...")
        master_index = self.generate_enhanced_master_index(all_weekly_content)
        
        # Save master index
        with open(self.output_path / "enhanced_master_index.md", 'w', encoding='utf-8') as f:
            f.write(master_index)
        
        # Save processing summary
        self.save_processing_summary(all_weekly_content)
        
        print(f"\nEnhanced processing complete!")
        print(f"- {len(all_weekly_content)} weeks processed")
        print(f"- Files saved to: {self.output_path}")
        print(f"- Master index: enhanced_master_index.md")
        print(f"- Individual notes: week_XX_enhanced_integrated_notes.md")
        
        return all_weekly_content

    def save_processing_summary(self, all_weekly_content: List[WeeklyContent]):
        """Save enhanced processing summary and statistics"""
        summary_data = {
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "total_weeks": len(all_weekly_content),
                "textbook_markdown_available": self.processing_status["textbook_markdown_available"],
                "sources_processed": {
                    "textbook_chapters": len(self.textbook_chapters),
                    "lecture_slides": len([w for w in all_weekly_content if w.lecture_content]),
                    "transcripts": len([w for w in all_weekly_content if w.transcript_content])
                },
                "transcript_directories": [str(d) for d in self.transcript_directories]
            },
            "content_statistics": {
                "total_key_concepts": sum(len(w.key_concepts) for w in all_weekly_content),
                "total_learning_objectives": sum(len(w.learning_objectives) for w in all_weekly_content),
                "total_gaps_identified": sum(len(w.gaps_identified) for w in all_weekly_content),
                "weeks_with_textbook": len([w for w in all_weekly_content if w.textbook_content]),
                "weeks_with_transcripts": len([w for w in all_weekly_content if w.transcript_content])
            },
            "weekly_summary": [
                {
                    "week": w.week_number,
                    "textbook_chapter": w.textbook_chapter,
                    "lecture_title": w.lecture_title,
                    "textbook_available": w.textbook_content is not None,
                    "transcript_sources": len(w.transcript_sources),
                    "key_concepts_count": len(w.key_concepts),
                    "learning_objectives_count": len(w.learning_objectives),
                    "gaps_count": len(w.gaps_identified)
                }
                for w in all_weekly_content
            ]
        }
        
        with open(self.output_path / "enhanced_processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

def main():
    """Main function"""
    project_root = Path.cwd()
    
    processor = EnhancedThreeSourceProcessor(str(project_root))
    processor.run_enhanced_processing()

if __name__ == "__main__":
    main()