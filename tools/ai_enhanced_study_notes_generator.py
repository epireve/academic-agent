#!/usr/bin/env python3
"""
AI-Enhanced Study Notes Generator for WOC7017 Security Risk Analysis and Evaluation

This tool creates comprehensive study notes by combining textbook chapters and lecture slides,
enhanced with AI-powered summaries and strategically placed diagrams using Gemini 2.5 Pro via Kilocode.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class WeeklyStudyContent:
    """Represents study content for a specific week"""
    week_number: int
    chapter_title: str
    textbook_content: str
    lecture_content: str
    available_diagrams: List[str]
    generated_notes: Optional[str] = None
    processing_time: float = 0.0
    tokens_used: int = 0
    success: bool = False

class KilocodeClient:
    """Enhanced Kilocode client for study notes generation"""
    
    def __init__(self):
        self.token = os.getenv('KILOCODE_TOKEN')
        if not self.token:
            raise ValueError("KILOCODE_TOKEN not found in environment variables")
        
        # Initialize OpenAI client with Kilocode configuration
        self.client = OpenAI(
            base_url="https://kilocode.ai/api/openrouter",
            api_key=self.token,
            default_headers={
                "HTTP-Referer": "https://kilocode.ai",
                "X-Title": "Kilo Code",
                "X-KiloCode-Version": "1.0.0",
            }
        )
        
        self.model = "google/gemini-2.5-pro-preview"
        self.temperature = 0.7
        self.max_tokens = 8000

    def generate_study_notes(self, textbook_content: str, lecture_content: str, 
                           available_diagrams: List[str], week_number: int, 
                           chapter_title: str) -> Dict[str, Any]:
        """Generate comprehensive study notes using AI"""
        
        # Create diagram descriptions
        diagram_descriptions = []
        for diagram in available_diagrams:
            diagram_name = Path(diagram).name
            diagram_descriptions.append(f"- {diagram_name}")
        
        diagram_list = "\n".join(diagram_descriptions) if diagram_descriptions else "No diagrams available"
        
        system_prompt = """You are an expert academic content synthesizer specializing in cybersecurity and risk assessment education. 

Your task is to create comprehensive study notes by naturally integrating textbook and lecture content.

Guidelines:
- Synthesize content naturally without artificial structure
- Embed diagrams only where they genuinely enhance understanding
- Use varied formatting (bullets, paragraphs, tables) based on content needs
- Focus on clear, comprehensive explanations
- Create one high-level concept diagram using Mermaid.js showing main relationships
- Include relevant diagrams from available sources with brief contextual captions
- Eliminate redundancy while preserving important details
- No references or bibliography
- General applicability (not exam-specific)

Output format should be clean markdown with:
1. High-level concept overview (Mermaid diagram)
2. Executive summary
3. Key concepts with natural explanations
4. Strategic diagram placement
5. Key takeaways
"""

        user_prompt = f"""
Create comprehensive study notes for Week {week_number}: {chapter_title}

TEXTBOOK CONTENT:
{textbook_content[:4000]}...

LECTURE CONTENT:
{lecture_content}

AVAILABLE DIAGRAMS:
{diagram_list}

Requirements:
- Create a high-level Mermaid.js concept diagram showing main relationships
- Naturally integrate both textbook and lecture content
- Embed diagrams only where they enhance understanding (use format: ![Description](diagram_filename))
- Use appropriate formatting (bullets, tables, paragraphs) based on content needs
- Focus on key concepts without artificial "why important" structure
- Provide clear, comprehensive explanations
- Include key takeaways section
- No references or bibliography

Generate comprehensive study notes that synthesize both sources effectively.
"""

        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "study_notes": response.choices[0].message.content,
                "processing_time": processing_time,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "success": True
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating study notes for week {week_number}: {e}")
            return {
                "study_notes": f"Error generating study notes: {str(e)}",
                "processing_time": 0.0,
                "tokens_used": 0,
                "success": False
            }

class AIEnhancedStudyNotesGenerator:
    """Main generator class for AI-enhanced study notes"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Define textbook chapters structure
        self.textbook_chapters = [
            {"number": 1, "title": "Introduction"},
            {"number": 2, "title": "Information Security Risk Assessment Basics"},
            {"number": 3, "title": "Project Definition"},
            {"number": 4, "title": "Security Risk Assessment Preparation"},
            {"number": 5, "title": "Data Gathering"},
            {"number": 6, "title": "Administrative Data Gathering"},
            {"number": 7, "title": "Technical Data Gathering"},
            {"number": 8, "title": "Physical Data Gathering"},
            {"number": 9, "title": "Security Risk Analysis"},
            {"number": 10, "title": "Security Risk Mitigation"},
            {"number": 11, "title": "Security Risk Assessment Reporting"},
            {"number": 12, "title": "Security Risk Assessment Project Management"},
            {"number": 13, "title": "Security Risk Assessment Approaches"},
        ]
        
        # Define paths
        self.textbook_path = self.project_root / "output" / "sra" / "textbook" / "markdown"
        self.lecture_path = self.project_root / "output" / "sra" / "lectures" / "markdown"
        self.output_path = self.project_root / "output" / "sra" / "ai_enhanced_study_notes"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kilocode client
        self.kilocode = KilocodeClient()
        
        # Processing results
        self.processing_results = []

    def load_textbook_content(self, week: int) -> Tuple[str, List[str]]:
        """Load textbook content and available diagrams for a specific week"""
        chapter_info = self.textbook_chapters[week - 1]
        chapter_dir = self.textbook_path / f"Chapter_{week}_{chapter_info['title'].replace(' ', '_')}"
        
        # Find markdown file
        md_files = list(chapter_dir.glob("*.md"))
        if not md_files:
            return "", []
        
        # Read markdown content
        try:
            with open(md_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading textbook content for week {week}: {e}")
            return "", []
        
        # Find available diagrams
        diagrams = []
        for img_file in chapter_dir.glob("*.jpeg"):
            diagrams.append(str(img_file))
        
        return content, diagrams

    def load_lecture_content(self, week: int) -> str:
        """Load lecture content for a specific week"""
        note_file = self.lecture_path / f"Note{week}.md"
        
        if not note_file.exists():
            return ""
        
        try:
            with open(note_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading lecture content for week {week}: {e}")
            return ""

    def process_week(self, week: int) -> WeeklyStudyContent:
        """Process study notes for a specific week"""
        chapter_info = self.textbook_chapters[week - 1]
        chapter_title = chapter_info["title"]
        
        print(f"🔄 Processing Week {week}: {chapter_title}")
        
        # Load content
        textbook_content, available_diagrams = self.load_textbook_content(week)
        lecture_content = self.load_lecture_content(week)
        
        # Create weekly content object
        weekly_content = WeeklyStudyContent(
            week_number=week,
            chapter_title=chapter_title,
            textbook_content=textbook_content,
            lecture_content=lecture_content,
            available_diagrams=available_diagrams
        )
        
        # Generate study notes with AI
        result = self.kilocode.generate_study_notes(
            textbook_content=textbook_content,
            lecture_content=lecture_content,
            available_diagrams=available_diagrams,
            week_number=week,
            chapter_title=chapter_title
        )
        
        # Update weekly content with results
        weekly_content.generated_notes = result["study_notes"]
        weekly_content.processing_time = result["processing_time"]
        weekly_content.tokens_used = result["tokens_used"]
        weekly_content.success = result["success"]
        
        # Save individual week notes
        if weekly_content.success:
            week_file = self.output_path / f"week_{week:02d}_comprehensive_study_notes.md"
            try:
                with open(week_file, 'w', encoding='utf-8') as f:
                    f.write(weekly_content.generated_notes)
                print(f"✅ Week {week} completed successfully")
                print(f"   📄 Output: {week_file}")
                print(f"   ⏱️  Processing time: {weekly_content.processing_time:.1f}s")
                print(f"   🎯 Tokens used: {weekly_content.tokens_used}")
            except Exception as e:
                print(f"❌ Error saving week {week}: {e}")
                weekly_content.success = False
        else:
            print(f"❌ Week {week} failed to process")
        
        return weekly_content

    def process_all_weeks(self) -> List[WeeklyStudyContent]:
        """Process all weeks sequentially"""
        print("🚀 Starting AI-Enhanced Study Notes Generation")
        print("=" * 60)
        
        all_weekly_content = []
        
        for week in range(1, 14):
            # Process week
            weekly_content = self.process_week(week)
            all_weekly_content.append(weekly_content)
            self.processing_results.append(weekly_content)
            
            # Small delay to be respectful to API
            time.sleep(2)
            
            print()  # Add spacing between weeks
        
        return all_weekly_content

    def generate_master_index(self, all_weekly_content: List[WeeklyStudyContent]) -> str:
        """Generate master index for all study notes"""
        index = []
        
        index.append("# WOC7017 Security Risk Analysis and Evaluation")
        index.append("## AI-Enhanced Comprehensive Study Notes")
        index.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        index.append("**AI Model**: Gemini 2.5 Pro (via Kilocode)")
        index.append("")
        
        # Course overview
        index.append("## Course Overview")
        index.append("These comprehensive study notes combine textbook chapters and lecture slides,")
        index.append("enhanced with AI-powered synthesis and strategically placed diagrams.")
        index.append("")
        
        # Processing statistics
        successful_weeks = [w for w in all_weekly_content if w.success]
        total_tokens = sum(w.tokens_used for w in successful_weeks)
        total_time = sum(w.processing_time for w in successful_weeks)
        
        index.append("## Processing Statistics")
        index.append(f"- **Total Weeks**: {len(all_weekly_content)}")
        index.append(f"- **Successfully Processed**: {len(successful_weeks)}")
        index.append(f"- **Total Processing Time**: {total_time:.1f} seconds")
        index.append(f"- **Total Tokens Used**: {total_tokens:,}")
        index.append(f"- **Average Tokens per Week**: {total_tokens/len(successful_weeks):.0f}")
        index.append("")
        
        # Weekly breakdown
        index.append("## Weekly Study Notes Index")
        index.append("")
        
        for week_content in all_weekly_content:
            status = "✅" if week_content.success else "❌"
            index.append(f"### Week {week_content.week_number}: {week_content.chapter_title}")
            index.append(f"- **Status**: {status}")
            index.append(f"- **Textbook Content**: {'✅ Available' if week_content.textbook_content else '❌ Not Available'}")
            index.append(f"- **Lecture Content**: {'✅ Available' if week_content.lecture_content else '❌ Not Available'}")
            index.append(f"- **Diagrams Available**: {len(week_content.available_diagrams)}")
            
            if week_content.success:
                index.append(f"- **Processing Time**: {week_content.processing_time:.1f}s")
                index.append(f"- **Tokens Used**: {week_content.tokens_used:,}")
                index.append(f"- **Study Notes**: `week_{week_content.week_number:02d}_comprehensive_study_notes.md`")
            else:
                index.append(f"- **Status**: Failed to process")
            
            index.append("")
        
        # Usage instructions
        index.append("## How to Use These Notes")
        index.append("1. **Sequential Study**: Start with Week 1 and progress through each week")
        index.append("2. **Concept Mapping**: Use the high-level diagrams to understand relationships")
        index.append("3. **Visual Learning**: Pay attention to embedded diagrams and their contexts")
        index.append("4. **Key Takeaways**: Review the summary points at the end of each week")
        index.append("5. **Cross-References**: Look for connections between weeks")
        index.append("")
        
        # Features
        index.append("## Features")
        index.append("- **AI-Synthesized Content**: Natural integration of textbook and lecture materials")
        index.append("- **Strategic Diagrams**: Visuals embedded where they enhance understanding")
        index.append("- **Flexible Formatting**: Bullets, paragraphs, and tables as appropriate")
        index.append("- **Comprehensive Coverage**: All key concepts from both sources")
        index.append("- **Study-Optimized**: Focused on understanding rather than memorization")
        index.append("")
        
        return "\n".join(index)

    def save_processing_summary(self, all_weekly_content: List[WeeklyStudyContent]):
        """Save processing summary and statistics"""
        summary_data = {
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "total_weeks": len(all_weekly_content),
                "successful_weeks": len([w for w in all_weekly_content if w.success]),
                "model_used": self.kilocode.model,
                "total_processing_time": sum(w.processing_time for w in all_weekly_content),
                "total_tokens_used": sum(w.tokens_used for w in all_weekly_content)
            },
            "weekly_results": [
                {
                    "week": w.week_number,
                    "chapter_title": w.chapter_title,
                    "success": w.success,
                    "processing_time": w.processing_time,
                    "tokens_used": w.tokens_used,
                    "textbook_available": bool(w.textbook_content),
                    "lecture_available": bool(w.lecture_content),
                    "diagrams_count": len(w.available_diagrams)
                }
                for w in all_weekly_content
            ]
        }
        
        with open(self.output_path / "processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    def run_complete_generation(self):
        """Run the complete AI-enhanced study notes generation pipeline"""
        print("🤖 AI-Enhanced Study Notes Generator")
        print("=" * 60)
        
        # Process all weeks
        all_weekly_content = self.process_all_weeks()
        
        # Generate master index
        print("📚 Generating master index...")
        master_index = self.generate_master_index(all_weekly_content)
        
        # Save master index
        with open(self.output_path / "master_index.md", 'w', encoding='utf-8') as f:
            f.write(master_index)
        
        # Save processing summary
        self.save_processing_summary(all_weekly_content)
        
        # Final summary
        successful_weeks = len([w for w in all_weekly_content if w.success])
        total_tokens = sum(w.tokens_used for w in all_weekly_content if w.success)
        
        print("=" * 60)
        print("🎉 AI-Enhanced Study Notes Generation Complete!")
        print(f"📊 Successfully processed: {successful_weeks}/13 weeks")
        print(f"🎯 Total tokens used: {total_tokens:,}")
        print(f"📁 Output directory: {self.output_path}")
        print(f"📚 Master index: master_index.md")
        print(f"📄 Individual notes: week_XX_comprehensive_study_notes.md")
        
        return all_weekly_content

def main():
    """Main function"""
    project_root = Path.cwd()
    
    generator = AIEnhancedStudyNotesGenerator(str(project_root))
    results = generator.run_complete_generation()

if __name__ == "__main__":
    main()