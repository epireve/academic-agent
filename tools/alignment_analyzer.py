#!/usr/bin/env python3
"""
Alignment Analyzer for WOC7017 Security Risk Analysis and Evaluation

This tool analyzes the alignment between textbook chapters and lecture notes
using the Kilocode API to provide detailed content comparison.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

@dataclass
class ContentSource:
    """Represents a content source (textbook chapter or lecture note)"""
    id: str
    title: str
    content: str
    source_type: str  # 'textbook' or 'lecture'
    week_number: int

@dataclass
class AlignmentResult:
    """Result of alignment analysis between sources"""
    week_number: int
    textbook_chapter: str
    lecture_note: str
    alignment_score: float  # 0.0 to 1.0
    content_overlap: List[str]
    gaps_in_lecture: List[str]
    gaps_in_textbook: List[str]
    recommendations: List[str]

class KilocodeClient:
    """Client for interacting with Kilocode API"""
    
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
        
        self.model = os.getenv('KILOCODE_DEFAULT_MODEL', 'google/gemini-2.5-pro-preview')
        self.temperature = float(os.getenv('KILOCODE_DEFAULT_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('KILOCODE_DEFAULT_MAX_TOKENS', '2000'))

    def analyze_alignment(self, textbook_content: str, lecture_content: str, week_number: int) -> Dict[str, Any]:
        """Analyze alignment between textbook chapter and lecture content"""
        
        system_prompt = """You are an expert academic content analyzer specializing in cybersecurity and risk assessment education. 

Your task is to analyze the alignment between textbook chapters and lecture notes for a Security Risk Analysis and Evaluation course (WOC7017).

Provide a detailed analysis in the following JSON format:
{
    "alignment_score": 0.85,
    "content_overlap": ["topic1", "topic2", "topic3"],
    "gaps_in_lecture": ["missing_topic1", "missing_topic2"],
    "gaps_in_textbook": ["extra_lecture_topic1"],
    "recommendations": ["recommendation1", "recommendation2"],
    "detailed_analysis": "Comprehensive analysis of the alignment..."
}

Focus on:
1. Content coverage alignment
2. Learning objectives alignment
3. Key concepts and terminology
4. Practical applications and examples
5. Assessment preparation adequacy"""

        user_prompt = f"""
Analyze the alignment between Week {week_number} textbook chapter and lecture notes:

TEXTBOOK CHAPTER CONTENT:
{textbook_content[:3000]}...

LECTURE NOTES CONTENT:
{lecture_content}

Please provide a comprehensive alignment analysis focusing on:
1. How well the lecture covers the textbook material
2. What key concepts might be missing from either source
3. Areas where the lecture adds value beyond the textbook
4. Recommendations for improving alignment
5. Overall alignment quality score (0.0 to 1.0)
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            print(f"Error analyzing alignment for week {week_number}: {e}")
            return {
                "alignment_score": 0.0,
                "content_overlap": [],
                "gaps_in_lecture": [],
                "gaps_in_textbook": [],
                "recommendations": [f"Analysis failed: {str(e)}"],
                "detailed_analysis": f"Error occurred during analysis: {str(e)}"
            }

class AlignmentAnalyzer:
    """Main analyzer class for content alignment"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.kilocode = KilocodeClient()
        
        # Define textbook chapters
        self.textbook_chapters = [
            {"number": 1, "title": "Introduction", "start_page": 22, "end_page": 43},
            {"number": 2, "title": "Information Security Risk Assessment Basics", "start_page": 44, "end_page": 58},
            {"number": 3, "title": "Project Definition", "start_page": 60, "end_page": 93},
            {"number": 4, "title": "Security Risk Assessment Preparation", "start_page": 94, "end_page": 131},
            {"number": 5, "title": "Data Gathering", "start_page": 132, "end_page": 164},
            {"number": 6, "title": "Administrative Data Gathering", "start_page": 166, "end_page": 235},
            {"number": 7, "title": "Technical Data Gathering", "start_page": 236, "end_page": 307},
            {"number": 8, "title": "Physical Data Gathering", "start_page": 308, "end_page": 384},
            {"number": 9, "title": "Security Risk Analysis", "start_page": 386, "end_page": 400},
            {"number": 10, "title": "Security Risk Mitigation", "start_page": 402, "end_page": 414},
            {"number": 11, "title": "Security Risk Assessment Reporting", "start_page": 416, "end_page": 428},
            {"number": 12, "title": "Security Risk Assessment Project Management", "start_page": 430, "end_page": 454},
            {"number": 13, "title": "Security Risk Assessment Approaches", "start_page": 456, "end_page": 474},
        ]
        
        # Paths
        self.lecture_notes_path = self.project_root / "output" / "sra" / "lectures" / "markdown"
        self.textbook_chapters_path = self.project_root / "Split_Chapters"
        self.output_path = self.project_root / "output" / "sra" / "alignment_analysis"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_lecture_notes(self) -> Dict[int, ContentSource]:
        """Load all lecture notes"""
        lecture_notes = {}
        
        for week in range(1, 14):
            note_file = self.lecture_notes_path / f"Note{week}.md"
            if note_file.exists():
                with open(note_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title from first line or use default
                lines = content.split('\n')
                title = lines[0].strip('#').strip() if lines else f"Note {week}"
                
                lecture_notes[week] = ContentSource(
                    id=f"note_{week}",
                    title=title,
                    content=content,
                    source_type="lecture",
                    week_number=week
                )
            else:
                print(f"Warning: Note{week}.md not found")
        
        return lecture_notes

    def load_textbook_chapters(self) -> Dict[int, ContentSource]:
        """Load textbook chapters (for now, return metadata as content is in PDF)"""
        textbook_chapters = {}
        
        for chapter in self.textbook_chapters:
            chapter_num = chapter["number"]
            textbook_chapters[chapter_num] = ContentSource(
                id=f"chapter_{chapter_num}",
                title=chapter["title"],
                content=f"Chapter {chapter_num}: {chapter['title']} (Pages {chapter['start_page']}-{chapter['end_page']})",
                source_type="textbook",
                week_number=chapter_num
            )
        
        return textbook_chapters

    def analyze_weekly_alignment(self, week: int, textbook_chapter: ContentSource, lecture_note: ContentSource) -> AlignmentResult:
        """Analyze alignment for a specific week"""
        
        print(f"Analyzing Week {week}: {textbook_chapter.title} vs {lecture_note.title}")
        
        # Use Kilocode API to analyze alignment
        analysis = self.kilocode.analyze_alignment(
            textbook_chapter.content,
            lecture_note.content,
            week
        )
        
        return AlignmentResult(
            week_number=week,
            textbook_chapter=textbook_chapter.title,
            lecture_note=lecture_note.title,
            alignment_score=analysis.get("alignment_score", 0.0),
            content_overlap=analysis.get("content_overlap", []),
            gaps_in_lecture=analysis.get("gaps_in_lecture", []),
            gaps_in_textbook=analysis.get("gaps_in_textbook", []),
            recommendations=analysis.get("recommendations", [])
        )

    def run_full_analysis(self) -> List[AlignmentResult]:
        """Run complete alignment analysis for all weeks"""
        
        print("Loading lecture notes...")
        lecture_notes = self.load_lecture_notes()
        
        print("Loading textbook chapters...")
        textbook_chapters = self.load_textbook_chapters()
        
        print(f"Found {len(lecture_notes)} lecture notes and {len(textbook_chapters)} textbook chapters")
        
        results = []
        
        for week in range(1, 14):
            if week in lecture_notes and week in textbook_chapters:
                result = self.analyze_weekly_alignment(
                    week,
                    textbook_chapters[week],
                    lecture_notes[week]
                )
                results.append(result)
            else:
                print(f"Warning: Missing content for week {week}")
        
        return results

    def generate_report(self, results: List[AlignmentResult]) -> str:
        """Generate a comprehensive alignment report"""
        
        report = []
        report.append("# WOC7017 Security Risk Analysis and Evaluation")
        report.append("## Content Alignment Analysis Report")
        report.append(f"## Generated: {asyncio.get_event_loop().time()}")
        report.append("")
        
        # Summary
        avg_alignment = sum(r.alignment_score for r in results) / len(results) if results else 0
        report.append(f"## Executive Summary")
        report.append(f"- **Average Alignment Score**: {avg_alignment:.2f}/1.0")
        report.append(f"- **Total Weeks Analyzed**: {len(results)}")
        report.append(f"- **High Alignment (>0.8)**: {len([r for r in results if r.alignment_score > 0.8])}")
        report.append(f"- **Medium Alignment (0.6-0.8)**: {len([r for r in results if 0.6 <= r.alignment_score <= 0.8])}")
        report.append(f"- **Low Alignment (<0.6)**: {len([r for r in results if r.alignment_score < 0.6])}")
        report.append("")
        
        # Detailed analysis for each week
        for result in results:
            report.append(f"## Week {result.week_number}: {result.textbook_chapter}")
            report.append(f"**Lecture Topic**: {result.lecture_note}")
            report.append(f"**Alignment Score**: {result.alignment_score:.2f}/1.0")
            report.append("")
            
            if result.content_overlap:
                report.append("### Content Overlap")
                for item in result.content_overlap:
                    report.append(f"- {item}")
                report.append("")
            
            if result.gaps_in_lecture:
                report.append("### Gaps in Lecture Notes")
                for item in result.gaps_in_lecture:
                    report.append(f"- {item}")
                report.append("")
            
            if result.gaps_in_textbook:
                report.append("### Additional Lecture Content")
                for item in result.gaps_in_textbook:
                    report.append(f"- {item}")
                report.append("")
            
            if result.recommendations:
                report.append("### Recommendations")
                for item in result.recommendations:
                    report.append(f"- {item}")
                report.append("")
            
            report.append("---")
            report.append("")
        
        return "\n".join(report)

    def save_results(self, results: List[AlignmentResult], report: str):
        """Save analysis results and report"""
        
        # Save JSON results
        json_data = {
            "analysis_metadata": {
                "course": "WOC7017 Security Risk Analysis and Evaluation",
                "total_weeks": len(results),
                "average_alignment": sum(r.alignment_score for r in results) / len(results) if results else 0
            },
            "weekly_results": [asdict(result) for result in results]
        }
        
        with open(self.output_path / "alignment_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save markdown report
        with open(self.output_path / "alignment_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Results saved to {self.output_path}")

def main():
    """Main function"""
    
    # Get project root (current directory)
    project_root = Path.cwd()
    
    try:
        analyzer = AlignmentAnalyzer(str(project_root))
        
        print("Starting alignment analysis...")
        results = analyzer.run_full_analysis()
        
        print("Generating report...")
        report = analyzer.generate_report(results)
        
        print("Saving results...")
        analyzer.save_results(results, report)
        
        print("Analysis complete!")
        print(f"Average alignment score: {sum(r.alignment_score for r in results) / len(results):.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()