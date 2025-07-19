#!/usr/bin/env python3
"""
Quick Alignment Check for WOC7017 Security Risk Analysis and Evaluation

This tool provides a basic alignment analysis between textbook chapters and lecture notes.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class QuickAlignmentAnalyzer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Define textbook chapters
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
        
        # Paths
        self.lecture_notes_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "lectures" / get_processed_output_path(ContentType.MARKDOWN)
        self.output_path = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "alignment_analysis"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_lecture_titles(self) -> Dict[int, str]:
        """Extract lecture titles from markdown files"""
        lecture_titles = {}
        
        for week in range(1, 14):
            note_file = self.lecture_notes_path / f"Note{week}.md"
            if note_file.exists():
                with open(note_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title from first line
                lines = content.split('\n')
                if lines:
                    title_line = lines[0].strip('#').strip()
                    # Clean up the title
                    if 'WOA7107' in title_line:
                        parts = title_line.split('-')
                        if len(parts) > 1:
                            title = parts[-1].strip()
                        else:
                            title = f"Note {week}"
                    else:
                        title = title_line
                    lecture_titles[week] = title
                else:
                    lecture_titles[week] = f"Note {week}"
            else:
                lecture_titles[week] = f"Note {week} (Missing)"
        
        return lecture_titles

    def analyze_basic_alignment(self) -> List[Dict]:
        """Perform basic alignment analysis"""
        lecture_titles = self.get_lecture_titles()
        
        alignment_results = []
        
        for chapter in self.textbook_chapters:
            week = chapter["number"]
            textbook_title = chapter["title"]
            lecture_title = lecture_titles.get(week, "Missing")
            
            # Basic alignment scoring based on title similarity
            alignment_score = self.calculate_title_similarity(textbook_title, lecture_title)
            
            # Determine alignment quality
            if alignment_score > 0.8:
                alignment_quality = "High"
            elif alignment_score > 0.6:
                alignment_quality = "Medium"
            else:
                alignment_quality = "Low"
            
            result = {
                "week": week,
                "textbook_chapter": textbook_title,
                "textbook_pages": chapter["pages"],
                "lecture_title": lecture_title,
                "alignment_score": alignment_score,
                "alignment_quality": alignment_quality,
                "notes": self.get_alignment_notes(textbook_title, lecture_title)
            }
            
            alignment_results.append(result)
        
        return alignment_results

    def calculate_title_similarity(self, textbook_title: str, lecture_title: str) -> float:
        """Calculate similarity between textbook chapter title and lecture title"""
        if "Missing" in lecture_title:
            return 0.0
        
        # Convert to lowercase for comparison
        textbook_lower = textbook_title.lower()
        lecture_lower = lecture_title.lower()
        
        # Direct keyword matching
        keywords_textbook = set(textbook_lower.split())
        keywords_lecture = set(lecture_lower.split())
        
        # Remove common words
        common_words = {"and", "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by"}
        keywords_textbook -= common_words
        keywords_lecture -= common_words
        
        if not keywords_textbook or not keywords_lecture:
            return 0.5  # Neutral if no meaningful keywords
        
        # Calculate Jaccard similarity
        intersection = keywords_textbook & keywords_lecture
        union = keywords_textbook | keywords_lecture
        
        if not union:
            return 0.0
        
        similarity = len(intersection) / len(union)
        
        # Boost score for exact matches or high relevance
        if "introduction" in textbook_lower and "introduction" in lecture_lower:
            similarity = max(similarity, 0.9)
        elif "risk assessment" in textbook_lower and "risk assessment" in lecture_lower:
            similarity = max(similarity, 0.85)
        elif "project definition" in textbook_lower and "project definition" in lecture_lower:
            similarity = max(similarity, 0.9)
        
        return min(similarity, 1.0)

    def get_alignment_notes(self, textbook_title: str, lecture_title: str) -> str:
        """Generate alignment notes"""
        if "Missing" in lecture_title:
            return "Lecture notes not available for this week"
        
        if textbook_title.lower() == lecture_title.lower():
            return "Perfect title match"
        
        # Check for specific patterns
        if "introduction" in textbook_title.lower() and "introduction" in lecture_title.lower():
            return "Strong conceptual alignment - both cover introductory material"
        elif "risk assessment" in textbook_title.lower() and "risk assessment" in lecture_title.lower():
            return "Good alignment on risk assessment concepts"
        elif "project definition" in textbook_title.lower() and "project definition" in lecture_title.lower():
            return "Excellent alignment on project definition"
        elif "data gathering" in textbook_title.lower() and "data gathering" in lecture_title.lower():
            return "Well-aligned on data gathering methodology"
        else:
            return "Requires detailed content review to assess alignment"

    def generate_summary_report(self, results: List[Dict]) -> str:
        """Generate a summary report"""
        report = []
        report.append("# WOC7017 Security Risk Analysis and Evaluation")
        report.append("## Quick Alignment Analysis Report")
        report.append("")
        
        # Summary statistics
        total_weeks = len(results)
        high_alignment = len([r for r in results if r["alignment_quality"] == "High"])
        medium_alignment = len([r for r in results if r["alignment_quality"] == "Medium"])
        low_alignment = len([r for r in results if r["alignment_quality"] == "Low"])
        avg_score = sum(r["alignment_score"] for r in results) / total_weeks
        
        report.append("## Executive Summary")
        report.append(f"- **Total Weeks Analyzed**: {total_weeks}")
        report.append(f"- **Average Alignment Score**: {avg_score:.2f}/1.0")
        report.append(f"- **High Alignment**: {high_alignment} weeks ({high_alignment/total_weeks*100:.1f}%)")
        report.append(f"- **Medium Alignment**: {medium_alignment} weeks ({medium_alignment/total_weeks*100:.1f}%)")
        report.append(f"- **Low Alignment**: {low_alignment} weeks ({low_alignment/total_weeks*100:.1f}%)")
        report.append("")
        
        # Detailed breakdown
        report.append("## Weekly Alignment Breakdown")
        report.append("")
        
        for result in results:
            report.append(f"### Week {result['week']}: {result['alignment_quality']} Alignment")
            report.append(f"- **Textbook**: {result['textbook_chapter']} (Pages {result['textbook_pages']})")
            report.append(f"- **Lecture**: {result['lecture_title']}")
            report.append(f"- **Score**: {result['alignment_score']:.2f}/1.0")
            report.append(f"- **Notes**: {result['notes']}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if low_alignment > 0:
            report.append("### Priority Actions")
            low_alignment_weeks = [r for r in results if r["alignment_quality"] == "Low"]
            for week in low_alignment_weeks:
                report.append(f"- **Week {week['week']}**: Review alignment between '{week['textbook_chapter']}' and '{week['lecture_title']}'")
            report.append("")
        
        report.append("### General Recommendations")
        report.append("- Conduct detailed content analysis using AI-powered tools")
        report.append("- Review lecture notes for completeness and accuracy")
        report.append("- Ensure all textbook chapters have corresponding lecture coverage")
        report.append("- Consider supplementing lectures with additional examples from textbook")
        report.append("- Integrate weekly transcripts to enhance content analysis")
        report.append("")
        
        return "\n".join(report)

    def save_results(self, results: List[Dict], report: str):
        """Save results to files"""
        # Save JSON results
        summary_data = {
            "course": "WOC7017 Security Risk Analysis and Evaluation",
            "analysis_type": "Quick Alignment Check",
            "total_weeks": len(results),
            "average_alignment_score": sum(r["alignment_score"] for r in results) / len(results),
            "alignment_distribution": {
                "high": len([r for r in results if r["alignment_quality"] == "High"]),
                "medium": len([r for r in results if r["alignment_quality"] == "Medium"]),
                "low": len([r for r in results if r["alignment_quality"] == "Low"])
            },
            "weekly_results": results
        }
        
        with open(self.output_path / "quick_alignment_results.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        # Save markdown report
        with open(self.output_path / "quick_alignment_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Results saved to {self.output_path}")

def main():
    project_root = Path.cwd()
    
    analyzer = QuickAlignmentAnalyzer(str(project_root))
    
    print("Analyzing alignment between textbook chapters and lecture notes...")
    results = analyzer.analyze_basic_alignment()
    
    print("Generating summary report...")
    report = analyzer.generate_summary_report(results)
    
    print("Saving results...")
    analyzer.save_results(results, report)
    
    print("\nQuick Alignment Analysis Complete!")
    print(f"Average alignment score: {sum(r['alignment_score'] for r in results) / len(results):.2f}/1.0")
    
    # Show summary
    alignment_counts = {}
    for result in results:
        quality = result["alignment_quality"]
        alignment_counts[quality] = alignment_counts.get(quality, 0) + 1
    
    print("Alignment distribution:")
    for quality, count in alignment_counts.items():
        print(f"  {quality}: {count} weeks")

if __name__ == "__main__":
    main()