#!/usr/bin/env python3
"""
Generate final master index for all completed AI-enhanced study notes
"""

import os
import json
from pathlib import Path
from datetime import datetime

def generate_final_master_index():
    """Generate comprehensive master index"""
    
    project_root = Path.cwd()
    output_path = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    
    # Chapter titles
    chapters = [
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
    
    index = []
    
    index.append("# WOC7017 Security Risk Analysis and Evaluation")
    index.append("## AI-Enhanced Comprehensive Study Notes")
    index.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    index.append("**AI Model**: Gemini 2.5 Pro (via Kilocode)")
    index.append("**Content Sources**: Textbook + Lecture Slides + Strategic Diagrams")
    index.append("")
    
    index.append("## 🎯 Overview")
    index.append("These comprehensive study notes represent the complete integration of:")
    index.append("- **📚 Textbook Content**: The Security Risk Assessment Handbook (2nd Edition)")
    index.append("- **🎓 Lecture Slides**: Professor's presentation materials")
    index.append("- **🧠 AI Enhancement**: Gemini 2.5 Pro synthesis and analysis")
    index.append("- **📊 Strategic Diagrams**: Mermaid.js concept maps and embedded figures")
    index.append("")
    
    index.append("## 🚀 Features")
    index.append("✅ **Natural Integration**: Seamless combination of textbook depth and lecture clarity")
    index.append("✅ **Smart Diagram Placement**: Visuals embedded where they enhance understanding")
    index.append("✅ **Concept Mapping**: High-level Mermaid diagrams showing relationships")
    index.append("✅ **Flexible Formatting**: Bullets, paragraphs, and tables as appropriate")
    index.append("✅ **Exam-Ready**: Comprehensive coverage optimized for understanding")
    index.append("")
    
    index.append("## 📚 Complete Study Notes Index")
    index.append("")
    
    # Generate index for each week
    for chapter in chapters:
        week_num = chapter["number"]
        title = chapter["title"]
        filename = f"week_{week_num:02d}_comprehensive_study_notes.md"
        
        # Check if file exists and get size
        file_path = output_path / filename
        if file_path.exists():
            file_size = file_path.stat().st_size
            size_kb = file_size / 1024
            status = "✅"
        else:
            size_kb = 0
            status = "❌"
        
        index.append(f"### Week {week_num}: {title}")
        index.append(f"- **Status**: {status} {'Available' if status == '✅' else 'Missing'}")
        index.append(f"- **File**: `{filename}`")
        index.append(f"- **Size**: {size_kb:.1f} KB")
        index.append("")
    
    # Statistics
    total_files = len([f for f in output_path.glob("week_*_comprehensive_study_notes.md")])
    total_size = sum(f.stat().st_size for f in output_path.glob("week_*_comprehensive_study_notes.md"))
    
    index.append("## 📊 Statistics")
    index.append(f"- **Total Weeks**: 13")
    index.append(f"- **Completed**: {total_files}")
    index.append(f"- **Total Size**: {total_size/1024:.1f} KB ({total_size/1024/1024:.1f} MB)")
    index.append(f"- **Average Size**: {total_size/total_files/1024:.1f} KB per week")
    index.append("")
    
    index.append("## 🎓 How to Use These Notes")
    index.append("### For Sequential Learning")
    index.append("1. **Start with Week 1**: Foundation concepts are built upon progressively")
    index.append("2. **Study the Concept Diagrams**: Each week begins with a high-level Mermaid diagram")
    index.append("3. **Follow the Natural Flow**: Content is organized for optimal understanding")
    index.append("4. **Review Key Takeaways**: End each week with the summary points")
    index.append("")
    
    index.append("### For Exam Preparation")
    index.append("1. **Focus on Key Concepts**: Each section identifies the most important topics")
    index.append("2. **Study the Diagrams**: Visual aids are strategically placed to enhance comprehension")
    index.append("3. **Use Tables for Comparisons**: Complex comparisons are presented in tabular format")
    index.append("4. **Cross-Reference**: Look for connections between weeks and concepts")
    index.append("")
    
    index.append("### For Quick Review")
    index.append("1. **Executive Summaries**: Each week starts with a concise overview")
    index.append("2. **Key Takeaways**: Bullet-point summaries at the end of each week")
    index.append("3. **Concept Diagrams**: Visual representations of main relationships")
    index.append("")
    
    index.append("## 🔧 Technical Details")
    index.append("- **AI Model**: Google Gemini 2.5 Pro via Kilocode API")
    index.append("- **Diagram Format**: Mermaid.js for concept maps")
    index.append("- **Image Integration**: Strategic placement of textbook figures")
    index.append("- **Content Synthesis**: Natural language processing for optimal integration")
    index.append("")
    
    index.append("## 🎯 Study Recommendations")
    index.append("### Week-by-Week Approach")
    index.append("- **Week 1-2**: Foundation concepts and risk management principles")
    index.append("- **Week 3-4**: Project planning and assessment preparation")
    index.append("- **Week 5-8**: Data gathering methodologies (comprehensive coverage)")
    index.append("- **Week 9-10**: Risk analysis and mitigation strategies")
    index.append("- **Week 11-13**: Reporting, management, and assessment approaches")
    index.append("")
    
    index.append("### Key Integration Points")
    index.append("- **Risk Assessment Lifecycle**: Covered across multiple weeks")
    index.append("- **Data Gathering Methods**: Detailed in weeks 5-8")
    index.append("- **Practical Applications**: Integrated throughout all weeks")
    index.append("")
    
    index.append("---")
    index.append("*Generated with AI-enhanced content synthesis for optimal learning outcomes*")
    index.append("")
    
    # Save master index
    with open(output_path / "master_index.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(index))
    
    print("✅ Final master index generated successfully!")
    print(f"📁 Location: {output_path / 'master_index.md'}")
    print(f"📚 Total study notes: {total_files}/13 weeks")
    print(f"📊 Total content: {total_size/1024/1024:.1f} MB")

if __name__ == "__main__":
    generate_final_master_index()