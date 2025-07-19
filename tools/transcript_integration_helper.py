#!/usr/bin/env python3
"""
Transcript Integration Helper for WOC7017 Security Risk Analysis and Evaluation

This tool helps integrate weekly transcripts from multiple directories into the 
three-source processing pipeline. It handles various naming conventions and 
provides suggestions for organizing transcript files.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import shutil
import glob
from datetime import datetime

class TranscriptIntegrationHelper:
    """Helper class for integrating transcripts into the processing pipeline"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Multiple transcript directories as mentioned by user
        self.transcript_directories = [
            self.project_root / str(get_output_manager().outputs_dir) / "sra" / "transcripts" / get_processed_output_path(ContentType.MARKDOWN),
            Path("/Users/invoture/dev.local/mse-st/sra")
        ]
        
        # Standardized output directory
        self.standardized_output = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "transcripts" / "standardized"
        
        # Create directories
        self.standardized_output.mkdir(parents=True, exist_ok=True)
        
        # Week mapping for course structure
        self.week_mapping = {
            i: {
                "week": i,
                "textbook_chapter": f"Chapter {i}",
                "expected_topics": []
            }
            for i in range(1, 14)
        }
        
        # Common naming patterns to look for
        self.naming_patterns = [
            "week_{week}_transcript.md",
            "week{week}_transcript.md",
            "transcript_week_{week}.md",
            "Week{week}.md",
            "week_{week}.md",
            "w{week}.md",
            "lecture_{week}.md",
            "class_{week}.md",
            "session_{week}.md",
            "notes_{week}.md",
            "{week}.md"
        ]

    def scan_transcript_directories(self) -> Dict[str, List[str]]:
        """Scan all transcript directories for files"""
        found_files = {}
        
        for transcript_dir in self.transcript_directories:
            dir_name = str(transcript_dir)
            found_files[dir_name] = []
            
            if not transcript_dir.exists():
                print(f"⚠️  Directory not found: {transcript_dir}")
                continue
            
            # Find all markdown files
            md_files = list(transcript_dir.glob("*.md"))
            
            for md_file in md_files:
                found_files[dir_name].append({
                    "filename": md_file.name,
                    "full_path": str(md_file),
                    "size": md_file.stat().st_size,
                    "modified": datetime.fromtimestamp(md_file.stat().st_mtime).isoformat()
                })
        
        return found_files

    def analyze_filename_patterns(self, found_files: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze filename patterns to suggest week mappings"""
        analysis = {
            "total_files": 0,
            "potential_matches": {},
            "unmapped_files": [],
            "suggested_mappings": {}
        }
        
        for dir_name, files in found_files.items():
            analysis["total_files"] += len(files)
            
            for file_info in files:
                filename = file_info["filename"]
                matched_week = None
                
                # Try to match against patterns
                for week in range(1, 14):
                    for pattern in self.naming_patterns:
                        expected_name = pattern.format(week=week)
                        
                        # Exact match
                        if filename == expected_name:
                            matched_week = week
                            break
                        
                        # Partial match (case insensitive)
                        if filename.lower() in expected_name.lower() or expected_name.lower() in filename.lower():
                            # Check if it contains week number
                            if str(week) in filename:
                                matched_week = week
                                break
                    
                    if matched_week:
                        break
                
                if matched_week:
                    if matched_week not in analysis["potential_matches"]:
                        analysis["potential_matches"][matched_week] = []
                    
                    analysis["potential_matches"][matched_week].append({
                        "filename": filename,
                        "directory": dir_name,
                        "full_path": file_info["full_path"],
                        "confidence": "high" if any(pattern.format(week=matched_week) == filename for pattern in self.naming_patterns) else "medium"
                    })
                else:
                    analysis["unmapped_files"].append({
                        "filename": filename,
                        "directory": dir_name,
                        "full_path": file_info["full_path"]
                    })
        
        # Generate suggestions
        for week in range(1, 14):
            if week in analysis["potential_matches"]:
                # Sort by confidence
                matches = sorted(analysis["potential_matches"][week], key=lambda x: x["confidence"], reverse=True)
                analysis["suggested_mappings"][week] = matches[0]  # Best match
        
        return analysis

    def generate_transcript_discovery_report(self) -> str:
        """Generate a comprehensive transcript discovery report"""
        found_files = self.scan_transcript_directories()
        analysis = self.analyze_filename_patterns(found_files)
        
        report = []
        report.append("# Transcript Discovery Report")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Course**: WOC7017 Security Risk Analysis and Evaluation")
        report.append("")
        
        # Directory scan results
        report.append("## Directory Scan Results")
        for dir_name, files in found_files.items():
            report.append(f"### {dir_name}")
            report.append(f"- **Status**: {'✅ Exists' if files else '❌ Not Found'}")
            report.append(f"- **Files Found**: {len(files)}")
            
            if files:
                report.append("- **File List**:")
                for file_info in files:
                    size_kb = file_info["size"] / 1024
                    report.append(f"  - `{file_info['filename']}` ({size_kb:.1f} KB)")
            report.append("")
        
        # Analysis summary
        report.append("## Analysis Summary")
        report.append(f"- **Total Files Found**: {analysis['total_files']}")
        report.append(f"- **Weeks with Potential Matches**: {len(analysis['potential_matches'])}")
        report.append(f"- **Unmapped Files**: {len(analysis['unmapped_files'])}")
        report.append(f"- **Suggested Mappings**: {len(analysis['suggested_mappings'])}")
        report.append("")
        
        # Suggested mappings
        if analysis["suggested_mappings"]:
            report.append("## Suggested Week Mappings")
            for week in range(1, 14):
                if week in analysis["suggested_mappings"]:
                    mapping = analysis["suggested_mappings"][week]
                    report.append(f"### Week {week}")
                    report.append(f"- **File**: `{mapping['filename']}`")
                    report.append(f"- **Directory**: `{Path(mapping['directory']).name}`")
                    report.append(f"- **Confidence**: {mapping['confidence']}")
                    report.append("")
                else:
                    report.append(f"### Week {week}")
                    report.append(f"- **Status**: ❌ No transcript found")
                    report.append("")
        
        # Unmapped files
        if analysis["unmapped_files"]:
            report.append("## Unmapped Files")
            report.append("These files were found but couldn't be automatically mapped to specific weeks:")
            for file_info in analysis["unmapped_files"]:
                report.append(f"- `{file_info['filename']}` in `{Path(file_info['directory']).name}`")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if analysis["total_files"] == 0:
            report.append("### No Transcripts Found")
            report.append("- Check if transcript files exist in the specified directories")
            report.append("- Verify file extensions (.md, .txt, etc.)")
            report.append("- Consider adding transcripts to the expected directories")
        
        elif len(analysis["suggested_mappings"]) < 13:
            missing_weeks = [w for w in range(1, 14) if w not in analysis["suggested_mappings"]]
            report.append("### Missing Weeks")
            report.append(f"Transcripts not found for weeks: {', '.join(map(str, missing_weeks))}")
            report.append("- Review unmapped files for potential matches")
            report.append("- Check if these weeks had cancelled/postponed classes")
            report.append("- Consider manual mapping of similar-named files")
        
        if analysis["unmapped_files"]:
            report.append("### Unmapped Files")
            report.append("- Review these files manually to determine appropriate week mapping")
            report.append("- Consider renaming files to follow standard naming convention")
            report.append("- Check if files contain multiple weeks or other content")
        
        report.append("")
        report.append("### Standardization Suggestions")
        report.append("- Use consistent naming: `week_XX_transcript.md`")
        report.append("- Consolidate files into single transcript directory")
        report.append("- Add metadata headers to transcript files")
        report.append("")
        
        return "\n".join(report)

    def create_standardized_transcript_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized transcript structure"""
        standardization_results = {
            "processed_weeks": [],
            "skipped_weeks": [],
            "errors": []
        }
        
        for week in range(1, 14):
            if week in analysis["suggested_mappings"]:
                mapping = analysis["suggested_mappings"][week]
                source_path = Path(mapping["full_path"])
                
                try:
                    # Create standardized filename
                    standard_filename = f"week_{week:02d}_transcript.md"
                    target_path = self.standardized_output / standard_filename
                    
                    # Copy file with standardized name
                    shutil.copy2(source_path, target_path)
                    
                    # Add metadata header
                    self._add_metadata_header(target_path, week, mapping)
                    
                    standardization_results["processed_weeks"].append({
                        "week": week,
                        "source_file": mapping["filename"],
                        "source_directory": mapping["directory"],
                        "target_file": standard_filename,
                        "confidence": mapping["confidence"]
                    })
                    
                    print(f"✅ Week {week}: {mapping['filename']} → {standard_filename}")
                    
                except Exception as e:
                    error_msg = f"Week {week}: Error processing {mapping['filename']} - {str(e)}"
                    standardization_results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
            else:
                standardization_results["skipped_weeks"].append(week)
                print(f"⏭️  Week {week}: No transcript found")
        
        return standardization_results

    def _add_metadata_header(self, file_path: Path, week: int, mapping: Dict[str, Any]):
        """Add metadata header to transcript file"""
        try:
            # Read existing content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata header
            metadata = [
                "---",
                f"week: {week}",
                f"course: WOC7017",
                f"title: Week {week} Class Transcript",
                f"source_file: {mapping['filename']}",
                f"source_directory: {Path(mapping['directory']).name}",
                f"confidence: {mapping['confidence']}",
                f"processed_date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "---",
                ""
            ]
            
            # Write metadata + content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(metadata) + content)
                
        except Exception as e:
            print(f"Warning: Could not add metadata to {file_path}: {e}")

    def generate_integration_instructions(self) -> str:
        """Generate instructions for integrating transcripts"""
        instructions = []
        
        instructions.append("# Transcript Integration Instructions")
        instructions.append("")
        
        instructions.append("## Step 1: Review Discovery Report")
        instructions.append("1. Check the transcript discovery report")
        instructions.append("2. Verify suggested week mappings")
        instructions.append("3. Manually map any unmapped files")
        instructions.append("")
        
        instructions.append("## Step 2: Standardize File Structure")
        instructions.append("1. Run the standardization process")
        instructions.append("2. Review standardized files in `output/sra/transcripts/standardized/`")
        instructions.append("3. Manually copy any missing transcripts")
        instructions.append("")
        
        instructions.append("## Step 3: Re-run Enhanced Processing")
        instructions.append("1. Update transcript directories in enhanced processor")
        instructions.append("2. Run enhanced three-source processor")
        instructions.append("3. Verify transcript integration in generated notes")
        instructions.append("")
        
        instructions.append("## File Naming Convention")
        instructions.append("Use this standard format for transcript files:")
        instructions.append("```")
        instructions.append("week_01_transcript.md")
        instructions.append("week_02_transcript.md")
        instructions.append("...")
        instructions.append("week_13_transcript.md")
        instructions.append("```")
        instructions.append("")
        
        instructions.append("## Metadata Header Format")
        instructions.append("Add this header to each transcript file:")
        instructions.append("```yaml")
        instructions.append("---")
        instructions.append("week: 1")
        instructions.append("course: WOC7017")
        instructions.append("title: Week 1 Class Transcript")
        instructions.append("source_file: original_filename.md")
        instructions.append("source_directory: original_directory")
        instructions.append("confidence: high")
        instructions.append("processed_date: 2025-01-17 22:35:00")
        instructions.append("---")
        instructions.append("```")
        instructions.append("")
        
        return "\n".join(instructions)

    def run_complete_analysis(self):
        """Run complete transcript integration analysis"""
        print("🔍 Starting transcript integration analysis...")
        
        # Generate discovery report
        print("📊 Generating discovery report...")
        discovery_report = self.generate_transcript_discovery_report()
        
        # Save discovery report
        with open(self.standardized_output / "transcript_discovery_report.md", 'w', encoding='utf-8') as f:
            f.write(discovery_report)
        
        # Analyze files
        found_files = self.scan_transcript_directories()
        analysis = self.analyze_filename_patterns(found_files)
        
        # Create standardized structure
        print("📁 Creating standardized transcript structure...")
        standardization_results = self.create_standardized_transcript_structure(analysis)
        
        # Generate integration instructions
        print("📝 Generating integration instructions...")
        instructions = self.generate_integration_instructions()
        
        # Save instructions
        with open(self.standardized_output / "integration_instructions.md", 'w', encoding='utf-8') as f:
            f.write(instructions)
        
        # Save results
        results_data = {
            "analysis": analysis,
            "standardization_results": standardization_results,
            "summary": {
                "total_files_found": analysis["total_files"],
                "weeks_with_transcripts": len(analysis["suggested_mappings"]),
                "processed_weeks": len(standardization_results["processed_weeks"]),
                "skipped_weeks": len(standardization_results["skipped_weeks"]),
                "errors": len(standardization_results["errors"])
            }
        }
        
        with open(self.standardized_output / "integration_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print("\n🎉 Transcript integration analysis complete!")
        print(f"📁 Results saved to: {self.standardized_output}")
        print(f"📊 Files found: {analysis['total_files']}")
        print(f"✅ Weeks processed: {len(standardization_results['processed_weeks'])}")
        print(f"⏭️  Weeks skipped: {len(standardization_results['skipped_weeks'])}")
        
        if standardization_results["errors"]:
            print(f"❌ Errors: {len(standardization_results['errors'])}")
        
        return results_data

def main():
    """Main function"""
    project_root = Path.cwd()
    
    helper = TranscriptIntegrationHelper(str(project_root))
    results = helper.run_complete_analysis()

if __name__ == "__main__":
    main()