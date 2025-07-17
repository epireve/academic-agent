#!/usr/bin/env python
"""
Demonstration script for the Content Consolidation Agent
Shows how the agent would work with the existing project structure
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict


class ConsolidationDemo:
    """Demonstration of consolidation logic without dependencies"""
    
    def __init__(self):
        self.search_patterns = {
            "week": [
                r"week[-_]?(\d+)",
                r"w(\d+)",
                r"(\d+)[-_]?week",
                r"lecture[-_]?(\d+)",
                r"class[-_]?(\d+)"
            ],
            "transcript": [
                r"transcript",
                r"notes",
                r"summary",
                r"class",
                r"lecture"
            ],
            "course": [
                r"woc7017",
                r"sra",
                r"security[-_]?risk",
                r"risk[-_]?assessment"
            ]
        }
        
        self.unified_structure = {
            "transcripts": "transcripts/standardized",
            "lectures": "lectures/markdown", 
            "notes": "notes/markdown",
            "textbook": "textbook/markdown",
            "images": "images/consolidated",
            "metadata": "metadata/consolidated"
        }
        
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
    
    def extract_week_number(self, filename: str) -> Optional[int]:
        """Extract week number from filename"""
        filename_lower = filename.lower()
        
        for pattern in self.search_patterns["week"]:
            match = re.search(pattern, filename_lower)
            if match:
                try:
                    week_num = int(match.group(1))
                    if 1 <= week_num <= 15:
                        return week_num
                except (ValueError, IndexError):
                    continue
        return None
    
    def determine_content_type(self, filename: str, file_path: str) -> str:
        """Determine content type from filename and path"""
        filename_lower = filename.lower()
        path_lower = file_path.lower()
        
        for pattern in self.search_patterns["transcript"]:
            if re.search(pattern, filename_lower) or re.search(pattern, path_lower):
                return "transcript"
        
        if "lecture" in filename_lower or "lecture" in path_lower:
            return "lecture"
        
        if "notes" in filename_lower or "notes" in path_lower:
            return "notes"
        
        if "chapter" in filename_lower or "textbook" in path_lower:
            return "textbook"
        
        return "unknown"
    
    def calculate_confidence(self, filename: str, file_path: str, content_type: str) -> float:
        """Calculate confidence score for file classification"""
        confidence = 0.0
        filename_lower = filename.lower()
        path_lower = file_path.lower()
        
        # Week number extraction confidence
        if self.extract_week_number(filename):
            confidence += 0.3
        
        # Content type confidence
        if content_type != "unknown":
            confidence += 0.2
        
        # Course identifier confidence
        for pattern in self.search_patterns["course"]:
            if re.search(pattern, filename_lower) or re.search(pattern, path_lower):
                confidence += 0.2
                break
        
        # File extension confidence
        if filename.endswith('.md'):
            confidence += 0.1
        elif filename.endswith('.txt'):
            confidence += 0.05
        
        # Structured path confidence
        if any(indicator in path_lower for indicator in ['week', 'transcript', 'lecture', 'notes']):
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def scan_directory(self, directory: str) -> List[Dict[str, Any]]:
        """Scan directory for files and analyze them"""
        discovered_files = []
        
        if not os.path.exists(directory):
            return discovered_files
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.md', '.txt', '.pdf')):
                    file_path = os.path.join(root, file)
                    
                    # Extract information
                    week_number = self.extract_week_number(file)
                    content_type = self.determine_content_type(file, file_path)
                    confidence = self.calculate_confidence(file, file_path, content_type)
                    
                    file_info = {
                        "filename": file,
                        "full_path": file_path,
                        "relative_path": os.path.relpath(file_path, directory),
                        "week_number": week_number,
                        "content_type": content_type,
                        "confidence": confidence,
                        "base_directory": os.path.basename(directory)
                    }
                    
                    discovered_files.append(file_info)
        
        return discovered_files
    
    def analyze_current_structure(self, base_path: str) -> Dict[str, Any]:
        """Analyze the current project structure"""
        search_paths = [
            os.path.join(base_path, "output/sra/transcripts/markdown"),
            os.path.join(base_path, "output/sra/transcripts/standardized"),
            os.path.join(base_path, "markdown"),
            os.path.join(base_path, "output/sra/textbook/markdown"),
            os.path.join(base_path, "output/sra/lectures/markdown"),
            os.path.join(base_path, "output/sra/notes/markdown")
        ]
        
        all_files = []
        for search_path in search_paths:
            files = self.scan_directory(search_path)
            all_files.extend(files)
        
        # Group by week and content type
        week_groups = defaultdict(lambda: defaultdict(list))
        content_type_stats = defaultdict(int)
        confidence_stats = defaultdict(int)
        
        for file_info in all_files:
            week = file_info["week_number"]
            content_type = file_info["content_type"]
            confidence = file_info["confidence"]
            
            if week:
                week_groups[week][content_type].append(file_info)
            
            content_type_stats[content_type] += 1
            
            if confidence >= self.confidence_thresholds["high"]:
                confidence_stats["high"] += 1
            elif confidence >= self.confidence_thresholds["medium"]:
                confidence_stats["medium"] += 1
            else:
                confidence_stats["low"] += 1
        
        return {
            "total_files": len(all_files),
            "week_groups": dict(week_groups),
            "content_type_stats": dict(content_type_stats),
            "confidence_stats": dict(confidence_stats),
            "all_files": all_files
        }
    
    def identify_conflicts(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify naming conflicts and consolidation opportunities"""
        conflicts = []
        week_groups = analysis["week_groups"]
        
        for week, content_types in week_groups.items():
            for content_type, files in content_types.items():
                if len(files) > 1:
                    # Sort by confidence
                    sorted_files = sorted(files, key=lambda f: f["confidence"], reverse=True)
                    
                    conflict = {
                        "week": week,
                        "content_type": content_type,
                        "file_count": len(files),
                        "best_file": sorted_files[0],
                        "duplicates": sorted_files[1:],
                        "resolution": "choose_best"
                    }
                    
                    conflicts.append(conflict)
        
        return conflicts
    
    def generate_consolidation_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate consolidation plan"""
        conflicts = self.identify_conflicts(analysis)
        
        # Calculate statistics
        total_files = analysis["total_files"]
        files_with_conflicts = sum(c["file_count"] for c in conflicts)
        unique_files = total_files - files_with_conflicts + len(conflicts)
        
        # Generate target structure
        target_structure = {}
        for week, content_types in analysis["week_groups"].items():
            for content_type, files in content_types.items():
                target_filename = f"week_{week:02d}_{content_type}.md"
                target_dir = self.unified_structure.get(content_type, "unknown")
                target_path = os.path.join(target_dir, target_filename)
                
                # Choose best file if multiple exist
                best_file = max(files, key=lambda f: f["confidence"])
                
                target_structure[target_path] = {
                    "source_file": best_file["filename"],
                    "source_path": best_file["full_path"],
                    "week": week,
                    "content_type": content_type,
                    "confidence": best_file["confidence"],
                    "alternatives": [f["filename"] for f in files if f != best_file]
                }
        
        return {
            "conflicts": conflicts,
            "target_structure": target_structure,
            "statistics": {
                "total_source_files": total_files,
                "files_with_conflicts": files_with_conflicts,
                "unique_target_files": unique_files,
                "consolidation_ratio": unique_files / total_files if total_files > 0 else 0
            }
        }


def main():
    """Main demonstration function"""
    print("Content Consolidation Agent Demonstration")
    print("=" * 50)
    
    # Get base path
    base_path = "/Users/invoture/dev.local/academic-agent"
    
    # Create demo instance
    demo = ConsolidationDemo()
    
    # Analyze current structure
    print("Analyzing current project structure...")
    analysis = demo.analyze_current_structure(base_path)
    
    print(f"\nCurrent Structure Analysis:")
    print(f"Total files found: {analysis['total_files']}")
    print(f"Content type distribution:")
    for content_type, count in analysis['content_type_stats'].items():
        print(f"  {content_type}: {count} files")
    
    print(f"\nConfidence distribution:")
    for level, count in analysis['confidence_stats'].items():
        print(f"  {level}: {count} files")
    
    # Show week distribution
    print(f"\nWeek distribution:")
    week_groups = analysis['week_groups']
    for week in sorted(week_groups.keys()):
        content_types = week_groups[week]
        total_files = sum(len(files) for files in content_types.values())
        print(f"  Week {week}: {total_files} files")
        for content_type, files in content_types.items():
            print(f"    {content_type}: {len(files)} files")
    
    # Generate consolidation plan
    print("\nGenerating consolidation plan...")
    plan = demo.generate_consolidation_plan(analysis)
    
    print(f"\nConsolidation Plan:")
    print(f"Source files: {plan['statistics']['total_source_files']}")
    print(f"Target files: {plan['statistics']['unique_target_files']}")
    print(f"Consolidation ratio: {plan['statistics']['consolidation_ratio']:.2f}")
    
    # Show conflicts
    if plan['conflicts']:
        print(f"\nNaming Conflicts ({len(plan['conflicts'])} found):")
        for conflict in plan['conflicts']:
            print(f"  Week {conflict['week']} {conflict['content_type']}: {conflict['file_count']} files")
            print(f"    Best: {conflict['best_file']['filename']} (confidence: {conflict['best_file']['confidence']:.2f})")
            for dup in conflict['duplicates']:
                print(f"    Skip: {dup['filename']} (confidence: {dup['confidence']:.2f})")
            print()
    
    # Show target structure preview
    print(f"\nTarget Structure Preview:")
    target_structure = plan['target_structure']
    
    # Group by content type
    by_content_type = defaultdict(list)
    for target_path, info in target_structure.items():
        by_content_type[info['content_type']].append((target_path, info))
    
    for content_type, items in by_content_type.items():
        print(f"\n{content_type.upper()}:")
        for target_path, info in sorted(items):
            print(f"  {target_path}")
            print(f"    Source: {info['source_file']} (conf: {info['confidence']:.2f})")
            if info['alternatives']:
                print(f"    Alternatives: {', '.join(info['alternatives'])}")
    
    # Sample file analysis
    print(f"\nSample File Analysis:")
    sample_files = analysis['all_files'][:5]  # Show first 5 files
    for file_info in sample_files:
        print(f"  {file_info['filename']}")
        print(f"    Week: {file_info['week_number']}")
        print(f"    Type: {file_info['content_type']}")
        print(f"    Confidence: {file_info['confidence']:.2f}")
        print(f"    Path: {file_info['full_path']}")
        print()


if __name__ == "__main__":
    main()