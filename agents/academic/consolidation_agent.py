#!/usr/bin/env python
"""
Content Consolidation Agent - Specialized agent for merging transcripts from multiple locations
and resolving naming inconsistencies.
Part of the Academic Agent system
"""

import os
import re
import json
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

from .base_agent import BaseAgent, AgentMessage


@dataclass
class FileMapping:
    """Represents a mapping between source and target files"""
    source_path: str
    target_path: str
    confidence: float
    week_number: Optional[int] = None
    content_type: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ConsolidationResult:
    """Result of consolidation process"""
    success: bool
    processed_files: List[FileMapping]
    skipped_files: List[str]
    errors: List[Dict[str, Any]]
    consolidation_report: Dict[str, Any]
    unified_structure: Dict[str, Any]


class ContentConsolidationAgent(BaseAgent):
    """Agent responsible for consolidating content from multiple locations"""

    def __init__(self):
        super().__init__("consolidation_agent")
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
        
        # Standard directory structure for unified content
        self.unified_structure = {
            "transcripts": "transcripts/standardized",
            "lectures": "lectures/markdown", 
            "notes": "notes/markdown",
            "textbook": "textbook/markdown",
            "images": "images/consolidated",
            "metadata": "metadata/consolidated"
        }
        
        # Quality thresholds for confidence scoring
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }

    def scan_locations(self, base_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Scan multiple locations for content files
        
        Args:
            base_paths: List of base directory paths to scan
            
        Returns:
            List of discovered files with metadata
        """
        discovered_files = []
        
        for base_path in base_paths:
            if not os.path.exists(base_path):
                self.logger.warning(f"Path does not exist: {base_path}")
                continue
                
            self.logger.info(f"Scanning location: {base_path}")
            
            try:
                for root, dirs, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(('.md', '.txt', '.pdf')):
                            file_path = os.path.join(root, file)
                            file_info = self._analyze_file(file_path, base_path)
                            if file_info:
                                discovered_files.append(file_info)
                                
            except Exception as e:
                self.logger.error(f"Error scanning {base_path}: {str(e)}")
                continue
                
        self.logger.info(f"Discovered {len(discovered_files)} files across all locations")
        return discovered_files

    def _analyze_file(self, file_path: str, base_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single file to extract metadata and determine content type
        
        Args:
            file_path: Path to the file
            base_path: Base directory path
            
        Returns:
            File analysis result with metadata
        """
        try:
            filename = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, base_path)
            
            # Extract week number
            week_number = self._extract_week_number(filename)
            
            # Determine content type
            content_type = self._determine_content_type(filename, file_path)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(filename, file_path, content_type)
            
            # Get file stats
            stat_info = os.stat(file_path)
            
            file_info = {
                "filename": filename,
                "full_path": file_path,
                "relative_path": relative_path,
                "base_path": base_path,
                "week_number": week_number,
                "content_type": content_type,
                "confidence": confidence,
                "file_size": stat_info.st_size,
                "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "discovered_time": datetime.now().isoformat()
            }
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return None

    def _extract_week_number(self, filename: str) -> Optional[int]:
        """Extract week number from filename using various patterns"""
        filename_lower = filename.lower()
        
        for pattern in self.search_patterns["week"]:
            match = re.search(pattern, filename_lower)
            if match:
                try:
                    week_num = int(match.group(1))
                    if 1 <= week_num <= 15:  # Reasonable range for academic weeks
                        return week_num
                except (ValueError, IndexError):
                    continue
                    
        return None

    def _determine_content_type(self, filename: str, file_path: str) -> str:
        """Determine the type of content based on filename and path"""
        filename_lower = filename.lower()
        path_lower = file_path.lower()
        
        # Check for transcript indicators
        for pattern in self.search_patterns["transcript"]:
            if re.search(pattern, filename_lower) or re.search(pattern, path_lower):
                return "transcript"
                
        # Check for lecture indicators
        if "lecture" in filename_lower or "lecture" in path_lower:
            return "lecture"
            
        # Check for notes indicators
        if "notes" in filename_lower or "notes" in path_lower:
            return "notes"
            
        # Check for textbook indicators
        if "chapter" in filename_lower or "textbook" in path_lower:
            return "textbook"
            
        # Default to unknown
        return "unknown"

    def _calculate_confidence(self, filename: str, file_path: str, content_type: str) -> float:
        """Calculate confidence score for file classification"""
        confidence = 0.0
        filename_lower = filename.lower()
        path_lower = file_path.lower()
        
        # Week number extraction confidence
        if self._extract_week_number(filename):
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

    def resolve_naming_conflicts(self, discovered_files: List[Dict[str, Any]]) -> List[FileMapping]:
        """
        Resolve naming conflicts and create file mappings
        
        Args:
            discovered_files: List of discovered files with metadata
            
        Returns:
            List of file mappings with resolved names
        """
        mappings = []
        week_groups = defaultdict(list)
        
        # Group files by week number
        for file_info in discovered_files:
            week_num = file_info.get("week_number")
            if week_num:
                week_groups[week_num].append(file_info)
            else:
                # Handle files without week numbers
                self.logger.warning(f"No week number found for file: {file_info['filename']}")
        
        # Process each week group
        for week_num, files in week_groups.items():
            if len(files) == 1:
                # Single file for this week
                file_info = files[0]
                mapping = self._create_file_mapping(file_info, week_num)
                mappings.append(mapping)
            else:
                # Multiple files for this week - resolve conflicts
                resolved_mappings = self._resolve_week_conflicts(files, week_num)
                mappings.extend(resolved_mappings)
                
        return mappings

    def _create_file_mapping(self, file_info: Dict[str, Any], week_num: int) -> FileMapping:
        """Create a file mapping for a single file"""
        content_type = file_info["content_type"]
        
        # Generate standardized filename
        target_filename = f"week_{week_num:02d}_{content_type}.md"
        
        # Determine target directory based on content type
        if content_type == "transcript":
            target_dir = self.unified_structure["transcripts"]
        elif content_type == "lecture":
            target_dir = self.unified_structure["lectures"]
        elif content_type == "notes":
            target_dir = self.unified_structure["notes"]
        elif content_type == "textbook":
            target_dir = self.unified_structure["textbook"]
        else:
            target_dir = "unknown"
            
        target_path = os.path.join(target_dir, target_filename)
        
        return FileMapping(
            source_path=file_info["full_path"],
            target_path=target_path,
            confidence=file_info["confidence"],
            week_number=week_num,
            content_type=content_type,
            metadata=file_info
        )

    def _resolve_week_conflicts(self, files: List[Dict[str, Any]], week_num: int) -> List[FileMapping]:
        """Resolve conflicts when multiple files exist for the same week"""
        mappings = []
        
        # Sort files by confidence (highest first)
        sorted_files = sorted(files, key=lambda f: f["confidence"], reverse=True)
        
        # Group by content type
        content_groups = defaultdict(list)
        for file_info in sorted_files:
            content_groups[file_info["content_type"]].append(file_info)
        
        # Process each content type group
        for content_type, content_files in content_groups.items():
            if len(content_files) == 1:
                # Single file for this content type
                mapping = self._create_file_mapping(content_files[0], week_num)
                mappings.append(mapping)
            else:
                # Multiple files of same type - merge or choose best
                mapping = self._handle_duplicate_content(content_files, week_num, content_type)
                mappings.append(mapping)
                
        return mappings

    def _handle_duplicate_content(self, files: List[Dict[str, Any]], week_num: int, content_type: str) -> FileMapping:
        """Handle multiple files of the same content type for the same week"""
        # Choose the file with highest confidence
        best_file = max(files, key=lambda f: f["confidence"])
        
        # Log the decision
        self.logger.info(f"Week {week_num} {content_type}: Selected {best_file['filename']} (confidence: {best_file['confidence']:.2f})")
        
        # Note other files that will be skipped
        for file_info in files:
            if file_info != best_file:
                self.logger.info(f"  Skipping: {file_info['filename']} (confidence: {file_info['confidence']:.2f})")
        
        # Create mapping for the best file
        mapping = self._create_file_mapping(best_file, week_num)
        
        # Add information about duplicates to metadata
        mapping.metadata = mapping.metadata or {}
        mapping.metadata["duplicates"] = [
            {
                "filename": f["filename"],
                "confidence": f["confidence"],
                "path": f["full_path"]
            }
            for f in files if f != best_file
        ]
        
        return mapping

    def merge_content(self, mappings: List[FileMapping], output_base_path: str) -> ConsolidationResult:
        """
        Merge content according to the file mappings
        
        Args:
            mappings: List of file mappings
            output_base_path: Base path for output
            
        Returns:
            Consolidation result with success status and details
        """
        processed_files = []
        skipped_files = []
        errors = []
        
        # Create unified directory structure
        unified_dirs = self._create_unified_structure(output_base_path)
        
        # Process each mapping
        for mapping in mappings:
            try:
                result = self._process_file_mapping(mapping, output_base_path)
                if result["success"]:
                    processed_files.append(mapping)
                    self.logger.info(f"Processed: {mapping.source_path} -> {mapping.target_path}")
                else:
                    skipped_files.append(mapping.source_path)
                    errors.append({
                        "source_path": mapping.source_path,
                        "target_path": mapping.target_path,
                        "error": result["error"]
                    })
                    
            except Exception as e:
                error_info = {
                    "source_path": mapping.source_path,
                    "target_path": mapping.target_path,
                    "error": str(e)
                }
                errors.append(error_info)
                skipped_files.append(mapping.source_path)
                self.logger.error(f"Error processing {mapping.source_path}: {str(e)}")
        
        # Generate consolidation report
        consolidation_report = self._generate_consolidation_report(
            processed_files, skipped_files, errors
        )
        
        # Save consolidation report
        report_path = os.path.join(output_base_path, "consolidation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(consolidation_report, f, indent=2, default=str)
        
        return ConsolidationResult(
            success=len(errors) == 0,
            processed_files=processed_files,
            skipped_files=skipped_files,
            errors=errors,
            consolidation_report=consolidation_report,
            unified_structure=unified_dirs
        )

    def _create_unified_structure(self, base_path: str) -> Dict[str, str]:
        """Create the unified directory structure"""
        unified_dirs = {}
        
        for content_type, relative_path in self.unified_structure.items():
            full_path = os.path.join(base_path, relative_path)
            os.makedirs(full_path, exist_ok=True)
            unified_dirs[content_type] = full_path
            
        return unified_dirs

    def _process_file_mapping(self, mapping: FileMapping, base_path: str) -> Dict[str, Any]:
        """Process a single file mapping"""
        try:
            source_path = mapping.source_path
            target_path = os.path.join(base_path, mapping.target_path)
            
            # Ensure target directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Read source content
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add metadata header if it's a markdown file
            if target_path.endswith('.md'):
                content = self._add_metadata_header(content, mapping)
            
            # Write to target location
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _add_metadata_header(self, content: str, mapping: FileMapping) -> str:
        """Add standardized metadata header to content"""
        metadata = mapping.metadata or {}
        
        # Create YAML front matter
        header_lines = [
            "---",
            f"week: {mapping.week_number}",
            f"content_type: {mapping.content_type}",
            f"confidence: {mapping.confidence:.2f}",
            f"source_file: {os.path.basename(mapping.source_path)}",
            f"source_path: {mapping.source_path}",
            f"processed_date: {datetime.now().isoformat()}",
            f"consolidation_agent: {self.agent_id}"
        ]
        
        # Add original metadata if available
        if metadata.get("modified_time"):
            header_lines.append(f"original_modified: {metadata['modified_time']}")
        
        if metadata.get("file_size"):
            header_lines.append(f"file_size: {metadata['file_size']}")
            
        header_lines.append("---")
        header_lines.append("")
        
        # Remove existing front matter if present
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].lstrip()
        
        return "\n".join(header_lines) + content

    def _generate_consolidation_report(self, processed_files: List[FileMapping], 
                                     skipped_files: List[str], 
                                     errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive consolidation report"""
        # Group by content type and week
        content_type_stats = defaultdict(int)
        week_stats = defaultdict(int)
        
        for mapping in processed_files:
            content_type_stats[mapping.content_type] += 1
            if mapping.week_number:
                week_stats[mapping.week_number] += 1
        
        # Calculate confidence distribution
        confidence_distribution = {
            "high": 0,
            "medium": 0,
            "low": 0
        }
        
        for mapping in processed_files:
            if mapping.confidence >= self.confidence_thresholds["high"]:
                confidence_distribution["high"] += 1
            elif mapping.confidence >= self.confidence_thresholds["medium"]:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1
        
        return {
            "consolidation_summary": {
                "total_files_processed": len(processed_files),
                "total_files_skipped": len(skipped_files),
                "total_errors": len(errors),
                "processing_date": datetime.now().isoformat(),
                "agent_id": self.agent_id
            },
            "content_type_distribution": dict(content_type_stats),
            "week_distribution": dict(week_stats),
            "confidence_distribution": confidence_distribution,
            "processed_files": [
                {
                    "source_path": mapping.source_path,
                    "target_path": mapping.target_path,
                    "week_number": mapping.week_number,
                    "content_type": mapping.content_type,
                    "confidence": mapping.confidence
                }
                for mapping in processed_files
            ],
            "skipped_files": skipped_files,
            "errors": errors,
            "unified_structure": self.unified_structure
        }

    def track_progress(self, operation: str, current: int, total: int, details: str = ""):
        """Track and log progress of consolidation operations"""
        percentage = (current / total) * 100 if total > 0 else 0
        
        self.logger.info(f"{operation}: {current}/{total} ({percentage:.1f}%) - {details}")
        
        # Log metrics for monitoring
        metrics = {
            "operation": operation,
            "progress_percentage": percentage,
            "current_count": current,
            "total_count": total,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_metrics(metrics)

    def organize_unified_structure(self, base_path: str) -> Dict[str, Any]:
        """Organize content into unified directory structure"""
        try:
            # Create unified structure
            unified_dirs = self._create_unified_structure(base_path)
            
            # Create index files for each content type
            index_files = {}
            for content_type, dir_path in unified_dirs.items():
                index_path = os.path.join(dir_path, "index.md")
                index_content = self._generate_index_content(content_type, dir_path)
                
                with open(index_path, 'w', encoding='utf-8') as f:
                    f.write(index_content)
                
                index_files[content_type] = index_path
            
            # Create master index
            master_index_path = os.path.join(base_path, "master_index.md")
            master_index_content = self._generate_master_index(unified_dirs)
            
            with open(master_index_path, 'w', encoding='utf-8') as f:
                f.write(master_index_content)
            
            return {
                "success": True,
                "unified_structure": unified_dirs,
                "index_files": index_files,
                "master_index": master_index_path
            }
            
        except Exception as e:
            self.logger.error(f"Error organizing unified structure: {str(e)}")
            return {"success": False, "error": str(e)}

    def _generate_index_content(self, content_type: str, dir_path: str) -> str:
        """Generate index content for a content type directory"""
        lines = [
            f"# {content_type.title()} Index",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Files in this directory:",
            ""
        ]
        
        # List all files in the directory
        try:
            files = sorted([f for f in os.listdir(dir_path) if f.endswith('.md') and f != 'index.md'])
            
            for file in files:
                file_path = os.path.join(dir_path, file)
                lines.append(f"- [{file}]({file})")
                
                # Try to extract title from file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        first_lines = f.read(500)
                        
                    # Look for title in content
                    title_match = re.search(r'^#\s+(.+)$', first_lines, re.MULTILINE)
                    if title_match:
                        lines.append(f"  - {title_match.group(1)}")
                except:
                    pass
                    
        except Exception as e:
            lines.append(f"Error listing files: {str(e)}")
        
        return "\n".join(lines)

    def _generate_master_index(self, unified_dirs: Dict[str, str]) -> str:
        """Generate master index for all content"""
        lines = [
            "# Master Content Index",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Unified Content Structure",
            ""
        ]
        
        for content_type, dir_path in unified_dirs.items():
            lines.append(f"### {content_type.title()}")
            lines.append(f"Location: `{dir_path}`")
            
            # Count files in directory
            try:
                files = [f for f in os.listdir(dir_path) if f.endswith('.md')]
                lines.append(f"Files: {len(files)}")
                lines.append(f"[View Index]({os.path.join(dir_path, 'index.md')})")
            except Exception as e:
                lines.append(f"Error: {str(e)}")
                
            lines.append("")
        
        return "\n".join(lines)

    def consolidate_workflow(self, search_paths: List[str], output_path: str) -> ConsolidationResult:
        """
        Main consolidation workflow
        
        Args:
            search_paths: List of paths to search for content
            output_path: Path for consolidated output
            
        Returns:
            ConsolidationResult with complete workflow results
        """
        self.logger.info("Starting content consolidation workflow")
        
        try:
            # Step 1: Scan all locations
            self.logger.info("Step 1: Scanning locations for content files")
            discovered_files = self.scan_locations(search_paths)
            self.track_progress("File Discovery", len(discovered_files), len(discovered_files), 
                              f"Found {len(discovered_files)} files")
            
            if not discovered_files:
                self.logger.warning("No files discovered in search paths")
                return ConsolidationResult(
                    success=False,
                    processed_files=[],
                    skipped_files=[],
                    errors=[{"error": "No files discovered"}],
                    consolidation_report={},
                    unified_structure={}
                )
            
            # Step 2: Resolve naming conflicts
            self.logger.info("Step 2: Resolving naming conflicts")
            file_mappings = self.resolve_naming_conflicts(discovered_files)
            self.track_progress("Naming Resolution", len(file_mappings), len(discovered_files),
                              f"Created {len(file_mappings)} mappings")
            
            # Step 3: Merge content
            self.logger.info("Step 3: Merging content")
            result = self.merge_content(file_mappings, output_path)
            self.track_progress("Content Merging", len(result.processed_files), len(file_mappings),
                              f"Processed {len(result.processed_files)} files")
            
            # Step 4: Organize unified structure
            self.logger.info("Step 4: Organizing unified structure")
            structure_result = self.organize_unified_structure(output_path)
            
            if structure_result["success"]:
                result.unified_structure = structure_result["unified_structure"]
                
            self.logger.info("Content consolidation workflow completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in consolidation workflow: {str(e)}")
            return ConsolidationResult(
                success=False,
                processed_files=[],
                skipped_files=[],
                errors=[{"error": str(e)}],
                consolidation_report={},
                unified_structure={}
            )

    def check_quality(self, content: Dict[str, Any]) -> float:
        """Check quality of consolidation process"""
        quality_score = 1.0
        
        if "consolidation_result" in content:
            result = content["consolidation_result"]
            
            # Check success rate
            total_files = len(result.processed_files) + len(result.skipped_files)
            if total_files > 0:
                success_rate = len(result.processed_files) / total_files
                quality_score *= success_rate
            
            # Check for errors
            if result.errors:
                error_penalty = min(0.5, len(result.errors) * 0.1)
                quality_score -= error_penalty
                
            # Check confidence distribution
            if result.consolidation_report.get("confidence_distribution"):
                conf_dist = result.consolidation_report["confidence_distribution"]
                high_conf_ratio = conf_dist.get("high", 0) / max(1, sum(conf_dist.values()))
                quality_score *= (0.5 + 0.5 * high_conf_ratio)
        
        return max(0.0, min(1.0, quality_score))

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for consolidation"""
        if not isinstance(input_data, dict):
            return False
            
        required_fields = ["search_paths", "output_path"]
        if not all(field in input_data for field in required_fields):
            return False
            
        # Validate search paths
        if not isinstance(input_data["search_paths"], list):
            return False
            
        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data from consolidation"""
        if not isinstance(output_data, ConsolidationResult):
            return False
            
        # Check required fields
        required_fields = ["success", "processed_files", "skipped_files", "errors"]
        return all(hasattr(output_data, field) for field in required_fields)


def main():
    """Main entry point for the consolidation agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Content Consolidation Agent")
    parser.add_argument("--search-paths", nargs='+', required=True,
                       help="Paths to search for content files")
    parser.add_argument("--output-path", required=True,
                       help="Path for consolidated output")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Create consolidation agent
    agent = ContentConsolidationAgent()
    
    # Run consolidation workflow
    result = agent.consolidate_workflow(args.search_paths, args.output_path)
    
    # Print results
    print(f"\nConsolidation Results:")
    print(f"Success: {result.success}")
    print(f"Processed files: {len(result.processed_files)}")
    print(f"Skipped files: {len(result.skipped_files)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()