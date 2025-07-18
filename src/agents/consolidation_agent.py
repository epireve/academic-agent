#!/usr/bin/env python
"""
Content Consolidation Agent - Specialized agent for merging transcripts from multiple locations
and resolving naming inconsistencies.

Migrated to use unified base agent architecture.
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
    """Results of content consolidation operation"""
    total_files_processed: int
    successful_mappings: int
    failed_mappings: int
    conflicts_resolved: int
    file_mappings: List[FileMapping]
    errors: List[str]
    processing_time: float


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
            "notes": [
                r"notes?",
                r"note[-_]?(\d+)",
                r"summary",
                r"outline"
            ]
        }
        
        # Initialize consolidation tracking
        self.file_mappings: List[FileMapping] = []
        self.conflicts: List[Dict[str, Any]] = []
        
        self.logger.info("ContentConsolidationAgent initialized with unified architecture")

    async def consolidate_directory(self, source_dir: str, target_dir: str) -> ConsolidationResult:
        """Consolidate files from source directory to target directory"""
        try:
            start_time = datetime.now()
            self.logger.info(f"Starting consolidation: {source_dir} -> {target_dir}")
            
            if not await self.validate_input({"source_dir": source_dir, "target_dir": target_dir}):
                raise ValueError("Invalid input directories")
            
            source_path = Path(source_dir)
            target_path = Path(target_dir)
            
            # Create target directory if it doesn't exist
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Discover and analyze files
            file_mappings = await self._discover_file_mappings(source_path, target_path)
            
            # Resolve conflicts
            resolved_mappings = await self._resolve_conflicts(file_mappings)
            
            # Execute consolidation
            results = await self._execute_consolidation(resolved_mappings)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            consolidation_result = ConsolidationResult(
                total_files_processed=len(file_mappings),
                successful_mappings=sum(1 for r in results if r["success"]),
                failed_mappings=sum(1 for r in results if not r["success"]),
                conflicts_resolved=len(self.conflicts),
                file_mappings=resolved_mappings,
                errors=[r["error"] for r in results if not r["success"]],
                processing_time=processing_time
            )
            
            if await self.validate_output(consolidation_result):
                self.logger.info(f"Consolidation completed successfully in {processing_time:.2f}s")
                return consolidation_result
            else:
                raise ValueError("Output validation failed")
                
        except Exception as e:
            self.logger.error(f"Error during consolidation: {e}")
            return ConsolidationResult(
                total_files_processed=0,
                successful_mappings=0,
                failed_mappings=1,
                conflicts_resolved=0,
                file_mappings=[],
                errors=[str(e)],
                processing_time=0.0
            )

    async def _discover_file_mappings(self, source_path: Path, target_path: Path) -> List[FileMapping]:
        """Discover and create file mappings between source and target"""
        mappings = []
        
        # Find all relevant files in source directory
        for file_path in source_path.rglob('*'):
            if file_path.is_file() and self._is_relevant_file(file_path):
                mapping = await self._create_file_mapping(file_path, target_path)
                if mapping:
                    mappings.append(mapping)
        
        return mappings

    def _is_relevant_file(self, file_path: Path) -> bool:
        """Check if file is relevant for consolidation"""
        # Check file extensions
        relevant_extensions = {'.md', '.txt', '.json', '.pdf'}
        if file_path.suffix.lower() not in relevant_extensions:
            return False
        
        # Check if filename matches patterns
        filename_lower = file_path.name.lower()
        for pattern_group in self.search_patterns.values():
            for pattern in pattern_group:
                if re.search(pattern, filename_lower):
                    return True
        
        return False

    async def _create_file_mapping(self, source_file: Path, target_base: Path) -> Optional[FileMapping]:
        """Create a file mapping for a source file"""
        try:
            # Extract week number and content type
            week_number = self._extract_week_number(source_file.name)
            content_type = self._determine_content_type(source_file.name)
            
            # Generate target path
            target_file = self._generate_target_path(source_file, target_base, week_number, content_type)
            
            # Calculate confidence based on pattern matching
            confidence = self._calculate_mapping_confidence(source_file.name, week_number, content_type)
            
            mapping = FileMapping(
                source_path=str(source_file),
                target_path=str(target_file),
                confidence=confidence,
                week_number=week_number,
                content_type=content_type,
                metadata={
                    "source_size": source_file.stat().st_size,
                    "source_modified": source_file.stat().st_mtime
                }
            )
            
            return mapping
            
        except Exception as e:
            self.logger.error(f"Error creating mapping for {source_file}: {e}")
            return None

    def _extract_week_number(self, filename: str) -> Optional[int]:
        """Extract week number from filename"""
        filename_lower = filename.lower()
        
        for pattern in self.search_patterns["week"]:
            match = re.search(pattern, filename_lower)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None

    def _determine_content_type(self, filename: str) -> Optional[str]:
        """Determine content type from filename"""
        filename_lower = filename.lower()
        
        if any(re.search(pattern, filename_lower) for pattern in self.search_patterns["transcript"]):
            return "transcript"
        elif any(re.search(pattern, filename_lower) for pattern in self.search_patterns["notes"]):
            return "notes"
        
        return "unknown"

    def _generate_target_path(self, source_file: Path, target_base: Path, 
                            week_number: Optional[int], content_type: Optional[str]) -> Path:
        """Generate standardized target path"""
        # Create standardized filename
        if week_number:
            if content_type and content_type != "unknown":
                filename = f"week_{week_number:02d}_{content_type}{source_file.suffix}"
            else:
                filename = f"week_{week_number:02d}{source_file.suffix}"
        else:
            # Fallback to original name with cleanup
            cleaned_name = re.sub(r'[^\w\-_.]', '_', source_file.name)
            filename = cleaned_name
        
        # Determine subdirectory based on content type
        if content_type == "transcript":
            subdir = "transcripts"
        elif content_type == "notes":
            subdir = "notes"
        else:
            subdir = "other"
        
        return target_base / subdir / filename

    def _calculate_mapping_confidence(self, filename: str, week_number: Optional[int], 
                                    content_type: Optional[str]) -> float:
        """Calculate confidence score for file mapping"""
        confidence = 0.0
        
        # Week number detection adds confidence
        if week_number:
            confidence += 0.5
        
        # Content type detection adds confidence
        if content_type and content_type != "unknown":
            confidence += 0.3
        
        # Pattern matching strength
        pattern_matches = 0
        filename_lower = filename.lower()
        for pattern_group in self.search_patterns.values():
            for pattern in pattern_group:
                if re.search(pattern, filename_lower):
                    pattern_matches += 1
        
        confidence += min(pattern_matches * 0.1, 0.2)
        
        return min(confidence, 1.0)

    async def _resolve_conflicts(self, mappings: List[FileMapping]) -> List[FileMapping]:
        """Resolve conflicts in file mappings"""
        # Group mappings by target path
        target_groups = defaultdict(list)
        for mapping in mappings:
            target_groups[mapping.target_path].append(mapping)
        
        resolved_mappings = []
        
        for target_path, conflicting_mappings in target_groups.items():
            if len(conflicting_mappings) == 1:
                # No conflict
                resolved_mappings.append(conflicting_mappings[0])
            else:
                # Resolve conflict by choosing highest confidence mapping
                best_mapping = max(conflicting_mappings, key=lambda m: m.confidence)
                
                # Log conflict
                conflict = {
                    "target_path": target_path,
                    "conflicting_sources": [m.source_path for m in conflicting_mappings],
                    "resolution": best_mapping.source_path,
                    "confidence_scores": [m.confidence for m in conflicting_mappings]
                }
                self.conflicts.append(conflict)
                
                # Rename conflicting files
                for i, mapping in enumerate(conflicting_mappings):
                    if mapping != best_mapping:
                        # Add suffix to distinguish conflicts
                        path = Path(mapping.target_path)
                        new_path = path.parent / f"{path.stem}_conflict_{i}{path.suffix}"
                        mapping.target_path = str(new_path)
                        resolved_mappings.append(mapping)
                
                resolved_mappings.append(best_mapping)
        
        return resolved_mappings

    async def _execute_consolidation(self, mappings: List[FileMapping]) -> List[Dict[str, Any]]:
        """Execute the actual file consolidation"""
        results = []
        
        for mapping in mappings:
            try:
                source_path = Path(mapping.source_path)
                target_path = Path(mapping.target_path)
                
                # Create target directory if needed
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, target_path)
                
                results.append({
                    "success": True,
                    "source": mapping.source_path,
                    "target": mapping.target_path,
                    "confidence": mapping.confidence
                })
                
                self.logger.debug(f"Successfully copied: {source_path} -> {target_path}")
                
            except Exception as e:
                results.append({
                    "success": False,
                    "source": mapping.source_path,
                    "target": mapping.target_path,
                    "error": str(e)
                })
                
                self.logger.error(f"Failed to copy {mapping.source_path}: {e}")
        
        return results

    async def validate_input(self, input_data: Any) -> bool:
        """Validate input parameters"""
        if isinstance(input_data, dict):
            source_dir = input_data.get("source_dir")
            target_dir = input_data.get("target_dir")
            
            if source_dir and target_dir:
                source_path = Path(source_dir)
                return source_path.exists() and source_path.is_dir()
        
        return False

    async def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if isinstance(output_data, ConsolidationResult):
            return output_data.total_files_processed >= 0
        return False

    def get_consolidation_report(self) -> Dict[str, Any]:
        """Generate consolidation report"""
        return {
            "mappings_count": len(self.file_mappings),
            "conflicts_count": len(self.conflicts),
            "conflicts": self.conflicts,
            "average_confidence": sum(m.confidence for m in self.file_mappings) / len(self.file_mappings) if self.file_mappings else 0,
            "content_types": list(set(m.content_type for m in self.file_mappings if m.content_type)),
            "week_range": [
                min(m.week_number for m in self.file_mappings if m.week_number),
                max(m.week_number for m in self.file_mappings if m.week_number)
            ] if any(m.week_number for m in self.file_mappings) else None
        }