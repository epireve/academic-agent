#!/usr/bin/env python
"""
Week Resolver - Specialized processor for detecting and correcting week numbering discrepancies
in academic content. Addresses the week-13/14 issue and other week numbering inconsistencies.

Part of the Academic Agent system - Task 8 implementation
"""

import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import difflib


@dataclass
class WeekDetection:
    """Represents a detected week number with confidence and context"""
    week_number: int
    confidence: float
    source: str  # "filename", "content", "metadata", "pattern"
    pattern_matched: str
    context: str
    position: int = 0  # Position in filename or content


@dataclass
class WeekMapping:
    """Represents a week number mapping from detected to resolved"""
    original_week: Optional[int]
    resolved_week: int
    file_path: str
    content_type: str
    resolution_method: str
    confidence: float
    detected_weeks: List[WeekDetection]
    metadata: Dict[str, Any] = None


@dataclass
class WeekResolutionResult:
    """Result of week resolution process"""
    success: bool
    mappings: List[WeekMapping]
    discrepancies: List[Dict[str, Any]]
    resolution_report: Dict[str, Any]
    manual_review_needed: List[str]
    errors: List[Dict[str, Any]]


class WeekResolver:
    """
    Detects and corrects week numbering discrepancies in academic content.
    
    This class implements sophisticated logic to:
    1. Detect week numbers from various sources (filename, content, metadata)
    2. Identify numbering inconsistencies and gaps
    3. Resolve conflicts using contextual analysis and content mapping
    4. Apply manual overrides when configured
    5. Generate comprehensive logging and reports
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the WeekResolver with configuration
        
        Args:
            config_path: Optional path to week resolution configuration file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Week detection patterns (in order of reliability)
        self.week_patterns = {
            "explicit_week": [
                r'week[-_\s]*(\d+)',
                r'w(\d+)',
                r'(\d+)[-_\s]*week'
            ],
            "lecture_number": [
                r'lecture[-_\s]*(\d+)',
                r'class[-_\s]*(\d+)',
                r'session[-_\s]*(\d+)'
            ],
            "date_based": [
                r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})',  # Date patterns
            ],
            "ordinal": [
                r'(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth)',
                r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|13th|14th|15th)'
            ]
        }
        
        # Content-based week detection patterns
        self.content_patterns = {
            "topic_markers": [
                r'chapter[-_\s]*(\d+)',
                r'topic[-_\s]*(\d+)',
                r'part[-_\s]*(\d+)'
            ],
            "schedule_indicators": [
                r'week[-_\s]*(\d+)[-_\s]*:',
                r'week[-_\s]*(\d+)[-_\s]*-',
                r'week[-_\s]*(\d+)[-_\s]*\.',
            ]
        }
        
        # Manual week mapping overrides from configuration
        self.manual_overrides = self.config.get("manual_overrides", {})
        
        # Initialize tracking structures
        self.detected_weeks = defaultdict(list)
        self.content_analysis = {}
        self.resolution_log = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load week resolver configuration"""
        default_config = {
            "confidence_thresholds": {
                "high": 0.9,
                "medium": 0.7,
                "low": 0.5,
                "minimum": 0.3
            },
            "resolution_strategies": {
                "gap_filling": True,
                "content_analysis": True,
                "sequential_inference": True,
                "manual_override": True
            },
            "academic_calendar": {
                "semester_weeks": 15,
                "valid_range": [1, 15],
                "common_gaps": [13],  # Known problematic weeks
                "holiday_weeks": []
            },
            "manual_overrides": {
                # Example: "week-14.md": 13
                # This would map week-14 to actually be week 13
            },
            "content_mapping": {
                "enable": True,
                "similarity_threshold": 0.8,
                "topic_continuity_weight": 0.3
            },
            "logging": {
                "level": "INFO",
                "log_resolutions": True,
                "detailed_analysis": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def detect_week_numbers(self, file_path: str, content: Optional[str] = None) -> List[WeekDetection]:
        """
        Detect week numbers from filename, content, and metadata
        
        Args:
            file_path: Path to the file
            content: Optional file content for analysis
            
        Returns:
            List of detected week numbers with confidence scores
        """
        detections = []
        filename = os.path.basename(file_path)
        
        # 1. Filename-based detection (highest confidence)
        filename_detections = self._detect_from_filename(filename)
        detections.extend(filename_detections)
        
        # 2. Content-based detection (medium confidence)
        if content:
            content_detections = self._detect_from_content(content, file_path)
            detections.extend(content_detections)
        
        # 3. Metadata-based detection (if available)
        metadata_detections = self._detect_from_metadata(file_path)
        detections.extend(metadata_detections)
        
        # 4. Path-based detection (folder structure)
        path_detections = self._detect_from_path(file_path)
        detections.extend(path_detections)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Log detections
        if detections:
            self.logger.debug(f"Detected weeks for {filename}: {[d.week_number for d in detections]}")
        else:
            self.logger.warning(f"No week numbers detected for {filename}")
        
        return detections
    
    def _detect_from_filename(self, filename: str) -> List[WeekDetection]:
        """Detect week numbers from filename patterns"""
        detections = []
        filename_lower = filename.lower()
        
        # Try explicit week patterns first (highest confidence)
        for pattern in self.week_patterns["explicit_week"]:
            matches = re.finditer(pattern, filename_lower, re.IGNORECASE)
            for match in matches:
                try:
                    week_num = int(match.group(1))
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.95,
                            source="filename",
                            pattern_matched=pattern,
                            context=f"Matched '{match.group()}' in filename",
                            position=match.start()
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Try lecture/class patterns (medium-high confidence)
        for pattern in self.week_patterns["lecture_number"]:
            matches = re.finditer(pattern, filename_lower, re.IGNORECASE)
            for match in matches:
                try:
                    week_num = int(match.group(1))
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.85,
                            source="filename",
                            pattern_matched=pattern,
                            context=f"Inferred from lecture number '{match.group()}' in filename",
                            position=match.start()
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Try ordinal patterns (medium confidence)
        ordinal_map = {
            'first': 1, '1st': 1, 'second': 2, '2nd': 2, 'third': 3, '3rd': 3,
            'fourth': 4, '4th': 4, 'fifth': 5, '5th': 5, 'sixth': 6, '6th': 6,
            'seventh': 7, '7th': 7, 'eighth': 8, '8th': 8, 'ninth': 9, '9th': 9,
            'tenth': 10, '10th': 10, 'eleventh': 11, '11th': 11, 'twelfth': 12, '12th': 12,
            'thirteenth': 13, '13th': 13, 'fourteenth': 14, '14th': 14, 'fifteenth': 15, '15th': 15
        }
        
        for pattern in self.week_patterns["ordinal"]:
            matches = re.finditer(pattern, filename_lower, re.IGNORECASE)
            for match in matches:
                ordinal = match.group(1).lower()
                if ordinal in ordinal_map:
                    week_num = ordinal_map[ordinal]
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.75,
                            source="filename",
                            pattern_matched=pattern,
                            context=f"Ordinal '{match.group()}' in filename",
                            position=match.start()
                        ))
        
        return detections
    
    def _detect_from_content(self, content: str, file_path: str) -> List[WeekDetection]:
        """Detect week numbers from file content"""
        detections = []
        content_lower = content.lower()
        
        # Look for week indicators in the first few lines (higher confidence)
        lines = content.split('\n')
        header_content = '\n'.join(lines[:10]).lower()  # First 10 lines
        
        # Check for explicit week mentions in headers
        for pattern in self.week_patterns["explicit_week"]:
            matches = re.finditer(pattern, header_content, re.IGNORECASE)
            for match in matches:
                try:
                    week_num = int(match.group(1))
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.85,
                            source="content",
                            pattern_matched=pattern,
                            context=f"Found '{match.group()}' in content header",
                            position=match.start()
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Check for schedule indicators
        for pattern in self.content_patterns["schedule_indicators"]:
            matches = re.finditer(pattern, header_content, re.IGNORECASE)
            for match in matches:
                try:
                    week_num = int(match.group(1))
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.80,
                            source="content",
                            pattern_matched=pattern,
                            context=f"Schedule indicator '{match.group()}' in content",
                            position=match.start()
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Look for topic/chapter markers (lower confidence)
        for pattern in self.content_patterns["topic_markers"]:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                try:
                    week_num = int(match.group(1))
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.60,
                            source="content",
                            pattern_matched=pattern,
                            context=f"Topic marker '{match.group()}' in content",
                            position=match.start()
                        ))
                except (ValueError, IndexError):
                    continue
        
        # Check YAML frontmatter if present
        if content.startswith('---'):
            frontmatter_detections = self._detect_from_frontmatter(content)
            detections.extend(frontmatter_detections)
        
        return detections
    
    def _detect_from_frontmatter(self, content: str) -> List[WeekDetection]:
        """Extract week numbers from YAML frontmatter"""
        detections = []
        
        try:
            # Extract frontmatter
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                
                # Look for week field
                week_match = re.search(r'^week:\s*(\d+)', frontmatter, re.MULTILINE)
                if week_match:
                    week_num = int(week_match.group(1))
                    if self._is_valid_week(week_num):
                        detections.append(WeekDetection(
                            week_number=week_num,
                            confidence=0.95,
                            source="metadata",
                            pattern_matched="yaml_frontmatter",
                            context=f"YAML frontmatter week field: {week_num}",
                            position=0
                        ))
        except Exception as e:
            self.logger.debug(f"Error parsing frontmatter: {e}")
        
        return detections
    
    def _detect_from_metadata(self, file_path: str) -> List[WeekDetection]:
        """Detect week numbers from metadata files"""
        detections = []
        
        # Look for companion metadata files
        metadata_paths = [
            file_path + '_meta.json',
            file_path.replace('.md', '_meta.json'),
            os.path.join(os.path.dirname(file_path), 'metadata.json')
        ]
        
        for meta_path in metadata_paths:
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Check for week field
                    if 'week' in metadata:
                        week_num = int(metadata['week'])
                        if self._is_valid_week(week_num):
                            detections.append(WeekDetection(
                                week_number=week_num,
                                confidence=0.90,
                                source="metadata",
                                pattern_matched="metadata_file",
                                context=f"Metadata file week field: {week_num}",
                                position=0
                            ))
                    
                    # Check for other week indicators in metadata
                    for key, value in metadata.items():
                        if isinstance(value, str) and 'week' in key.lower():
                            week_match = re.search(r'(\d+)', str(value))
                            if week_match:
                                week_num = int(week_match.group(1))
                                if self._is_valid_week(week_num):
                                    detections.append(WeekDetection(
                                        week_number=week_num,
                                        confidence=0.75,
                                        source="metadata",
                                        pattern_matched="metadata_field",
                                        context=f"Metadata field {key}: {value}",
                                        position=0
                                    ))
                    
                except Exception as e:
                    self.logger.debug(f"Error reading metadata from {meta_path}: {e}")
        
        return detections
    
    def _detect_from_path(self, file_path: str) -> List[WeekDetection]:
        """Detect week numbers from directory path structure"""
        detections = []
        path_parts = Path(file_path).parts
        
        for part in path_parts:
            part_lower = part.lower()
            for pattern in self.week_patterns["explicit_week"]:
                matches = re.finditer(pattern, part_lower)
                for match in matches:
                    try:
                        week_num = int(match.group(1))
                        if self._is_valid_week(week_num):
                            detections.append(WeekDetection(
                                week_number=week_num,
                                confidence=0.70,
                                source="path",
                                pattern_matched=pattern,
                                context=f"Directory name '{part}' contains week {week_num}",
                                position=0
                            ))
                    except (ValueError, IndexError):
                        continue
        
        return detections
    
    def _is_valid_week(self, week_num: int) -> bool:
        """Check if a week number is within valid academic range"""
        valid_range = self.config["academic_calendar"]["valid_range"]
        return valid_range[0] <= week_num <= valid_range[1]
    
    def identify_discrepancies(self, file_detections: Dict[str, List[WeekDetection]]) -> List[Dict[str, Any]]:
        """
        Identify week numbering discrepancies across files
        
        Args:
            file_detections: Mapping of file paths to their week detections
            
        Returns:
            List of identified discrepancies
        """
        discrepancies = []
        
        # Build week distribution
        week_distribution = defaultdict(list)
        for file_path, detections in file_detections.items():
            if detections:
                best_detection = detections[0]  # Highest confidence
                week_distribution[best_detection.week_number].append({
                    'file_path': file_path,
                    'detection': best_detection
                })
        
        detected_weeks = set(week_distribution.keys())
        expected_weeks = set(range(
            self.config["academic_calendar"]["valid_range"][0],
            self.config["academic_calendar"]["valid_range"][1] + 1
        ))
        
        # Identify gaps
        missing_weeks = expected_weeks - detected_weeks
        if missing_weeks:
            discrepancies.append({
                'type': 'missing_weeks',
                'weeks': sorted(missing_weeks),
                'description': f"Missing weeks: {sorted(missing_weeks)}",
                'severity': 'medium' if len(missing_weeks) <= 2 else 'high'
            })
        
        # Identify extra weeks
        extra_weeks = detected_weeks - expected_weeks
        if extra_weeks:
            discrepancies.append({
                'type': 'extra_weeks',
                'weeks': sorted(extra_weeks),
                'description': f"Unexpected weeks: {sorted(extra_weeks)}",
                'severity': 'low'
            })
        
        # Identify duplicates
        duplicates = {week: files for week, files in week_distribution.items() if len(files) > 1}
        if duplicates:
            discrepancies.append({
                'type': 'duplicate_weeks',
                'weeks': duplicates,
                'description': f"Duplicate weeks found: {list(duplicates.keys())}",
                'severity': 'medium'
            })
        
        # Check for suspicious gaps (like missing week 13 but having week 14)
        known_gaps = self.config["academic_calendar"]["common_gaps"]
        for gap_week in known_gaps:
            if gap_week in missing_weeks and (gap_week + 1) in detected_weeks:
                discrepancies.append({
                    'type': 'suspicious_gap',
                    'week': gap_week,
                    'description': f"Week {gap_week} missing but week {gap_week + 1} present - possible mislabeling",
                    'severity': 'high',
                    'suggested_resolution': f"Likely week {gap_week + 1} should be week {gap_week}"
                })
        
        self.logger.info(f"Identified {len(discrepancies)} week numbering discrepancies")
        return discrepancies
    
    def resolve_week_numbers(self, file_detections: Dict[str, List[WeekDetection]], 
                           discrepancies: List[Dict[str, Any]]) -> List[WeekMapping]:
        """
        Resolve week numbering conflicts and create mappings
        
        Args:
            file_detections: Detected week numbers for each file
            discrepancies: Identified discrepancies
            
        Returns:
            List of week mappings for resolution
        """
        mappings = []
        
        for file_path, detections in file_detections.items():
            if not detections:
                continue
            
            original_week = detections[0].week_number
            resolved_week = original_week
            resolution_method = "no_change"
            confidence = detections[0].confidence
            
            # Check for manual overrides first
            filename = os.path.basename(file_path)
            if filename in self.manual_overrides:
                resolved_week = self.manual_overrides[filename]
                resolution_method = "manual_override"
                confidence = 1.0
                self.logger.info(f"Applied manual override: {filename} -> week {resolved_week}")
            
            # Apply suspicious gap resolution
            elif self._is_suspicious_gap_candidate(original_week, discrepancies):
                resolved_week = original_week - 1
                resolution_method = "gap_resolution"
                confidence = 0.85
                self.logger.info(f"Resolved suspicious gap: {filename} week {original_week} -> {resolved_week}")
            
            # Apply content-based resolution if enabled
            elif self.config["resolution_strategies"]["content_analysis"]:
                content_week = self._analyze_content_for_week(file_path, detections)
                if content_week and content_week != original_week:
                    resolved_week = content_week
                    resolution_method = "content_analysis"
                    confidence = 0.75
                    self.logger.info(f"Content analysis resolution: {filename} week {original_week} -> {resolved_week}")
            
            # Determine content type
            content_type = self._determine_content_type(file_path)
            
            mapping = WeekMapping(
                original_week=original_week,
                resolved_week=resolved_week,
                file_path=file_path,
                content_type=content_type,
                resolution_method=resolution_method,
                confidence=confidence,
                detected_weeks=detections,
                metadata={
                    'filename': filename,
                    'resolution_timestamp': datetime.now().isoformat(),
                    'discrepancies_considered': len(discrepancies)
                }
            )
            
            mappings.append(mapping)
            
            # Log the resolution
            self.resolution_log.append({
                'file_path': file_path,
                'original_week': original_week,
                'resolved_week': resolved_week,
                'method': resolution_method,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        
        return mappings
    
    def _is_suspicious_gap_candidate(self, week_num: int, discrepancies: List[Dict[str, Any]]) -> bool:
        """Check if a week number is a candidate for suspicious gap resolution"""
        for discrepancy in discrepancies:
            if (discrepancy['type'] == 'suspicious_gap' and 
                week_num == discrepancy['week'] + 1):
                return True
        return False
    
    def _analyze_content_for_week(self, file_path: str, detections: List[WeekDetection]) -> Optional[int]:
        """Analyze file content to infer the correct week number"""
        if not self.config["content_mapping"]["enable"]:
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key topics and concepts
            topics = self._extract_topics(content)
            
            # Compare with known week-topic mappings
            # This would require a topic mapping database - simplified for now
            
            # Look for sequential content indicators
            previous_refs = re.findall(r'previous\s+week|last\s+week|week\s+(\d+)', content.lower())
            next_refs = re.findall(r'next\s+week|following\s+week|week\s+(\d+)', content.lower())
            
            # Basic heuristic: if content mentions "previous week X", this might be week X+1
            for ref in previous_refs:
                if ref.isdigit():
                    inferred_week = int(ref) + 1
                    if self._is_valid_week(inferred_week):
                        return inferred_week
            
        except Exception as e:
            self.logger.debug(f"Error analyzing content for {file_path}: {e}")
        
        return None
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content for analysis"""
        # Simple topic extraction - could be enhanced with NLP
        lines = content.split('\n')
        topics = []
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('#') and len(line.split()) <= 10:  # Likely a header
                topics.append(line.lstrip('#').strip())
            elif line.startswith('**') and line.endswith('**'):  # Bold topics
                topics.append(line.strip('*').strip())
        
        return topics
    
    def _determine_content_type(self, file_path: str) -> str:
        """Determine content type from file path and name"""
        path_lower = file_path.lower()
        filename = os.path.basename(file_path).lower()
        
        if 'transcript' in path_lower or 'transcript' in filename:
            return 'transcript'
        elif 'lecture' in path_lower or 'lecture' in filename:
            return 'lecture'
        elif 'notes' in path_lower or 'notes' in filename:
            return 'notes'
        elif 'textbook' in path_lower or 'chapter' in filename:
            return 'textbook'
        else:
            return 'unknown'
    
    def apply_resolutions(self, mappings: List[WeekMapping], output_path: str) -> WeekResolutionResult:
        """
        Apply week number resolutions by renaming/moving files
        
        Args:
            mappings: Week mappings to apply
            output_path: Output directory for resolved files
            
        Returns:
            Resolution result with success status and details
        """
        processed_mappings = []
        errors = []
        manual_review_needed = []
        
        # Create output directory structure
        os.makedirs(output_path, exist_ok=True)
        
        for mapping in mappings:
            try:
                if mapping.original_week != mapping.resolved_week:
                    # Week number changed - need to rename
                    result = self._apply_week_resolution(mapping, output_path)
                    if result['success']:
                        processed_mappings.append(mapping)
                        self.logger.info(f"Applied resolution: {mapping.file_path} week {mapping.original_week} -> {mapping.resolved_week}")
                    else:
                        errors.append(result)
                        if mapping.confidence < self.config["confidence_thresholds"]["medium"]:
                            manual_review_needed.append(mapping.file_path)
                else:
                    # No change needed - copy file as is
                    result = self._copy_file_unchanged(mapping, output_path)
                    if result['success']:
                        processed_mappings.append(mapping)
                    else:
                        errors.append(result)
                        
            except Exception as e:
                error_info = {
                    'file_path': mapping.file_path,
                    'error': str(e),
                    'mapping': asdict(mapping)
                }
                errors.append(error_info)
                self.logger.error(f"Error applying resolution for {mapping.file_path}: {e}")
        
        # Generate resolution report
        resolution_report = self._generate_resolution_report(processed_mappings, errors)
        
        # Save resolution log
        log_path = os.path.join(output_path, 'week_resolution_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.resolution_log, f, indent=2, default=str)
        
        return WeekResolutionResult(
            success=len(errors) == 0,
            mappings=processed_mappings,
            discrepancies=[],  # Filled by calling method
            resolution_report=resolution_report,
            manual_review_needed=manual_review_needed,
            errors=errors
        )
    
    def _apply_week_resolution(self, mapping: WeekMapping, output_path: str) -> Dict[str, Any]:
        """Apply week number resolution to a specific file"""
        try:
            # Read original file
            with open(mapping.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate new filename
            filename = os.path.basename(mapping.file_path)
            new_filename = self._generate_resolved_filename(filename, mapping.resolved_week, mapping.content_type)
            
            # Create content type subdirectory
            content_subdir = os.path.join(output_path, mapping.content_type)
            os.makedirs(content_subdir, exist_ok=True)
            
            new_path = os.path.join(content_subdir, new_filename)
            
            # Update content metadata if it's a markdown file
            if new_path.endswith('.md'):
                content = self._update_content_metadata(content, mapping)
            
            # Write resolved file
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'success': True,
                'original_path': mapping.file_path,
                'new_path': new_path,
                'resolution_method': mapping.resolution_method
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': mapping.file_path
            }
    
    def _copy_file_unchanged(self, mapping: WeekMapping, output_path: str) -> Dict[str, Any]:
        """Copy file without changes"""
        try:
            # Read original file
            with open(mapping.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Preserve original filename but organize by content type
            filename = os.path.basename(mapping.file_path)
            content_subdir = os.path.join(output_path, mapping.content_type)
            os.makedirs(content_subdir, exist_ok=True)
            
            new_path = os.path.join(content_subdir, filename)
            
            # Write file
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'success': True,
                'original_path': mapping.file_path,
                'new_path': new_path,
                'resolution_method': 'unchanged'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': mapping.file_path
            }
    
    def _generate_resolved_filename(self, original_filename: str, week_number: int, content_type: str) -> str:
        """Generate a standardized filename with resolved week number"""
        # Remove extension
        name_part = os.path.splitext(original_filename)[0]
        
        # Generate standardized name
        resolved_name = f"week_{week_number:02d}_{content_type}"
        
        # Preserve any additional descriptive parts from original filename
        # Remove week-related parts and keep other descriptors
        cleaned_name = re.sub(r'week[-_\s]*\d+', '', name_part.lower())
        cleaned_name = re.sub(r'w\d+', '', cleaned_name)
        cleaned_name = re.sub(r'\d+[-_\s]*week', '', cleaned_name)
        cleaned_name = re.sub(r'[^a-z0-9]+', '_', cleaned_name).strip('_')
        
        if cleaned_name and cleaned_name != content_type:
            resolved_name += f"_{cleaned_name}"
        
        return resolved_name + '.md'
    
    def _update_content_metadata(self, content: str, mapping: WeekMapping) -> str:
        """Update or add metadata header to content"""
        # Check if content has YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                # Update existing frontmatter
                frontmatter = parts[1]
                body = parts[2]
                
                # Update week field
                frontmatter = re.sub(r'^week:\s*\d+', f'week: {mapping.resolved_week}', frontmatter, flags=re.MULTILINE)
                
                # Add resolution metadata
                if 'week_resolution:' not in frontmatter:
                    resolution_info = f"""week_resolution:
  original_week: {mapping.original_week}
  resolved_week: {mapping.resolved_week}
  resolution_method: {mapping.resolution_method}
  confidence: {mapping.confidence:.2f}
  resolved_date: {datetime.now().isoformat()}"""
                    frontmatter += "\n" + resolution_info
                
                return f"---{frontmatter}---{body}"
        
        # Add new frontmatter
        metadata_header = f"""---
week: {mapping.resolved_week}
content_type: {mapping.content_type}
original_file: {os.path.basename(mapping.file_path)}
week_resolution:
  original_week: {mapping.original_week}
  resolved_week: {mapping.resolved_week}
  resolution_method: {mapping.resolution_method}
  confidence: {mapping.confidence:.2f}
  resolved_date: {datetime.now().isoformat()}
---

"""
        
        return metadata_header + content
    
    def _generate_resolution_report(self, processed_mappings: List[WeekMapping], 
                                  errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive resolution report"""
        
        # Count resolutions by method
        resolution_methods = Counter(mapping.resolution_method for mapping in processed_mappings)
        
        # Count by content type
        content_types = Counter(mapping.content_type for mapping in processed_mappings)
        
        # Calculate confidence distribution
        confidence_ranges = {
            'high': sum(1 for m in processed_mappings if m.confidence >= self.config["confidence_thresholds"]["high"]),
            'medium': sum(1 for m in processed_mappings if self.config["confidence_thresholds"]["medium"] <= m.confidence < self.config["confidence_thresholds"]["high"]),
            'low': sum(1 for m in processed_mappings if m.confidence < self.config["confidence_thresholds"]["medium"])
        }
        
        # Count actual changes
        changes_made = sum(1 for m in processed_mappings if m.original_week != m.resolved_week)
        
        return {
            'summary': {
                'total_files_processed': len(processed_mappings),
                'files_with_changes': changes_made,
                'files_unchanged': len(processed_mappings) - changes_made,
                'total_errors': len(errors),
                'processing_date': datetime.now().isoformat()
            },
            'resolution_methods': dict(resolution_methods),
            'content_type_distribution': dict(content_types),
            'confidence_distribution': confidence_ranges,
            'week_changes': [
                {
                    'file': os.path.basename(m.file_path),
                    'original_week': m.original_week,
                    'resolved_week': m.resolved_week,
                    'method': m.resolution_method,
                    'confidence': m.confidence
                }
                for m in processed_mappings if m.original_week != m.resolved_week
            ],
            'error_summary': [
                {
                    'file': error.get('file_path', 'unknown'),
                    'error': error.get('error', 'unknown error')
                }
                for error in errors
            ]
        }
    
    def resolve_academic_content(self, search_paths: List[str], output_path: str) -> WeekResolutionResult:
        """
        Main workflow to resolve week numbering issues across academic content
        
        Args:
            search_paths: List of paths to search for content files
            output_path: Output directory for resolved content
            
        Returns:
            Complete resolution result
        """
        self.logger.info("Starting week number resolution workflow")
        
        # Step 1: Discover and analyze files
        file_detections = {}
        for search_path in search_paths:
            if not os.path.exists(search_path):
                self.logger.warning(f"Search path does not exist: {search_path}")
                continue
            
            self.logger.info(f"Analyzing files in: {search_path}")
            
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.endswith(('.md', '.txt')):
                        file_path = os.path.join(root, file)
                        
                        # Read content for analysis
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception as e:
                            self.logger.warning(f"Could not read {file_path}: {e}")
                            content = None
                        
                        # Detect week numbers
                        detections = self.detect_week_numbers(file_path, content)
                        if detections:
                            file_detections[file_path] = detections
        
        self.logger.info(f"Analyzed {len(file_detections)} files with week information")
        
        # Step 2: Identify discrepancies
        discrepancies = self.identify_discrepancies(file_detections)
        
        # Step 3: Resolve week numbers
        mappings = self.resolve_week_numbers(file_detections, discrepancies)
        
        # Step 4: Apply resolutions
        result = self.apply_resolutions(mappings, output_path)
        result.discrepancies = discrepancies
        
        # Step 5: Generate final report
        report_path = os.path.join(output_path, 'week_resolution_report.json')
        final_report = {
            'discrepancies': discrepancies,
            'resolution_result': result.resolution_report,
            'configuration': self.config,
            'files_analyzed': len(file_detections),
            'files_processed': len(result.mappings),
            'manual_review_needed': result.manual_review_needed,
            'workflow_completed': datetime.now().isoformat()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        self.logger.info(f"Week resolution workflow completed. Report saved to: {report_path}")
        return result


def main():
    """Command line interface for the Week Resolver"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Week Number Resolution Tool")
    parser.add_argument("--search-paths", nargs='+', required=True,
                       help="Paths to search for academic content files")
    parser.add_argument("--output-path", required=True,
                       help="Output directory for resolved files")
    parser.add_argument("--config", 
                       help="Path to week resolver configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create resolver and run workflow
    resolver = WeekResolver(config_path=args.config)
    result = resolver.resolve_academic_content(args.search_paths, args.output_path)
    
    # Print summary
    print(f"\nWeek Resolution Summary:")
    print(f"Success: {result.success}")
    print(f"Files processed: {len(result.mappings)}")
    print(f"Discrepancies found: {len(result.discrepancies)}")
    print(f"Manual review needed: {len(result.manual_review_needed)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.discrepancies:
        print(f"\nDiscrepancies found:")
        for disc in result.discrepancies:
            print(f"  - {disc['type']}: {disc['description']}")
    
    if result.manual_review_needed:
        print(f"\nFiles needing manual review:")
        for file_path in result.manual_review_needed:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()