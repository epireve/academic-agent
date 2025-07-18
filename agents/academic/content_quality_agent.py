#!/usr/bin/env python
"""
Content Quality Assurance Agent - Comprehensive quality validation for academic content.

This module implements advanced quality assessment capabilities including:
- Content completeness validation
- Formatting consistency checks
- Academic content quality scoring
- Automated quality improvement suggestions
- Quality analytics and reporting
- Integration with content consolidation workflows

Part of the Academic Agent system - Task 9 Implementation
"""

import os
import re
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import hashlib
import difflib

from .base_agent import BaseAgent, AgentMessage
from .quality_manager import QualityManager, QualityMetrics, QualityEvaluation


@dataclass
class ContentStructure:
    """Represents the structure analysis of academic content"""
    headers: List[str]
    header_levels: Dict[int, int]  # level -> count
    paragraphs: int
    lists: int
    images: int
    tables: int
    code_blocks: int
    mermaid_diagrams: int
    citations: int
    word_count: int
    character_count: int


@dataclass
class FormattingIssue:
    """Represents a formatting consistency issue"""
    issue_type: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    line_number: Optional[int]
    suggestion: str
    auto_fixable: bool
    context: str = ""


@dataclass
class CompletenessCheck:
    """Represents completeness validation results"""
    required_sections: List[str]
    missing_sections: List[str]
    incomplete_sections: List[str]
    completeness_score: float
    week_number: Optional[int]
    content_type: str


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    content_id: str
    file_path: str
    content_type: str
    week_number: Optional[int]
    
    # Quality scores
    overall_quality_score: float
    completeness_score: float
    formatting_score: float
    consistency_score: float
    academic_quality_score: float
    
    # Detailed analysis
    content_structure: ContentStructure
    completeness_check: CompletenessCheck
    formatting_issues: List[FormattingIssue]
    quality_evaluation: QualityEvaluation
    
    # Improvement suggestions
    improvement_suggestions: List[str]
    priority_fixes: List[str]
    auto_fixes_available: List[str]
    
    # Metadata
    assessment_date: datetime
    processing_time: float
    file_size: int
    content_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "content_id": self.content_id,
            "file_path": self.file_path,
            "content_type": self.content_type,
            "week_number": self.week_number,
            "overall_quality_score": self.overall_quality_score,
            "completeness_score": self.completeness_score,
            "formatting_score": self.formatting_score,
            "consistency_score": self.consistency_score,
            "academic_quality_score": self.academic_quality_score,
            "content_structure": asdict(self.content_structure),
            "completeness_check": asdict(self.completeness_check),
            "formatting_issues": [asdict(issue) for issue in self.formatting_issues],
            "quality_evaluation": asdict(self.quality_evaluation),
            "improvement_suggestions": self.improvement_suggestions,
            "priority_fixes": self.priority_fixes,
            "auto_fixes_available": self.auto_fixes_available,
            "assessment_date": self.assessment_date.isoformat(),
            "processing_time": self.processing_time,
            "file_size": self.file_size,
            "content_hash": self.content_hash
        }


@dataclass
class QualityAnalytics:
    """Analytics across multiple quality assessments"""
    total_files_assessed: int
    average_quality_score: float
    quality_distribution: Dict[str, int]  # quality_range -> count
    common_issues: List[Dict[str, Any]]
    content_type_quality: Dict[str, float]
    week_quality_trends: Dict[int, float]
    improvement_opportunities: List[str]
    quality_trends: List[Dict[str, Any]]
    assessment_summary: Dict[str, Any]


class ContentQualityAgent(BaseAgent):
    """
    Advanced Content Quality Assurance Agent
    
    Provides comprehensive quality validation for academic content including
    completeness checks, formatting validation, and quality improvement suggestions.
    """

    def __init__(self):
        super().__init__("content_quality_agent")
        
        # Initialize quality manager
        self.quality_manager = QualityManager(quality_threshold=0.7)
        
        # Quality assessment configuration
        self.quality_config = {
            "weights": {
                "completeness": 0.25,
                "formatting": 0.20,
                "consistency": 0.20,
                "academic_quality": 0.35
            },
            "thresholds": {
                "excellent": 0.90,
                "good": 0.75,
                "acceptable": 0.60,
                "poor": 0.40
            },
            "required_sections": {
                "transcript": ["overview", "key_concepts", "summary"],
                "lecture": ["introduction", "main_content", "conclusion"],
                "notes": ["summary", "key_points"],
                "textbook": ["chapter_title", "sections", "summary"],
                "comprehensive_study_notes": [
                    "high-level concept overview", 
                    "executive summary", 
                    "key concepts",
                    "detailed analysis",
                    "practical applications",
                    "exam focus areas",
                    "review questions"
                ]
            },
            "formatting_rules": {
                "max_line_length": 120,
                "header_consistency": True,
                "list_formatting": True,
                "image_alt_text": True,
                "proper_emphasis": True,
                "link_validity": False  # Disabled for now
            }
        }
        
        # Academic quality patterns
        self.academic_patterns = {
            "concept_introduction": r"(?i)(define|definition|concept|principle|theory)",
            "examples": r"(?i)(example|instance|case study|illustration)",
            "analysis": r"(?i)(analysis|analyze|examine|evaluate|assess)",
            "synthesis": r"(?i)(synthesis|synthesize|combine|integrate)",
            "application": r"(?i)(application|apply|implement|use|utilize)",
            "citation_patterns": r"\[(.*?)\]|\((.*?)\)|(?:see|cf\.|compare|reference)",
            "technical_terms": r"[A-Z]{2,}|[a-z]+[-_][a-z]+",
            "question_patterns": r"\?|(?i)(question|problem|challenge|issue)"
        }
        
        # Quality assessment history
        self.assessment_history: List[QualityReport] = []
        self.quality_analytics: Optional[QualityAnalytics] = None
        
        self.logger.info("Content Quality Agent initialized with comprehensive validation rules")

    def assess_content_quality(self, file_path: str, content: str = None, 
                             content_type: str = None, week_number: int = None) -> QualityReport:
        """
        Perform comprehensive quality assessment of academic content
        
        Args:
            file_path: Path to the content file
            content: Content string (if not provided, will read from file)
            content_type: Type of content (transcript, lecture, notes, etc.)
            week_number: Week number if applicable
            
        Returns:
            Comprehensive quality report
        """
        start_time = datetime.now()
        
        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Extract metadata from content or filename
            if content_type is None:
                content_type = self._detect_content_type(file_path, content)
            
            if week_number is None:
                week_number = self._extract_week_number(file_path, content)
            
            # Generate content ID
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_id = f"{content_type}_{week_number or 'unknown'}_{content_hash[:8]}"
            
            # Perform detailed analysis
            self.logger.info(f"Assessing quality for {file_path} (type: {content_type}, week: {week_number})")
            
            # Structure analysis
            structure = self._analyze_content_structure(content)
            
            # Completeness check
            completeness = self._check_completeness(content, content_type, week_number)
            
            # Formatting validation
            formatting_issues = self._validate_formatting(content)
            
            # Consistency check
            consistency_score = self._check_consistency(content, content_type)
            
            # Academic quality assessment
            academic_quality = self._assess_academic_quality(content, content_type)
            
            # Quality evaluation using existing quality manager
            quality_evaluation = self.quality_manager.evaluate_content(content, content_type)
            
            # Calculate composite scores
            completeness_score = completeness.completeness_score
            formatting_score = self._calculate_formatting_score(formatting_issues)
            overall_quality = self._calculate_overall_quality(
                completeness_score, formatting_score, consistency_score, academic_quality
            )
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(
                completeness, formatting_issues, consistency_score, academic_quality
            )
            
            # Create quality report
            processing_time = (datetime.now() - start_time).total_seconds()
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else len(content.encode())
            
            report = QualityReport(
                content_id=content_id,
                file_path=file_path,
                content_type=content_type,
                week_number=week_number,
                overall_quality_score=overall_quality,
                completeness_score=completeness_score,
                formatting_score=formatting_score,
                consistency_score=consistency_score,
                academic_quality_score=academic_quality,
                content_structure=structure,
                completeness_check=completeness,
                formatting_issues=formatting_issues,
                quality_evaluation=quality_evaluation,
                improvement_suggestions=suggestions["general"],
                priority_fixes=suggestions["priority"],
                auto_fixes_available=suggestions["auto_fixable"],
                assessment_date=datetime.now(),
                processing_time=processing_time,
                file_size=file_size,
                content_hash=content_hash
            )
            
            # Store in history
            self.assessment_history.append(report)
            
            # Log metrics
            self.log_metrics({
                "operation": "quality_assessment",
                "quality_score": overall_quality,
                "completeness_score": completeness_score,
                "formatting_score": formatting_score,
                "consistency_score": consistency_score,
                "academic_quality_score": academic_quality,
                "processing_time": processing_time,
                "content_type": content_type,
                "success": True
            })
            
            self.logger.info(
                f"Quality assessment completed: {overall_quality:.2f} "
                f"(C:{completeness_score:.2f}, F:{formatting_score:.2f}, "
                f"Cs:{consistency_score:.2f}, A:{academic_quality:.2f})"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error assessing content quality for {file_path}: {str(e)}")
            self.handle_error(e, {"operation": "assess_content_quality", "file_path": file_path})
            raise

    def _detect_content_type(self, file_path: str, content: str) -> str:
        """Detect content type from file path and content"""
        file_path_lower = file_path.lower()
        content_lower = content.lower()
        
        # Check file path
        if "transcript" in file_path_lower:
            return "transcript"
        elif "lecture" in file_path_lower:
            return "lecture"
        elif "notes" in file_path_lower:
            return "notes"
        elif "chapter" in file_path_lower or "textbook" in file_path_lower:
            return "textbook"
        elif "comprehensive_study_notes" in file_path_lower:
            return "comprehensive_study_notes"
        
        # Check content patterns
        if "# chapter" in content_lower:
            return "textbook"
        elif "transcript" in content_lower[:500]:
            return "transcript"
        elif "lecture" in content_lower[:500]:
            return "lecture"
        elif "comprehensive study notes" in content_lower[:500]:
            return "comprehensive_study_notes"
        elif "notes" in content_lower[:500]:
            return "notes"
        
        return "unknown"

    def _extract_week_number(self, file_path: str, content: str) -> Optional[int]:
        """Extract week number from file path or content"""
        # Check filename first
        filename = os.path.basename(file_path)
        week_patterns = [
            r"week[-_]?(\d+)",
            r"w(\d+)",
            r"(\d+)[-_]?week",
            r"lecture[-_]?(\d+)",
            r"class[-_]?(\d+)"
        ]
        
        for pattern in week_patterns:
            match = re.search(pattern, filename.lower())
            if match:
                try:
                    week_num = int(match.group(1))
                    if 1 <= week_num <= 15:
                        return week_num
                except (ValueError, IndexError):
                    continue
        
        # Check content frontmatter
        if content.startswith("---"):
            frontmatter_match = re.search(r"week:\s*(\d+)", content[:500])
            if frontmatter_match:
                try:
                    return int(frontmatter_match.group(1))
                except ValueError:
                    pass
        
        return None

    def _analyze_content_structure(self, content: str) -> ContentStructure:
        """Analyze the structural elements of content"""
        lines = content.split('\n')
        
        # Headers
        headers = []
        header_levels = defaultdict(int)
        for line in lines:
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if level <= 6:  # Valid markdown header
                    headers.append(line.strip())
                    header_levels[level] += 1
        
        # Count structural elements
        paragraphs = len([line for line in lines if line.strip() and not line.startswith(('#', '-', '*', '>', '```', '|'))])
        lists = len(re.findall(r'^[-*+]\s', content, re.MULTILINE))
        lists += len(re.findall(r'^\d+\.\s', content, re.MULTILINE))
        
        images = len(re.findall(r'!\[.*?\]\(.*?\)', content))
        tables = len(re.findall(r'^\|.*\|', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```', content)) // 2
        mermaid_diagrams = len(re.findall(r'```mermaid', content))
        
        # Citations (basic patterns)
        citations = len(re.findall(self.academic_patterns["citation_patterns"], content))
        
        # Word and character counts
        word_count = len(content.split())
        character_count = len(content)
        
        return ContentStructure(
            headers=headers,
            header_levels=dict(header_levels),
            paragraphs=paragraphs,
            lists=lists,
            images=images,
            tables=tables,
            code_blocks=code_blocks,
            mermaid_diagrams=mermaid_diagrams,
            citations=citations,
            word_count=word_count,
            character_count=character_count
        )

    def _check_completeness(self, content: str, content_type: str, week_number: int = None) -> CompletenessCheck:
        """Check if content meets completeness requirements"""
        required_sections = self.quality_config["required_sections"].get(content_type, [])
        content_lower = content.lower()
        
        missing_sections = []
        incomplete_sections = []
        
        for section in required_sections:
            section_pattern = section.replace(" ", "[-_\\s]*").replace("-", "[-_\\s]*")
            if not re.search(section_pattern, content_lower):
                missing_sections.append(section)
            else:
                # Check if section has substantial content
                section_match = re.search(f"#{1,6}.*{section_pattern}.*", content_lower, re.MULTILINE)
                if section_match:
                    # Find content between this section and next section
                    start_pos = section_match.end()
                    next_section = re.search(r'^#{1,6}\s', content[start_pos:], re.MULTILINE)
                    section_content = content[start_pos:start_pos + next_section.start()] if next_section else content[start_pos:]
                    
                    # Check if section has enough content (arbitrary threshold)
                    if len(section_content.strip()) < 100:  # Less than 100 characters
                        incomplete_sections.append(section)
        
        # Calculate completeness score
        total_required = len(required_sections)
        if total_required == 0:
            completeness_score = 1.0
        else:
            missing_penalty = len(missing_sections) / total_required
            incomplete_penalty = (len(incomplete_sections) * 0.5) / total_required
            completeness_score = max(0.0, 1.0 - missing_penalty - incomplete_penalty)
        
        return CompletenessCheck(
            required_sections=required_sections,
            missing_sections=missing_sections,
            incomplete_sections=incomplete_sections,
            completeness_score=completeness_score,
            week_number=week_number,
            content_type=content_type
        )

    def _validate_formatting(self, content: str) -> List[FormattingIssue]:
        """Validate formatting consistency and identify issues"""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > self.quality_config["formatting_rules"]["max_line_length"]:
                issues.append(FormattingIssue(
                    issue_type="long_line",
                    severity="low",
                    description=f"Line exceeds maximum length ({len(line)} > {self.quality_config['formatting_rules']['max_line_length']})",
                    line_number=line_num,
                    suggestion="Consider breaking the line or using proper markdown formatting",
                    auto_fixable=False,
                    context=line[:50] + "..." if len(line) > 50 else line
                ))
            
            # Check header formatting
            if line.strip().startswith('#'):
                if not re.match(r'^#{1,6}\s+\S', line):
                    issues.append(FormattingIssue(
                        issue_type="header_formatting",
                        severity="medium",
                        description="Header should have space after # and non-empty title",
                        line_number=line_num,
                        suggestion="Use format: '# Title' with space after #",
                        auto_fixable=True,
                        context=line
                    ))
            
            # Check list formatting
            if re.match(r'^[-*+]\s*\S', line):
                if not re.match(r'^[-*+]\s+\S', line):
                    issues.append(FormattingIssue(
                        issue_type="list_formatting",
                        severity="low",
                        description="List item should have space after bullet",
                        line_number=line_num,
                        suggestion="Use format: '- item' with space after bullet",
                        auto_fixable=True,
                        context=line
                    ))
        
        # Check for images without alt text
        images_without_alt = re.findall(r'!\[\s*\]\([^)]+\)', content)
        if images_without_alt:
            issues.append(FormattingIssue(
                issue_type="missing_alt_text",
                severity="medium",
                description=f"Found {len(images_without_alt)} images without alt text",
                line_number=None,
                suggestion="Add descriptive alt text for all images",
                auto_fixable=False,
                context="Multiple images"
            ))
        
        # Check for inconsistent emphasis
        bold_patterns = [r'\*\*[^*]+\*\*', r'__[^_]+__']
        italic_patterns = [r'\*[^*]+\*', r'_[^_]+_']
        
        bold_counts = sum(len(re.findall(pattern, content)) for pattern in bold_patterns)
        italic_counts = sum(len(re.findall(pattern, content)) for pattern in italic_patterns)
        
        if bold_counts > 0 and italic_counts > 0:
            # Check for mixed emphasis styles
            asterisk_bold = len(re.findall(r'\*\*[^*]+\*\*', content))
            underscore_bold = len(re.findall(r'__[^_]+__', content))
            
            if asterisk_bold > 0 and underscore_bold > 0:
                issues.append(FormattingIssue(
                    issue_type="inconsistent_emphasis",
                    severity="low",
                    description="Mixed bold formatting styles (** and __)",
                    line_number=None,
                    suggestion="Use consistent formatting style throughout document",
                    auto_fixable=True,
                    context="Document-wide"
                ))
        
        return issues

    def _check_consistency(self, content: str, content_type: str) -> float:
        """Check formatting and style consistency"""
        consistency_score = 1.0
        penalties = []
        
        # Header consistency
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        if headers:
            # Check for consistent header hierarchy
            header_levels = [len(h[0]) for h in headers]
            if len(set(header_levels)) > 1:
                # Check if levels are sequential
                sorted_levels = sorted(set(header_levels))
                if sorted_levels != list(range(sorted_levels[0], sorted_levels[-1] + 1)):
                    penalties.append(0.1)  # Non-sequential header levels
        
        # List consistency
        unordered_bullets = re.findall(r'^([-*+])\s', content, re.MULTILINE)
        if unordered_bullets:
            unique_bullets = set(unordered_bullets)
            if len(unique_bullets) > 1:
                penalties.append(0.05)  # Mixed bullet styles
        
        # Link formatting consistency
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
        if links:
            # Check for consistent link text patterns
            link_texts = [link[0] for link in links]
            # Basic check for consistent capitalization
            if len(set(text.istitle() for text in link_texts)) > 1:
                penalties.append(0.03)  # Inconsistent link text capitalization
        
        # Calculate final consistency score
        total_penalty = sum(penalties)
        consistency_score = max(0.0, consistency_score - total_penalty)
        
        return consistency_score

    def _assess_academic_quality(self, content: str, content_type: str) -> float:
        """Assess academic quality of content"""
        quality_score = 0.0
        content_lower = content.lower()
        
        # Content depth indicators
        concept_matches = len(re.findall(self.academic_patterns["concept_introduction"], content_lower))
        example_matches = len(re.findall(self.academic_patterns["examples"], content_lower))
        analysis_matches = len(re.findall(self.academic_patterns["analysis"], content_lower))
        
        # Normalize by content length (per 1000 words)
        word_count = len(content.split())
        normalization_factor = max(1, word_count / 1000)
        
        # Concept introduction score (0-0.3)
        concept_score = min(0.3, (concept_matches / normalization_factor) * 0.1)
        quality_score += concept_score
        
        # Examples and illustrations score (0-0.2)
        example_score = min(0.2, (example_matches / normalization_factor) * 0.05)
        quality_score += example_score
        
        # Analysis depth score (0-0.3)
        analysis_score = min(0.3, (analysis_matches / normalization_factor) * 0.1)
        quality_score += analysis_score
        
        # Structure and organization score (0-0.2)
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        if headers:
            structure_score = min(0.2, len(headers) / 10 * 0.2)
            quality_score += structure_score
        
        # Ensure score is between 0 and 1
        return min(1.0, quality_score)

    def _calculate_formatting_score(self, formatting_issues: List[FormattingIssue]) -> float:
        """Calculate formatting score based on issues found"""
        if not formatting_issues:
            return 1.0
        
        # Weight penalties by severity
        severity_weights = {"low": 0.02, "medium": 0.05, "high": 0.1, "critical": 0.2}
        total_penalty = sum(severity_weights.get(issue.severity, 0.05) for issue in formatting_issues)
        
        # Cap the penalty to avoid negative scores
        total_penalty = min(total_penalty, 0.8)
        
        return max(0.2, 1.0 - total_penalty)

    def _calculate_overall_quality(self, completeness: float, formatting: float, 
                                 consistency: float, academic: float) -> float:
        """Calculate weighted overall quality score"""
        weights = self.quality_config["weights"]
        
        overall = (
            completeness * weights["completeness"] +
            formatting * weights["formatting"] +
            consistency * weights["consistency"] +
            academic * weights["academic_quality"]
        )
        
        return min(1.0, max(0.0, overall))

    def _generate_improvement_suggestions(self, completeness: CompletenessCheck, 
                                        formatting_issues: List[FormattingIssue],
                                        consistency_score: float, 
                                        academic_quality: float) -> Dict[str, List[str]]:
        """Generate actionable improvement suggestions"""
        suggestions = {"general": [], "priority": [], "auto_fixable": []}
        
        # Completeness suggestions
        if completeness.missing_sections:
            suggestions["priority"].append(
                f"Add missing required sections: {', '.join(completeness.missing_sections)}"
            )
        
        if completeness.incomplete_sections:
            suggestions["general"].append(
                f"Expand incomplete sections: {', '.join(completeness.incomplete_sections)}"
            )
        
        # Formatting suggestions
        critical_formatting = [issue for issue in formatting_issues if issue.severity == "critical"]
        auto_fixable_formatting = [issue for issue in formatting_issues if issue.auto_fixable]
        
        if critical_formatting:
            suggestions["priority"].extend([issue.suggestion for issue in critical_formatting])
        
        if auto_fixable_formatting:
            suggestions["auto_fixable"].extend([issue.suggestion for issue in auto_fixable_formatting])
        
        # Consistency suggestions
        if consistency_score < 0.7:
            suggestions["general"].append("Improve formatting consistency throughout document")
        
        # Academic quality suggestions
        if academic_quality < 0.6:
            suggestions["general"].extend([
                "Add more detailed concept explanations",
                "Include more examples and practical applications",
                "Provide deeper analysis of key topics"
            ])
        
        return suggestions

    def validate_batch_content(self, file_paths: List[str], 
                             output_dir: str = None) -> Dict[str, Any]:
        """
        Validate quality for multiple content files
        
        Args:
            file_paths: List of file paths to validate
            output_dir: Directory to save detailed reports
            
        Returns:
            Batch validation results with analytics
        """
        self.logger.info(f"Starting batch quality validation for {len(file_paths)} files")
        
        batch_results = {
            "total_files": len(file_paths),
            "processed_files": 0,
            "failed_files": 0,
            "reports": [],
            "summary": {},
            "analytics": None
        }
        
        # Process each file
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    report = self.assess_content_quality(file_path)
                    batch_results["reports"].append(report.to_dict())
                    batch_results["processed_files"] += 1
                else:
                    self.logger.warning(f"File not found: {file_path}")
                    batch_results["failed_files"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                batch_results["failed_files"] += 1
        
        # Generate analytics
        if self.assessment_history:
            batch_results["analytics"] = self.generate_quality_analytics()
        
        # Generate summary
        if batch_results["reports"]:
            batch_results["summary"] = self._generate_batch_summary(batch_results["reports"])
        
        # Save detailed report if output directory specified
        if output_dir:
            self._save_batch_report(batch_results, output_dir)
        
        self.logger.info(
            f"Batch validation completed: {batch_results['processed_files']} processed, "
            f"{batch_results['failed_files']} failed"
        )
        
        return batch_results

    def generate_quality_analytics(self) -> QualityAnalytics:
        """Generate comprehensive quality analytics from assessment history"""
        if not self.assessment_history:
            return QualityAnalytics(
                total_files_assessed=0,
                average_quality_score=0.0,
                quality_distribution={},
                common_issues=[],
                content_type_quality={},
                week_quality_trends={},
                improvement_opportunities=[],
                quality_trends=[],
                assessment_summary={}
            )
        
        # Basic statistics
        total_files = len(self.assessment_history)
        quality_scores = [report.overall_quality_score for report in self.assessment_history]
        average_quality = statistics.mean(quality_scores)
        
        # Quality distribution
        quality_ranges = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0, "very_poor": 0}
        thresholds = self.quality_config["thresholds"]
        
        for score in quality_scores:
            if score >= thresholds["excellent"]:
                quality_ranges["excellent"] += 1
            elif score >= thresholds["good"]:
                quality_ranges["good"] += 1
            elif score >= thresholds["acceptable"]:
                quality_ranges["acceptable"] += 1
            elif score >= thresholds["poor"]:
                quality_ranges["poor"] += 1
            else:
                quality_ranges["very_poor"] += 1
        
        # Content type quality
        content_type_scores = defaultdict(list)
        for report in self.assessment_history:
            content_type_scores[report.content_type].append(report.overall_quality_score)
        
        content_type_quality = {
            content_type: statistics.mean(scores)
            for content_type, scores in content_type_scores.items()
        }
        
        # Week quality trends
        week_scores = defaultdict(list)
        for report in self.assessment_history:
            if report.week_number:
                week_scores[report.week_number].append(report.overall_quality_score)
        
        week_quality_trends = {
            week: statistics.mean(scores)
            for week, scores in week_scores.items()
        }
        
        # Common issues analysis
        all_issues = []
        for report in self.assessment_history:
            all_issues.extend([issue.issue_type for issue in report.formatting_issues])
        
        issue_counts = Counter(all_issues)
        common_issues = [
            {"issue_type": issue, "count": count, "percentage": (count / total_files) * 100}
            for issue, count in issue_counts.most_common(10)
        ]
        
        # Improvement opportunities
        improvement_opportunities = []
        low_completeness = [r for r in self.assessment_history if r.completeness_score < 0.7]
        if low_completeness:
            improvement_opportunities.append(
                f"Completeness: {len(low_completeness)} files need better section coverage"
            )
        
        low_academic_quality = [r for r in self.assessment_history if r.academic_quality_score < 0.6]
        if low_academic_quality:
            improvement_opportunities.append(
                f"Academic Quality: {len(low_academic_quality)} files need deeper analysis"
            )
        
        # Quality trends (last 10 assessments)
        recent_reports = self.assessment_history[-10:] if len(self.assessment_history) >= 10 else self.assessment_history
        quality_trends = [
            {
                "assessment_date": report.assessment_date.isoformat(),
                "quality_score": report.overall_quality_score,
                "content_type": report.content_type,
                "week_number": report.week_number
            }
            for report in recent_reports
        ]
        
        # Assessment summary
        assessment_summary = {
            "average_quality_score": average_quality,
            "highest_quality_score": max(quality_scores),
            "lowest_quality_score": min(quality_scores),
            "quality_std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
            "files_above_threshold": sum(1 for score in quality_scores if score >= 0.7),
            "pass_rate": (sum(1 for score in quality_scores if score >= 0.7) / total_files) * 100
        }
        
        analytics = QualityAnalytics(
            total_files_assessed=total_files,
            average_quality_score=average_quality,
            quality_distribution=quality_ranges,
            common_issues=common_issues,
            content_type_quality=content_type_quality,
            week_quality_trends=week_quality_trends,
            improvement_opportunities=improvement_opportunities,
            quality_trends=quality_trends,
            assessment_summary=assessment_summary
        )
        
        self.quality_analytics = analytics
        return analytics

    def _generate_batch_summary(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""
        if not reports:
            return {}
        
        quality_scores = [report["overall_quality_score"] for report in reports]
        
        return {
            "total_reports": len(reports),
            "average_quality": statistics.mean(quality_scores),
            "median_quality": statistics.median(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
            "files_above_threshold": sum(1 for score in quality_scores if score >= 0.7),
            "pass_rate": (sum(1 for score in quality_scores if score >= 0.7) / len(quality_scores)) * 100,
            "content_types": list(set(report["content_type"] for report in reports)),
            "weeks_covered": sorted(list(set(report["week_number"] for report in reports if report["week_number"]))),
            "total_issues": sum(len(report["formatting_issues"]) for report in reports),
            "auto_fixable_issues": sum(
                len([issue for issue in report["formatting_issues"] if issue["auto_fixable"]])
                for report in reports
            )
        }

    def _save_batch_report(self, batch_results: Dict[str, Any], output_dir: str) -> None:
        """Save detailed batch report to output directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"quality_assessment_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        # Save analytics if available
        if batch_results.get("analytics"):
            analytics_file = os.path.join(output_dir, f"quality_analytics_{timestamp}.json")
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(batch_results["analytics"]), f, indent=2, default=str)
        
        self.logger.info(f"Batch report saved to {report_file}")

    def auto_fix_issues(self, file_path: str, backup: bool = True) -> Dict[str, Any]:
        """
        Automatically fix auto-fixable formatting issues
        
        Args:
            file_path: Path to the file to fix
            backup: Whether to create backup before fixing
            
        Returns:
            Results of auto-fix operation
        """
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Create backup if requested
            if backup:
                backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                self.logger.info(f"Backup created: {backup_path}")
            
            # Assess current quality
            initial_report = self.assess_content_quality(file_path, original_content)
            auto_fixable_issues = [issue for issue in initial_report.formatting_issues if issue.auto_fixable]
            
            if not auto_fixable_issues:
                return {
                    "success": True,
                    "fixes_applied": 0,
                    "issues_fixed": [],
                    "message": "No auto-fixable issues found"
                }
            
            # Apply fixes
            fixed_content = original_content
            issues_fixed = []
            
            # Fix header formatting
            header_issues = [issue for issue in auto_fixable_issues if issue.issue_type == "header_formatting"]
            if header_issues:
                fixed_content = re.sub(r'^(#{1,6})([^\s#])', r'\1 \2', fixed_content, flags=re.MULTILINE)
                issues_fixed.extend(["header_formatting"] * len(header_issues))
            
            # Fix list formatting
            list_issues = [issue for issue in auto_fixable_issues if issue.issue_type == "list_formatting"]
            if list_issues:
                fixed_content = re.sub(r'^([-*+])([^\s])', r'\1 \2', fixed_content, flags=re.MULTILINE)
                issues_fixed.extend(["list_formatting"] * len(list_issues))
            
            # Fix inconsistent emphasis
            emphasis_issues = [issue for issue in auto_fixable_issues if issue.issue_type == "inconsistent_emphasis"]
            if emphasis_issues:
                # Convert all bold to ** style
                fixed_content = re.sub(r'__([^_]+)__', r'**\1**', fixed_content)
                # Convert all italic to * style
                fixed_content = re.sub(r'\b_([^_]+)_\b', r'*\1*', fixed_content)
                issues_fixed.extend(["inconsistent_emphasis"] * len(emphasis_issues))
            
            # Write fixed content
            if fixed_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                # Verify improvements
                final_report = self.assess_content_quality(file_path, fixed_content)
                
                return {
                    "success": True,
                    "fixes_applied": len(issues_fixed),
                    "issues_fixed": list(set(issues_fixed)),
                    "quality_improvement": final_report.overall_quality_score - initial_report.overall_quality_score,
                    "backup_path": backup_path if backup else None,
                    "message": f"Successfully applied {len(issues_fixed)} auto-fixes"
                }
            else:
                return {
                    "success": True,
                    "fixes_applied": 0,
                    "issues_fixed": [],
                    "message": "No changes needed"
                }
                
        except Exception as e:
            self.logger.error(f"Error auto-fixing {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "fixes_applied": 0,
                "issues_fixed": []
            }

    def integrate_with_consolidation(self, consolidation_result: Any) -> Dict[str, Any]:
        """
        Integrate quality assessment with content consolidation results
        
        Args:
            consolidation_result: Result from ContentConsolidationAgent
            
        Returns:
            Integrated quality and consolidation analysis
        """
        try:
            if not hasattr(consolidation_result, 'processed_files'):
                raise ValueError("Invalid consolidation result format")
            
            # Assess quality for all consolidated files
            quality_results = []
            
            for file_mapping in consolidation_result.processed_files:
                try:
                    target_path = file_mapping.target_path
                    if os.path.exists(target_path):
                        quality_report = self.assess_content_quality(
                            target_path,
                            content_type=file_mapping.content_type,
                            week_number=file_mapping.week_number
                        )
                        quality_results.append(quality_report)
                except Exception as e:
                    self.logger.error(f"Error assessing quality for {target_path}: {str(e)}")
            
            # Generate integrated analysis
            integration_result = {
                "consolidation_success": consolidation_result.success,
                "total_consolidated_files": len(consolidation_result.processed_files),
                "quality_assessed_files": len(quality_results),
                "quality_reports": [report.to_dict() for report in quality_results],
                "integrated_metrics": self._calculate_integrated_metrics(
                    consolidation_result, quality_results
                ),
                "recommendations": self._generate_integrated_recommendations(
                    consolidation_result, quality_results
                ),
                "assessment_timestamp": datetime.now().isoformat()
            }
            
            return integration_result
            
        except Exception as e:
            self.logger.error(f"Error integrating with consolidation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "consolidation_success": False,
                "quality_assessed_files": 0
            }

    def _calculate_integrated_metrics(self, consolidation_result: Any, 
                                    quality_results: List[QualityReport]) -> Dict[str, Any]:
        """Calculate metrics that combine consolidation and quality data"""
        if not quality_results:
            return {}
        
        # Quality metrics
        quality_scores = [report.overall_quality_score for report in quality_results]
        completeness_scores = [report.completeness_score for report in quality_results]
        
        # Content type analysis
        content_type_quality = defaultdict(list)
        content_type_counts = defaultdict(int)
        
        for report in quality_results:
            content_type_quality[report.content_type].append(report.overall_quality_score)
            content_type_counts[report.content_type] += 1
        
        # Week coverage analysis
        week_coverage = {}
        for report in quality_results:
            if report.week_number:
                week_coverage[report.week_number] = report.overall_quality_score
        
        return {
            "overall_quality_score": statistics.mean(quality_scores),
            "overall_completeness_score": statistics.mean(completeness_scores),
            "quality_score_range": {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "std_dev": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
            },
            "content_type_distribution": dict(content_type_counts),
            "content_type_quality_scores": {
                content_type: statistics.mean(scores)
                for content_type, scores in content_type_quality.items()
            },
            "week_coverage": week_coverage,
            "weeks_covered": len(week_coverage),
            "consolidation_efficiency": len(quality_results) / len(consolidation_result.processed_files),
            "files_above_quality_threshold": sum(1 for score in quality_scores if score >= 0.7),
            "quality_pass_rate": (sum(1 for score in quality_scores if score >= 0.7) / len(quality_scores)) * 100
        }

    def _generate_integrated_recommendations(self, consolidation_result: Any, 
                                           quality_results: List[QualityReport]) -> List[str]:
        """Generate recommendations based on both consolidation and quality results"""
        recommendations = []
        
        # Consolidation-based recommendations
        if consolidation_result.errors:
            recommendations.append(
                f"Address {len(consolidation_result.errors)} consolidation errors to improve content integration"
            )
        
        if consolidation_result.skipped_files:
            recommendations.append(
                f"Review {len(consolidation_result.skipped_files)} skipped files for potential content gaps"
            )
        
        # Quality-based recommendations
        if quality_results:
            low_quality_files = [r for r in quality_results if r.overall_quality_score < 0.6]
            if low_quality_files:
                recommendations.append(
                    f"Improve quality for {len(low_quality_files)} files with scores below 0.6"
                )
            
            # Content type specific recommendations
            content_type_quality = defaultdict(list)
            for report in quality_results:
                content_type_quality[report.content_type].append(report.overall_quality_score)
            
            for content_type, scores in content_type_quality.items():
                avg_score = statistics.mean(scores)
                if avg_score < 0.7:
                    recommendations.append(
                        f"Focus on improving {content_type} content quality (current average: {avg_score:.2f})"
                    )
            
            # Week coverage recommendations
            week_numbers = [r.week_number for r in quality_results if r.week_number]
            if week_numbers:
                missing_weeks = set(range(1, 14)) - set(week_numbers)
                if missing_weeks:
                    recommendations.append(
                        f"Consider adding content for missing weeks: {sorted(missing_weeks)}"
                    )
        
        return recommendations

    def check_quality(self, content: Dict[str, Any]) -> float:
        """Implementation of base class abstract method"""
        if "file_path" in content:
            report = self.assess_content_quality(content["file_path"])
            return report.overall_quality_score
        elif "content_text" in content:
            # Create temporary assessment for text content
            structure = self._analyze_content_structure(content["content_text"])
            completeness = self._check_completeness(
                content["content_text"], 
                content.get("content_type", "unknown")
            )
            formatting_issues = self._validate_formatting(content["content_text"])
            consistency_score = self._check_consistency(
                content["content_text"], 
                content.get("content_type", "unknown")
            )
            academic_quality = self._assess_academic_quality(
                content["content_text"], 
                content.get("content_type", "unknown")
            )
            
            return self._calculate_overall_quality(
                completeness.completeness_score,
                self._calculate_formatting_score(formatting_issues),
                consistency_score,
                academic_quality
            )
        
        return 0.0

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for quality assessment"""
        if isinstance(input_data, str):
            # File path input
            return os.path.exists(input_data)
        elif isinstance(input_data, dict):
            # Content dictionary input
            required_fields = ["content_text"] if "content_text" in input_data else ["file_path"]
            return all(field in input_data for field in required_fields)
        elif isinstance(input_data, list):
            # Batch processing input
            return all(self.validate_input(item) for item in input_data)
        
        return False

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data from quality assessment"""
        if isinstance(output_data, QualityReport):
            return (
                hasattr(output_data, 'overall_quality_score') and
                0.0 <= output_data.overall_quality_score <= 1.0 and
                hasattr(output_data, 'content_structure') and
                hasattr(output_data, 'formatting_issues')
            )
        elif isinstance(output_data, dict):
            # Batch results validation
            required_fields = ["total_files", "processed_files", "reports"]
            return all(field in output_data for field in required_fields)
        
        return False


def main():
    """Main entry point for the content quality agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Content Quality Assurance Agent")
    parser.add_argument("--file", help="Single file to assess")
    parser.add_argument("--batch", nargs='+', help="Batch of files to assess")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--auto-fix", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Create quality agent
    agent = ContentQualityAgent()
    
    if args.file:
        # Single file assessment
        report = agent.assess_content_quality(args.file)
        print(f"\nQuality Assessment for {args.file}:")
        print(f"Overall Quality Score: {report.overall_quality_score:.2f}")
        print(f"Completeness Score: {report.completeness_score:.2f}")
        print(f"Formatting Score: {report.formatting_score:.2f}")
        print(f"Consistency Score: {report.consistency_score:.2f}")
        print(f"Academic Quality Score: {report.academic_quality_score:.2f}")
        
        if report.improvement_suggestions:
            print("\nImprovement Suggestions:")
            for suggestion in report.improvement_suggestions:
                print(f"  - {suggestion}")
        
        if args.auto_fix:
            fix_results = agent.auto_fix_issues(args.file)
            print(f"\nAuto-fix Results: {fix_results['fixes_applied']} fixes applied")
    
    elif args.batch:
        # Batch assessment
        results = agent.validate_batch_content(args.batch, args.output_dir)
        print(f"\nBatch Assessment Results:")
        print(f"Total Files: {results['total_files']}")
        print(f"Processed: {results['processed_files']}")
        print(f"Failed: {results['failed_files']}")
        
        if results.get('summary'):
            summary = results['summary']
            print(f"Average Quality: {summary['average_quality']:.2f}")
            print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    
    else:
        print("Please specify either --file or --batch option")


if __name__ == "__main__":
    main()