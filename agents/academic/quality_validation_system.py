#!/usr/bin/env python3
"""
Quality Validation System for Academic Agent PDF-to-Markdown Conversion

This module implements a comprehensive validation framework to ensure 90% accuracy
in PDF-to-markdown conversion through:
- Accuracy measurement and validation
- Automated quality testing with benchmarks
- Regression testing for PDF processing
- Quality metrics collection and reporting
- Continuous validation pipeline
- Quality improvement feedback loops

Task 17 Implementation - Quality Validation System
"""

import os
import json
import time
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import difflib
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from ...src.agents.base_agent import BaseAgent
from .content_quality_agent import ContentQualityAgent, QualityReport
from ...src.agents.quality_manager import QualityManager, QualityMetrics
from ...src.agents.communication_manager import CommunicationManager


@dataclass
class ValidationBenchmark:
    """Represents a validation benchmark for accuracy measurement"""
    benchmark_id: str
    name: str
    description: str
    source_pdf_path: str
    expected_markdown_path: str
    expected_metadata_path: Optional[str] = None
    accuracy_threshold: float = 0.9
    content_type: str = "academic"
    difficulty_level: str = "medium"  # easy, medium, hard, expert
    tags: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    validation_count: int = 0
    average_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "description": self.description,
            "source_pdf_path": self.source_pdf_path,
            "expected_markdown_path": self.expected_markdown_path,
            "expected_metadata_path": self.expected_metadata_path,
            "accuracy_threshold": self.accuracy_threshold,
            "content_type": self.content_type,
            "difficulty_level": self.difficulty_level,
            "tags": self.tags,
            "created_date": self.created_date.isoformat(),
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "validation_count": self.validation_count,
            "average_accuracy": self.average_accuracy
        }


@dataclass
class ValidationResult:
    """Represents the result of a validation test"""
    benchmark_id: str
    test_id: str
    timestamp: datetime
    accuracy_score: float
    passed: bool
    processing_time: float
    
    # Detailed accuracy breakdown
    content_accuracy: float
    structure_accuracy: float
    formatting_accuracy: float
    metadata_accuracy: float
    
    # Comparison details
    content_diff: str
    structure_diff: str
    formatting_issues: List[str]
    metadata_diff: str
    
    # Error information
    errors: List[str]
    warnings: List[str]
    
    # Processing metadata
    pdf_processor_version: str
    processing_config: Dict[str, Any]
    system_info: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "benchmark_id": self.benchmark_id,
            "test_id": self.test_id,
            "timestamp": self.timestamp.isoformat(),
            "accuracy_score": self.accuracy_score,
            "passed": self.passed,
            "processing_time": self.processing_time,
            "content_accuracy": self.content_accuracy,
            "structure_accuracy": self.structure_accuracy,
            "formatting_accuracy": self.formatting_accuracy,
            "metadata_accuracy": self.metadata_accuracy,
            "content_diff": self.content_diff,
            "structure_diff": self.structure_diff,
            "formatting_issues": self.formatting_issues,
            "metadata_diff": self.metadata_diff,
            "errors": self.errors,
            "warnings": self.warnings,
            "pdf_processor_version": self.pdf_processor_version,
            "processing_config": self.processing_config,
            "system_info": self.system_info
        }


@dataclass
class QualityTrend:
    """Represents quality trends over time"""
    period: str  # daily, weekly, monthly
    start_date: datetime
    end_date: datetime
    average_accuracy: float
    test_count: int
    pass_rate: float
    improvement_rate: float
    regression_count: int
    benchmark_performance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "period": self.period,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "average_accuracy": self.average_accuracy,
            "test_count": self.test_count,
            "pass_rate": self.pass_rate,
            "improvement_rate": self.improvement_rate,
            "regression_count": self.regression_count,
            "benchmark_performance": self.benchmark_performance
        }


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    report_id: str
    generation_date: datetime
    report_period: Tuple[datetime, datetime]
    
    # Overall statistics
    total_tests: int
    passed_tests: int
    failed_tests: int
    overall_accuracy: float
    pass_rate: float
    
    # Benchmark performance
    benchmark_results: Dict[str, Dict[str, Any]]
    performance_by_difficulty: Dict[str, Dict[str, Any]]
    performance_by_content_type: Dict[str, Dict[str, Any]]
    
    # Trend analysis
    accuracy_trends: List[QualityTrend]
    regression_analysis: Dict[str, Any]
    improvement_opportunities: List[str]
    
    # Quality metrics integration
    quality_scores: Dict[str, float]
    quality_distribution: Dict[str, int]
    
    # Recommendations
    recommendations: List[str]
    action_items: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "report_id": self.report_id,
            "generation_date": self.generation_date.isoformat(),
            "report_period": [self.report_period[0].isoformat(), self.report_period[1].isoformat()],
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "overall_accuracy": self.overall_accuracy,
            "pass_rate": self.pass_rate,
            "benchmark_results": self.benchmark_results,
            "performance_by_difficulty": self.performance_by_difficulty,
            "performance_by_content_type": self.performance_by_content_type,
            "accuracy_trends": [trend.to_dict() for trend in self.accuracy_trends],
            "regression_analysis": self.regression_analysis,
            "improvement_opportunities": self.improvement_opportunities,
            "quality_scores": self.quality_scores,
            "quality_distribution": self.quality_distribution,
            "recommendations": self.recommendations,
            "action_items": self.action_items
        }


class QualityValidationSystem(BaseAgent):
    """
    Comprehensive Quality Validation System for PDF-to-Markdown Conversion
    
    Implements a robust framework for ensuring 90% accuracy in content conversion
    through automated testing, continuous monitoring, and regression detection.
    """

    def __init__(self, validation_config: Optional[Dict[str, Any]] = None):
        super().__init__("quality_validation_system")
        
        # Initialize configuration
        self.config = self._load_validation_config(validation_config)
        
        # Initialize components
        self.quality_agent = ContentQualityAgent()
        self.quality_manager = QualityManager(quality_threshold=0.9)
        self.communication_manager = CommunicationManager()
        
        # Validation data
        self.benchmarks: Dict[str, ValidationBenchmark] = {}
        self.validation_results: List[ValidationResult] = []
        self.validation_history: Dict[str, List[ValidationResult]] = defaultdict(list)
        
        # Performance tracking
        self.accuracy_threshold = self.config.get("accuracy_threshold", 0.9)
        self.regression_threshold = self.config.get("regression_threshold", 0.05)
        self.max_concurrent_tests = self.config.get("max_concurrent_tests", 4)
        
        # Paths and directories
        self.validation_dir = Path(self.config.get("validation_dir", "validation"))
        self.benchmarks_dir = self.validation_dir / "benchmarks"
        self.results_dir = self.validation_dir / "results"
        self.reports_dir = self.validation_dir / "reports"
        
        # Create directories
        for dir_path in [self.validation_dir, self.benchmarks_dir, self.results_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing benchmarks and results
        self._load_existing_data()
        
        self.logger.info(f"Quality Validation System initialized with {len(self.benchmarks)} benchmarks")

    def _load_validation_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            "accuracy_threshold": 0.9,
            "regression_threshold": 0.05,
            "max_concurrent_tests": 4,
            "validation_dir": "validation",
            "benchmark_timeout": 300,  # 5 minutes per benchmark
            "content_similarity_weight": 0.4,
            "structure_similarity_weight": 0.3,
            "formatting_similarity_weight": 0.2,
            "metadata_similarity_weight": 0.1,
            "enable_continuous_validation": True,
            "validation_schedule": "daily",
            "regression_detection_window": 7,  # days
            "quality_improvement_threshold": 0.02
        }
        
        if config:
            default_config.update(config)
        
        return default_config

    def _load_existing_data(self) -> None:
        """Load existing benchmarks and validation results"""
        # Load benchmarks
        benchmarks_file = self.benchmarks_dir / "benchmarks.json"
        if benchmarks_file.exists():
            try:
                with open(benchmarks_file, 'r', encoding='utf-8') as f:
                    benchmarks_data = json.load(f)
                
                for benchmark_data in benchmarks_data.get("benchmarks", []):
                    benchmark = ValidationBenchmark(**benchmark_data)
                    benchmark.created_date = datetime.fromisoformat(benchmark_data["created_date"])
                    if benchmark_data.get("last_validated"):
                        benchmark.last_validated = datetime.fromisoformat(benchmark_data["last_validated"])
                    self.benchmarks[benchmark.benchmark_id] = benchmark
                
                self.logger.info(f"Loaded {len(self.benchmarks)} existing benchmarks")
            except Exception as e:
                self.logger.error(f"Error loading benchmarks: {str(e)}")
        
        # Load recent validation results
        results_pattern = self.results_dir / "validation_results_*.json"
        for results_file in self.results_dir.glob("validation_results_*.json"):
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results_data = json.load(f)
                
                for result_data in results_data.get("results", []):
                    result = ValidationResult(
                        benchmark_id=result_data["benchmark_id"],
                        test_id=result_data["test_id"],
                        timestamp=datetime.fromisoformat(result_data["timestamp"]),
                        accuracy_score=result_data["accuracy_score"],
                        passed=result_data["passed"],
                        processing_time=result_data["processing_time"],
                        content_accuracy=result_data["content_accuracy"],
                        structure_accuracy=result_data["structure_accuracy"],
                        formatting_accuracy=result_data["formatting_accuracy"],
                        metadata_accuracy=result_data["metadata_accuracy"],
                        content_diff=result_data["content_diff"],
                        structure_diff=result_data["structure_diff"],
                        formatting_issues=result_data["formatting_issues"],
                        metadata_diff=result_data["metadata_diff"],
                        errors=result_data["errors"],
                        warnings=result_data["warnings"],
                        pdf_processor_version=result_data["pdf_processor_version"],
                        processing_config=result_data["processing_config"],
                        system_info=result_data["system_info"]
                    )
                    
                    self.validation_results.append(result)
                    self.validation_history[result.benchmark_id].append(result)
                
                self.logger.info(f"Loaded {len(results_data.get('results', []))} results from {results_file}")
            except Exception as e:
                self.logger.error(f"Error loading results from {results_file}: {str(e)}")

    def create_benchmark(self, name: str, description: str, source_pdf_path: str,
                        expected_markdown_path: str, expected_metadata_path: Optional[str] = None,
                        accuracy_threshold: float = 0.9, content_type: str = "academic",
                        difficulty_level: str = "medium", tags: List[str] = None) -> ValidationBenchmark:
        """
        Create a new validation benchmark
        
        Args:
            name: Descriptive name for the benchmark
            description: Detailed description of what this benchmark tests
            source_pdf_path: Path to the source PDF file
            expected_markdown_path: Path to the expected markdown output
            expected_metadata_path: Optional path to expected metadata
            accuracy_threshold: Minimum accuracy required to pass (default: 0.9)
            content_type: Type of content being tested
            difficulty_level: Difficulty level of the benchmark
            tags: Optional tags for categorization
            
        Returns:
            Created ValidationBenchmark
        """
        try:
            # Validate input paths
            if not os.path.exists(source_pdf_path):
                raise ValueError(f"Source PDF file not found: {source_pdf_path}")
            
            if not os.path.exists(expected_markdown_path):
                raise ValueError(f"Expected markdown file not found: {expected_markdown_path}")
            
            if expected_metadata_path and not os.path.exists(expected_metadata_path):
                raise ValueError(f"Expected metadata file not found: {expected_metadata_path}")
            
            # Generate benchmark ID
            content_hash = hashlib.md5(f"{name}_{source_pdf_path}".encode()).hexdigest()[:8]
            benchmark_id = f"benchmark_{content_hash}_{int(time.time())}"
            
            # Create benchmark
            benchmark = ValidationBenchmark(
                benchmark_id=benchmark_id,
                name=name,
                description=description,
                source_pdf_path=source_pdf_path,
                expected_markdown_path=expected_markdown_path,
                expected_metadata_path=expected_metadata_path,
                accuracy_threshold=accuracy_threshold,
                content_type=content_type,
                difficulty_level=difficulty_level,
                tags=tags or []
            )
            
            # Store benchmark
            self.benchmarks[benchmark_id] = benchmark
            self._save_benchmarks()
            
            self.logger.info(f"Created new benchmark: {name} ({benchmark_id})")
            
            # Log metrics
            self.log_metrics({
                "operation": "create_benchmark",
                "benchmark_id": benchmark_id,
                "content_type": content_type,
                "difficulty_level": difficulty_level,
                "accuracy_threshold": accuracy_threshold,
                "success": True
            })
            
            return benchmark
            
        except Exception as e:
            self.logger.error(f"Error creating benchmark: {str(e)}")
            self.handle_error(e, {"operation": "create_benchmark", "name": name})
            raise

    def validate_benchmark(self, benchmark_id: str, pdf_processor_func: callable,
                         processor_config: Dict[str, Any] = None) -> ValidationResult:
        """
        Run validation test against a specific benchmark
        
        Args:
            benchmark_id: ID of the benchmark to test
            pdf_processor_func: Function to process PDF (should return markdown content and metadata)
            processor_config: Configuration for the PDF processor
            
        Returns:
            ValidationResult with detailed accuracy analysis
        """
        try:
            if benchmark_id not in self.benchmarks:
                raise ValueError(f"Benchmark not found: {benchmark_id}")
            
            benchmark = self.benchmarks[benchmark_id]
            test_id = f"test_{benchmark_id}_{int(time.time())}"
            start_time = time.time()
            
            self.logger.info(f"Running validation test {test_id} for benchmark {benchmark.name}")
            
            # Process the PDF
            try:
                processing_start = time.time()
                result = pdf_processor_func(benchmark.source_pdf_path, processor_config or {})
                processing_time = time.time() - processing_start
                
                if not isinstance(result, dict) or 'markdown' not in result:
                    raise ValueError("PDF processor must return dict with 'markdown' key")
                
                actual_markdown = result['markdown']
                actual_metadata = result.get('metadata', {})
                
            except Exception as e:
                # Create failed validation result
                return self._create_failed_validation_result(
                    benchmark_id, test_id, start_time, str(e), processor_config or {}
                )
            
            # Load expected outputs
            with open(benchmark.expected_markdown_path, 'r', encoding='utf-8') as f:
                expected_markdown = f.read()
            
            expected_metadata = {}
            if benchmark.expected_metadata_path:
                with open(benchmark.expected_metadata_path, 'r', encoding='utf-8') as f:
                    expected_metadata = json.load(f)
            
            # Calculate accuracy scores
            accuracy_scores = self._calculate_accuracy_scores(
                expected_markdown, actual_markdown,
                expected_metadata, actual_metadata
            )
            
            # Generate detailed comparisons
            content_diff = self._generate_content_diff(expected_markdown, actual_markdown)
            structure_diff = self._generate_structure_diff(expected_markdown, actual_markdown)
            formatting_issues = self._identify_formatting_issues(expected_markdown, actual_markdown)
            metadata_diff = self._generate_metadata_diff(expected_metadata, actual_metadata)
            
            # Calculate overall accuracy
            weights = self.config
            overall_accuracy = (
                accuracy_scores['content'] * weights.get('content_similarity_weight', 0.4) +
                accuracy_scores['structure'] * weights.get('structure_similarity_weight', 0.3) +
                accuracy_scores['formatting'] * weights.get('formatting_similarity_weight', 0.2) +
                accuracy_scores['metadata'] * weights.get('metadata_similarity_weight', 0.1)
            )
            
            # Determine if test passed
            passed = overall_accuracy >= benchmark.accuracy_threshold
            
            # Collect system information
            system_info = self._collect_system_info()
            
            # Create validation result
            validation_result = ValidationResult(
                benchmark_id=benchmark_id,
                test_id=test_id,
                timestamp=datetime.now(),
                accuracy_score=overall_accuracy,
                passed=passed,
                processing_time=processing_time,
                content_accuracy=accuracy_scores['content'],
                structure_accuracy=accuracy_scores['structure'],
                formatting_accuracy=accuracy_scores['formatting'],
                metadata_accuracy=accuracy_scores['metadata'],
                content_diff=content_diff,
                structure_diff=structure_diff,
                formatting_issues=formatting_issues,
                metadata_diff=metadata_diff,
                errors=[],
                warnings=[],
                pdf_processor_version=system_info.get('processor_version', 'unknown'),
                processing_config=processor_config or {},
                system_info=system_info
            )
            
            # Update benchmark statistics
            self._update_benchmark_stats(benchmark_id, validation_result)
            
            # Store validation result
            self.validation_results.append(validation_result)
            self.validation_history[benchmark_id].append(validation_result)
            self._save_validation_results()
            
            # Log metrics
            self.log_metrics({
                "operation": "validate_benchmark",
                "benchmark_id": benchmark_id,
                "test_id": test_id,
                "accuracy_score": overall_accuracy,
                "passed": passed,
                "processing_time": processing_time,
                "content_accuracy": accuracy_scores['content'],
                "structure_accuracy": accuracy_scores['structure'],
                "formatting_accuracy": accuracy_scores['formatting'],
                "metadata_accuracy": accuracy_scores['metadata'],
                "success": True
            })
            
            self.logger.info(
                f"Validation test completed: {test_id} - "
                f"Accuracy: {overall_accuracy:.3f}, Passed: {passed}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in validation test: {str(e)}")
            self.handle_error(e, {
                "operation": "validate_benchmark",
                "benchmark_id": benchmark_id
            })
            raise

    def run_full_validation_suite(self, pdf_processor_func: callable,
                                processor_config: Dict[str, Any] = None,
                                benchmark_filter: Dict[str, Any] = None) -> List[ValidationResult]:
        """
        Run validation tests against all benchmarks or filtered subset
        
        Args:
            pdf_processor_func: Function to process PDFs
            processor_config: Configuration for the PDF processor
            benchmark_filter: Optional filter criteria (content_type, difficulty_level, tags)
            
        Returns:
            List of ValidationResults for all tested benchmarks
        """
        try:
            # Filter benchmarks
            benchmarks_to_test = self._filter_benchmarks(benchmark_filter or {})
            
            if not benchmarks_to_test:
                self.logger.warning("No benchmarks match the filter criteria")
                return []
            
            self.logger.info(f"Running full validation suite on {len(benchmarks_to_test)} benchmarks")
            
            results = []
            
            # Run tests concurrently
            with ThreadPoolExecutor(max_workers=self.max_concurrent_tests) as executor:
                future_to_benchmark = {
                    executor.submit(
                        self.validate_benchmark,
                        benchmark_id,
                        pdf_processor_func,
                        processor_config
                    ): benchmark_id
                    for benchmark_id in benchmarks_to_test
                }
                
                for future in as_completed(future_to_benchmark):
                    benchmark_id = future_to_benchmark[future]
                    try:
                        result = future.result(timeout=self.config.get('benchmark_timeout', 300))
                        results.append(result)
                        
                        # Check for regression
                        if self._detect_regression(benchmark_id, result):
                            self.logger.warning(f"Regression detected in benchmark {benchmark_id}")
                        
                    except Exception as e:
                        self.logger.error(f"Error in benchmark {benchmark_id}: {str(e)}")
                        # Create failed result
                        failed_result = self._create_failed_validation_result(
                            benchmark_id,
                            f"failed_{benchmark_id}_{int(time.time())}",
                            time.time(),
                            str(e),
                            processor_config or {}
                        )
                        results.append(failed_result)
            
            # Generate summary statistics
            self._log_validation_suite_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in full validation suite: {str(e)}")
            self.handle_error(e, {"operation": "run_full_validation_suite"})
            raise

    def detect_quality_regressions(self, window_days: int = 7) -> Dict[str, Any]:
        """
        Detect quality regressions over a specified time window
        
        Args:
            window_days: Number of days to look back for regression analysis
            
        Returns:
            Dictionary with regression analysis results
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=window_days)
            
            # Get recent results
            recent_results = [
                result for result in self.validation_results
                if result.timestamp >= cutoff_date
            ]
            
            if not recent_results:
                return {"regressions": [], "message": "No recent validation results for analysis"}
            
            # Group results by benchmark
            results_by_benchmark = defaultdict(list)
            for result in recent_results:
                results_by_benchmark[result.benchmark_id].append(result)
            
            regressions = []
            
            for benchmark_id, benchmark_results in results_by_benchmark.items():
                if len(benchmark_results) < 2:
                    continue
                
                # Sort by timestamp
                benchmark_results.sort(key=lambda x: x.timestamp)
                
                # Calculate trend
                scores = [result.accuracy_score for result in benchmark_results]
                
                # Check for significant drop
                recent_avg = statistics.mean(scores[-3:])  # Last 3 results
                historical_avg = statistics.mean(scores[:-3]) if len(scores) > 3 else recent_avg
                
                if historical_avg - recent_avg > self.regression_threshold:
                    regression_info = {
                        "benchmark_id": benchmark_id,
                        "benchmark_name": self.benchmarks[benchmark_id].name if benchmark_id in self.benchmarks else "Unknown",
                        "historical_accuracy": historical_avg,
                        "recent_accuracy": recent_avg,
                        "regression_magnitude": historical_avg - recent_avg,
                        "failing_tests": len([r for r in benchmark_results[-3:] if not r.passed]),
                        "last_failure": max([r.timestamp for r in benchmark_results[-3:] if not r.passed], default=None)
                    }
                    regressions.append(regression_info)
            
            # Overall system regression analysis
            all_scores = [result.accuracy_score for result in recent_results]
            system_trend = self._calculate_trend(all_scores)
            
            regression_analysis = {
                "analysis_period": {
                    "start_date": cutoff_date.isoformat(),
                    "end_date": datetime.now().isoformat(),
                    "window_days": window_days
                },
                "regressions": regressions,
                "regression_count": len(regressions),
                "system_trend": system_trend,
                "overall_accuracy": statistics.mean(all_scores) if all_scores else 0.0,
                "total_tests": len(recent_results),
                "failed_tests": len([r for r in recent_results if not r.passed]),
                "recommendations": self._generate_regression_recommendations(regressions, system_trend)
            }
            
            # Log metrics
            self.log_metrics({
                "operation": "detect_regressions",
                "window_days": window_days,
                "regression_count": len(regressions),
                "system_trend": system_trend,
                "overall_accuracy": regression_analysis["overall_accuracy"],
                "success": True
            })
            
            return regression_analysis
            
        except Exception as e:
            self.logger.error(f"Error detecting regressions: {str(e)}")
            self.handle_error(e, {"operation": "detect_quality_regressions"})
            raise

    def generate_validation_report(self, period_days: int = 30) -> ValidationReport:
        """
        Generate comprehensive validation report
        
        Args:
            period_days: Number of days to include in the report
            
        Returns:
            ValidationReport with detailed analysis
        """
        try:
            report_id = f"validation_report_{int(time.time())}"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            
            # Filter results for the period
            period_results = [
                result for result in self.validation_results
                if start_date <= result.timestamp <= end_date
            ]
            
            if not period_results:
                self.logger.warning(f"No validation results found for period {start_date} to {end_date}")
                return self._create_empty_report(report_id, start_date, end_date)
            
            # Calculate overall statistics
            total_tests = len(period_results)
            passed_tests = len([r for r in period_results if r.passed])
            failed_tests = total_tests - passed_tests
            overall_accuracy = statistics.mean([r.accuracy_score for r in period_results])
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Analyze performance by benchmark
            benchmark_results = self._analyze_benchmark_performance(period_results)
            
            # Analyze performance by difficulty and content type
            performance_by_difficulty = self._analyze_performance_by_difficulty(period_results)
            performance_by_content_type = self._analyze_performance_by_content_type(period_results)
            
            # Generate accuracy trends
            accuracy_trends = self._calculate_accuracy_trends(period_results, start_date, end_date)
            
            # Perform regression analysis
            regression_analysis = self.detect_quality_regressions(period_days)
            
            # Identify improvement opportunities
            improvement_opportunities = self._identify_improvement_opportunities(period_results)
            
            # Integrate with quality system
            quality_scores = self._calculate_integrated_quality_scores(period_results)
            quality_distribution = self._calculate_quality_distribution(period_results)
            
            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                period_results, regression_analysis, improvement_opportunities
            )
            
            # Generate action items
            action_items = self._generate_action_items(
                period_results, regression_analysis, recommendations
            )
            
            # Create validation report
            validation_report = ValidationReport(
                report_id=report_id,
                generation_date=datetime.now(),
                report_period=(start_date, end_date),
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                overall_accuracy=overall_accuracy,
                pass_rate=pass_rate,
                benchmark_results=benchmark_results,
                performance_by_difficulty=performance_by_difficulty,
                performance_by_content_type=performance_by_content_type,
                accuracy_trends=accuracy_trends,
                regression_analysis=regression_analysis,
                improvement_opportunities=improvement_opportunities,
                quality_scores=quality_scores,
                quality_distribution=quality_distribution,
                recommendations=recommendations,
                action_items=action_items
            )
            
            # Save report
            self._save_validation_report(validation_report)
            
            # Log metrics
            self.log_metrics({
                "operation": "generate_validation_report",
                "report_id": report_id,
                "period_days": period_days,
                "total_tests": total_tests,
                "overall_accuracy": overall_accuracy,
                "pass_rate": pass_rate,
                "success": True
            })
            
            self.logger.info(
                f"Generated validation report {report_id}: "
                f"{total_tests} tests, {overall_accuracy:.3f} accuracy, {pass_rate:.1f}% pass rate"
            )
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error generating validation report: {str(e)}")
            self.handle_error(e, {"operation": "generate_validation_report"})
            raise

    def setup_continuous_validation(self, pdf_processor_func: callable,
                                  processor_config: Dict[str, Any] = None,
                                  schedule: str = "daily") -> None:
        """
        Set up continuous validation pipeline
        
        Args:
            pdf_processor_func: Function to process PDFs
            processor_config: Configuration for the PDF processor
            schedule: Validation schedule (daily, weekly, hourly)
        """
        # This would typically integrate with a scheduler like celery or APScheduler
        # For now, we provide the framework and logging
        
        self.logger.info(f"Setting up continuous validation with {schedule} schedule")
        
        # Store configuration for continuous validation
        continuous_config = {
            "enabled": True,
            "schedule": schedule,
            "processor_config": processor_config or {},
            "last_run": None,
            "next_run": None
        }
        
        # Save continuous validation configuration
        config_file = self.validation_dir / "continuous_validation_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(continuous_config, f, indent=2, default=str)
        
        self.logger.info("Continuous validation configuration saved")
        
        # Log metrics
        self.log_metrics({
            "operation": "setup_continuous_validation",
            "schedule": schedule,
            "enabled": True,
            "success": True
        })

    # Helper methods for accuracy calculation and analysis

    def _calculate_accuracy_scores(self, expected_markdown: str, actual_markdown: str,
                                 expected_metadata: Dict[str, Any], actual_metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculate detailed accuracy scores for different aspects"""
        scores = {}
        
        # Content accuracy (text similarity)
        scores['content'] = self._calculate_content_similarity(expected_markdown, actual_markdown)
        
        # Structure accuracy (header hierarchy, sections)
        scores['structure'] = self._calculate_structure_similarity(expected_markdown, actual_markdown)
        
        # Formatting accuracy (markdown formatting)
        scores['formatting'] = self._calculate_formatting_similarity(expected_markdown, actual_markdown)
        
        # Metadata accuracy
        scores['metadata'] = self._calculate_metadata_similarity(expected_metadata, actual_metadata)
        
        return scores

    def _calculate_content_similarity(self, expected: str, actual: str) -> float:
        """Calculate content similarity using sequence matching"""
        # Remove formatting for pure content comparison
        expected_clean = re.sub(r'[#*_`\[\]()]+', '', expected).strip()
        actual_clean = re.sub(r'[#*_`\[\]()]+', '', actual).strip()
        
        # Use SequenceMatcher for similarity
        matcher = difflib.SequenceMatcher(None, expected_clean.lower(), actual_clean.lower())
        return matcher.ratio()

    def _calculate_structure_similarity(self, expected: str, actual: str) -> float:
        """Calculate structure similarity based on headers and sections"""
        # Extract headers
        expected_headers = re.findall(r'^(#{1,6})\s*(.+)$', expected, re.MULTILINE)
        actual_headers = re.findall(r'^(#{1,6})\s*(.+)$', actual, re.MULTILINE)
        
        if not expected_headers and not actual_headers:
            return 1.0
        
        if not expected_headers or not actual_headers:
            return 0.0
        
        # Compare header structure
        expected_structure = [(len(h[0]), h[1].strip().lower()) for h in expected_headers]
        actual_structure = [(len(h[0]), h[1].strip().lower()) for h in actual_headers]
        
        # Use sequence matching on structure
        matcher = difflib.SequenceMatcher(None, expected_structure, actual_structure)
        return matcher.ratio()

    def _calculate_formatting_similarity(self, expected: str, actual: str) -> float:
        """Calculate formatting similarity"""
        formatting_elements = [
            (r'\*\*[^*]+\*\*', 'bold'),  # Bold
            (r'\*[^*]+\*', 'italic'),    # Italic
            (r'`[^`]+`', 'code'),        # Inline code
            (r'^```', 'code_block'),     # Code blocks
            (r'^\s*[-*+]\s', 'list'),    # Lists
            (r'^\s*\d+\.\s', 'numbered_list'),  # Numbered lists
            (r'!\[[^\]]*\]\([^)]+\)', 'image'),  # Images
            (r'\[[^\]]+\]\([^)]+\)', 'link')     # Links
        ]
        
        expected_formats = {}
        actual_formats = {}
        
        for pattern, name in formatting_elements:
            expected_formats[name] = len(re.findall(pattern, expected, re.MULTILINE))
            actual_formats[name] = len(re.findall(pattern, actual, re.MULTILINE))
        
        # Calculate similarity for each formatting type
        similarities = []
        for name in expected_formats:
            expected_count = expected_formats[name]
            actual_count = actual_formats[name]
            
            if expected_count == 0 and actual_count == 0:
                similarities.append(1.0)
            elif expected_count == 0 or actual_count == 0:
                similarities.append(0.0)
            else:
                similarity = min(expected_count, actual_count) / max(expected_count, actual_count)
                similarities.append(similarity)
        
        return statistics.mean(similarities) if similarities else 1.0

    def _calculate_metadata_similarity(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> float:
        """Calculate metadata similarity"""
        if not expected and not actual:
            return 1.0
        
        if not expected or not actual:
            return 0.0
        
        # Compare common fields
        common_fields = set(expected.keys()) & set(actual.keys())
        if not common_fields:
            return 0.0
        
        matches = 0
        total = len(common_fields)
        
        for field in common_fields:
            expected_val = str(expected[field]).lower()
            actual_val = str(actual[field]).lower()
            
            if expected_val == actual_val:
                matches += 1
            else:
                # Use partial matching for text fields
                matcher = difflib.SequenceMatcher(None, expected_val, actual_val)
                if matcher.ratio() > 0.8:
                    matches += 0.5
        
        return matches / total if total > 0 else 0.0

    def _generate_content_diff(self, expected: str, actual: str) -> str:
        """Generate detailed content diff"""
        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile='expected',
            tofile='actual',
            n=3
        )
        return ''.join(diff)

    def _generate_structure_diff(self, expected: str, actual: str) -> str:
        """Generate structure comparison"""
        expected_headers = re.findall(r'^(#{1,6})\s*(.+)$', expected, re.MULTILINE)
        actual_headers = re.findall(r'^(#{1,6})\s*(.+)$', actual, re.MULTILINE)
        
        expected_structure = [f"{h[0]} {h[1]}" for h in expected_headers]
        actual_structure = [f"{h[0]} {h[1]}" for h in actual_headers]
        
        diff = difflib.unified_diff(
            expected_structure,
            actual_structure,
            fromfile='expected_structure',
            tofile='actual_structure',
            lineterm=''
        )
        return '\n'.join(diff)

    def _identify_formatting_issues(self, expected: str, actual: str) -> List[str]:
        """Identify specific formatting issues"""
        issues = []
        
        # Check for missing bold formatting
        expected_bold = len(re.findall(r'\*\*[^*]+\*\*', expected))
        actual_bold = len(re.findall(r'\*\*[^*]+\*\*', actual))
        
        if expected_bold > actual_bold:
            issues.append(f"Missing bold formatting: expected {expected_bold}, found {actual_bold}")
        
        # Check for missing italic formatting
        expected_italic = len(re.findall(r'\*[^*]+\*', expected))
        actual_italic = len(re.findall(r'\*[^*]+\*', actual))
        
        if expected_italic > actual_italic:
            issues.append(f"Missing italic formatting: expected {expected_italic}, found {actual_italic}")
        
        # Check for missing code blocks
        expected_code = len(re.findall(r'^```', expected, re.MULTILINE))
        actual_code = len(re.findall(r'^```', actual, re.MULTILINE))
        
        if expected_code > actual_code:
            issues.append(f"Missing code blocks: expected {expected_code}, found {actual_code}")
        
        # Check for missing lists
        expected_lists = len(re.findall(r'^\s*[-*+]\s', expected, re.MULTILINE))
        actual_lists = len(re.findall(r'^\s*[-*+]\s', actual, re.MULTILINE))
        
        if expected_lists > actual_lists:
            issues.append(f"Missing list items: expected {expected_lists}, found {actual_lists}")
        
        return issues

    def _generate_metadata_diff(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> str:
        """Generate metadata comparison"""
        expected_str = json.dumps(expected, indent=2, sort_keys=True)
        actual_str = json.dumps(actual, indent=2, sort_keys=True)
        
        diff = difflib.unified_diff(
            expected_str.splitlines(keepends=True),
            actual_str.splitlines(keepends=True),
            fromfile='expected_metadata',
            tofile='actual_metadata',
            n=3
        )
        return ''.join(diff)

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for validation context"""
        import platform
        import sys
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor_version": "marker-pdf-1.0",  # This would be dynamic
            "timestamp": datetime.now().isoformat(),
            "memory_usage": "unknown",  # Could add psutil for memory info
            "cpu_count": os.cpu_count()
        }

    def _create_failed_validation_result(self, benchmark_id: str, test_id: str,
                                       start_time: float, error_message: str,
                                       processor_config: Dict[str, Any]) -> ValidationResult:
        """Create a failed validation result"""
        return ValidationResult(
            benchmark_id=benchmark_id,
            test_id=test_id,
            timestamp=datetime.now(),
            accuracy_score=0.0,
            passed=False,
            processing_time=time.time() - start_time,
            content_accuracy=0.0,
            structure_accuracy=0.0,
            formatting_accuracy=0.0,
            metadata_accuracy=0.0,
            content_diff="",
            structure_diff="",
            formatting_issues=[],
            metadata_diff="",
            errors=[error_message],
            warnings=[],
            pdf_processor_version="unknown",
            processing_config=processor_config,
            system_info=self._collect_system_info()
        )

    def _update_benchmark_stats(self, benchmark_id: str, result: ValidationResult) -> None:
        """Update benchmark statistics with new result"""
        benchmark = self.benchmarks[benchmark_id]
        benchmark.validation_count += 1
        benchmark.last_validated = result.timestamp
        
        # Update average accuracy
        if benchmark.validation_count == 1:
            benchmark.average_accuracy = result.accuracy_score
        else:
            # Moving average
            old_avg = benchmark.average_accuracy
            benchmark.average_accuracy = old_avg + (result.accuracy_score - old_avg) / benchmark.validation_count

    def _filter_benchmarks(self, filter_criteria: Dict[str, Any]) -> List[str]:
        """Filter benchmarks based on criteria"""
        filtered_benchmarks = []
        
        for benchmark_id, benchmark in self.benchmarks.items():
            include = True
            
            # Filter by content type
            if 'content_type' in filter_criteria:
                if benchmark.content_type != filter_criteria['content_type']:
                    include = False
            
            # Filter by difficulty level
            if 'difficulty_level' in filter_criteria:
                if benchmark.difficulty_level != filter_criteria['difficulty_level']:
                    include = False
            
            # Filter by tags
            if 'tags' in filter_criteria:
                required_tags = filter_criteria['tags']
                if not all(tag in benchmark.tags for tag in required_tags):
                    include = False
            
            # Filter by accuracy threshold
            if 'min_accuracy_threshold' in filter_criteria:
                if benchmark.accuracy_threshold < filter_criteria['min_accuracy_threshold']:
                    include = False
            
            if include:
                filtered_benchmarks.append(benchmark_id)
        
        return filtered_benchmarks

    def _detect_regression(self, benchmark_id: str, current_result: ValidationResult) -> bool:
        """Detect if current result represents a regression"""
        if benchmark_id not in self.validation_history:
            return False
        
        history = self.validation_history[benchmark_id]
        if len(history) < 2:
            return False
        
        # Get recent historical average (excluding current result)
        recent_results = history[-5:]  # Last 5 results
        if len(recent_results) < 2:
            return False
        
        historical_avg = statistics.mean([r.accuracy_score for r in recent_results])
        
        # Check if current result is significantly worse
        return historical_avg - current_result.accuracy_score > self.regression_threshold

    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate trend direction from scores"""
        if len(scores) < 2:
            return "stable"
        
        # Simple linear trend
        first_half_avg = statistics.mean(scores[:len(scores)//2])
        second_half_avg = statistics.mean(scores[len(scores)//2:])
        
        diff = second_half_avg - first_half_avg
        
        if abs(diff) < 0.01:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"

    def _generate_regression_recommendations(self, regressions: List[Dict[str, Any]], 
                                           system_trend: str) -> List[str]:
        """Generate recommendations based on regression analysis"""
        recommendations = []
        
        if regressions:
            recommendations.append(f"Investigate {len(regressions)} benchmarks showing regression")
            
            # Group by difficulty level if available
            difficulty_groups = defaultdict(int)
            for regression in regressions:
                benchmark_id = regression['benchmark_id']
                if benchmark_id in self.benchmarks:
                    difficulty = self.benchmarks[benchmark_id].difficulty_level
                    difficulty_groups[difficulty] += 1
            
            if difficulty_groups:
                for difficulty, count in difficulty_groups.items():
                    recommendations.append(f"Focus on {difficulty} difficulty benchmarks ({count} regressions)")
        
        if system_trend == "declining":
            recommendations.append("System-wide performance decline detected - review recent changes")
        elif system_trend == "stable" and regressions:
            recommendations.append("Isolated regressions detected - investigate specific benchmark issues")
        
        return recommendations

    def _log_validation_suite_summary(self, results: List[ValidationResult]) -> None:
        """Log summary of validation suite run"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.passed])
        failed_tests = total_tests - passed_tests
        
        if total_tests > 0:
            overall_accuracy = statistics.mean([r.accuracy_score for r in results])
            pass_rate = (passed_tests / total_tests) * 100
            
            self.logger.info(
                f"Validation suite completed: {total_tests} tests, "
                f"{passed_tests} passed, {failed_tests} failed, "
                f"Overall accuracy: {overall_accuracy:.3f}, Pass rate: {pass_rate:.1f}%"
            )
            
            # Log metrics
            self.log_metrics({
                "operation": "validation_suite_summary",
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "overall_accuracy": overall_accuracy,
                "pass_rate": pass_rate,
                "success": True
            })

    def _save_benchmarks(self) -> None:
        """Save benchmarks to file"""
        benchmarks_data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "benchmarks": [benchmark.to_dict() for benchmark in self.benchmarks.values()]
        }
        
        benchmarks_file = self.benchmarks_dir / "benchmarks.json"
        with open(benchmarks_file, 'w', encoding='utf-8') as f:
            json.dump(benchmarks_data, f, indent=2, default=str)

    def _save_validation_results(self) -> None:
        """Save recent validation results to file"""
        # Keep only recent results (last 30 days) in memory
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_results = [
            result for result in self.validation_results
            if result.timestamp >= cutoff_date
        ]
        
        results_data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "results": [result.to_dict() for result in recent_results]
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"validation_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)

    def _save_validation_report(self, report: ValidationReport) -> None:
        """Save validation report to file"""
        report_file = self.reports_dir / f"{report.report_id}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved: {report_file}")

    # Additional helper methods for report generation
    
    def _create_empty_report(self, report_id: str, start_date: datetime, end_date: datetime) -> ValidationReport:
        """Create empty validation report when no data is available"""
        return ValidationReport(
            report_id=report_id,
            generation_date=datetime.now(),
            report_period=(start_date, end_date),
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            overall_accuracy=0.0,
            pass_rate=0.0,
            benchmark_results={},
            performance_by_difficulty={},
            performance_by_content_type={},
            accuracy_trends=[],
            regression_analysis={},
            improvement_opportunities=["No validation data available for analysis"],
            quality_scores={},
            quality_distribution={},
            recommendations=["Set up validation benchmarks and run initial tests"],
            action_items=["Create validation benchmarks", "Run baseline validation tests"]
        )

    def _analyze_benchmark_performance(self, results: List[ValidationResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by individual benchmarks"""
        benchmark_performance = defaultdict(list)
        
        for result in results:
            benchmark_performance[result.benchmark_id].append(result)
        
        analysis = {}
        for benchmark_id, benchmark_results in benchmark_performance.items():
            benchmark_name = self.benchmarks[benchmark_id].name if benchmark_id in self.benchmarks else "Unknown"
            
            scores = [r.accuracy_score for r in benchmark_results]
            passed_count = len([r for r in benchmark_results if r.passed])
            
            analysis[benchmark_id] = {
                "name": benchmark_name,
                "test_count": len(benchmark_results),
                "average_accuracy": statistics.mean(scores),
                "min_accuracy": min(scores),
                "max_accuracy": max(scores),
                "pass_count": passed_count,
                "pass_rate": (passed_count / len(benchmark_results)) * 100,
                "std_deviation": statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
        
        return analysis

    def _analyze_performance_by_difficulty(self, results: List[ValidationResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance grouped by difficulty level"""
        difficulty_results = defaultdict(list)
        
        for result in results:
            if result.benchmark_id in self.benchmarks:
                difficulty = self.benchmarks[result.benchmark_id].difficulty_level
                difficulty_results[difficulty].append(result)
        
        analysis = {}
        for difficulty, diff_results in difficulty_results.items():
            scores = [r.accuracy_score for r in diff_results]
            passed_count = len([r for r in diff_results if r.passed])
            
            analysis[difficulty] = {
                "test_count": len(diff_results),
                "average_accuracy": statistics.mean(scores),
                "pass_count": passed_count,
                "pass_rate": (passed_count / len(diff_results)) * 100,
                "benchmark_count": len(set(r.benchmark_id for r in diff_results))
            }
        
        return analysis

    def _analyze_performance_by_content_type(self, results: List[ValidationResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance grouped by content type"""
        content_type_results = defaultdict(list)
        
        for result in results:
            if result.benchmark_id in self.benchmarks:
                content_type = self.benchmarks[result.benchmark_id].content_type
                content_type_results[content_type].append(result)
        
        analysis = {}
        for content_type, type_results in content_type_results.items():
            scores = [r.accuracy_score for r in type_results]
            passed_count = len([r for r in type_results if r.passed])
            
            analysis[content_type] = {
                "test_count": len(type_results),
                "average_accuracy": statistics.mean(scores),
                "pass_count": passed_count,
                "pass_rate": (passed_count / len(type_results)) * 100,
                "benchmark_count": len(set(r.benchmark_id for r in type_results))
            }
        
        return analysis

    def _calculate_accuracy_trends(self, results: List[ValidationResult], 
                                 start_date: datetime, end_date: datetime) -> List[QualityTrend]:
        """Calculate accuracy trends over time"""
        # Group results by week
        weekly_results = defaultdict(list)
        
        for result in results:
            # Get week start date
            days_since_monday = result.timestamp.weekday()
            week_start = result.timestamp - timedelta(days=days_since_monday)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            weekly_results[week_start].append(result)
        
        trends = []
        sorted_weeks = sorted(weekly_results.keys())
        
        for i, week_start in enumerate(sorted_weeks):
            week_end = week_start + timedelta(days=6)
            week_results = weekly_results[week_start]
            
            scores = [r.accuracy_score for r in week_results]
            passed_count = len([r for r in week_results if r.passed])
            
            # Calculate improvement rate
            improvement_rate = 0.0
            if i > 0:
                prev_week_results = weekly_results[sorted_weeks[i-1]]
                prev_avg = statistics.mean([r.accuracy_score for r in prev_week_results])
                current_avg = statistics.mean(scores)
                improvement_rate = current_avg - prev_avg
            
            # Analyze benchmark performance for the week
            benchmark_performance = {}
            for result in week_results:
                if result.benchmark_id not in benchmark_performance:
                    benchmark_performance[result.benchmark_id] = []
                benchmark_performance[result.benchmark_id].append(result.accuracy_score)
            
            # Average performance per benchmark
            for benchmark_id in benchmark_performance:
                benchmark_performance[benchmark_id] = statistics.mean(benchmark_performance[benchmark_id])
            
            trend = QualityTrend(
                period="weekly",
                start_date=week_start,
                end_date=week_end,
                average_accuracy=statistics.mean(scores),
                test_count=len(week_results),
                pass_rate=(passed_count / len(week_results)) * 100,
                improvement_rate=improvement_rate,
                regression_count=len([r for r in week_results if r.accuracy_score < 0.8]),
                benchmark_performance=benchmark_performance
            )
            
            trends.append(trend)
        
        return trends

    def _identify_improvement_opportunities(self, results: List[ValidationResult]) -> List[str]:
        """Identify opportunities for quality improvement"""
        opportunities = []
        
        # Analyze failed tests
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            # Group failures by benchmark
            failures_by_benchmark = defaultdict(int)
            for result in failed_results:
                failures_by_benchmark[result.benchmark_id] += 1
            
            # Identify problematic benchmarks
            total_benchmarks = len(set(r.benchmark_id for r in results))
            if failures_by_benchmark:
                worst_benchmark = max(failures_by_benchmark.items(), key=lambda x: x[1])
                benchmark_name = self.benchmarks.get(worst_benchmark[0], {}).get('name', 'Unknown')
                opportunities.append(f"Focus on benchmark '{benchmark_name}' with {worst_benchmark[1]} failures")
        
        # Analyze accuracy scores by component
        content_scores = [r.content_accuracy for r in results]
        structure_scores = [r.structure_accuracy for r in results]
        formatting_scores = [r.formatting_accuracy for r in results]
        metadata_scores = [r.metadata_accuracy for r in results]
        
        avg_content = statistics.mean(content_scores)
        avg_structure = statistics.mean(structure_scores)
        avg_formatting = statistics.mean(formatting_scores)
        avg_metadata = statistics.mean(metadata_scores)
        
        if avg_content < 0.85:
            opportunities.append("Improve content extraction accuracy")
        if avg_structure < 0.85:
            opportunities.append("Improve document structure preservation")
        if avg_formatting < 0.85:
            opportunities.append("Improve markdown formatting accuracy")
        if avg_metadata < 0.85:
            opportunities.append("Improve metadata extraction")
        
        # Check for processing time issues
        processing_times = [r.processing_time for r in results]
        avg_processing_time = statistics.mean(processing_times)
        if avg_processing_time > 60:  # More than 1 minute average
            opportunities.append("Optimize processing performance")
        
        return opportunities

    def _calculate_integrated_quality_scores(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate quality scores integrated with existing quality system"""
        if not results:
            return {}
        
        return {
            "overall_validation_score": statistics.mean([r.accuracy_score for r in results]),
            "content_quality_score": statistics.mean([r.content_accuracy for r in results]),
            "structure_quality_score": statistics.mean([r.structure_accuracy for r in results]),
            "formatting_quality_score": statistics.mean([r.formatting_accuracy for r in results]),
            "metadata_quality_score": statistics.mean([r.metadata_accuracy for r in results]),
            "processing_efficiency": 1.0 / statistics.mean([r.processing_time for r in results])
        }

    def _calculate_quality_distribution(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Calculate distribution of quality scores"""
        distribution = {
            "excellent": 0,  # >= 0.95
            "good": 0,       # >= 0.9
            "acceptable": 0, # >= 0.8
            "poor": 0,       # >= 0.7
            "failing": 0     # < 0.7
        }
        
        for result in results:
            score = result.accuracy_score
            if score >= 0.95:
                distribution["excellent"] += 1
            elif score >= 0.9:
                distribution["good"] += 1
            elif score >= 0.8:
                distribution["acceptable"] += 1
            elif score >= 0.7:
                distribution["poor"] += 1
            else:
                distribution["failing"] += 1
        
        return distribution

    def _generate_validation_recommendations(self, results: List[ValidationResult],
                                           regression_analysis: Dict[str, Any],
                                           improvement_opportunities: List[str]) -> List[str]:
        """Generate comprehensive validation recommendations"""
        recommendations = []
        
        if not results:
            recommendations.append("Set up validation benchmarks and run initial tests")
            return recommendations
        
        overall_accuracy = statistics.mean([r.accuracy_score for r in results])
        pass_rate = (len([r for r in results if r.passed]) / len(results)) * 100
        
        # Overall performance recommendations
        if overall_accuracy < 0.9:
            recommendations.append(f"Overall accuracy ({overall_accuracy:.3f}) below 90% target - review PDF processing pipeline")
        
        if pass_rate < 85:
            recommendations.append(f"Pass rate ({pass_rate:.1f}%) below target - investigate failing benchmarks")
        
        # Regression-based recommendations
        if regression_analysis.get("regression_count", 0) > 0:
            recommendations.append("Address quality regressions before adding new features")
        
        # Component-specific recommendations
        content_avg = statistics.mean([r.content_accuracy for r in results])
        structure_avg = statistics.mean([r.structure_accuracy for r in results])
        formatting_avg = statistics.mean([r.formatting_accuracy for r in results])
        
        if content_avg < structure_avg and content_avg < formatting_avg:
            recommendations.append("Prioritize content extraction improvements")
        elif structure_avg < content_avg and structure_avg < formatting_avg:
            recommendations.append("Focus on document structure preservation")
        elif formatting_avg < content_avg and formatting_avg < structure_avg:
            recommendations.append("Improve markdown formatting accuracy")
        
        # Processing efficiency recommendations
        avg_processing_time = statistics.mean([r.processing_time for r in results])
        if avg_processing_time > 30:
            recommendations.append("Optimize processing performance for faster validation cycles")
        
        # Add improvement opportunities
        recommendations.extend(improvement_opportunities)
        
        return recommendations

    def _generate_action_items(self, results: List[ValidationResult],
                             regression_analysis: Dict[str, Any],
                             recommendations: List[str]) -> List[str]:
        """Generate specific action items"""
        action_items = []
        
        # Failed benchmark action items
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            failed_benchmarks = set(r.benchmark_id for r in failed_results)
            for benchmark_id in list(failed_benchmarks)[:3]:  # Top 3 failing benchmarks
                benchmark_name = self.benchmarks.get(benchmark_id, {}).get('name', 'Unknown')
                action_items.append(f"Debug and fix issues in benchmark: {benchmark_name}")
        
        # Regression action items
        if regression_analysis.get("regression_count", 0) > 0:
            action_items.append("Investigate recent code changes causing regressions")
            action_items.append("Run detailed diff analysis on regressed benchmarks")
        
        # Accuracy improvement action items
        overall_accuracy = statistics.mean([r.accuracy_score for r in results]) if results else 0
        if overall_accuracy < 0.9:
            action_items.append("Review and tune PDF processor configuration")
            action_items.append("Analyze common failure patterns across benchmarks")
        
        # Process improvement action items
        if len(self.benchmarks) < 10:
            action_items.append("Create additional validation benchmarks for better coverage")
        
        if not any("continuous" in rec.lower() for rec in recommendations):
            action_items.append("Set up continuous validation pipeline")
        
        return action_items

    def check_quality(self, content: Dict[str, Any]) -> float:
        """Implementation of base class abstract method for quality checking"""
        # This integrates with the base agent quality checking interface
        if "validation_results" in content:
            results = content["validation_results"]
            if results:
                return statistics.mean([r.accuracy_score for r in results])
        
        return 0.0

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for validation system"""
        if isinstance(input_data, dict):
            if "pdf_path" in input_data and "expected_output" in input_data:
                return os.path.exists(input_data["pdf_path"])
        
        return False

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data from validation system"""
        if isinstance(output_data, ValidationResult):
            return (
                hasattr(output_data, 'accuracy_score') and
                0.0 <= output_data.accuracy_score <= 1.0 and
                hasattr(output_data, 'benchmark_id') and
                hasattr(output_data, 'timestamp')
            )
        elif isinstance(output_data, ValidationReport):
            return (
                hasattr(output_data, 'overall_accuracy') and
                hasattr(output_data, 'total_tests') and
                hasattr(output_data, 'report_id')
            )
        
        return False


def main():
    """Main entry point for the Quality Validation System"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Validation System for PDF-to-Markdown Conversion")
    parser.add_argument("--create-benchmark", help="Create a new validation benchmark")
    parser.add_argument("--run-validation", action="store_true", help="Run full validation suite")
    parser.add_argument("--generate-report", action="store_true", help="Generate validation report")
    parser.add_argument("--detect-regressions", action="store_true", help="Detect quality regressions")
    parser.add_argument("--setup-continuous", action="store_true", help="Setup continuous validation")
    parser.add_argument("--config", help="Path to validation configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # Create validation system
    validation_system = QualityValidationSystem(config)
    
    if args.create_benchmark:
        print("Benchmark creation requires interactive input - use API directly")
        
    elif args.run_validation:
        def dummy_processor(pdf_path, config):
            # This would be replaced with actual PDF processor
            return {
                "markdown": "# Test Document\n\nProcessed content here",
                "metadata": {"title": "Test", "pages": 1}
            }
        
        results = validation_system.run_full_validation_suite(dummy_processor)
        print(f"Validation completed: {len(results)} tests run")
        
    elif args.generate_report:
        report = validation_system.generate_validation_report()
        print(f"Generated validation report: {report.report_id}")
        print(f"Overall accuracy: {report.overall_accuracy:.3f}")
        print(f"Pass rate: {report.pass_rate:.1f}%")
        
    elif args.detect_regressions:
        regression_analysis = validation_system.detect_quality_regressions()
        regression_count = regression_analysis.get("regression_count", 0)
        print(f"Regression analysis completed: {regression_count} regressions detected")
        
    elif args.setup_continuous:
        def dummy_processor(pdf_path, config):
            return {
                "markdown": "# Test Document\n\nProcessed content here",
                "metadata": {"title": "Test", "pages": 1}
            }
        
        validation_system.setup_continuous_validation(dummy_processor)
        print("Continuous validation setup completed")
        
    else:
        print("Please specify an action (--run-validation, --generate-report, etc.)")


if __name__ == "__main__":
    main()