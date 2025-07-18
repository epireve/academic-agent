"""
Unit tests for Quality Validation System.

This module tests the comprehensive quality validation framework for ensuring
90% accuracy in PDF-to-markdown conversion through automated testing,
regression detection, and continuous validation.
"""

import pytest
import json
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import asdict

from agents.academic.quality_validation_system import (
    QualityValidationSystem,
    ValidationBenchmark,
    ValidationResult,
    QualityTrend,
    ValidationReport
)


class TestValidationBenchmark:
    """Test ValidationBenchmark data class."""
    
    def test_validation_benchmark_creation(self):
        """Test creation of ValidationBenchmark."""
        benchmark = ValidationBenchmark(
            benchmark_id="test_001",
            name="Test Benchmark",
            description="A test benchmark for validation",
            source_pdf_path="/test/source.pdf",
            expected_markdown_path="/test/expected.md",
            accuracy_threshold=0.9,
            content_type="academic",
            difficulty_level="medium"
        )
        
        assert benchmark.benchmark_id == "test_001"
        assert benchmark.name == "Test Benchmark"
        assert benchmark.accuracy_threshold == 0.9
        assert benchmark.content_type == "academic"
        assert benchmark.difficulty_level == "medium"
        assert benchmark.validation_count == 0
        assert benchmark.average_accuracy == 0.0
    
    def test_validation_benchmark_to_dict(self):
        """Test ValidationBenchmark serialization to dictionary."""
        benchmark = ValidationBenchmark(
            benchmark_id="test_001",
            name="Test Benchmark",
            description="A test benchmark",
            source_pdf_path="/test/source.pdf",
            expected_markdown_path="/test/expected.md",
            tags=["test", "academic"]
        )
        
        benchmark_dict = benchmark.to_dict()
        
        assert benchmark_dict["benchmark_id"] == "test_001"
        assert benchmark_dict["name"] == "Test Benchmark"
        assert benchmark_dict["tags"] == ["test", "academic"]
        assert "created_date" in benchmark_dict
        assert isinstance(benchmark_dict["created_date"], str)


class TestValidationResult:
    """Test ValidationResult data class."""
    
    def test_validation_result_creation(self):
        """Test creation of ValidationResult."""
        result = ValidationResult(
            benchmark_id="test_001",
            test_id="test_run_001",
            timestamp=datetime.now(),
            accuracy_score=0.85,
            passed=True,
            processing_time=2.5,
            content_accuracy=0.9,
            structure_accuracy=0.8,
            formatting_accuracy=0.85,
            metadata_accuracy=0.9,
            content_diff="",
            structure_diff="",
            formatting_issues=[],
            metadata_diff="",
            errors=[],
            warnings=[],
            pdf_processor_version="1.0",
            processing_config={},
            system_info={}
        )
        
        assert result.benchmark_id == "test_001"
        assert result.accuracy_score == 0.85
        assert result.passed is True
        assert result.processing_time == 2.5
        assert len(result.errors) == 0
    
    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization to dictionary."""
        result = ValidationResult(
            benchmark_id="test_001",
            test_id="test_run_001",
            timestamp=datetime.now(),
            accuracy_score=0.85,
            passed=True,
            processing_time=2.5,
            content_accuracy=0.9,
            structure_accuracy=0.8,
            formatting_accuracy=0.85,
            metadata_accuracy=0.9,
            content_diff="line added",
            structure_diff="header missing",
            formatting_issues=["missing bold"],
            metadata_diff="title different",
            errors=["minor error"],
            warnings=["performance warning"],
            pdf_processor_version="1.0",
            processing_config={"param": "value"},
            system_info={"platform": "test"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["benchmark_id"] == "test_001"
        assert result_dict["accuracy_score"] == 0.85
        assert result_dict["formatting_issues"] == ["missing bold"]
        assert result_dict["processing_config"] == {"param": "value"}
        assert "timestamp" in result_dict


class TestQualityValidationSystem:
    """Test QualityValidationSystem main class."""
    
    @pytest.fixture
    def temp_validation_dir(self):
        """Create temporary validation directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def validation_config(self, temp_validation_dir):
        """Create test validation configuration."""
        return {
            "accuracy_threshold": 0.9,
            "regression_threshold": 0.05,
            "max_concurrent_tests": 2,
            "validation_dir": str(temp_validation_dir),
            "benchmark_timeout": 60
        }
    
    @pytest.fixture
    def validation_system(self, validation_config):
        """Create QualityValidationSystem instance for testing."""
        return QualityValidationSystem(validation_config)
    
    @pytest.fixture
    def sample_pdf_file(self, temp_validation_dir):
        """Create sample PDF file for testing."""
        pdf_path = temp_validation_dir / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF")
        return pdf_path
    
    @pytest.fixture
    def sample_markdown_file(self, temp_validation_dir):
        """Create sample expected markdown file for testing."""
        md_path = temp_validation_dir / "expected.md"
        md_path.write_text("# Test Document\n\nThis is a test document.\n\n## Section 1\n\nContent here.\n")
        return md_path
    
    def test_validation_system_initialization(self, validation_system, temp_validation_dir):
        """Test QualityValidationSystem initialization."""
        assert validation_system.accuracy_threshold == 0.9
        assert validation_system.regression_threshold == 0.05
        assert validation_system.max_concurrent_tests == 2
        assert validation_system.validation_dir == Path(temp_validation_dir)
        
        # Check that directories were created
        assert validation_system.benchmarks_dir.exists()
        assert validation_system.results_dir.exists()
        assert validation_system.reports_dir.exists()
    
    def test_create_benchmark_success(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test successful benchmark creation."""
        benchmark = validation_system.create_benchmark(
            name="Test Benchmark",
            description="A test benchmark for validation",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            accuracy_threshold=0.85,
            content_type="academic",
            difficulty_level="easy",
            tags=["test", "basic"]
        )
        
        assert benchmark.name == "Test Benchmark"
        assert benchmark.accuracy_threshold == 0.85
        assert benchmark.content_type == "academic"
        assert benchmark.difficulty_level == "easy"
        assert benchmark.tags == ["test", "basic"]
        assert benchmark.benchmark_id in validation_system.benchmarks
    
    def test_create_benchmark_missing_source_pdf(self, validation_system, sample_markdown_file):
        """Test benchmark creation with missing source PDF."""
        with pytest.raises(ValueError, match="Source PDF file not found"):
            validation_system.create_benchmark(
                name="Test Benchmark",
                description="A test benchmark",
                source_pdf_path="/nonexistent/file.pdf",
                expected_markdown_path=str(sample_markdown_file)
            )
    
    def test_create_benchmark_missing_expected_markdown(self, validation_system, sample_pdf_file):
        """Test benchmark creation with missing expected markdown."""
        with pytest.raises(ValueError, match="Expected markdown file not found"):
            validation_system.create_benchmark(
                name="Test Benchmark",
                description="A test benchmark",
                source_pdf_path=str(sample_pdf_file),
                expected_markdown_path="/nonexistent/file.md"
            )
    
    def test_validate_benchmark_success(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test successful benchmark validation."""
        # Create benchmark
        benchmark = validation_system.create_benchmark(
            name="Test Benchmark",
            description="A test benchmark",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file)
        )
        
        # Mock PDF processor function
        def mock_pdf_processor(pdf_path, config):
            return {
                "markdown": "# Test Document\n\nThis is a test document.\n\n## Section 1\n\nContent here.\n",
                "metadata": {"title": "Test Document", "pages": 1}
            }
        
        # Run validation
        result = validation_system.validate_benchmark(
            benchmark.benchmark_id,
            mock_pdf_processor,
            {"test_config": "value"}
        )
        
        assert result.benchmark_id == benchmark.benchmark_id
        assert result.accuracy_score > 0.8  # Should be high due to exact match
        assert result.passed is True
        assert result.processing_time > 0
        assert len(result.errors) == 0
    
    def test_validate_benchmark_processor_failure(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test benchmark validation with processor failure."""
        # Create benchmark
        benchmark = validation_system.create_benchmark(
            name="Test Benchmark",
            description="A test benchmark",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file)
        )
        
        # Mock failing PDF processor function
        def failing_pdf_processor(pdf_path, config):
            raise RuntimeError("PDF processing failed")
        
        # Run validation
        result = validation_system.validate_benchmark(
            benchmark.benchmark_id,
            failing_pdf_processor
        )
        
        assert result.benchmark_id == benchmark.benchmark_id
        assert result.accuracy_score == 0.0
        assert result.passed is False
        assert len(result.errors) == 1
        assert "PDF processing failed" in result.errors[0]
    
    def test_validate_benchmark_nonexistent(self, validation_system):
        """Test validation of nonexistent benchmark."""
        def mock_pdf_processor(pdf_path, config):
            return {"markdown": "test", "metadata": {}}
        
        with pytest.raises(ValueError, match="Benchmark not found"):
            validation_system.validate_benchmark(
                "nonexistent_benchmark",
                mock_pdf_processor
            )
    
    def test_calculate_content_similarity(self, validation_system):
        """Test content similarity calculation."""
        expected = "This is a test document with some content."
        actual = "This is a test document with similar content."
        
        similarity = validation_system._calculate_content_similarity(expected, actual)
        
        assert 0.7 < similarity < 1.0  # Should be high similarity
    
    def test_calculate_structure_similarity(self, validation_system):
        """Test structure similarity calculation."""
        expected = "# Title\n\n## Section 1\n\nContent\n\n### Subsection\n\nMore content"
        actual = "# Title\n\n## Section 1\n\nContent\n\n### Subsection\n\nMore content"
        
        similarity = validation_system._calculate_structure_similarity(expected, actual)
        
        assert similarity == 1.0  # Exact structure match
    
    def test_calculate_structure_similarity_different(self, validation_system):
        """Test structure similarity with different structures."""
        expected = "# Title\n\n## Section 1\n\n### Subsection"
        actual = "# Title\n\n### Different Section"
        
        similarity = validation_system._calculate_structure_similarity(expected, actual)
        
        assert 0.0 < similarity < 1.0  # Partial similarity
    
    def test_calculate_formatting_similarity(self, validation_system):
        """Test formatting similarity calculation."""
        expected = "This is **bold** and *italic* text with `code` and [link](url)."
        actual = "This is **bold** and *italic* text with `code` and [link](url)."
        
        similarity = validation_system._calculate_formatting_similarity(expected, actual)
        
        assert similarity == 1.0  # Perfect formatting match
    
    def test_calculate_formatting_similarity_missing_formatting(self, validation_system):
        """Test formatting similarity with missing formatting."""
        expected = "This is **bold** and *italic* text with `code`."
        actual = "This is bold and italic text with code."
        
        similarity = validation_system._calculate_formatting_similarity(expected, actual)
        
        assert similarity < 0.5  # Low similarity due to missing formatting
    
    def test_calculate_metadata_similarity(self, validation_system):
        """Test metadata similarity calculation."""
        expected = {"title": "Test Document", "author": "Test Author", "pages": 5}
        actual = {"title": "Test Document", "author": "Test Author", "pages": 5}
        
        similarity = validation_system._calculate_metadata_similarity(expected, actual)
        
        assert similarity == 1.0  # Perfect match
    
    def test_calculate_metadata_similarity_partial(self, validation_system):
        """Test metadata similarity with partial match."""
        expected = {"title": "Test Document", "author": "Test Author", "pages": 5}
        actual = {"title": "Test Document", "author": "Different Author", "pages": 5}
        
        similarity = validation_system._calculate_metadata_similarity(expected, actual)
        
        assert 0.5 < similarity < 1.0  # Partial match
    
    def test_generate_content_diff(self, validation_system):
        """Test content diff generation."""
        expected = "Line 1\nLine 2\nLine 3"
        actual = "Line 1\nModified Line 2\nLine 3"
        
        diff = validation_system._generate_content_diff(expected, actual)
        
        assert "Line 2" in diff
        assert "Modified Line 2" in diff
        assert "@@" in diff  # Unified diff format
    
    def test_identify_formatting_issues(self, validation_system):
        """Test formatting issue identification."""
        expected = "This has **bold** and *italic* and `code` and - list item"
        actual = "This has bold and italic and code and list item"
        
        issues = validation_system._identify_formatting_issues(expected, actual)
        
        assert len(issues) > 0
        assert any("bold" in issue.lower() for issue in issues)
        assert any("italic" in issue.lower() for issue in issues)
    
    def test_filter_benchmarks_by_content_type(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test benchmark filtering by content type."""
        # Create benchmarks with different content types
        validation_system.create_benchmark(
            name="Academic Benchmark",
            description="Academic content",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            content_type="academic"
        )
        
        validation_system.create_benchmark(
            name="Technical Benchmark",
            description="Technical content",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            content_type="technical"
        )
        
        # Filter by content type
        academic_benchmarks = validation_system._filter_benchmarks({"content_type": "academic"})
        technical_benchmarks = validation_system._filter_benchmarks({"content_type": "technical"})
        
        assert len(academic_benchmarks) == 1
        assert len(technical_benchmarks) == 1
    
    def test_filter_benchmarks_by_difficulty(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test benchmark filtering by difficulty level."""
        # Create benchmarks with different difficulty levels
        validation_system.create_benchmark(
            name="Easy Benchmark",
            description="Easy content",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            difficulty_level="easy"
        )
        
        validation_system.create_benchmark(
            name="Hard Benchmark",
            description="Hard content",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            difficulty_level="hard"
        )
        
        # Filter by difficulty
        easy_benchmarks = validation_system._filter_benchmarks({"difficulty_level": "easy"})
        hard_benchmarks = validation_system._filter_benchmarks({"difficulty_level": "hard"})
        
        assert len(easy_benchmarks) == 1
        assert len(hard_benchmarks) == 1
    
    def test_filter_benchmarks_by_tags(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test benchmark filtering by tags."""
        # Create benchmarks with different tags
        validation_system.create_benchmark(
            name="Tagged Benchmark",
            description="Benchmark with tags",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            tags=["test", "regression"]
        )
        
        validation_system.create_benchmark(
            name="Other Benchmark",
            description="Benchmark with other tags",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            tags=["performance"]
        )
        
        # Filter by tags
        test_benchmarks = validation_system._filter_benchmarks({"tags": ["test"]})
        regression_benchmarks = validation_system._filter_benchmarks({"tags": ["regression"]})
        performance_benchmarks = validation_system._filter_benchmarks({"tags": ["performance"]})
        
        assert len(test_benchmarks) == 1
        assert len(regression_benchmarks) == 1
        assert len(performance_benchmarks) == 1
    
    def test_detect_regression_with_decline(self, validation_system):
        """Test regression detection with declining performance."""
        benchmark_id = "test_benchmark"
        
        # Create historical results showing decline
        base_time = datetime.now()
        historical_results = [
            ValidationResult(
                benchmark_id=benchmark_id,
                test_id=f"test_{i}",
                timestamp=base_time - timedelta(hours=i),
                accuracy_score=0.9 - (i * 0.02),  # Declining scores
                passed=True,
                processing_time=1.0,
                content_accuracy=0.9,
                structure_accuracy=0.9,
                formatting_accuracy=0.9,
                metadata_accuracy=0.9,
                content_diff="",
                structure_diff="",
                formatting_issues=[],
                metadata_diff="",
                errors=[],
                warnings=[],
                pdf_processor_version="1.0",
                processing_config={},
                system_info={}
            )
            for i in range(5)
        ]
        
        # Add historical results
        validation_system.validation_history[benchmark_id] = historical_results
        
        # Create current result with significant decline
        current_result = ValidationResult(
            benchmark_id=benchmark_id,
            test_id="current_test",
            timestamp=base_time,
            accuracy_score=0.75,  # Significant drop
            passed=True,
            processing_time=1.0,
            content_accuracy=0.75,
            structure_accuracy=0.75,
            formatting_accuracy=0.75,
            metadata_accuracy=0.75,
            content_diff="",
            structure_diff="",
            formatting_issues=[],
            metadata_diff="",
            errors=[],
            warnings=[],
            pdf_processor_version="1.0",
            processing_config={},
            system_info={}
        )
        
        # Test regression detection
        is_regression = validation_system._detect_regression(benchmark_id, current_result)
        
        assert is_regression is True
    
    def test_detect_regression_no_decline(self, validation_system):
        """Test regression detection with stable performance."""
        benchmark_id = "test_benchmark"
        
        # Create historical results with stable performance
        base_time = datetime.now()
        historical_results = [
            ValidationResult(
                benchmark_id=benchmark_id,
                test_id=f"test_{i}",
                timestamp=base_time - timedelta(hours=i),
                accuracy_score=0.9,  # Stable scores
                passed=True,
                processing_time=1.0,
                content_accuracy=0.9,
                structure_accuracy=0.9,
                formatting_accuracy=0.9,
                metadata_accuracy=0.9,
                content_diff="",
                structure_diff="",
                formatting_issues=[],
                metadata_diff="",
                errors=[],
                warnings=[],
                pdf_processor_version="1.0",
                processing_config={},
                system_info={}
            )
            for i in range(5)
        ]
        
        # Add historical results
        validation_system.validation_history[benchmark_id] = historical_results
        
        # Create current result with similar performance
        current_result = ValidationResult(
            benchmark_id=benchmark_id,
            test_id="current_test",
            timestamp=base_time,
            accuracy_score=0.89,  # Slight variation but not regression
            passed=True,
            processing_time=1.0,
            content_accuracy=0.89,
            structure_accuracy=0.89,
            formatting_accuracy=0.89,
            metadata_accuracy=0.89,
            content_diff="",
            structure_diff="",
            formatting_issues=[],
            metadata_diff="",
            errors=[],
            warnings=[],
            pdf_processor_version="1.0",
            processing_config={},
            system_info={}
        )
        
        # Test regression detection
        is_regression = validation_system._detect_regression(benchmark_id, current_result)
        
        assert is_regression is False
    
    def test_calculate_trend_improving(self, validation_system):
        """Test trend calculation for improving scores."""
        scores = [0.7, 0.75, 0.8, 0.85, 0.9]
        trend = validation_system._calculate_trend(scores)
        assert trend == "improving"
    
    def test_calculate_trend_declining(self, validation_system):
        """Test trend calculation for declining scores."""
        scores = [0.9, 0.85, 0.8, 0.75, 0.7]
        trend = validation_system._calculate_trend(scores)
        assert trend == "declining"
    
    def test_calculate_trend_stable(self, validation_system):
        """Test trend calculation for stable scores."""
        scores = [0.85, 0.84, 0.86, 0.85, 0.85]
        trend = validation_system._calculate_trend(scores)
        assert trend == "stable"
    
    def test_run_full_validation_suite(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test running full validation suite."""
        # Create multiple benchmarks
        for i in range(3):
            validation_system.create_benchmark(
                name=f"Test Benchmark {i}",
                description=f"Test benchmark {i}",
                source_pdf_path=str(sample_pdf_file),
                expected_markdown_path=str(sample_markdown_file),
                content_type="academic",
                difficulty_level="medium"
            )
        
        # Mock PDF processor
        def mock_pdf_processor(pdf_path, config):
            return {
                "markdown": "# Test Document\n\nThis is a test document.\n\n## Section 1\n\nContent here.\n",
                "metadata": {"title": "Test Document", "pages": 1}
            }
        
        # Run full validation suite
        results = validation_system.run_full_validation_suite(
            mock_pdf_processor,
            {"test_config": "value"}
        )
        
        assert len(results) == 3
        assert all(isinstance(result, ValidationResult) for result in results)
        assert all(result.accuracy_score > 0.8 for result in results)
    
    def test_run_full_validation_suite_with_filter(self, validation_system, sample_pdf_file, sample_markdown_file):
        """Test running validation suite with filters."""
        # Create benchmarks with different content types
        validation_system.create_benchmark(
            name="Academic Benchmark",
            description="Academic content",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            content_type="academic"
        )
        
        validation_system.create_benchmark(
            name="Technical Benchmark",
            description="Technical content",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            content_type="technical"
        )
        
        # Mock PDF processor
        def mock_pdf_processor(pdf_path, config):
            return {
                "markdown": "# Test Document\n\nContent here.\n",
                "metadata": {"title": "Test", "pages": 1}
            }
        
        # Run validation suite with filter
        results = validation_system.run_full_validation_suite(
            mock_pdf_processor,
            {},
            {"content_type": "academic"}
        )
        
        assert len(results) == 1
        assert results[0].benchmark_id in [bid for bid, b in validation_system.benchmarks.items() if b.content_type == "academic"]
    
    def test_collect_system_info(self, validation_system):
        """Test system information collection."""
        system_info = validation_system._collect_system_info()
        
        assert "python_version" in system_info
        assert "platform" in system_info
        assert "processor_version" in system_info
        assert "timestamp" in system_info
        assert "cpu_count" in system_info
    
    def test_check_quality_interface(self, validation_system):
        """Test quality checking interface implementation."""
        # Test with validation results
        mock_results = [Mock(accuracy_score=0.85), Mock(accuracy_score=0.9)]
        content = {"validation_results": mock_results}
        
        quality_score = validation_system.check_quality(content)
        assert quality_score == 0.875  # Average of 0.85 and 0.9
        
        # Test with empty content
        empty_quality = validation_system.check_quality({})
        assert empty_quality == 0.0
    
    def test_validate_input_interface(self, validation_system, sample_pdf_file):
        """Test input validation interface implementation."""
        # Valid input
        valid_input = {
            "pdf_path": str(sample_pdf_file),
            "expected_output": "test"
        }
        assert validation_system.validate_input(valid_input) is True
        
        # Invalid input - missing file
        invalid_input = {
            "pdf_path": "/nonexistent/file.pdf",
            "expected_output": "test"
        }
        assert validation_system.validate_input(invalid_input) is False
        
        # Invalid input - wrong structure
        wrong_structure = {"wrong": "structure"}
        assert validation_system.validate_input(wrong_structure) is False
    
    def test_validate_output_interface(self, validation_system):
        """Test output validation interface implementation."""
        # Valid ValidationResult
        valid_result = ValidationResult(
            benchmark_id="test",
            test_id="test",
            timestamp=datetime.now(),
            accuracy_score=0.85,
            passed=True,
            processing_time=1.0,
            content_accuracy=0.85,
            structure_accuracy=0.85,
            formatting_accuracy=0.85,
            metadata_accuracy=0.85,
            content_diff="",
            structure_diff="",
            formatting_issues=[],
            metadata_diff="",
            errors=[],
            warnings=[],
            pdf_processor_version="1.0",
            processing_config={},
            system_info={}
        )
        assert validation_system.validate_output(valid_result) is True
        
        # Valid ValidationReport
        valid_report = ValidationReport(
            report_id="test_report",
            generation_date=datetime.now(),
            report_period=(datetime.now(), datetime.now()),
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            overall_accuracy=0.85,
            pass_rate=80.0,
            benchmark_results={},
            performance_by_difficulty={},
            performance_by_content_type={},
            accuracy_trends=[],
            regression_analysis={},
            improvement_opportunities=[],
            quality_scores={},
            quality_distribution={},
            recommendations=[],
            action_items=[]
        )
        assert validation_system.validate_output(valid_report) is True
        
        # Invalid output
        invalid_output = {"invalid": "output"}
        assert validation_system.validate_output(invalid_output) is False


class TestValidationReportGeneration:
    """Test validation report generation functionality."""
    
    @pytest.fixture
    def validation_system_with_data(self, validation_config):
        """Create validation system with sample data."""
        system = QualityValidationSystem(validation_config)
        
        # Add sample validation results
        base_time = datetime.now()
        for i in range(10):
            result = ValidationResult(
                benchmark_id=f"benchmark_{i % 3}",
                test_id=f"test_{i}",
                timestamp=base_time - timedelta(days=i),
                accuracy_score=0.8 + (i % 3) * 0.05,
                passed=(0.8 + (i % 3) * 0.05) >= 0.85,
                processing_time=1.0 + i * 0.1,
                content_accuracy=0.8 + (i % 3) * 0.05,
                structure_accuracy=0.85,
                formatting_accuracy=0.8,
                metadata_accuracy=0.9,
                content_diff="",
                structure_diff="",
                formatting_issues=[],
                metadata_diff="",
                errors=[],
                warnings=[],
                pdf_processor_version="1.0",
                processing_config={},
                system_info={}
            )
            system.validation_results.append(result)
            system.validation_history[result.benchmark_id].append(result)
        
        return system
    
    def test_generate_validation_report_with_data(self, validation_system_with_data):
        """Test validation report generation with sample data."""
        report = validation_system_with_data.generate_validation_report(period_days=15)
        
        assert report.total_tests == 10
        assert report.overall_accuracy > 0.7
        assert report.pass_rate >= 0.0
        assert len(report.benchmark_results) > 0
        assert len(report.recommendations) > 0
        assert len(report.action_items) > 0
    
    def test_generate_empty_validation_report(self, validation_config):
        """Test validation report generation with no data."""
        system = QualityValidationSystem(validation_config)
        report = system.generate_validation_report(period_days=30)
        
        assert report.total_tests == 0
        assert report.overall_accuracy == 0.0
        assert report.pass_rate == 0.0
        assert len(report.recommendations) > 0  # Should have setup recommendations
        assert "No validation data available" in report.improvement_opportunities[0]
    
    def test_detect_quality_regressions_with_data(self, validation_system_with_data):
        """Test regression detection with sample data."""
        # Add some regressing results
        base_time = datetime.now()
        regressing_results = [
            ValidationResult(
                benchmark_id="benchmark_0",
                test_id=f"regress_{i}",
                timestamp=base_time - timedelta(hours=i),
                accuracy_score=0.7 - i * 0.02,  # Declining
                passed=False,
                processing_time=1.0,
                content_accuracy=0.7,
                structure_accuracy=0.7,
                formatting_accuracy=0.7,
                metadata_accuracy=0.7,
                content_diff="",
                structure_diff="",
                formatting_issues=[],
                metadata_diff="",
                errors=[],
                warnings=[],
                pdf_processor_version="1.0",
                processing_config={},
                system_info={}
            )
            for i in range(3)
        ]
        
        validation_system_with_data.validation_results.extend(regressing_results)
        for result in regressing_results:
            validation_system_with_data.validation_history[result.benchmark_id].append(result)
        
        regression_analysis = validation_system_with_data.detect_quality_regressions(window_days=1)
        
        assert "regression_count" in regression_analysis
        assert "system_trend" in regression_analysis
        assert "recommendations" in regression_analysis
    
    def test_detect_quality_regressions_no_data(self, validation_config):
        """Test regression detection with no data."""
        system = QualityValidationSystem(validation_config)
        regression_analysis = system.detect_quality_regressions(window_days=7)
        
        assert regression_analysis["regressions"] == []
        assert "No recent validation results" in regression_analysis["message"]


class TestContinuousValidation:
    """Test continuous validation functionality."""
    
    def test_setup_continuous_validation(self, validation_config):
        """Test continuous validation setup."""
        system = QualityValidationSystem(validation_config)
        
        def mock_processor(pdf_path, config):
            return {"markdown": "test", "metadata": {}}
        
        # Setup continuous validation
        system.setup_continuous_validation(
            mock_processor,
            {"param": "value"},
            "daily"
        )
        
        # Check that configuration file was created
        config_file = system.validation_dir / "continuous_validation_config.json"
        assert config_file.exists()
        
        # Check configuration content
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        assert config["enabled"] is True
        assert config["schedule"] == "daily"
        assert config["processor_config"] == {"param": "value"}


class TestValidationSystemPersistence:
    """Test validation system data persistence."""
    
    def test_save_and_load_benchmarks(self, validation_config, sample_pdf_file, sample_markdown_file):
        """Test saving and loading benchmarks."""
        # Create system and add benchmark
        system1 = QualityValidationSystem(validation_config)
        benchmark = system1.create_benchmark(
            name="Persistent Benchmark",
            description="Test persistence",
            source_pdf_path=str(sample_pdf_file),
            expected_markdown_path=str(sample_markdown_file),
            tags=["persistence", "test"]
        )
        
        # Create new system instance (simulates restart)
        system2 = QualityValidationSystem(validation_config)
        
        # Check that benchmark was loaded
        assert benchmark.benchmark_id in system2.benchmarks
        loaded_benchmark = system2.benchmarks[benchmark.benchmark_id]
        assert loaded_benchmark.name == "Persistent Benchmark"
        assert loaded_benchmark.tags == ["persistence", "test"]
    
    def test_save_validation_results(self, validation_config):
        """Test saving validation results."""
        system = QualityValidationSystem(validation_config)
        
        # Add sample validation result
        result = ValidationResult(
            benchmark_id="test_benchmark",
            test_id="test_001",
            timestamp=datetime.now(),
            accuracy_score=0.85,
            passed=True,
            processing_time=2.0,
            content_accuracy=0.85,
            structure_accuracy=0.85,
            formatting_accuracy=0.85,
            metadata_accuracy=0.85,
            content_diff="",
            structure_diff="",
            formatting_issues=[],
            metadata_diff="",
            errors=[],
            warnings=[],
            pdf_processor_version="1.0",
            processing_config={},
            system_info={}
        )
        
        system.validation_results.append(result)
        system._save_validation_results()
        
        # Check that results file was created
        results_files = list(system.results_dir.glob("validation_results_*.json"))
        assert len(results_files) > 0
        
        # Check file content
        with open(results_files[0], 'r') as f:
            results_data = json.load(f)
        
        assert len(results_data["results"]) == 1
        assert results_data["results"][0]["benchmark_id"] == "test_benchmark"


if __name__ == "__main__":
    pytest.main([__file__])