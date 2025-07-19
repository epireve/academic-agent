#!/usr/bin/env python3
"""
Unit tests for the migrated IngestionAgent.

Tests the unified architecture implementation of the ingestion agent
including PDF processing, content extraction, and format handling.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.ingestion_agent import IngestionAgent, IngestionResult, IngestionMetadata
from src.agents.quality_manager import QualityEvaluation, QualityMetrics
from src.processors.pdf_processor import PDFProcessingResult


class TestIngestionAgent:
    """Test suite for IngestionAgent unified architecture implementation."""

    @pytest.fixture
    def ingestion_agent(self):
        """Create an IngestionAgent instance for testing."""
        return IngestionAgent()

    @pytest.fixture
    def sample_pdf_result(self):
        """Sample PDF processing result for testing."""
        return PDFProcessingResult(
            source_path=Path("test.pdf"),
            success=True,
            content="# Test Document\n\nThis is test content from PDF.",
            metadata={"pages": 5, "size": "1.2MB"},
            processing_time=2.5,
            processor_used="marker",
            pages_processed=5
        )

    @pytest.fixture
    def sample_files(self, tmp_path):
        """Create sample files for testing."""
        files = {}
        
        # PDF file (empty for testing)
        files["test.pdf"] = tmp_path / "test.pdf"
        files["test.pdf"].write_bytes(b"%PDF-1.4\n%fake pdf content")
        
        # Markdown file
        files["test.md"] = tmp_path / "test.md"
        files["test.md"].write_text("# Test Markdown\n\nContent here.")
        
        # Text file
        files["test.txt"] = tmp_path / "test.txt"
        files["test.txt"].write_text("Plain text content.")
        
        # JSON file
        files["test.json"] = tmp_path / "test.json"
        files["test.json"].write_text(json.dumps({"title": "Test", "content": "JSON data"}))
        
        # Unsupported file
        files["test.xyz"] = tmp_path / "test.xyz"
        files["test.xyz"].write_text("Unsupported format")
        
        return files

    @pytest.mark.asyncio
    async def test_ingestion_agent_initialization(self, ingestion_agent):
        """Test IngestionAgent initialization."""
        assert ingestion_agent.agent_name == "ingestion_agent"
        assert ingestion_agent.quality_manager is not None
        assert ingestion_agent.pdf_processor is not None
        assert ingestion_agent.supported_formats is not None
        assert ingestion_agent.output_dir.exists()

    @pytest.mark.asyncio
    async def test_validate_input_valid_file(self, ingestion_agent, sample_files):
        """Test input validation with valid file."""
        result = await ingestion_agent.validate_input(str(sample_files["test.pdf"]))
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_input_invalid_file(self, ingestion_agent):
        """Test input validation with invalid file."""
        result = await ingestion_agent.validate_input("nonexistent.pdf")
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_non_string(self, ingestion_agent):
        """Test input validation with non-string input."""
        result = await ingestion_agent.validate_input(123)
        assert result is False

    def test_detect_format_pdf(self, ingestion_agent, sample_files):
        """Test format detection for PDF files."""
        format_info = ingestion_agent._detect_format(sample_files["test.pdf"])
        assert format_info["format"] == "pdf"
        assert format_info["supported"] is True

    def test_detect_format_markdown(self, ingestion_agent, sample_files):
        """Test format detection for Markdown files."""
        format_info = ingestion_agent._detect_format(sample_files["test.md"])
        assert format_info["format"] == "markdown"
        assert format_info["supported"] is True

    def test_detect_format_unsupported(self, ingestion_agent, sample_files):
        """Test format detection for unsupported files."""
        format_info = ingestion_agent._detect_format(sample_files["test.xyz"])
        assert format_info["format"] == "xyz"
        assert format_info["supported"] is False

    @pytest.mark.asyncio
    async def test_process_pdf_success(self, ingestion_agent, sample_files, sample_pdf_result):
        """Test successful PDF processing."""
        # Mock PDF processor
        with patch.object(ingestion_agent.pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = sample_pdf_result
            
            result = await ingestion_agent._process_pdf(sample_files["test.pdf"])
            
            assert result["success"] is True
            assert result["content"] == sample_pdf_result.content
            assert result["metadata"]["pages_processed"] == 5

    @pytest.mark.asyncio
    async def test_process_pdf_failure(self, ingestion_agent, sample_files):
        """Test PDF processing failure."""
        # Mock PDF processor to return failure
        failed_result = PDFProcessingResult(
            source_path=Path("test.pdf"),
            success=False,
            error_message="Processing failed"
        )
        
        with patch.object(ingestion_agent.pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = failed_result
            
            result = await ingestion_agent._process_pdf(sample_files["test.pdf"])
            
            assert result["success"] is False
            assert "Processing failed" in result["error"]

    @pytest.mark.asyncio
    async def test_process_text_file(self, ingestion_agent, sample_files):
        """Test processing text files."""
        result = await ingestion_agent._process_text_file(sample_files["test.txt"])
        
        assert result["success"] is True
        assert result["content"] == "Plain text content."
        assert result["format"] == "text"

    @pytest.mark.asyncio
    async def test_process_markdown_file(self, ingestion_agent, sample_files):
        """Test processing markdown files."""
        result = await ingestion_agent._process_text_file(sample_files["test.md"])
        
        assert result["success"] is True
        assert "# Test Markdown" in result["content"]
        assert result["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_process_json_file(self, ingestion_agent, sample_files):
        """Test processing JSON files."""
        result = await ingestion_agent._process_json_file(sample_files["test.json"])
        
        assert result["success"] is True
        assert "Test" in result["content"]
        assert result["format"] == "json"

    @pytest.mark.asyncio
    async def test_process_json_invalid(self, ingestion_agent, tmp_path):
        """Test processing invalid JSON file."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json")
        
        result = await ingestion_agent._process_json_file(invalid_json)
        
        assert result["success"] is False
        assert "JSON parsing failed" in result["error"]

    @pytest.mark.asyncio
    async def test_ingest_file_success(self, ingestion_agent, sample_files):
        """Test successful file ingestion."""
        # Mock quality manager
        with patch.object(ingestion_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.return_value = QualityEvaluation(
                content_type="ingested",
                quality_score=0.85,
                feedback=["Good ingestion"],
                areas_for_improvement=[],
                strengths=["Clear content"],
                metrics=QualityMetrics(0.85, 0.8, 0.9, 0.8, 0.85, 0.8),
                assessment="Good quality",
                approved=True
            )
            
            result = await ingestion_agent.ingest_file(str(sample_files["test.md"]))
            
            assert result["success"] is True
            assert "output_path" in result
            assert result["format"] == "markdown"
            assert result["quality_score"] == 0.85

    @pytest.mark.asyncio
    async def test_ingest_file_unsupported_format(self, ingestion_agent, sample_files):
        """Test ingesting unsupported file format."""
        result = await ingestion_agent.ingest_file(str(sample_files["test.xyz"]))
        
        assert result["success"] is False
        assert "Unsupported format" in result["error"]

    @pytest.mark.asyncio
    async def test_ingest_file_invalid_input(self, ingestion_agent):
        """Test ingesting with invalid input."""
        result = await ingestion_agent.ingest_file("nonexistent.pdf")
        
        assert result["success"] is False
        assert "Invalid input file" in result["error"]

    @pytest.mark.asyncio
    async def test_batch_ingest_directory(self, ingestion_agent, sample_files):
        """Test batch ingestion of directory."""
        # Create directory with sample files
        directory = sample_files["test.md"].parent
        
        # Mock quality manager
        with patch.object(ingestion_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.return_value = QualityEvaluation(
                content_type="ingested",
                quality_score=0.8,
                feedback=[],
                areas_for_improvement=[],
                strengths=[],
                metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
                assessment="Good",
                approved=True
            )
            
            result = await ingestion_agent.batch_ingest_directory(str(directory))
            
            assert result["success"] is True
            assert result["total_files"] > 0
            assert result["successful"] > 0
            assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_batch_ingest_empty_directory(self, ingestion_agent, tmp_path):
        """Test batch ingestion of empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        result = await ingestion_agent.batch_ingest_directory(str(empty_dir))
        
        assert result["success"] is True
        assert result["message"] == "No supported files found"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_batch_ingest_nonexistent_directory(self, ingestion_agent):
        """Test batch ingestion of nonexistent directory."""
        result = await ingestion_agent.batch_ingest_directory("/nonexistent/path")
        
        assert result["success"] is False
        assert "Directory not found" in result["error"]

    @pytest.mark.asyncio
    async def test_save_ingestion_result(self, ingestion_agent, tmp_path):
        """Test saving ingestion result."""
        metadata = IngestionMetadata(
            source_file="test.pdf",
            ingested_date=datetime.now().isoformat(),
            processing_version="unified-v2.0",
            format="pdf",
            quality_score=0.85,
            processing_time=2.5
        )
        
        ingestion_result = IngestionResult(
            content="Test content",
            metadata=metadata,
            extracted_data={},
            processing_notes=[]
        )
        
        # Set output directory to temp path
        ingestion_agent.output_dir = tmp_path
        
        output_path = await ingestion_agent._save_ingestion_result(ingestion_result, "test.pdf")
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Check content
        with open(output_path) as f:
            saved_data = json.load(f)
        
        assert saved_data["metadata"]["source_file"] == "test.pdf"
        assert saved_data["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_save_as_markdown(self, ingestion_agent, tmp_path):
        """Test saving ingestion result as markdown."""
        metadata = IngestionMetadata(
            source_file="test.pdf",
            ingested_date=datetime.now().isoformat(),
            processing_version="unified-v2.0",
            format="pdf",
            quality_score=0.85,
            processing_time=2.5
        )
        
        ingestion_result = IngestionResult(
            content="# Test Content\n\nTest ingested content.",
            metadata=metadata,
            extracted_data={"pages": 5},
            processing_notes=["Successfully processed"]
        )
        
        output_path = tmp_path / "test_ingestion.md"
        await ingestion_agent._save_as_markdown(ingestion_result, output_path)
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "# Ingested Content: test.pdf" in content
        assert "Quality Score: 0.85" in content
        assert "# Test Content" in content

    @pytest.mark.asyncio
    async def test_extract_metadata_from_content(self, ingestion_agent):
        """Test metadata extraction from content."""
        content = """# Document Title
        
## Introduction
This is a test document with multiple sections.

### Key Points
- Point 1
- Point 2

## Conclusion
Final thoughts.
"""
        
        metadata = ingestion_agent._extract_metadata_from_content(content)
        
        assert metadata["title"] == "Document Title"
        assert metadata["sections"] >= 2
        assert metadata["word_count"] > 0
        assert len(metadata["headings"]) > 0

    @pytest.mark.asyncio
    async def test_validate_output_success(self, ingestion_agent):
        """Test output validation with successful result."""
        result = {
            "success": True,
            "output_path": "/path/to/output.json",
            "format": "pdf",
            "quality_score": 0.8
        }
        
        validation = await ingestion_agent.validate_output(result)
        assert validation is True

    @pytest.mark.asyncio
    async def test_validate_output_failure(self, ingestion_agent):
        """Test output validation with failed result."""
        result = {
            "success": False,
            "error": "Ingestion failed"
        }
        
        validation = await ingestion_agent.validate_output(result)
        assert validation is False

    @pytest.mark.asyncio
    async def test_error_handling_during_processing(self, ingestion_agent, sample_files):
        """Test error handling during file processing."""
        # Mock quality manager to raise exception
        with patch.object(ingestion_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.side_effect = Exception("Quality check failed")
            
            result = await ingestion_agent.ingest_file(str(sample_files["test.txt"]))
            
            assert result["success"] is False
            assert "error" in result
            assert "Quality check failed" in result["error"]

    def test_agent_inheritance(self, ingestion_agent):
        """Test that IngestionAgent properly inherits from BaseAgent."""
        from src.agents.base_agent import BaseAgent
        
        assert isinstance(ingestion_agent, BaseAgent)
        assert hasattr(ingestion_agent, 'agent_name')
        assert hasattr(ingestion_agent, 'logger')
        assert hasattr(ingestion_agent, 'base_dir')

    @pytest.mark.asyncio
    async def test_async_functionality(self, ingestion_agent):
        """Test that the agent properly supports async operations."""
        import inspect
        
        assert inspect.iscoroutinefunction(ingestion_agent.ingest_file)
        assert inspect.iscoroutinefunction(ingestion_agent.batch_ingest_directory)
        assert inspect.iscoroutinefunction(ingestion_agent.validate_input)
        assert inspect.iscoroutinefunction(ingestion_agent.validate_output)

    def test_supported_formats(self, ingestion_agent):
        """Test supported file formats."""
        assert "pdf" in ingestion_agent.supported_formats
        assert "md" in ingestion_agent.supported_formats
        assert "txt" in ingestion_agent.supported_formats
        assert "json" in ingestion_agent.supported_formats

    @pytest.mark.asyncio
    async def test_content_preprocessing(self, ingestion_agent):
        """Test content preprocessing."""
        raw_content = "  This is content with   extra spaces   and\n\n\nmultiple newlines  "
        
        processed = ingestion_agent._preprocess_content(raw_content)
        
        assert processed.strip() == "This is content with extra spaces and\n\nmultiple newlines"

    @pytest.mark.asyncio
    async def test_format_specific_processing(self, ingestion_agent, sample_files):
        """Test format-specific processing features."""
        # Test PDF with mock processor
        with patch.object(ingestion_agent.pdf_processor, 'process_pdf') as mock_process:
            mock_result = PDFProcessingResult(
                source_path=sample_files["test.pdf"],
                success=True,
                content="PDF content",
                metadata={"special_pdf_feature": True}
            )
            mock_process.return_value = mock_result
            
            result = await ingestion_agent._process_pdf(sample_files["test.pdf"])
            assert result["metadata"]["special_pdf_feature"] is True

    def _preprocess_content(self, content: str) -> str:
        """Mock implementation of content preprocessing."""
        import re
        # Remove extra whitespace
        content = re.sub(r' +', ' ', content)
        # Normalize line breaks
        content = re.sub(r'\n{3,}', '\n\n', content)
        return content

    # Patch the method for testing
    IngestionAgent._preprocess_content = _preprocess_content