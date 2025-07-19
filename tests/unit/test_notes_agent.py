#!/usr/bin/env python3
"""
Unit tests for the migrated NotesAgent.

Tests the unified architecture implementation of the notes generation agent
including async functionality, quality management, and error handling.
"""

import unittest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.notes_agent import NotesAgent, NotesData, NotesSection, NotesMetadata
from src.agents.quality_manager import QualityEvaluation, QualityMetrics


class TestNotesAgent(unittest.TestCase):
    """Test suite for NotesAgent unified architecture implementation."""

    def setUp(self):
        """Create a NotesAgent instance for testing."""
        self.notes_agent = NotesAgent()

    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing."""
        return """# Test Document

## Introduction
This is a test document for the academic agent system.

## Main Content

### Section 1
Here is some content for section 1 with important information.
- Key point 1
- Key point 2

### Section 2
Here is some content for section 2 with more details.
1. Numbered point 1
2. Numbered point 2

## Conclusion
This concludes the test document.
"""

    @pytest.fixture
    def sample_json_content(self):
        """Sample JSON content for testing."""
        return {
            "title": "Test Document",
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "This is an introduction"
                },
                {
                    "heading": "Main Content",
                    "content": "This is the main content"
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_notes_agent_initialization(self, notes_agent):
        """Test NotesAgent initialization."""
        assert notes_agent.agent_name == "notes_agent"
        assert notes_agent.quality_manager is not None
        assert notes_agent.ai_enhancement_enabled is True
        assert notes_agent.quality_threshold == 0.7
        assert notes_agent.notes_output_dir.exists()

    @pytest.mark.asyncio
    async def test_validate_input_valid_file(self, notes_agent, tmp_path):
        """Test input validation with valid file."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test content")
        
        result = await notes_agent.validate_input(str(test_file))
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_input_invalid_file(self, notes_agent):
        """Test input validation with invalid file."""
        result = await notes_agent.validate_input("nonexistent_file.md")
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_input_non_string(self, notes_agent):
        """Test input validation with non-string input."""
        result = await notes_agent.validate_input(123)
        assert result is False

    @pytest.mark.asyncio
    async def test_read_content_markdown(self, notes_agent, tmp_path, sample_markdown_content):
        """Test reading markdown content."""
        test_file = tmp_path / "test.md"
        test_file.write_text(sample_markdown_content)
        
        content = await notes_agent._read_content(str(test_file))
        assert content == sample_markdown_content

    @pytest.mark.asyncio
    async def test_read_content_json(self, notes_agent, tmp_path, sample_json_content):
        """Test reading JSON content."""
        test_file = tmp_path / "test.json"
        test_file.write_text(json.dumps(sample_json_content))
        
        content = await notes_agent._read_content(str(test_file))
        assert "Test Document" in content
        assert "Introduction" in content

    @pytest.mark.asyncio
    async def test_read_content_text(self, notes_agent, tmp_path):
        """Test reading plain text content."""
        test_content = "This is plain text content"
        test_file = tmp_path / "test.txt"
        test_file.write_text(test_content)
        
        content = await notes_agent._read_content(str(test_file))
        assert content == test_content

    @pytest.mark.asyncio
    async def test_generate_sections_from_markdown(self, notes_agent, sample_markdown_content):
        """Test section generation from markdown content."""
        sections = await notes_agent._generate_sections(sample_markdown_content)
        
        assert len(sections) >= 3  # Introduction, Main Content, Conclusion
        
        # Check first section
        intro_section = sections[0]
        assert intro_section.title == "Test Document"
        assert "test document" in intro_section.content.lower()
        
        # Check that key points are extracted
        main_section = next((s for s in sections if "Main Content" in s.title), None)
        assert main_section is not None
        assert len(main_section.key_points) > 0

    def test_create_section(self, notes_agent):
        """Test section creation with analysis."""
        title = "Test Section"
        content = """This is a test section with content.
        
- First key point
- Second key point

Some more content with important concepts like Analysis and Processing.
"""
        
        section = notes_agent._create_section(title, content)
        
        assert section.title == title
        assert section.content == content
        assert section.level == 1
        assert len(section.key_points) == 2
        assert "First key point" in section.key_points[0]
        assert len(section.concepts) > 0

    def test_extract_concepts(self, notes_agent):
        """Test concept extraction from content."""
        content = "This document discusses Machine Learning and Data Processing with Algorithms."
        
        concepts = notes_agent._extract_concepts(content)
        
        assert len(concepts) > 0
        # Should extract longer words as concepts
        concept_text = " ".join(concepts).lower()
        assert any(len(word) > 5 for word in concepts)

    @pytest.mark.asyncio
    async def test_process_file_success(self, notes_agent, tmp_path, sample_markdown_content):
        """Test successful file processing."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text(sample_markdown_content)
        
        # Mock quality manager
        with patch.object(notes_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.return_value = QualityEvaluation(
                content_type="notes",
                quality_score=0.85,
                feedback=["Good quality content"],
                areas_for_improvement=[],
                strengths=["Well structured"],
                metrics=QualityMetrics(0.85, 0.8, 0.9, 0.8, 0.85, 0.8),
                assessment="Good quality",
                approved=True
            )
            
            result = await notes_agent.process_file(str(test_file))
            
            assert result["success"] is True
            assert "output_path" in result
            assert result["quality_score"] == 0.85
            assert result["sections_count"] > 0
            assert result["processing_time"] > 0

    @pytest.mark.asyncio
    async def test_process_file_invalid_input(self, notes_agent):
        """Test file processing with invalid input."""
        result = await notes_agent.process_file("nonexistent_file.md")
        
        assert result["success"] is False
        assert "error" in result
        assert "Invalid input file" in result["error"]

    @pytest.mark.asyncio
    async def test_save_notes(self, notes_agent, tmp_path):
        """Test saving notes data."""
        # Create test notes data
        metadata = NotesMetadata(
            source_file="test.md",
            generated_date=datetime.now().isoformat(),
            processing_version="unified-v2.0",
            quality_score=0.8,
            processing_time=1.5
        )
        
        sections = [
            NotesSection(
                title="Test Section",
                content="Test content",
                level=1,
                key_points=["Point 1", "Point 2"],
                summary="Test summary",
                concepts=["Concept1", "Concept2"]
            )
        ]
        
        notes_data = NotesData(
            sections=sections,
            metadata=metadata,
            outline=[],
            ai_enhancements={}
        )
        
        # Set output directory to temp path
        notes_agent.notes_output_dir = tmp_path
        
        output_path = await notes_agent._save_notes(notes_data, "test.md")
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Check markdown file was also created
        md_path = output_path.with_suffix('.md')
        assert md_path.exists()
        
        # Verify content
        with open(output_path) as f:
            saved_data = json.load(f)
        
        assert saved_data["metadata"]["source_file"] == "test.md"
        assert len(saved_data["sections"]) == 1

    @pytest.mark.asyncio
    async def test_save_as_markdown(self, notes_agent, tmp_path):
        """Test saving notes as markdown."""
        metadata = NotesMetadata(
            source_file="test.md",
            generated_date=datetime.now().isoformat(),
            processing_version="unified-v2.0",
            quality_score=0.8,
            processing_time=1.5
        )
        
        sections = [
            NotesSection(
                title="Test Section",
                content="Test content",
                level=1,
                key_points=["Point 1", "Point 2"],
                summary="Test summary",
                concepts=["Concept1", "Concept2"]
            )
        ]
        
        notes_data = NotesData(
            sections=sections,
            metadata=metadata,
            outline=[],
            ai_enhancements={}
        )
        
        output_path = tmp_path / "test_notes.md"
        await notes_agent._save_as_markdown(notes_data, output_path)
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "# Notes: test.md" in content
        assert "Quality Score: 0.80" in content
        assert "## Test Section" in content
        assert "**Key Points:**" in content
        assert "**Key Concepts:**" in content

    @pytest.mark.asyncio
    async def test_batch_process_directory(self, notes_agent, tmp_path):
        """Test batch processing of directory."""
        # Create test files
        (tmp_path / "test1.md").write_text("# Test 1\nContent 1")
        (tmp_path / "test2.txt").write_text("Test 2 content")
        (tmp_path / "test3.json").write_text('{"title": "Test 3"}')
        (tmp_path / "ignored.pdf").write_text("Binary content")  # Should be ignored
        
        # Mock quality manager
        with patch.object(notes_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.return_value = QualityEvaluation(
                content_type="notes",
                quality_score=0.8,
                feedback=[],
                areas_for_improvement=[],
                strengths=[],
                metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
                assessment="Good",
                approved=True
            )
            
            result = await notes_agent.batch_process_directory(str(tmp_path))
            
            assert result["success"] is True
            assert result["total_processed"] == 3  # Only supported formats
            assert result["successful"] == 3
            assert result["failed"] == 0
            assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_process_empty_directory(self, notes_agent, tmp_path):
        """Test batch processing of empty directory."""
        result = await notes_agent.batch_process_directory(str(tmp_path))
        
        assert result["success"] is True
        assert result["message"] == "No supported files found"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_batch_process_nonexistent_directory(self, notes_agent):
        """Test batch processing of nonexistent directory."""
        result = await notes_agent.batch_process_directory("/nonexistent/path")
        
        assert result["success"] is False
        assert "Directory not found" in result["error"]

    @pytest.mark.asyncio
    async def test_validate_output_success(self, notes_agent):
        """Test output validation with successful result."""
        result = {
            "success": True,
            "output_path": "/path/to/output.json",
            "quality_score": 0.8
        }
        
        validation = await notes_agent.validate_output(result)
        assert validation is True

    @pytest.mark.asyncio
    async def test_validate_output_failure(self, notes_agent):
        """Test output validation with failed result."""
        result = {
            "success": False,
            "error": "Processing failed"
        }
        
        validation = await notes_agent.validate_output(result)
        assert validation is False

    @pytest.mark.asyncio
    async def test_validate_output_invalid_format(self, notes_agent):
        """Test output validation with invalid format."""
        result = "invalid result format"
        
        validation = await notes_agent.validate_output(result)
        assert validation is False

    @pytest.mark.asyncio
    async def test_error_handling_during_processing(self, notes_agent, tmp_path):
        """Test error handling during file processing."""
        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test")
        
        # Mock quality manager to raise exception
        with patch.object(notes_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.side_effect = Exception("Quality check failed")
            
            result = await notes_agent.process_file(str(test_file))
            
            assert result["success"] is False
            assert "error" in result
            assert "Quality check failed" in result["error"]

    def test_agent_inheritance(self, notes_agent):
        """Test that NotesAgent properly inherits from BaseAgent."""
        from src.agents.base_agent import BaseAgent
        
        assert isinstance(notes_agent, BaseAgent)
        assert hasattr(notes_agent, 'agent_name')
        assert hasattr(notes_agent, 'logger')
        assert hasattr(notes_agent, 'base_dir')

    @pytest.mark.asyncio
    async def test_async_functionality(self, notes_agent):
        """Test that the agent properly supports async operations."""
        # Test that async methods are actually coroutines
        import inspect
        
        assert inspect.iscoroutinefunction(notes_agent.process_file)
        assert inspect.iscoroutinefunction(notes_agent.batch_process_directory)
        assert inspect.iscoroutinefunction(notes_agent.validate_input)
        assert inspect.iscoroutinefunction(notes_agent.validate_output)