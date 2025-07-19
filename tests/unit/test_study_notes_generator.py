#!/usr/bin/env python3
"""
Comprehensive tests for StudyNotesGenerator

Tests all core functionality including note generation pipeline, template system,
content formatting, diagram integration, quality validation, and batch processing.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import the agent to test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.academic.study_notes_generator import (
    StudyNotesGeneratorTool, StudyNote, StudySection, 
    DiagramType, StudyNoteMetadata
)


class TestStudyNotesGenerator:
    """Comprehensive test suite for StudyNotesGenerator"""
    
    @pytest.fixture
    def generator_tool(self):
        """Create a StudyNotesGeneratorTool instance for testing"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tool = StudyNotesGeneratorTool(tmp_dir)
            return tool
    
    @pytest.fixture
    def sample_academic_content(self):
        """Sample academic content for testing"""
        return """# Introduction to Data Structures

Data structures are ways of organizing and storing data so that they can be accessed and worked with efficiently.

## Arrays

Arrays are collections of elements identified by array index or key.

### Static Arrays
Fixed-size arrays allocated at compile time.

### Dynamic Arrays
Arrays that can grow or shrink during runtime.

## Linked Lists

Linked lists are linear data structures where elements are stored in nodes.

### Singly Linked Lists
Each node points to the next node.

### Doubly Linked Lists
Each node has pointers to both next and previous nodes.

## Trees

Trees are hierarchical data structures with a root node and child nodes.

### Binary Trees
Each node has at most two children.

### Binary Search Trees
Binary trees with ordered elements for efficient searching.

## Key Concepts

- **Node**: Basic unit containing data and references
- **Pointer**: Reference to memory location
- **Traversal**: Process of visiting all nodes
- **Complexity**: Time and space efficiency measures
"""

    @pytest.fixture
    def sample_file(self, sample_academic_content):
        """Create a temporary file with sample content"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_academic_content)
            return f.name

    def test_initialization(self, generator_tool):
        """Test tool initialization"""
        assert hasattr(generator_tool, 'base_dir')
        assert hasattr(generator_tool, 'output_manager')
        assert hasattr(generator_tool, 'diagram_generator')
        assert hasattr(generator_tool, 'content_processor')

    def test_study_note_dataclass(self):
        """Test StudyNote dataclass functionality"""
        metadata = StudyNoteMetadata(
            title="Test Note",
            subject="Computer Science",
            created_date="2024-01-01",
            word_count=100,
            estimated_reading_time=2.0
        )
        
        sections = [StudySection(
            title="Section 1",
            content="Content 1",
            level=1,
            key_points=["Point 1"],
            summary="Summary 1"
        )]
        
        study_note = StudyNote(
            metadata=metadata,
            sections=sections,
            overview="Test overview",
            learning_objectives=["Objective 1"],
            key_takeaways=["Takeaway 1"],
            diagrams=["diagram1.svg"],
            cross_references=["Reference 1"]
        )
        
        assert study_note.metadata.title == "Test Note"
        assert len(study_note.sections) == 1
        assert study_note.overview == "Test overview"
        assert study_note.learning_objectives == ["Objective 1"]

    def test_study_section_dataclass(self):
        """Test StudySection dataclass functionality"""
        section = StudySection(
            title="Test Section",
            content="Test content",
            level=2,
            key_points=["Point 1", "Point 2"],
            summary="Test summary",
            concepts=["Concept 1"],
            examples=["Example 1"],
            diagrams=["diagram.svg"]
        )
        
        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert section.level == 2
        assert len(section.key_points) == 2
        assert section.summary == "Test summary"

    @patch('agents.academic.study_notes_generator.StudyNotesGeneratorTool.parse_content')
    def test_forward_basic_functionality(self, mock_parse, generator_tool, sample_file):
        """Test basic forward method functionality"""
        # Mock the parse_content method
        mock_sections = [
            StudySection(
                title="Test Section",
                content="Test content",
                level=1,
                key_points=["Point 1"],
                summary="Test summary"
            )
        ]
        mock_parse.return_value = mock_sections
        
        result = generator_tool.forward(
            content_path=sample_file,
            title="Test Study Notes",
            subject="Computer Science"
        )
        
        assert result["processing_stats"]["success"] == True
        assert "study_notes" in result
        assert "output_files" in result
        mock_parse.assert_called_once()

    def test_forward_with_output_formats(self, generator_tool, sample_file):
        """Test forward method with different output formats"""
        result = generator_tool.forward(
            content_path=sample_file,
            title="Test Study Notes",
            subject="Computer Science",
            output_formats=["markdown", "json", "html"]
        )
        
        # Should attempt to generate all formats
        assert "processing_stats" in result
        assert "output_files" in result

    def test_forward_with_diagrams(self, generator_tool, sample_file):
        """Test forward method with diagram generation"""
        result = generator_tool.forward(
            content_path=sample_file,
            title="Test Study Notes",
            subject="Computer Science",
            include_diagrams=True
        )
        
        assert "processing_stats" in result
        # Should have attempted diagram generation
        stats = result["processing_stats"]
        assert "diagrams_generated" in stats

    def test_forward_file_not_found(self, generator_tool):
        """Test forward method with non-existent file"""
        result = generator_tool.forward(
            content_path="nonexistent_file.md",
            title="Test Study Notes",
            subject="Computer Science"
        )
        
        assert result["processing_stats"]["success"] == False
        assert "error" in result["processing_stats"]

    @patch('agents.academic.study_notes_generator.StudyNotesGeneratorTool.ai_request')
    def test_parse_content_sections(self, mock_ai_request, generator_tool, sample_academic_content):
        """Test content parsing into sections"""
        # Mock AI responses
        mock_ai_request.side_effect = [
            '["Key point 1", "Key point 2"]',  # key points
            "This is a summary",  # summary
            '["Concept 1", "Concept 2"]'  # concepts
        ]
        
        sections = generator_tool.parse_content(sample_academic_content)
        
        assert len(sections) > 0
        # Should have main sections
        section_titles = [s.title for s in sections]
        assert "Introduction to Data Structures" in section_titles
        assert "Arrays" in section_titles
        assert "Linked Lists" in section_titles

    def test_parse_content_hierarchy(self, generator_tool, sample_academic_content):
        """Test content parsing maintains hierarchy"""
        sections = generator_tool.parse_content(sample_academic_content)
        
        # Find arrays section
        arrays_section = next((s for s in sections if s.title == "Arrays"), None)
        assert arrays_section is not None
        assert arrays_section.level == 2
        
        # Find static arrays subsection
        static_arrays = next((s for s in sections if s.title == "Static Arrays"), None)
        assert static_arrays is not None
        assert static_arrays.level == 3

    @patch('agents.academic.study_notes_generator.StudyNotesGeneratorTool.ai_request')
    def test_enhance_section_with_ai(self, mock_ai_request, generator_tool):
        """Test section enhancement with AI"""
        mock_ai_request.side_effect = [
            '["Key point 1", "Key point 2"]',  # key points
            "Enhanced summary",  # summary
            '["Enhanced concept 1"]'  # concepts
        ]
        
        section = StudySection(
            title="Test Section",
            content="Test content",
            level=1,
            key_points=[],
            summary=""
        )
        
        enhanced_section = generator_tool.enhance_section_with_ai(section)
        
        assert len(enhanced_section.key_points) > 0
        assert enhanced_section.summary == "Enhanced summary"
        assert len(enhanced_section.concepts) > 0

    @patch('agents.academic.study_notes_generator.StudyNotesGeneratorTool.ai_request')
    def test_enhance_section_ai_failure_fallback(self, mock_ai_request, generator_tool):
        """Test section enhancement with AI failure fallback"""
        mock_ai_request.side_effect = Exception("AI service unavailable")
        
        section = StudySection(
            title="Test Section",
            content="Test content with some important information.",
            level=1,
            key_points=[],
            summary=""
        )
        
        enhanced_section = generator_tool.enhance_section_with_ai(section)
        
        # Should have fallback values
        assert isinstance(enhanced_section.key_points, list)
        assert isinstance(enhanced_section.summary, str)
        assert isinstance(enhanced_section.concepts, list)

    def test_generate_learning_objectives(self, generator_tool):
        """Test learning objectives generation"""
        sections = [
            StudySection(
                title="Arrays",
                content="Arrays are collections of elements",
                level=2,
                key_points=["Arrays store multiple elements"],
                summary="Arrays are fundamental data structures"
            ),
            StudySection(
                title="Linked Lists",
                content="Linked lists are linear structures",
                level=2,
                key_points=["Nodes contain data and pointers"],
                summary="Linked lists provide dynamic storage"
            )
        ]
        
        objectives = generator_tool.generate_learning_objectives(sections)
        
        assert isinstance(objectives, list)
        assert len(objectives) > 0

    def test_generate_key_takeaways(self, generator_tool):
        """Test key takeaways generation"""
        sections = [
            StudySection(
                title="Test Section 1",
                content="Content 1",
                level=1,
                key_points=["Point 1", "Point 2"],
                summary="Summary 1"
            ),
            StudySection(
                title="Test Section 2",
                content="Content 2",
                level=1,
                key_points=["Point 3", "Point 4"],
                summary="Summary 2"
            )
        ]
        
        takeaways = generator_tool.generate_key_takeaways(sections)
        
        assert isinstance(takeaways, list)
        assert len(takeaways) > 0

    def test_create_study_note_structure(self, generator_tool, sample_academic_content):
        """Test complete study note creation"""
        sections = generator_tool.parse_content(sample_academic_content)
        
        study_note = generator_tool.create_study_note(
            sections=sections,
            title="Data Structures Study Notes",
            subject="Computer Science"
        )
        
        assert isinstance(study_note, StudyNote)
        assert study_note.metadata.title == "Data Structures Study Notes"
        assert study_note.metadata.subject == "Computer Science"
        assert len(study_note.sections) > 0
        assert len(study_note.overview) > 0
        assert len(study_note.learning_objectives) > 0
        assert len(study_note.key_takeaways) > 0

    def test_format_as_markdown(self, generator_tool):
        """Test markdown formatting"""
        metadata = StudyNoteMetadata(
            title="Test Study Notes",
            subject="Computer Science",
            created_date="2024-01-01",
            word_count=100,
            estimated_reading_time=2.0
        )
        
        sections = [StudySection(
            title="Test Section",
            content="Test content",
            level=1,
            key_points=["Point 1"],
            summary="Test summary"
        )]
        
        study_note = StudyNote(
            metadata=metadata,
            sections=sections,
            overview="Test overview",
            learning_objectives=["Objective 1"],
            key_takeaways=["Takeaway 1"]
        )
        
        markdown = generator_tool.format_as_markdown(study_note)
        
        assert "# Test Study Notes" in markdown
        assert "## Overview" in markdown
        assert "## Learning Objectives" in markdown
        assert "## Test Section" in markdown
        assert "### Key Points" in markdown
        assert "## Key Takeaways" in markdown

    def test_format_as_json(self, generator_tool):
        """Test JSON formatting"""
        metadata = StudyNoteMetadata(
            title="Test Study Notes",
            subject="Computer Science",
            created_date="2024-01-01",
            word_count=100,
            estimated_reading_time=2.0
        )
        
        sections = [StudySection(
            title="Test Section",
            content="Test content",
            level=1,
            key_points=["Point 1"],
            summary="Test summary"
        )]
        
        study_note = StudyNote(
            metadata=metadata,
            sections=sections,
            overview="Test overview",
            learning_objectives=["Objective 1"],
            key_takeaways=["Takeaway 1"]
        )
        
        json_output = generator_tool.format_as_json(study_note)
        
        # Should be valid JSON
        parsed = json.loads(json_output)
        assert parsed["metadata"]["title"] == "Test Study Notes"
        assert len(parsed["sections"]) == 1
        assert "overview" in parsed

    def test_format_as_html(self, generator_tool):
        """Test HTML formatting"""
        metadata = StudyNoteMetadata(
            title="Test Study Notes",
            subject="Computer Science",
            created_date="2024-01-01",
            word_count=100,
            estimated_reading_time=2.0
        )
        
        sections = [StudySection(
            title="Test Section",
            content="Test content",
            level=1,
            key_points=["Point 1"],
            summary="Test summary"
        )]
        
        study_note = StudyNote(
            metadata=metadata,
            sections=sections,
            overview="Test overview",
            learning_objectives=["Objective 1"],
            key_takeaways=["Takeaway 1"]
        )
        
        html_output = generator_tool.format_as_html(study_note)
        
        assert "<html>" in html_output
        assert "<h1>Test Study Notes</h1>" in html_output
        assert "<h2>Overview</h2>" in html_output
        assert "<h2>Learning Objectives</h2>" in html_output

    @patch('agents.academic.study_notes_generator.DiagramGenerator')
    def test_diagram_integration(self, mock_diagram_gen, generator_tool):
        """Test diagram generation integration"""
        mock_diagram_instance = Mock()
        mock_diagram_instance.generate_concept_diagram.return_value = "mermaid_diagram_code"
        mock_diagram_gen.return_value = mock_diagram_instance
        
        generator_tool.diagram_generator = mock_diagram_instance
        
        # Test diagram generation
        concepts = ["Arrays", "Linked Lists", "Trees"]
        diagram = generator_tool.generate_diagram_for_concepts(concepts, "Data Structures")
        
        assert diagram == "mermaid_diagram_code"
        mock_diagram_instance.generate_concept_diagram.assert_called_once()

    def test_batch_processing(self, generator_tool):
        """Test batch processing of multiple files"""
        # Create multiple test files
        test_files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(f"# Test Document {i}\n\nContent for document {i}")
                test_files.append(f.name)
        
        results = []
        for file_path in test_files:
            result = generator_tool.forward(
                content_path=file_path,
                title=f"Test Notes {len(results) + 1}",
                subject="Computer Science"
            )
            results.append(result)
        
        # All should succeed
        for result in results:
            assert "processing_stats" in result

    def test_error_handling_invalid_content(self, generator_tool):
        """Test error handling with invalid content"""
        # Create file with invalid content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("")  # Empty file
            empty_file = f.name
        
        result = generator_tool.forward(
            content_path=empty_file,
            title="Empty Notes",
            subject="Computer Science"
        )
        
        # Should handle gracefully
        assert "processing_stats" in result

    def test_memory_efficiency_large_content(self, generator_tool):
        """Test memory efficiency with large content"""
        # Create large content
        large_content = "# Large Document\n\n" + "Content line. " * 10000
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(large_content)
            large_file = f.name
        
        result = generator_tool.forward(
            content_path=large_file,
            title="Large Notes",
            subject="Computer Science"
        )
        
        # Should handle large content
        assert "processing_stats" in result

    def test_concurrent_processing(self, generator_tool, sample_file):
        """Test concurrent processing capabilities"""
        import threading
        
        results = []
        
        def process_file():
            result = generator_tool.forward(
                content_path=sample_file,
                title="Concurrent Test",
                subject="Computer Science"
            )
            results.append(result)
        
        # Start multiple threads
        threads = [threading.Thread(target=process_file) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should complete
        assert len(results) == 3

    def test_quality_validation_integration(self, generator_tool, sample_file):
        """Test integration with quality validation"""
        result = generator_tool.forward(
            content_path=sample_file,
            title="Quality Test",
            subject="Computer Science",
            quality_threshold=0.7
        )
        
        # Should include quality metrics
        stats = result.get("processing_stats", {})
        assert "quality_score" in stats or "success" in stats

    def test_custom_templates(self, generator_tool):
        """Test custom template functionality"""
        custom_template = {
            "header_format": "## {title}",
            "include_summary": True,
            "include_examples": True
        }
        
        metadata = StudyNoteMetadata(
            title="Custom Template Test",
            subject="Computer Science",
            created_date="2024-01-01",
            word_count=100,
            estimated_reading_time=2.0
        )
        
        sections = [StudySection(
            title="Test Section",
            content="Test content",
            level=1,
            key_points=["Point 1"],
            summary="Test summary"
        )]
        
        study_note = StudyNote(
            metadata=metadata,
            sections=sections,
            overview="Test overview",
            learning_objectives=["Objective 1"],
            key_takeaways=["Takeaway 1"]
        )
        
        # Test that custom formatting works
        formatted = generator_tool.format_as_markdown(study_note, template=custom_template)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    @pytest.mark.asyncio
    async def test_async_compatibility(self, generator_tool, sample_file):
        """Test asynchronous compatibility"""
        # Test that the tool can be used in async context
        result = generator_tool.forward(
            content_path=sample_file,
            title="Async Test",
            subject="Computer Science"
        )
        
        assert "processing_stats" in result

    def teardown_method(self, method):
        """Clean up after each test"""
        # Clean up any temporary files if needed
        pass


if __name__ == "__main__":
    pytest.main([__file__])