#!/usr/bin/env python3
"""
Comprehensive tests for AnalysisAgent

Tests all core functionality including content analysis, markdown parsing,
section extraction, quality assessment, and error handling.
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

from agents.academic.analysis_agent import AnalysisAgent, AnalysisResult, ContentSection


class TestAnalysisAgent:
    """Comprehensive test suite for AnalysisAgent"""
    
    @pytest.fixture
    def analysis_agent(self):
        """Create an AnalysisAgent instance for testing"""
        agent = AnalysisAgent()
        return agent
    
    @pytest.fixture
    def sample_markdown_content(self):
        """Sample markdown content for testing"""
        return """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.

## Supervised Learning

Supervised learning uses labeled data to train models.

### Classification
Classification predicts discrete categories.

### Regression
Regression predicts continuous values.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data.

### Clustering
Clustering groups similar data points.

### Dimensionality Reduction
Reduces the number of features while preserving information.

## Key Concepts

- **Algorithm**: A set of rules for solving problems
- **Dataset**: Collection of data for training
- **Feature**: Individual measurable property
- **Model**: Mathematical representation of a process
"""

    @pytest.fixture
    def sample_file(self, sample_markdown_content):
        """Create a temporary file with sample content"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_markdown_content)
            return f.name

    def test_initialization(self, analysis_agent):
        """Test agent initialization"""
        assert analysis_agent.agent_id == "analysis_agent"
        assert hasattr(analysis_agent, 'logger')
        assert hasattr(analysis_agent, 'initialize')
        assert hasattr(analysis_agent, 'cleanup')

    @pytest.mark.asyncio
    async def test_initialize_and_cleanup(self, analysis_agent):
        """Test agent initialization and cleanup"""
        # Test initialization
        await analysis_agent.initialize()
        
        # Test cleanup
        await analysis_agent.cleanup()

    def test_validate_input_string(self, analysis_agent):
        """Test input validation with string input"""
        # Valid string input
        assert analysis_agent.validate_input("test content") == True
        assert analysis_agent.validate_input("# Header\nContent") == True
        
        # Invalid string input
        assert analysis_agent.validate_input("") == False
        assert analysis_agent.validate_input("   ") == False
        assert analysis_agent.validate_input(None) == False

    def test_validate_input_dict(self, analysis_agent):
        """Test input validation with dictionary input"""
        # Valid dict input
        valid_dict = {
            "content": "test content",
            "title": "test title"
        }
        assert analysis_agent.validate_input(valid_dict) == True
        
        # Invalid dict input - missing required fields
        invalid_dict1 = {"title": "test title"}
        assert analysis_agent.validate_input(invalid_dict1) == False
        
        invalid_dict2 = {"content": ""}
        assert analysis_agent.validate_input(invalid_dict2) == False

    def test_validate_output(self, analysis_agent):
        """Test output validation"""
        # Valid output
        valid_output = {
            "success": True,
            "analysis_result": {"sections": []},
            "quality_score": 0.8
        }
        assert analysis_agent.validate_output(valid_output) == True
        
        # Invalid output - missing required fields
        invalid_output1 = {"success": True}
        assert analysis_agent.validate_output(invalid_output1) == False
        
        invalid_output2 = {"analysis_result": {}}
        assert analysis_agent.validate_output(invalid_output2) == False
        
        # Invalid output - wrong type
        assert analysis_agent.validate_output("invalid") == False
        assert analysis_agent.validate_output(None) == False

    def test_check_quality_with_analysis_result(self, analysis_agent):
        """Test quality checking with analysis result"""
        # High quality analysis result
        high_quality_result = {
            "sections": [
                {"title": "Section 1", "content": "Good content"},
                {"title": "Section 2", "content": "More good content"}
            ],
            "summary": "Comprehensive summary",
            "key_points": ["Point 1", "Point 2", "Point 3"]
        }
        quality_score = analysis_agent.check_quality(high_quality_result)
        assert quality_score > 0.7
        
        # Low quality analysis result
        low_quality_result = {
            "sections": [],
            "summary": "",
            "key_points": []
        }
        quality_score = analysis_agent.check_quality(low_quality_result)
        assert quality_score < 0.3

    def test_check_quality_with_string_content(self, analysis_agent):
        """Test quality checking with string content"""
        # Good quality content
        good_content = "# Header\n\nThis is comprehensive content with multiple sections and detailed information."
        quality_score = analysis_agent.check_quality(good_content)
        assert quality_score > 0.5
        
        # Poor quality content
        poor_content = "Short"
        quality_score = analysis_agent.check_quality(poor_content)
        assert quality_score < 0.5

    def test_parse_markdown_headers(self, analysis_agent, sample_markdown_content):
        """Test markdown header parsing"""
        sections = analysis_agent.parse_markdown(sample_markdown_content)
        
        # Should have main sections
        assert len(sections) > 0
        
        # Check for expected sections
        section_titles = [s.title for s in sections]
        assert "Introduction to Machine Learning" in section_titles
        assert "Supervised Learning" in section_titles
        assert "Unsupervised Learning" in section_titles
        assert "Key Concepts" in section_titles

    def test_parse_markdown_hierarchy(self, analysis_agent, sample_markdown_content):
        """Test markdown hierarchy parsing"""
        sections = analysis_agent.parse_markdown(sample_markdown_content)
        
        # Find supervised learning section
        supervised_section = next((s for s in sections if s.title == "Supervised Learning"), None)
        assert supervised_section is not None
        assert supervised_section.level == 2
        
        # Check for subsections
        classification_section = next((s for s in sections if s.title == "Classification"), None)
        assert classification_section is not None
        assert classification_section.level == 3

    def test_extract_sections_content(self, analysis_agent, sample_markdown_content):
        """Test section content extraction"""
        sections = analysis_agent.parse_markdown(sample_markdown_content)
        
        # Find key concepts section
        key_concepts_section = next((s for s in sections if s.title == "Key Concepts"), None)
        assert key_concepts_section is not None
        
        # Should contain bullet points
        assert "Algorithm" in key_concepts_section.content
        assert "Dataset" in key_concepts_section.content
        assert "Feature" in key_concepts_section.content
        assert "Model" in key_concepts_section.content

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_generate_summary_success(self, mock_ai_request, analysis_agent, sample_markdown_content):
        """Test summary generation with successful AI response"""
        mock_ai_request.return_value = "This is a comprehensive overview of machine learning fundamentals."
        
        summary = analysis_agent.generate_summary(sample_markdown_content)
        
        assert summary == "This is a comprehensive overview of machine learning fundamentals."
        mock_ai_request.assert_called_once()

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_generate_summary_fallback(self, mock_ai_request, analysis_agent, sample_markdown_content):
        """Test summary generation with AI failure fallback"""
        mock_ai_request.side_effect = Exception("AI service unavailable")
        
        summary = analysis_agent.generate_summary(sample_markdown_content)
        
        # Should fallback to extracting first sentences
        assert len(summary) > 0
        assert "Machine learning is a subset" in summary

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_extract_key_points_success(self, mock_ai_request, analysis_agent, sample_markdown_content):
        """Test key points extraction with successful AI response"""
        mock_ai_request.return_value = '["Point 1", "Point 2", "Point 3"]'
        
        key_points = analysis_agent.extract_key_points(sample_markdown_content)
        
        assert key_points == ["Point 1", "Point 2", "Point 3"]
        mock_ai_request.assert_called_once()

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_extract_key_points_fallback(self, mock_ai_request, analysis_agent, sample_markdown_content):
        """Test key points extraction with AI failure fallback"""
        mock_ai_request.side_effect = Exception("AI service unavailable")
        
        key_points = analysis_agent.extract_key_points(sample_markdown_content)
        
        # Should fallback to manual extraction
        assert isinstance(key_points, list)
        assert len(key_points) > 0

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_extract_concepts_success(self, mock_ai_request, analysis_agent, sample_markdown_content):
        """Test concept extraction with successful AI response"""
        mock_ai_request.return_value = '["Machine Learning", "Supervised Learning", "Classification"]'
        
        concepts = analysis_agent.extract_concepts(sample_markdown_content)
        
        assert concepts == ["Machine Learning", "Supervised Learning", "Classification"]
        mock_ai_request.assert_called_once()

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_extract_concepts_fallback(self, mock_ai_request, analysis_agent, sample_markdown_content):
        """Test concept extraction with AI failure fallback"""
        mock_ai_request.side_effect = Exception("AI service unavailable")
        
        concepts = analysis_agent.extract_concepts(sample_markdown_content)
        
        # Should fallback to manual extraction
        assert isinstance(concepts, list)
        assert len(concepts) > 0

    def test_analyze_content_structure(self, analysis_agent, sample_markdown_content):
        """Test content structure analysis"""
        analysis = analysis_agent.analyze_content(sample_markdown_content)
        
        assert isinstance(analysis, AnalysisResult)
        assert analysis.title == "Introduction to Machine Learning"
        assert len(analysis.sections) > 0
        assert analysis.word_count > 0
        assert analysis.reading_time > 0
        assert len(analysis.summary) > 0
        assert len(analysis.key_points) > 0
        assert len(analysis.concepts) > 0

    def test_analyze_content_metadata(self, analysis_agent, sample_markdown_content):
        """Test content metadata extraction"""
        analysis = analysis_agent.analyze_content(sample_markdown_content)
        
        # Check metadata
        assert analysis.word_count > 50  # Should have substantial word count
        assert analysis.reading_time > 0  # Should have positive reading time
        assert len(analysis.sections) >= 4  # Should have multiple sections

    def test_analyze_file_success(self, analysis_agent, sample_file):
        """Test file analysis with valid file"""
        result = analysis_agent.analyze_file(sample_file)
        
        assert result["success"] == True
        assert "analysis_result" in result
        assert "quality_score" in result
        
        analysis = result["analysis_result"]
        assert analysis.title == "Introduction to Machine Learning"
        assert len(analysis.sections) > 0

    def test_analyze_file_not_found(self, analysis_agent):
        """Test file analysis with non-existent file"""
        result = analysis_agent.analyze_file("nonexistent_file.md")
        
        assert result["success"] == False
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_analyze_file_invalid_encoding(self, analysis_agent):
        """Test file analysis with encoding issues"""
        # Create a file with invalid encoding
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.md', delete=False) as f:
            f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8
            invalid_file = f.name
        
        result = analysis_agent.analyze_file(invalid_file)
        
        assert result["success"] == False
        assert "error" in result

    def test_process_batch_files(self, analysis_agent, sample_file):
        """Test batch processing of multiple files"""
        files = [sample_file, sample_file]  # Use same file twice for testing
        
        results = analysis_agent.process_batch(files)
        
        assert len(results) == 2
        for result in results:
            assert result["success"] == True
            assert "analysis_result" in result

    def test_process_batch_mixed_results(self, analysis_agent, sample_file):
        """Test batch processing with mixed success/failure"""
        files = [sample_file, "nonexistent_file.md"]
        
        results = analysis_agent.process_batch(files)
        
        assert len(results) == 2
        assert results[0]["success"] == True
        assert results[1]["success"] == False

    def test_empty_content_handling(self, analysis_agent):
        """Test handling of empty content"""
        analysis = analysis_agent.analyze_content("")
        
        assert isinstance(analysis, AnalysisResult)
        assert analysis.title == "Untitled Document"
        assert len(analysis.sections) == 0
        assert analysis.word_count == 0

    def test_malformed_markdown_handling(self, analysis_agent):
        """Test handling of malformed markdown"""
        malformed_content = """
        ### This is a subsection without a main section
        Some content here.
        
        ## This is out of order
        More content.
        
        # This comes after lower level headers
        Final content.
        """
        
        analysis = analysis_agent.analyze_content(malformed_content)
        
        assert isinstance(analysis, AnalysisResult)
        assert len(analysis.sections) > 0  # Should still parse sections

    def test_large_content_handling(self, analysis_agent):
        """Test handling of large content"""
        # Create large content (>10MB would be too much for testing)
        large_content = "# Large Document\n\n" + "This is content. " * 10000
        
        analysis = analysis_agent.analyze_content(large_content)
        
        assert isinstance(analysis, AnalysisResult)
        assert analysis.word_count > 10000

    @pytest.mark.asyncio
    async def test_async_processing(self, analysis_agent, sample_file):
        """Test asynchronous processing capabilities"""
        # Test that the agent can handle async operations
        await analysis_agent.initialize()
        
        # Test analysis in async context
        result = analysis_agent.analyze_file(sample_file)
        assert result["success"] == True
        
        await analysis_agent.cleanup()

    def test_concurrent_analysis(self, analysis_agent, sample_file):
        """Test concurrent analysis operations"""
        import threading
        
        results = []
        
        def analyze_file():
            result = analysis_agent.analyze_file(sample_file)
            results.append(result)
        
        # Start multiple threads
        threads = [threading.Thread(target=analyze_file) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result["success"] == True

    def test_content_section_dataclass(self):
        """Test ContentSection dataclass functionality"""
        section = ContentSection(
            title="Test Section",
            content="Test content",
            level=2,
            start_line=1,
            end_line=5
        )
        
        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert section.level == 2
        assert section.start_line == 1
        assert section.end_line == 5

    def test_analysis_result_dataclass(self):
        """Test AnalysisResult dataclass functionality"""
        sections = [ContentSection("Section 1", "Content 1", 1, 1, 2)]
        
        result = AnalysisResult(
            title="Test Analysis",
            sections=sections,
            summary="Test summary",
            key_points=["Point 1"],
            concepts=["Concept 1"],
            word_count=100,
            reading_time=1.5,
            complexity_score=0.7
        )
        
        assert result.title == "Test Analysis"
        assert len(result.sections) == 1
        assert result.summary == "Test summary"
        assert result.key_points == ["Point 1"]
        assert result.concepts == ["Concept 1"]
        assert result.word_count == 100
        assert result.reading_time == 1.5
        assert result.complexity_score == 0.7

    @patch('agents.academic.analysis_agent.AnalysisAgent.ai_request')
    def test_ai_request_error_handling(self, mock_ai_request, analysis_agent):
        """Test AI request error handling"""
        mock_ai_request.side_effect = Exception("AI service error")
        
        # Should not raise exception, but handle gracefully
        summary = analysis_agent.generate_summary("Test content")
        key_points = analysis_agent.extract_key_points("Test content")
        concepts = analysis_agent.extract_concepts("Test content")
        
        # Should get fallback responses
        assert isinstance(summary, str)
        assert isinstance(key_points, list)
        assert isinstance(concepts, list)

    def teardown_method(self, method):
        """Clean up after each test"""
        # Clean up any temporary files if needed
        pass


if __name__ == "__main__":
    pytest.main([__file__])