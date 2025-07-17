"""
Unit tests for the AcademicAgent and its tools.

This module tests the core functionality of the academic agent system including
PDF processing, content analysis, outline generation, and quality management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import os
from datetime import datetime

from agents.academic.academic_agent import (
    PDFIngestionTool,
    ContentAnalysisTool,
    OutlineGenerationTool,
    NotesGenerationTool,
    ContentUpdateTool,
    QualityManagerTool,
    setup_agent,
    process_with_quality_control
)
from tests.utils import (
    TestFileManager,
    MockLLMResponse,
    MockDoclingResult,
    create_sample_markdown_content,
    create_sample_analysis_result,
    PDFTestHelper,
    AnalysisTestHelper,
    assert_quality_score_valid
)


class TestPDFIngestionTool:
    """Test cases for PDFIngestionTool."""
    
    def test_pdf_ingestion_tool_initialization(self, tmp_path):
        """Test PDFIngestionTool initialization."""
        tool = PDFIngestionTool(tmp_path)
        
        assert tool.base_dir == tmp_path
        assert hasattr(tool, 'converter')
        assert tool.name == "pdf_ingestion_tool"
        assert tool.output_type == "object"
        
        # Check that required directories are created
        assert (tmp_path / "raw").exists()
        assert (tmp_path / "markdown").exists()
        assert (tmp_path / "metadata").exists()
    
    @patch('agents.academic.academic_agent.DocumentConverter')
    def test_pdf_ingestion_tool_with_mock_converter(self, mock_converter_class, tmp_path):
        """Test PDFIngestionTool with mocked DocumentConverter."""
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        tool = PDFIngestionTool(tmp_path)
        
        assert tool.converter == mock_converter
        mock_converter_class.assert_called_once()
    
    def test_ensure_directories(self, tmp_path):
        """Test directory creation in PDFIngestionTool."""
        tool = PDFIngestionTool(tmp_path)
        
        required_dirs = ["raw", "markdown", "metadata"]
        for dir_name in required_dirs:
            dir_path = tmp_path / dir_name
            assert dir_path.exists()
            assert dir_path.is_dir()
    
    @patch('agents.academic.academic_agent.DocumentConverter')
    def test_process_pdf_success(self, mock_converter_class, tmp_path):
        """Test successful PDF processing."""
        # Setup mock converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_result = MockDoclingResult(
            markdown_content="# Test Document\n\nContent here",
            title="Test Document",
            language="en"
        )
        mock_converter.convert.return_value = mock_result
        
        tool = PDFIngestionTool(tmp_path)
        
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        result = tool.process_pdf(str(test_pdf))
        
        assert result["status"] == "success"
        assert "markdown_path" in result
        assert "metadata_path" in result
        assert "metadata" in result
        
        # Check that files were created
        assert (tmp_path / "markdown" / "test.md").exists()
        assert (tmp_path / "metadata" / "test.json").exists()
        
        # Verify content
        markdown_content = (tmp_path / "markdown" / "test.md").read_text()
        assert markdown_content == "# Test Document\n\nContent here"
        
        metadata_content = json.loads((tmp_path / "metadata" / "test.json").read_text())
        assert metadata_content["title"] == "Test Document"
        assert metadata_content["language"] == "en"
    
    @patch('agents.academic.academic_agent.DocumentConverter')
    def test_process_pdf_error(self, mock_converter_class, tmp_path):
        """Test PDF processing with error."""
        # Setup mock converter to raise error
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        mock_converter.convert.side_effect = Exception("PDF processing error")
        
        tool = PDFIngestionTool(tmp_path)
        
        # Create a test PDF file
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        result = tool.process_pdf(str(test_pdf))
        
        assert result["status"] == "error"
        assert "error" in result
        assert "PDF processing error" in result["error"]
        assert result["file"] == str(test_pdf)
    
    @patch('agents.academic.academic_agent.DocumentConverter')
    def test_forward_single_file(self, mock_converter_class, tmp_path):
        """Test forward method with single file."""
        # Setup mock converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_result = MockDoclingResult("# Test\nContent")
        mock_converter.convert.return_value = mock_result
        
        tool = PDFIngestionTool(tmp_path)
        
        # Create test PDF
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        result = tool.forward(str(test_pdf))
        
        PDFTestHelper.assert_pdf_processing_result(result)
        assert result["stats"]["total"] == 1
        assert result["stats"]["success"] == 1
        assert result["stats"]["failed"] == 0
        assert len(result["processed_files"]) == 1
        assert len(result["errors"]) == 0
    
    @patch('agents.academic.academic_agent.DocumentConverter')
    def test_forward_directory(self, mock_converter_class, tmp_path):
        """Test forward method with directory."""
        # Setup mock converter
        mock_converter = Mock()
        mock_converter_class.return_value = mock_converter
        
        mock_result = MockDoclingResult("# Test\nContent")
        mock_converter.convert.return_value = mock_result
        
        tool = PDFIngestionTool(tmp_path)
        
        # Create test PDFs in directory
        test_dir = tmp_path / "pdfs"
        test_dir.mkdir()
        
        pdf1 = test_dir / "test1.pdf"
        pdf2 = test_dir / "test2.pdf"
        pdf1.write_bytes(b"fake pdf content 1")
        pdf2.write_bytes(b"fake pdf content 2")
        
        result = tool.forward(str(test_dir))
        
        PDFTestHelper.assert_pdf_processing_result(result)
        assert result["stats"]["total"] == 2
        assert result["stats"]["success"] == 2
        assert result["stats"]["failed"] == 0
        assert len(result["processed_files"]) == 2
        assert len(result["errors"]) == 0
    
    def test_forward_nonexistent_path(self, tmp_path):
        """Test forward method with nonexistent path."""
        tool = PDFIngestionTool(tmp_path)
        
        nonexistent_path = tmp_path / "nonexistent.pdf"
        
        with pytest.raises(FileNotFoundError):
            tool.forward(str(nonexistent_path))


class TestContentAnalysisTool:
    """Test cases for ContentAnalysisTool."""
    
    def test_content_analysis_tool_initialization(self, tmp_path):
        """Test ContentAnalysisTool initialization."""
        tool = ContentAnalysisTool(tmp_path)
        
        assert tool.base_dir == tmp_path
        assert hasattr(tool, 'markdown_processor')
        assert tool.name == "content_analysis_tool"
        assert tool.output_type == "object"
        
        # Check that required directories are created
        required_dirs = ["analysis", "raw", "markdown", "metadata"]
        for dir_name in required_dirs:
            assert (tmp_path / dir_name).exists()
    
    def test_analyze_single_file_success(self, tmp_path):
        """Test successful single file analysis."""
        tool = ContentAnalysisTool(tmp_path)
        
        # Mock the markdown processor
        mock_processor = Mock()
        mock_processor.clean_markdown.return_value = "Cleaned content"
        mock_processor.extract_sections.return_value = [
            {"title": "Section 1", "content": "Content 1"},
            {"title": "Section 2", "content": "Content 2"}
        ]
        mock_processor.save_analysis.return_value = {
            "markdown_path": "/test/analysis.md",
            "json_path": "/test/analysis.json"
        }
        tool.markdown_processor = mock_processor
        
        # Create test markdown file
        test_md = tmp_path / "test.md"
        test_md.write_text(create_sample_markdown_content())
        
        result = tool.analyze_single_file(str(test_md))
        
        assert "error" not in result
        assert "main_topics" in result
        assert "key_concepts" in result
        assert "structure" in result
        assert "summary" in result
        assert "sections" in result
        
        AnalysisTestHelper.assert_analysis_result(result)
    
    def test_analyze_single_file_error(self, tmp_path):
        """Test single file analysis with error."""
        tool = ContentAnalysisTool(tmp_path)
        
        # Use nonexistent file to trigger error
        nonexistent_file = tmp_path / "nonexistent.md"
        
        result = tool.analyze_single_file(str(nonexistent_file))
        
        assert "error" in result
        assert result["source_file"] == str(nonexistent_file)
    
    def test_forward_single_file(self, tmp_path):
        """Test forward method with single file."""
        tool = ContentAnalysisTool(tmp_path)
        
        # Mock the markdown processor
        mock_processor = Mock()
        mock_processor.clean_markdown.return_value = "Cleaned content"
        mock_processor.extract_sections.return_value = [
            {"title": "Section 1", "content": "Content 1"}
        ]
        mock_processor.save_analysis.return_value = {
            "markdown_path": "/test/analysis.md",
            "json_path": "/test/analysis.json"
        }
        tool.markdown_processor = mock_processor
        
        # Create test markdown file
        test_md = tmp_path / "test.md"
        test_md.write_text(create_sample_markdown_content())
        
        result = tool.forward(str(test_md))
        
        assert "files_analyzed" in result
        assert "errors" in result
        assert "stats" in result
        assert result["stats"]["total"] == 1
        assert result["stats"]["success"] == 1
        assert result["stats"]["failed"] == 0
    
    def test_forward_directory(self, tmp_path):
        """Test forward method with directory."""
        tool = ContentAnalysisTool(tmp_path)
        
        # Mock the markdown processor
        mock_processor = Mock()
        mock_processor.clean_markdown.return_value = "Cleaned content"
        mock_processor.extract_sections.return_value = [
            {"title": "Section 1", "content": "Content 1"}
        ]
        mock_processor.save_analysis.return_value = {
            "markdown_path": "/test/analysis.md",
            "json_path": "/test/analysis.json"
        }
        tool.markdown_processor = mock_processor
        
        # Create test markdown files
        test_dir = tmp_path / "markdown"
        test_dir.mkdir()
        
        md1 = test_dir / "test1.md"
        md2 = test_dir / "test2.md"
        md1.write_text(create_sample_markdown_content())
        md2.write_text(create_sample_markdown_content())
        
        result = tool.forward(str(test_dir))
        
        assert result["stats"]["total"] == 2
        assert result["stats"]["success"] == 2
        assert result["stats"]["failed"] == 0
        assert len(result["files_analyzed"]) == 2


class TestOutlineGenerationTool:
    """Test cases for OutlineGenerationTool."""
    
    def test_outline_generation_tool_initialization(self, tmp_path):
        """Test OutlineGenerationTool initialization."""
        with patch('agents.academic.academic_agent.OUTLINES_DIR', tmp_path):
            tool = OutlineGenerationTool()
            
            assert tool.name == "outline_generation_tool"
            assert tool.output_type == "object"
            assert hasattr(tool, 'markdown_processor')
    
    def test_forward_success(self, tmp_path):
        """Test successful outline generation."""
        with patch('agents.academic.academic_agent.OUTLINES_DIR', tmp_path):
            tool = OutlineGenerationTool()
            
            # Mock markdown processor
            mock_processor = Mock()
            mock_processor.clean_markdown.return_value = "Cleaned content"
            mock_processor.extract_sections.return_value = [
                {"title": "Introduction", "content": "Introduction content"},
                {"title": "Main Content", "content": "Main content here"}
            ]
            tool.markdown_processor = mock_processor
            
            # Create test markdown file
            test_md = tmp_path / "test.md"
            test_md.write_text(create_sample_markdown_content())
            
            result = tool.forward(str(test_md))
            
            assert "outline" in result
            assert "outline_path" in result
            assert "outline_json_path" in result
            
            outline = result["outline"]
            AnalysisTestHelper.assert_outline_result(outline)
            
            # Check that files were created
            assert (tmp_path / "test_outline.md").exists()
            assert (tmp_path / "test_outline.json").exists()
    
    def test_forward_error(self, tmp_path):
        """Test outline generation with error."""
        with patch('agents.academic.academic_agent.OUTLINES_DIR', tmp_path):
            tool = OutlineGenerationTool()
            
            # Use nonexistent file to trigger error
            nonexistent_file = tmp_path / "nonexistent.md"
            
            result = tool.forward(str(nonexistent_file))
            
            assert "error" in result
            assert result["file"] == str(nonexistent_file)


class TestNotesGenerationTool:
    """Test cases for NotesGenerationTool."""
    
    def test_notes_generation_tool_initialization(self, tmp_path):
        """Test NotesGenerationTool initialization."""
        with patch('agents.academic.academic_agent.NOTES_DIR', tmp_path):
            tool = NotesGenerationTool()
            
            assert tool.name == "notes_generation_tool"
            assert tool.output_type == "object"
            assert hasattr(tool, 'markdown_processor')
    
    def test_forward_success(self, tmp_path):
        """Test successful notes generation."""
        with patch('agents.academic.academic_agent.NOTES_DIR', tmp_path):
            tool = NotesGenerationTool()
            
            # Mock markdown processor
            mock_processor = Mock()
            mock_processor.clean_markdown.return_value = "Cleaned content"
            mock_processor.extract_sections.return_value = [
                {"title": "Introduction", "content": "Introduction content with details"},
                {"title": "Main Content", "content": "Main content with more information"}
            ]
            tool.markdown_processor = mock_processor
            
            # Create test markdown file
            test_md = tmp_path / "test.md"
            test_md.write_text(create_sample_markdown_content())
            
            result = tool.forward(str(test_md))
            
            assert "notes" in result
            assert "notes_path" in result
            assert "notes_json_path" in result
            
            notes = result["notes"]
            AnalysisTestHelper.assert_notes_result(notes)
            
            # Check that files were created
            assert (tmp_path / "test_notes.md").exists()
            assert (tmp_path / "test_notes.json").exists()
    
    def test_forward_error(self, tmp_path):
        """Test notes generation with error."""
        with patch('agents.academic.academic_agent.NOTES_DIR', tmp_path):
            tool = NotesGenerationTool()
            
            # Use nonexistent file to trigger error
            nonexistent_file = tmp_path / "nonexistent.md"
            
            result = tool.forward(str(nonexistent_file))
            
            assert "error" in result
            assert result["file"] == str(nonexistent_file)


class TestContentUpdateTool:
    """Test cases for ContentUpdateTool."""
    
    def test_content_update_tool_initialization(self):
        """Test ContentUpdateTool initialization."""
        tool = ContentUpdateTool()
        
        assert tool.name == "content_update_tool"
        assert tool.output_type == "object"
    
    def test_forward_basic_update(self, tmp_path):
        """Test basic content update functionality."""
        tool = ContentUpdateTool()
        
        # Create test files
        notes_file = tmp_path / "notes.json"
        notes_data = {"content": ["Original content"]}
        notes_file.write_text(json.dumps(notes_data))
        
        new_content_file = tmp_path / "new_content.md"
        new_content_file.write_text("New content to add")
        
        result = tool.forward(str(notes_file), str(new_content_file))
        
        assert "updated_notes" in result
        updated_notes = result["updated_notes"]
        assert "source_file" in updated_notes
        assert "update_date" in updated_notes
        assert "content" in updated_notes
        assert updated_notes["content"] == ["Original content"]


class TestQualityManagerTool:
    """Test cases for QualityManagerTool."""
    
    def test_quality_manager_tool_initialization(self):
        """Test QualityManagerTool initialization."""
        tool = QualityManagerTool()
        
        assert tool.name == "quality_manager_tool"
        assert tool.output_type == "object"
        assert tool.model_name == "groq/deepseek-r1-distill-llama-70b"
        assert "temperature" in tool.model_config
        assert "max_tokens" in tool.model_config
    
    def test_get_evaluation_prompt(self):
        """Test evaluation prompt generation."""
        tool = QualityManagerTool()
        
        prompt = tool.get_evaluation_prompt("markdown", "Test content")
        
        assert "markdown" in prompt
        assert "Test content" in prompt
        assert "Assessment:" in prompt
        assert "Reasoning:" in prompt
        assert "Suggestions:" in prompt
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        tool = QualityManagerTool()
        
        # Test different assessment types
        assert tool.calculate_quality_score("excellent work") == 0.9
        assert tool.calculate_quality_score("good quality") == 0.8
        assert tool.calculate_quality_score("adequate content") == 0.7
        assert tool.calculate_quality_score("needs improvement") == 0.5
        assert tool.calculate_quality_score("poor quality") == 0.3
        assert tool.calculate_quality_score("neutral content") == 0.6
    
    @patch('agents.academic.academic_agent.completion')
    def test_get_reasoning_evaluation(self, mock_completion):
        """Test reasoning evaluation with mocked LLM."""
        tool = QualityManagerTool()
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = """
        Assessment: Good quality content with clear structure
        Reasoning: The content is well-organized and demonstrates good understanding
        Suggestions: Consider adding more examples
        """
        mock_completion.return_value = mock_response
        
        result = tool.get_reasoning_evaluation("markdown", "Test content")
        
        assert "assessment" in result
        assert "reasoning" in result
        assert "suggestions" in result
        assert "quality_score" in result
        assert_quality_score_valid(result["quality_score"])
    
    def test_forward_markdown_content(self, tmp_path):
        """Test forward method with markdown content."""
        tool = QualityManagerTool()
        
        # Create test markdown file
        test_md = tmp_path / "test.md"
        test_md.write_text(create_sample_markdown_content())
        
        # Mock the reasoning evaluation
        with patch.object(tool, 'get_reasoning_evaluation') as mock_eval:
            mock_eval.return_value = {
                "assessment": "Good quality content",
                "reasoning": "Well structured",
                "suggestions": ["Add more examples"]
            }
            
            with patch.object(tool, 'calculate_quality_score') as mock_calc:
                mock_calc.return_value = 0.8
                
                result = tool.forward(str(test_md), "markdown")
                
                assert "evaluation" in result
                evaluation = result["evaluation"]
                assert "quality_score" in evaluation
                assert "feedback" in evaluation
                assert "reasoning" in evaluation
                assert "assessment" in evaluation
                assert "approved" in evaluation
                assert evaluation["approved"] is True  # 0.8 >= 0.7
    
    def test_forward_json_content(self, tmp_path):
        """Test forward method with JSON content."""
        tool = QualityManagerTool()
        
        # Create test JSON file
        test_json = tmp_path / "test.json"
        test_data = create_sample_analysis_result()
        test_json.write_text(json.dumps(test_data))
        
        # Mock the reasoning evaluation
        with patch.object(tool, 'get_reasoning_evaluation') as mock_eval:
            mock_eval.return_value = {
                "assessment": "Needs improvement",
                "reasoning": "Missing key elements",
                "suggestions": ["Add more detail", "Improve structure"]
            }
            
            with patch.object(tool, 'calculate_quality_score') as mock_calc:
                mock_calc.return_value = 0.5
                
                result = tool.forward(str(test_json), "analysis", quality_threshold=0.7)
                
                assert "evaluation" in result
                evaluation = result["evaluation"]
                assert evaluation["approved"] is False  # 0.5 < 0.7
                assert len(evaluation["improvement_suggestions"]) > 0


class TestAgentSetup:
    """Test cases for agent setup and configuration."""
    
    @patch('agents.academic.academic_agent.LiteLLMModel')
    @patch('agents.academic.academic_agent.CodeAgent')
    def test_setup_agent_success(self, mock_code_agent, mock_litellm_model, tmp_path):
        """Test successful agent setup."""
        mock_model = Mock()
        mock_litellm_model.return_value = mock_model
        
        mock_agent = Mock()
        mock_code_agent.return_value = mock_agent
        
        api_key = "test_api_key"
        
        with patch('agents.academic.academic_agent.PROCESSED_DIR', tmp_path):
            agent = setup_agent(api_key, tmp_path)
            
            assert agent == mock_agent
            mock_litellm_model.assert_called_once()
            mock_code_agent.assert_called_once()
            
            # Check that tools were attached to agent
            assert hasattr(agent, 'pdf_tool')
            assert hasattr(agent, 'analysis_tool')
            assert hasattr(agent, 'outline_tool')
            assert hasattr(agent, 'notes_tool')
            assert hasattr(agent, 'update_tool')
            assert hasattr(agent, 'quality_manager')
    
    @patch('agents.academic.academic_agent.LiteLLMModel')
    def test_setup_agent_model_error(self, mock_litellm_model, tmp_path):
        """Test agent setup with model error."""
        mock_litellm_model.side_effect = Exception("Model initialization error")
        
        api_key = "test_api_key"
        
        with pytest.raises(Exception) as exc_info:
            setup_agent(api_key, tmp_path)
        
        assert "Model initialization error" in str(exc_info.value)


class TestQualityControlWorkflow:
    """Test cases for quality control workflow."""
    
    def test_process_with_quality_control_success(self):
        """Test successful quality control process."""
        # Create mock agent
        mock_agent = Mock()
        mock_quality_manager = Mock()
        mock_agent.quality_manager = mock_quality_manager
        
        # Mock quality evaluation - passes threshold
        mock_quality_manager.forward.return_value = {
            "evaluation": {
                "quality_score": 0.8,
                "approved": True
            }
        }
        
        result = process_with_quality_control(
            mock_agent, 
            "/test/content.md", 
            "markdown",
            max_attempts=3,
            quality_threshold=0.7
        )
        
        assert result is True
        mock_quality_manager.forward.assert_called_once_with(
            "/test/content.md", "markdown", 0.7
        )
    
    def test_process_with_quality_control_failure(self):
        """Test quality control process with failure."""
        # Create mock agent
        mock_agent = Mock()
        mock_quality_manager = Mock()
        mock_agent.quality_manager = mock_quality_manager
        
        # Mock quality evaluation - fails threshold
        mock_quality_manager.forward.return_value = {
            "evaluation": {
                "quality_score": 0.5,
                "approved": False,
                "improvement_suggestions": ["Improve structure", "Add more detail"]
            }
        }
        
        result = process_with_quality_control(
            mock_agent,
            "/test/content.md",
            "markdown",
            max_attempts=1,
            quality_threshold=0.7
        )
        
        assert result is False
        mock_quality_manager.forward.assert_called_once()
    
    def test_process_with_quality_control_error(self):
        """Test quality control process with error."""
        # Create mock agent
        mock_agent = Mock()
        mock_quality_manager = Mock()
        mock_agent.quality_manager = mock_quality_manager
        
        # Mock quality evaluation error
        mock_quality_manager.forward.return_value = {
            "error": "Quality evaluation failed"
        }
        
        result = process_with_quality_control(
            mock_agent,
            "/test/content.md",
            "markdown",
            max_attempts=3,
            quality_threshold=0.7
        )
        
        assert result is False
        mock_quality_manager.forward.assert_called_once()


@pytest.mark.integration
class TestAcademicAgentIntegration:
    """Integration tests for the academic agent system."""
    
    def test_full_processing_pipeline_mock(self, tmp_path):
        """Test the full processing pipeline with mocked components."""
        # This would be an integration test that exercises the full pipeline
        # but with mocked external dependencies
        
        # Create test input files
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"fake pdf content")
        
        # Mock all external dependencies
        with patch('agents.academic.academic_agent.DocumentConverter') as mock_converter, \
             patch('agents.academic.academic_agent.completion') as mock_completion, \
             patch('agents.academic.academic_agent.LiteLLMModel') as mock_model:
            
            # Setup mocks
            mock_doc_converter = Mock()
            mock_converter.return_value = mock_doc_converter
            mock_result = MockDoclingResult("# Test\nContent")
            mock_doc_converter.convert.return_value = mock_result
            
            mock_llm_response = Mock()
            mock_llm_response.choices = [Mock()]
            mock_llm_response.choices[0].message.content = "Assessment: Good quality"
            mock_completion.return_value = mock_llm_response
            
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance
            
            # Test the setup
            api_key = "test_key"
            agent = setup_agent(api_key, tmp_path)
            
            # Verify agent was created with all tools
            assert hasattr(agent, 'pdf_tool')
            assert hasattr(agent, 'analysis_tool')
            assert hasattr(agent, 'outline_tool')
            assert hasattr(agent, 'notes_tool')
            assert hasattr(agent, 'quality_manager')
            
            # This confirms the integration setup works
            assert agent is not None