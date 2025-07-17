"""
Test utilities for the academic-agent test suite.

This module provides helper functions and utilities for testing various components
of the academic agent system.
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch, MagicMock
import pytest
from datetime import datetime


class TestFileManager:
    """Utility class for managing test files and directories."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize the test file manager."""
        self.base_path = base_path or Path(tempfile.mkdtemp())
        self.created_files = []
        self.created_dirs = []
    
    def create_file(self, relative_path: str, content: str = "") -> Path:
        """Create a test file with specified content."""
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        self.created_files.append(file_path)
        return file_path
    
    def create_json_file(self, relative_path: str, data: Dict[str, Any]) -> Path:
        """Create a test JSON file with specified data."""
        file_path = self.base_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, indent=2))
        self.created_files.append(file_path)
        return file_path
    
    def create_directory(self, relative_path: str) -> Path:
        """Create a test directory."""
        dir_path = self.base_path / relative_path
        dir_path.mkdir(parents=True, exist_ok=True)
        self.created_dirs.append(dir_path)
        return dir_path
    
    def cleanup(self) -> None:
        """Clean up all created files and directories."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        
        for dir_path in self.created_dirs:
            if dir_path.exists():
                shutil.rmtree(dir_path)
        
        if self.base_path.exists():
            shutil.rmtree(self.base_path)


class MockLLMResponse:
    """Mock LLM response for testing."""
    
    def __init__(self, content: str, usage: Optional[Dict[str, Any]] = None):
        self.content = content
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
    
    @property
    def choices(self):
        """Mock choices property."""
        choice = Mock()
        choice.message.content = self.content
        return [choice]


class MockDoclingResult:
    """Mock Docling processing result for testing."""
    
    def __init__(self, markdown_content: str, title: str = "Test Document", language: str = "en"):
        self.document = Mock()
        self.document.export_to_markdown.return_value = markdown_content
        self.document.title = title
        self.document.language = language


def create_sample_pdf_content() -> bytes:
    """Create sample PDF content for testing."""
    return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 50
>>
stream
BT
/F1 12 Tf
100 700 Td
(Sample PDF for Testing) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000189 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
291
%%EOF"""


def create_sample_markdown_content() -> str:
    """Create sample markdown content for testing."""
    return """# Academic Research Paper

## Abstract

This is a sample academic research paper for testing the academic agent system.
The paper covers various aspects of the research topic with detailed analysis.

## Introduction

The introduction section provides background information and context for the research.
It establishes the problem statement and research objectives.

### Research Questions

1. What are the key components of the academic agent system?
2. How can we ensure quality in automated academic processing?
3. What are the best practices for PDF processing in academic contexts?

## Literature Review

### Previous Work

Previous research in this area has focused on various aspects:
- Automated document processing
- Quality assurance in academic systems
- Integration of AI in educational tools

### Gaps in Current Research

Current research lacks comprehensive testing frameworks for academic agents.
This paper aims to address these gaps through systematic testing approaches.

## Methodology

### Data Collection

The methodology involves collecting various types of academic documents:
- Research papers
- Lecture notes
- Textbooks
- Presentation slides

### Analysis Framework

The analysis framework consists of several components:
1. Document ingestion and processing
2. Content analysis and extraction
3. Quality evaluation and feedback
4. Output generation and validation

## Results

### Processing Performance

The system demonstrates high accuracy in document processing:
- PDF conversion accuracy: 95%
- Content extraction quality: 92%
- Analysis depth score: 88%

### Quality Metrics

Quality evaluation shows consistent performance:
- Structural analysis: Excellent
- Content comprehension: Good
- Output formatting: Very Good

## Discussion

### Implications

The results have several important implications:
- Automated processing can achieve high quality standards
- Quality control mechanisms are essential
- Integration testing is crucial for system reliability

### Limitations

Some limitations were identified:
- Performance depends on document quality
- Complex mathematical content requires special handling
- Multi-language support needs improvement

## Conclusion

The academic agent system shows promise for automated academic document processing.
The comprehensive testing framework ensures reliable performance and quality control.

### Future Work

Future research should focus on:
- Expanding language support
- Improving mathematical content processing
- Developing more sophisticated quality metrics

## References

1. Smith, J. (2023). Automated Academic Processing Systems. Journal of AI Research.
2. Johnson, M. (2022). Quality Control in Educational AI. Educational Technology Review.
3. Brown, L. (2021). PDF Processing Best Practices. Document Processing Quarterly.
"""


def create_sample_analysis_result() -> Dict[str, Any]:
    """Create sample analysis result for testing."""
    return {
        "main_topics": [
            "Academic Research Paper",
            "Literature Review",
            "Methodology", 
            "Results",
            "Discussion",
            "Conclusion"
        ],
        "key_concepts": [
            "automated document processing",
            "quality assurance in academic systems",
            "PDF conversion accuracy",
            "content extraction quality",
            "analysis depth score",
            "structural analysis",
            "content comprehension",
            "output formatting"
        ],
        "structure": [
            {
                "title": "Abstract",
                "summary": "Overview of the research paper and its contributions"
            },
            {
                "title": "Introduction",
                "summary": "Background information and research objectives"
            },
            {
                "title": "Literature Review",
                "summary": "Previous work and gaps in current research"
            },
            {
                "title": "Methodology",
                "summary": "Data collection and analysis framework"
            },
            {
                "title": "Results",
                "summary": "Processing performance and quality metrics"
            },
            {
                "title": "Discussion",
                "summary": "Implications and limitations of the study"
            },
            {
                "title": "Conclusion",
                "summary": "Summary of findings and future work"
            }
        ],
        "summary": "A comprehensive research paper on academic agent systems with focus on automated document processing and quality assurance.",
        "source_file": "/test/sample_paper.md",
        "analysis_date": datetime.now().isoformat(),
        "sections": [
            {"title": "Abstract", "content": "This is a sample academic research paper..."},
            {"title": "Introduction", "content": "The introduction section provides background..."},
            {"title": "Literature Review", "content": "Previous research in this area..."},
            {"title": "Methodology", "content": "The methodology involves collecting..."},
            {"title": "Results", "content": "The system demonstrates high accuracy..."},
            {"title": "Discussion", "content": "The results have several important implications..."},
            {"title": "Conclusion", "content": "The academic agent system shows promise..."}
        ]
    }


def create_sample_outline_result() -> Dict[str, Any]:
    """Create sample outline result for testing."""
    return {
        "source_file": "/test/sample_paper.md",
        "generated_date": datetime.now().isoformat(),
        "sections": [
            {
                "title": "Abstract",
                "key_points": [
                    "Sample academic research paper",
                    "Testing the academic agent system",
                    "Detailed analysis of research topic"
                ],
                "subsections": []
            },
            {
                "title": "Introduction",
                "key_points": [
                    "Background information and context",
                    "Problem statement and research objectives"
                ],
                "subsections": [
                    {
                        "title": "Research Questions",
                        "key_points": [
                            "Key components of academic agent system",
                            "Quality assurance in automated processing",
                            "Best practices for PDF processing"
                        ]
                    }
                ]
            },
            {
                "title": "Literature Review",
                "key_points": [
                    "Previous work in academic processing",
                    "Gaps in current research"
                ],
                "subsections": [
                    {
                        "title": "Previous Work",
                        "key_points": [
                            "Automated document processing",
                            "Quality assurance systems",
                            "AI in educational tools"
                        ]
                    },
                    {
                        "title": "Gaps in Current Research",
                        "key_points": [
                            "Lack of comprehensive testing frameworks",
                            "Systematic testing approaches needed"
                        ]
                    }
                ]
            },
            {
                "title": "Methodology",
                "key_points": [
                    "Data collection from various sources",
                    "Multi-component analysis framework"
                ],
                "subsections": [
                    {
                        "title": "Data Collection",
                        "key_points": [
                            "Research papers",
                            "Lecture notes", 
                            "Textbooks",
                            "Presentation slides"
                        ]
                    },
                    {
                        "title": "Analysis Framework",
                        "key_points": [
                            "Document ingestion and processing",
                            "Content analysis and extraction",
                            "Quality evaluation and feedback",
                            "Output generation and validation"
                        ]
                    }
                ]
            }
        ]
    }


def create_sample_notes_result() -> Dict[str, Any]:
    """Create sample notes result for testing."""
    return {
        "source_file": "/test/sample_paper.md",
        "generated_date": datetime.now().isoformat(),
        "sections": [
            {
                "title": "Abstract",
                "summary": "Overview of sample academic research paper for testing academic agent system.",
                "key_concepts": [
                    "Sample academic research paper",
                    "Academic agent system testing",
                    "Detailed analysis coverage"
                ],
                "detailed_notes": [
                    "This is a sample academic research paper for testing the academic agent system.",
                    "The paper covers various aspects of the research topic with detailed analysis.",
                    "Serves as comprehensive test case for system validation."
                ]
            },
            {
                "title": "Introduction",
                "summary": "Provides background information and establishes research context and objectives.",
                "key_concepts": [
                    "Background information",
                    "Research context",
                    "Problem statement",
                    "Research objectives"
                ],
                "detailed_notes": [
                    "The introduction section provides background information and context for the research.",
                    "It establishes the problem statement and research objectives.",
                    "Forms foundation for subsequent analysis and methodology."
                ]
            },
            {
                "title": "Literature Review",
                "summary": "Examines previous work and identifies gaps in current research.",
                "key_concepts": [
                    "Previous research",
                    "Automated document processing",
                    "Quality assurance systems",
                    "Research gaps"
                ],
                "detailed_notes": [
                    "Previous research in this area has focused on various aspects:",
                    "- Automated document processing",
                    "- Quality assurance in academic systems", 
                    "- Integration of AI in educational tools",
                    "Current research lacks comprehensive testing frameworks for academic agents."
                ]
            }
        ]
    }


def assert_valid_json_structure(data: Dict[str, Any], required_keys: List[str]) -> None:
    """Assert that a JSON structure contains required keys."""
    assert isinstance(data, dict), "Data should be a dictionary"
    
    for key in required_keys:
        assert key in data, f"Required key '{key}' not found in data"
    
    # Additional validation for common academic agent data structures
    if "sections" in data:
        assert isinstance(data["sections"], list), "Sections should be a list"
        for section in data["sections"]:
            assert isinstance(section, dict), "Each section should be a dictionary"
            assert "title" in section, "Each section should have a title"


def assert_valid_markdown_output(content: str) -> None:
    """Assert that content is valid markdown."""
    assert isinstance(content, str), "Content should be a string"
    assert len(content.strip()) > 0, "Content should not be empty"
    
    # Check for basic markdown structure
    lines = content.split('\n')
    has_header = any(line.startswith('#') for line in lines)
    assert has_header, "Markdown should contain at least one header"


def assert_quality_score_valid(score: float) -> None:
    """Assert that a quality score is within valid range."""
    assert isinstance(score, (int, float)), "Quality score should be a number"
    assert 0.0 <= score <= 1.0, f"Quality score {score} should be between 0.0 and 1.0"


def mock_api_response(content: str, status_code: int = 200) -> Mock:
    """Create a mock API response."""
    response = Mock()
    response.status_code = status_code
    response.text = content
    response.json.return_value = {"content": content}
    return response


def create_mock_agent_with_tools() -> Mock:
    """Create a mock agent with all required tools."""
    agent = Mock()
    
    # Mock tools
    agent.pdf_tool = Mock()
    agent.analysis_tool = Mock()
    agent.outline_tool = Mock()
    agent.notes_tool = Mock()
    agent.update_tool = Mock()
    agent.quality_manager = Mock()
    
    # Mock tool responses
    agent.pdf_tool.forward.return_value = {
        "processed_files": [],
        "errors": [],
        "stats": {"total": 0, "success": 0, "failed": 0}
    }
    
    agent.analysis_tool.forward.return_value = {
        "files_analyzed": [],
        "errors": [],
        "stats": {"total": 0, "success": 0, "failed": 0}
    }
    
    agent.outline_tool.forward.return_value = {
        "outline": {},
        "outline_path": "/test/outline.md",
        "outline_json_path": "/test/outline.json"
    }
    
    agent.notes_tool.forward.return_value = {
        "notes": {},
        "notes_path": "/test/notes.md", 
        "notes_json_path": "/test/notes.json"
    }
    
    agent.quality_manager.forward.return_value = {
        "evaluation": {
            "quality_score": 0.8,
            "feedback": [],
            "reasoning": "Test reasoning",
            "assessment": "Good quality",
            "approved": True,
            "improvement_suggestions": []
        }
    }
    
    return agent


class PDFTestHelper:
    """Helper class for PDF-related tests."""
    
    @staticmethod
    def create_test_pdf(path: Path, content: str = "Test PDF Content") -> Path:
        """Create a test PDF file."""
        pdf_content = create_sample_pdf_content()
        path.write_bytes(pdf_content)
        return path
    
    @staticmethod
    def assert_pdf_processing_result(result: Dict[str, Any]) -> None:
        """Assert that PDF processing result is valid."""
        required_keys = ["processed_files", "errors", "stats"]
        assert_valid_json_structure(result, required_keys)
        
        assert isinstance(result["processed_files"], list)
        assert isinstance(result["errors"], list)
        assert isinstance(result["stats"], dict)
        
        stats = result["stats"]
        assert "total" in stats
        assert "success" in stats
        assert "failed" in stats
        assert stats["total"] == stats["success"] + stats["failed"]


class AnalysisTestHelper:
    """Helper class for analysis-related tests."""
    
    @staticmethod
    def assert_analysis_result(result: Dict[str, Any]) -> None:
        """Assert that analysis result is valid."""
        required_keys = ["main_topics", "key_concepts", "structure", "summary"]
        assert_valid_json_structure(result, required_keys)
        
        assert isinstance(result["main_topics"], list)
        assert isinstance(result["key_concepts"], list)
        assert isinstance(result["structure"], list)
        assert isinstance(result["summary"], str)
    
    @staticmethod
    def assert_outline_result(result: Dict[str, Any]) -> None:
        """Assert that outline result is valid."""
        required_keys = ["source_file", "generated_date", "sections"]
        assert_valid_json_structure(result, required_keys)
        
        assert isinstance(result["sections"], list)
        for section in result["sections"]:
            assert "title" in section
            assert "key_points" in section
            assert isinstance(section["key_points"], list)
    
    @staticmethod
    def assert_notes_result(result: Dict[str, Any]) -> None:
        """Assert that notes result is valid."""
        required_keys = ["source_file", "generated_date", "sections"]
        assert_valid_json_structure(result, required_keys)
        
        assert isinstance(result["sections"], list)
        for section in result["sections"]:
            assert "title" in section
            assert "summary" in section
            assert "key_concepts" in section
            assert "detailed_notes" in section