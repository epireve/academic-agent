"""
Pytest configuration and fixtures for the academic-agent test suite.

This module provides shared fixtures and configuration for all tests.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch
import pytest
from datetime import datetime

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.academic.base_agent import BaseAgent, AgentMessage
from agents.academic.communication_manager import CommunicationManager
from agents.academic.quality_manager import QualityManager


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_pdf_path(test_data_dir: Path) -> Path:
    """Return path to a sample PDF file for testing."""
    pdf_path = test_data_dir / "sample.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a minimal PDF file for testing if it doesn't exist
    if not pdf_path.exists():
        # Create a simple PDF content (this is a mock - in real scenarios you'd have actual PDFs)
        pdf_content = b"""%PDF-1.4
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
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
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
285
%%EOF"""
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)
    
    return pdf_path


@pytest.fixture(scope="session")
def sample_markdown_content() -> str:
    """Return sample markdown content for testing."""
    return """# Test Document

## Introduction

This is a test document for the academic agent system.

## Main Content

### Section 1

Here is some content for section 1 with important information.

### Section 2

Here is some content for section 2 with more details.

## Conclusion

This concludes the test document.
"""


@pytest.fixture(scope="function")
def sample_markdown_file(tmp_path: Path, sample_markdown_content: str) -> Path:
    """Create a temporary markdown file for testing."""
    md_file = tmp_path / "test_document.md"
    md_file.write_text(sample_markdown_content)
    return md_file


@pytest.fixture(scope="function")
def temp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory structure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    # Create subdirectories
    subdirs = ["raw", "markdown", "metadata", "analysis", "outlines", "notes", "processed"]
    for subdir in subdirs:
        (workspace / subdir).mkdir()
    
    return workspace


@pytest.fixture(scope="function")
def mock_groq_api_key() -> str:
    """Return a mock Groq API key for testing."""
    return "gsk_test_key_123456789"


@pytest.fixture(scope="function")
def mock_env_vars(mock_groq_api_key: str) -> Generator[Dict[str, str], None, None]:
    """Mock environment variables for testing."""
    env_vars = {
        "GROQ_API_KEY": mock_groq_api_key,
        "TESTING": "true",
        "LOG_LEVEL": "DEBUG"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture(scope="function")
def sample_analysis_data() -> Dict[str, Any]:
    """Return sample analysis data for testing."""
    return {
        "main_topics": ["Introduction", "Main Content", "Conclusion"],
        "key_concepts": [
            "Test document for academic agent system",
            "Important information in section 1",
            "More details in section 2"
        ],
        "structure": [
            {"title": "Introduction", "summary": "This is a test document..."},
            {"title": "Main Content", "summary": "Section 1 and 2 content..."},
            {"title": "Conclusion", "summary": "This concludes the test..."}
        ],
        "summary": "A comprehensive test document for the academic agent system",
        "source_file": "/path/to/test.md",
        "analysis_date": datetime.now().isoformat(),
        "sections": [
            {"title": "Introduction", "content": "This is a test document..."},
            {"title": "Main Content", "content": "Section 1 and 2 content..."},
            {"title": "Conclusion", "content": "This concludes the test..."}
        ]
    }


@pytest.fixture(scope="function")
def sample_outline_data() -> Dict[str, Any]:
    """Return sample outline data for testing."""
    return {
        "source_file": "/path/to/test.md",
        "generated_date": datetime.now().isoformat(),
        "sections": [
            {
                "title": "Introduction",
                "key_points": ["Test document overview", "Purpose statement"],
                "subsections": []
            },
            {
                "title": "Main Content",
                "key_points": ["Section 1 details", "Section 2 details"],
                "subsections": [
                    {"title": "Section 1", "key_points": ["Important information"]},
                    {"title": "Section 2", "key_points": ["More details"]}
                ]
            },
            {
                "title": "Conclusion",
                "key_points": ["Summary of findings", "Final thoughts"],
                "subsections": []
            }
        ]
    }


@pytest.fixture(scope="function")
def sample_notes_data() -> Dict[str, Any]:
    """Return sample notes data for testing."""
    return {
        "source_file": "/path/to/test.md",
        "generated_date": datetime.now().isoformat(),
        "sections": [
            {
                "title": "Introduction",
                "summary": "This is a test document for the academic agent system.",
                "key_concepts": ["Test document", "Academic agent system"],
                "detailed_notes": ["Line 1", "Line 2", "Line 3"]
            },
            {
                "title": "Main Content",
                "summary": "Contains two main sections with detailed information.",
                "key_concepts": ["Section 1", "Section 2", "Important information"],
                "detailed_notes": ["Content line 1", "Content line 2"]
            }
        ]
    }


@pytest.fixture(scope="function")
def mock_agent_message() -> AgentMessage:
    """Return a mock agent message for testing."""
    return AgentMessage(
        sender="test_agent",
        recipient="target_agent",
        message_type="test_message",
        content={"data": "test_data"},
        metadata={"test": True},
        timestamp=datetime.now(),
        priority=1
    )


@pytest.fixture(scope="function")
def mock_base_agent() -> BaseAgent:
    """Return a mock base agent for testing."""
    agent = BaseAgent("test_agent")
    agent.check_quality = Mock(return_value=0.8)
    agent.validate_input = Mock(return_value=True)
    agent.validate_output = Mock(return_value=True)
    return agent


@pytest.fixture(scope="function")
def mock_communication_manager() -> Mock:
    """Return a mock communication manager for testing."""
    manager = Mock(spec=CommunicationManager)
    manager.send_message = Mock(return_value=True)
    manager.receive_message = Mock(return_value=True)
    manager.broadcast_message = Mock(return_value=True)
    manager.get_agent_status = Mock(return_value={"status": "active"})
    return manager


@pytest.fixture(scope="function")
def mock_quality_manager() -> Mock:
    """Return a mock quality manager for testing."""
    manager = Mock(spec=QualityManager)
    manager.evaluate_quality = Mock(return_value={"score": 0.8, "feedback": []})
    manager.get_quality_threshold = Mock(return_value=0.7)
    manager.set_quality_threshold = Mock(return_value=True)
    return manager


@pytest.fixture(scope="function")
def mock_litellm_model() -> Mock:
    """Return a mock LiteLLM model for testing."""
    model = Mock()
    model.generate = Mock(return_value="Generated response")
    model.model_id = "groq/llama-3.3-70b-versatile"
    model.api_key = "test_key"
    return model


@pytest.fixture(scope="function")
def mock_docling_converter() -> Mock:
    """Return a mock Docling converter for testing."""
    converter = Mock()
    
    # Mock document result
    mock_document = Mock()
    mock_document.export_to_markdown.return_value = "# Test Document\n\nContent here"
    mock_document.title = "Test Document"
    mock_document.language = "en"
    
    mock_result = Mock()
    mock_result.document = mock_document
    
    converter.convert.return_value = mock_result
    return converter


@pytest.fixture(scope="function")
def mock_markdown_processor() -> Mock:
    """Return a mock markdown processor for testing."""
    processor = Mock()
    processor.clean_markdown = Mock(return_value="Cleaned markdown")
    processor.extract_sections = Mock(return_value=[
        {"title": "Section 1", "content": "Content 1"},
        {"title": "Section 2", "content": "Content 2"}
    ])
    processor.save_analysis = Mock(return_value={
        "markdown_path": "/path/to/analysis.md",
        "json_path": "/path/to/analysis.json"
    })
    return processor


@pytest.fixture(scope="function")
def quality_evaluation_response() -> Dict[str, Any]:
    """Return sample quality evaluation response."""
    return {
        "quality_score": 0.8,
        "feedback": ["Good structure", "Clear content"],
        "reasoning": "The content is well-organized and clearly written.",
        "assessment": "Good quality content with clear structure",
        "approved": True,
        "improvement_suggestions": []
    }


@pytest.fixture(scope="function")  
def pdf_processing_result() -> Dict[str, Any]:
    """Return sample PDF processing result."""
    return {
        "processed_files": [
            {
                "status": "success",
                "markdown_path": "/path/to/output.md",
                "metadata_path": "/path/to/output.json",
                "metadata": {
                    "source_file": "/path/to/input.pdf",
                    "processed_date": datetime.now().isoformat(),
                    "title": "Test Document",
                    "language": "en"
                }
            }
        ],
        "errors": [],
        "stats": {"total": 1, "success": 1, "failed": 0}
    }


@pytest.fixture(scope="function")
def analysis_processing_result() -> Dict[str, Any]:
    """Return sample analysis processing result."""
    return {
        "files_analyzed": [
            {
                "main_topics": ["Topic 1", "Topic 2"],
                "key_concepts": ["Concept 1", "Concept 2"],
                "structure": [{"title": "Section 1", "summary": "Summary"}],
                "summary": "Document summary",
                "source_file": "/path/to/input.md",
                "analysis_date": datetime.now().isoformat(),
                "sections": [{"title": "Section 1", "content": "Content"}]
            }
        ],
        "errors": [],
        "stats": {"total": 1, "success": 1, "failed": 0}
    }


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path) -> None:
    """Set up the test environment before each test."""
    # Create logs directory for testing
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Mock the logs directory path
    with patch('agents.academic.base_agent.Path') as mock_path:
        mock_path.return_value = logs_dir
        yield


@pytest.fixture(scope="function")
def disable_network() -> Generator[None, None, None]:
    """Disable network access for tests that shouldn't use it."""
    with patch('socket.socket') as mock_socket:
        mock_socket.side_effect = OSError("Network access disabled for testing")
        yield


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "pdf: PDF processing tests")
    config.addinivalue_line("markers", "agent: Agent-specific tests")
    config.addinivalue_line("markers", "quality: Quality control tests")
    config.addinivalue_line("markers", "network: Tests that require network access")
    config.addinivalue_line("markers", "requires_api: Tests that require API keys")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add markers based on test name
        if "pdf" in item.name.lower():
            item.add_marker(pytest.mark.pdf)
        if "agent" in item.name.lower():
            item.add_marker(pytest.mark.agent)
        if "quality" in item.name.lower():
            item.add_marker(pytest.mark.quality)
        if "network" in item.name.lower() or "api" in item.name.lower():
            item.add_marker(pytest.mark.network)
            item.add_marker(pytest.mark.requires_api)