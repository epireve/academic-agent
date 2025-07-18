#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Academic Agent v2.0 integration tests.
"""

import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_config():
    """Session-scoped test configuration."""
    return {
        "test_timeout": 60,
        "max_memory_usage_mb": 500,
        "enable_cleanup": True,
        "log_level": "DEBUG",
        "parallel_workers": 2
    }


@pytest.fixture(scope="function")
def temp_workspace():
    """Function-scoped temporary workspace."""
    temp_dir = Path(tempfile.mkdtemp(prefix="academic_agent_test_"))
    
    # Create standard test directories
    (temp_dir / "input").mkdir()
    (temp_dir / "output").mkdir()
    (temp_dir / "config").mkdir()
    (temp_dir / "logs").mkdir()
    
    yield temp_dir
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def mock_pdf_content():
    """Sample PDF content for testing."""
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
/Length 67
>>
stream
BT
/F1 12 Tf
72 720 Td
(Academic Agent Test PDF Content) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000195 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
312
%%EOF"""


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as memory intensive"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if "stress" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark memory intensive tests
        if "memory" in item.nodeid:
            item.add_marker(pytest.mark.memory_intensive)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Auto-used fixture to set up test environment."""
    # Suppress warnings during tests
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    
    yield
    
    # Cleanup after test
    pass  # Additional cleanup if needed