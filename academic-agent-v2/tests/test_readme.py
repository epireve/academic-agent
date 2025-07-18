"""Test README.md content and structure."""

import re
from pathlib import Path


def test_readme_exists():
    """Test that README.md exists."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md file should exist"


def test_readme_has_title():
    """Test that README.md has a proper title."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    assert content.startswith("# Academic Agent v2.0"), "README should start with project title"


def test_readme_has_required_sections():
    """Test that README.md contains all required sections."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    required_sections = [
        "## Project Overview",
        "## Features",
        "## Architecture",
        "## Installation",
        "## Usage",
        "## Development",
        "## Testing",
        "## Contributing",
        "## Performance",
        "## License",
    ]

    for section in required_sections:
        assert section in content, f"README should contain section: {section}"


def test_readme_has_installation_instructions():
    """Test that README.md contains installation instructions."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for key installation commands
    assert "git clone" in content, "README should contain git clone instruction"
    assert "poetry install" in content, "README should contain poetry install instruction"
    assert "poetry shell" in content, "README should contain poetry shell instruction"


def test_readme_has_usage_examples():
    """Test that README.md contains usage examples."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for usage examples
    assert "```python" in content, "README should contain Python code examples"
    assert "```bash" in content, "README should contain bash command examples"


def test_readme_has_testing_instructions():
    """Test that README.md contains testing instructions."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for testing commands
    assert "pytest" in content, "README should mention pytest"
    assert "coverage" in content, "README should mention coverage"


def test_readme_has_contributing_guidelines():
    """Test that README.md contains contributing guidelines."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for contributing guidelines
    assert "TDD" in content, "README should mention TDD"
    assert "pre-commit" in content, "README should mention pre-commit"
    assert "type hints" in content, "README should mention type hints"


def test_readme_has_performance_metrics():
    """Test that README.md contains performance information."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for performance metrics
    assert "25 pages/second" in content, "README should mention processing speed"
    assert "50%" in content, "README should mention performance improvement"


def test_readme_has_valid_markdown():
    """Test that README.md has valid markdown structure."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for proper markdown header structure
    lines = content.split("\n")
    header_lines = [line for line in lines if line.startswith("#")]

    assert len(header_lines) > 0, "README should have headers"

    # Check that main title is h1
    assert header_lines[0].startswith("# "), "First header should be h1"

    # Check for proper code blocks
    code_blocks = re.findall(r"```\w+", content)
    assert len(code_blocks) > 0, "README should have code blocks"


def test_readme_has_project_badges():
    """Test that README.md contains project badges."""
    readme_path = Path("README.md")
    content = readme_path.read_text()

    # Check for badges (shields.io style)
    assert "![" in content, "README should contain badges or images"
    assert "https://img.shields.io" in content, "README should contain shields.io badges"
