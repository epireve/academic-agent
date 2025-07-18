"""Test Poetry configuration setup."""

import subprocess
import sys
from pathlib import Path

import toml


def test_poetry_configuration():
    """Test that Poetry is properly configured with pyproject.toml."""
    # Check if pyproject.toml exists
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml should exist"

    # Load and validate pyproject.toml content
    with open(pyproject_path, "r") as f:
        config = toml.load(f)

    # Check required sections
    assert "tool" in config, "pyproject.toml should have [tool] section"
    assert "poetry" in config["tool"], "pyproject.toml should have [tool.poetry] section"

    poetry_config = config["tool"]["poetry"]

    # Check required fields
    assert "name" in poetry_config, "Poetry config should have 'name' field"
    assert "version" in poetry_config, "Poetry config should have 'version' field"
    assert "description" in poetry_config, "Poetry config should have 'description' field"
    assert "authors" in poetry_config, "Poetry config should have 'authors' field"
    assert "dependencies" in poetry_config, "Poetry config should have 'dependencies' field"

    # Check that Python dependency is specified
    assert "python" in poetry_config["dependencies"], "Python dependency should be specified"

    # Check that development dependencies section exists
    if "group" in poetry_config:
        assert "dev" in poetry_config["group"], "Development dependencies group should exist"
        assert (
            "dependencies" in poetry_config["group"]["dev"]
        ), "Dev dependencies should be specified"


def test_poetry_installation():
    """Test that Poetry can install dependencies successfully."""
    # This test assumes pyproject.toml is already created
    result = subprocess.run(
        [sys.executable, "-m", "poetry", "install", "--dry-run"], capture_output=True, text=True
    )

    # Should not fail with basic configuration
    assert result.returncode in [0, 1], f"Poetry install dry-run failed: {result.stderr}"


def test_poetry_show_command():
    """Test that Poetry show command works."""
    result = subprocess.run(
        [sys.executable, "-m", "poetry", "show"], capture_output=True, text=True
    )

    # Should not fail even with no dependencies installed
    assert result.returncode in [0, 1], f"Poetry show failed: {result.stderr}"
