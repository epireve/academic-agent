"""Tests for file utility functions."""

import tempfile
from pathlib import Path

import pytest

from src.core.exceptions import ValidationError
from src.utils.file_utils import (
    clean_filename,
    ensure_directory,
    get_file_size_mb,
    get_files_by_extension,
    safe_copy,
)


class TestEnsureDirectory:
    """Test ensure_directory function."""

    def test_create_new_directory(self):
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            new_dir = tmp_path / "new_directory"

            result = ensure_directory(new_dir)

            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_existing_directory(self):
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            result = ensure_directory(tmp_path)

            assert result == tmp_path
            assert tmp_path.exists()
            assert tmp_path.is_dir()

    def test_nested_directory_creation(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            nested_dir = tmp_path / "level1" / "level2" / "level3"

            result = ensure_directory(nested_dir)

            assert result == nested_dir
            assert nested_dir.exists()
            assert nested_dir.is_dir()

    def test_existing_file_raises_error(self):
        """Test that existing file raises ValidationError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "existing_file.txt"
            file_path.write_text("test content")

            with pytest.raises(ValidationError):
                ensure_directory(file_path)


class TestSafeCopy:
    """Test safe_copy function."""

    def test_copy_file_success(self):
        """Test successful file copy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.txt"
            destination = tmp_path / "destination.txt"

            source.write_text("test content")

            result = safe_copy(source, destination)

            assert result == destination
            assert destination.exists()
            assert destination.read_text() == "test content"

    def test_copy_nonexistent_source_raises_error(self):
        """Test that non-existent source raises ValidationError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "nonexistent.txt"
            destination = tmp_path / "destination.txt"

            with pytest.raises(ValidationError):
                safe_copy(source, destination)

    def test_copy_existing_destination_without_overwrite_raises_error(self):
        """Test that existing destination without overwrite raises ValidationError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.txt"
            destination = tmp_path / "destination.txt"

            source.write_text("source content")
            destination.write_text("destination content")

            with pytest.raises(ValidationError):
                safe_copy(source, destination, overwrite=False)

    def test_copy_existing_destination_with_overwrite_succeeds(self):
        """Test that existing destination with overwrite succeeds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.txt"
            destination = tmp_path / "destination.txt"

            source.write_text("source content")
            destination.write_text("destination content")

            result = safe_copy(source, destination, overwrite=True)

            assert result == destination
            assert destination.read_text() == "source content"

    def test_copy_creates_destination_directory(self):
        """Test that copy creates destination directory if needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source = tmp_path / "source.txt"
            destination = tmp_path / "nested" / "dir" / "destination.txt"

            source.write_text("test content")

            result = safe_copy(source, destination)

            assert result == destination
            assert destination.exists()
            assert destination.read_text() == "test content"


class TestGetFilesByExtension:
    """Test get_files_by_extension function."""

    def test_get_files_basic(self):
        """Test basic file retrieval by extension."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test files
            (tmp_path / "file1.txt").write_text("content1")
            (tmp_path / "file2.txt").write_text("content2")
            (tmp_path / "file3.py").write_text("content3")

            result = get_files_by_extension(tmp_path, ".txt")

            assert len(result) == 2
            assert all(f.suffix == ".txt" for f in result)

    def test_get_files_without_dot(self):
        """Test file retrieval with extension without dot."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            (tmp_path / "file1.py").write_text("content1")
            (tmp_path / "file2.py").write_text("content2")

            result = get_files_by_extension(tmp_path, "py")

            assert len(result) == 2
            assert all(f.suffix == ".py" for f in result)

    def test_get_files_recursive(self):
        """Test recursive file retrieval."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create nested structure
            (tmp_path / "file1.txt").write_text("content1")
            nested_dir = tmp_path / "nested"
            nested_dir.mkdir()
            (nested_dir / "file2.txt").write_text("content2")

            result = get_files_by_extension(tmp_path, ".txt", recursive=True)

            assert len(result) == 2

    def test_get_files_nonexistent_directory(self):
        """Test with non-existent directory."""
        nonexistent = Path("/nonexistent/directory")

        result = get_files_by_extension(nonexistent, ".txt")

        assert result == []


class TestCleanFilename:
    """Test clean_filename function."""

    def test_clean_basic_filename(self):
        """Test cleaning basic filename."""
        result = clean_filename("normal_filename.txt")
        assert result == "normal_filename.txt"

    def test_clean_filename_with_invalid_chars(self):
        """Test cleaning filename with invalid characters."""
        result = clean_filename("file<>name:with|invalid?chars*.txt")
        assert result == "file_name_with_invalid_chars.txt"

    def test_clean_filename_with_multiple_underscores(self):
        """Test cleaning filename with multiple consecutive underscores."""
        result = clean_filename("file____with____underscores.txt")
        assert result == "file_with_underscores.txt"

    def test_clean_filename_with_leading_trailing_underscores(self):
        """Test cleaning filename with leading/trailing underscores."""
        result = clean_filename("___filename___.txt")
        assert result == "filename.txt"

    def test_clean_empty_filename(self):
        """Test cleaning empty filename."""
        result = clean_filename("")
        assert result == ""

    def test_clean_filename_only_invalid_chars(self):
        """Test cleaning filename with only invalid characters."""
        result = clean_filename("<>:|?*")
        assert result == ""


class TestGetFileSizeMb:
    """Test get_file_size_mb function."""

    def test_get_file_size_existing_file(self):
        """Test getting size of existing file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test.txt"

            # Create file with known content
            content = "x" * 1024  # 1KB
            test_file.write_text(content)

            result = get_file_size_mb(test_file)

            # Should be approximately 1KB = 0.001MB
            assert 0.0009 < result < 0.0011

    def test_get_file_size_nonexistent_file(self):
        """Test getting size of non-existent file."""
        nonexistent = Path("/nonexistent/file.txt")

        result = get_file_size_mb(nonexistent)

        assert result == 0.0

    def test_get_file_size_empty_file(self):
        """Test getting size of empty file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            empty_file = tmp_path / "empty.txt"
            empty_file.write_text("")

            result = get_file_size_mb(empty_file)

            assert result == 0.0
