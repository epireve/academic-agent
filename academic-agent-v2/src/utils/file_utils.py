"""File utility functions for Academic Agent v2."""

import shutil
from pathlib import Path
from typing import List

from ..core.exceptions import ValidationError
from ..core.logging import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path to create

    Returns:
        The created directory path

    Raises:
        ValidationError: If path exists but is not a directory
    """
    if path.exists() and not path.is_dir():
        raise ValidationError(f"Path exists but is not a directory: {path}")

    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def safe_copy(source: Path, destination: Path, overwrite: bool = False) -> Path:
    """Safely copy a file with error handling.

    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        Destination path

    Raises:
        ValidationError: If source doesn't exist or destination exists and
            overwrite is False
    """
    if not source.exists():
        raise ValidationError(f"Source file does not exist: {source}")

    if destination.exists() and not overwrite:
        raise ValidationError(f"Destination exists and overwrite is False: {destination}")

    # Ensure destination directory exists
    ensure_directory(destination.parent)

    shutil.copy2(source, destination)
    logger.debug(f"Copied file: {source} -> {destination}")
    return destination


def get_files_by_extension(directory: Path, extension: str, recursive: bool = False) -> List[Path]:
    """Get all files with specified extension in directory.

    Args:
        directory: Directory to search
        extension: File extension (with or without dot)
        recursive: Whether to search recursively

    Returns:
        List of matching file paths
    """
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    # Normalize extension
    if not extension.startswith("."):
        extension = f".{extension}"

    pattern = f"**/*{extension}" if recursive else f"*{extension}"
    files = list(directory.glob(pattern))

    logger.debug(f"Found {len(files)} files with extension {extension} in {directory}")
    return files


def clean_filename(filename: str) -> str:
    """Clean filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Cleaned filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    cleaned = filename
    for char in invalid_chars:
        cleaned = cleaned.replace(char, "_")

    # Remove multiple consecutive underscores
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    # Handle leading/trailing underscores more carefully
    # Split on dots to preserve file extensions
    parts = cleaned.split(".")
    if len(parts) > 1:
        # Has extension - clean base name and extension separately
        base_parts = parts[:-1]
        extension = parts[-1]
        cleaned_base = ".".join(base_parts).strip("_")
        cleaned_extension = extension.strip("_")
        cleaned = f"{cleaned_base}.{cleaned_extension}" if cleaned_base else cleaned_extension
    else:
        # No extension - just clean the whole thing
        cleaned = cleaned.strip("_")

    return cleaned


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    if not file_path.exists():
        return 0.0

    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)
