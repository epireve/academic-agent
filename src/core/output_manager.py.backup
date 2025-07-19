#!/usr/bin/env python3
"""
Centralized Output Management System for Academic Agent

This module provides a unified approach to managing all output files,
replacing hardcoded paths with a configurable, organized structure.
Part of the unified architecture refactoring effort.
"""

import os
import shutil
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import json

from .logging import get_logger
from .exceptions import AcademicAgentError, ValidationError


class OutputCategory(Enum):
    """Standard output categories for organized file management."""
    FINAL = "final"
    PROCESSED = "processed"
    ANALYSIS = "analysis"
    ASSETS = "assets"
    WORKING = "working"
    LOGS = "logs"
    METADATA = "metadata"
    CACHE = "cache"


class ContentType(Enum):
    """Content types for subcategory organization."""
    # Final outputs
    STUDY_NOTES = "study_notes"
    REPORTS = "reports"
    EXPORTS = "exports"
    SUMMARIES = "summaries"
    
    # Processed content
    MARKDOWN = "markdown"
    ENHANCED = "enhanced"
    INTEGRATED = "integrated"
    RESOLVED = "resolved"
    
    # Analysis outputs
    ALIGNMENT = "alignment"
    QUALITY = "quality"
    CONSOLIDATION = "consolidation"
    VALIDATION = "validation"
    
    # Assets
    IMAGES = "images"
    DIAGRAMS = "diagrams"
    TABLES = "tables"
    MEDIA = "media"
    
    # Working files
    TEMP = "temp"
    STAGING = "staging"


@dataclass
class OutputConfig:
    """Configuration for output management."""
    base_directory: str = "outputs"
    working_directory: str = "working"
    logs_directory: str = "logs"
    metadata_directory: str = "metadata"
    auto_create_dirs: bool = True
    cleanup_temp_files: bool = True
    temp_file_max_age_hours: int = 24
    archive_old_outputs: bool = True
    archive_max_age_days: int = 30
    max_cache_size_mb: int = 1000


@dataclass
class OutputLocation:
    """Represents a specific output location with metadata."""
    path: Path
    category: OutputCategory
    content_type: Optional[ContentType] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    file_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OutputManager:
    """
    Centralized manager for all output file operations.
    
    Provides:
    - Standardized output directory structure
    - Path resolution with validation
    - Automatic directory creation
    - Temporary file cleanup
    - Output organization and archiving
    """
    
    def __init__(self, project_root: Union[str, Path], config: Optional[OutputConfig] = None):
        """
        Initialize the output manager.
        
        Args:
            project_root: Root directory of the project
            config: Optional configuration for output management
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or OutputConfig()
        self.logger = get_logger("output_manager")
        
        # Initialize directory structure
        self._setup_directories()
        
        # Track output locations
        self._output_locations: Dict[str, OutputLocation] = {}
        
        self.logger.info(f"Output manager initialized with root: {self.project_root}")
    
    def _setup_directories(self):
        """Set up the standard directory structure."""
        # Main output directories
        self.outputs_dir = self.project_root / self.config.base_directory
        self.working_dir = self.project_root / self.config.working_directory
        self.logs_dir = self.project_root / self.config.logs_directory
        self.metadata_dir = self.project_root / self.config.metadata_directory
        
        # Specific category directories
        self.final_dir = self.outputs_dir / "final"
        self.processed_dir = self.outputs_dir / "processed"
        self.analysis_dir = self.outputs_dir / "analysis"
        self.assets_dir = self.outputs_dir / "assets"
        self.cache_dir = self.working_dir / "cache"
        self.temp_dir = self.working_dir / "temp"
        self.staging_dir = self.working_dir / "staging"
        
        # Create directories if enabled
        if self.config.auto_create_dirs:
            self._create_standard_directories()
    
    def _create_standard_directories(self):
        """Create all standard directories."""
        directories = [
            # Main directories
            self.outputs_dir,
            self.working_dir,
            self.logs_dir,
            self.metadata_dir,
            
            # Final outputs
            self.final_dir / "study_notes",
            self.final_dir / "reports", 
            self.final_dir / "exports",
            self.final_dir / "summaries",
            
            # Processed content
            self.processed_dir / "markdown",
            self.processed_dir / "enhanced",
            self.processed_dir / "integrated",
            self.processed_dir / "resolved",
            
            # Analysis outputs
            self.analysis_dir / "alignment",
            self.analysis_dir / "quality",
            self.analysis_dir / "consolidation",
            self.analysis_dir / "validation",
            
            # Assets
            self.assets_dir / "images",
            self.assets_dir / "diagrams",
            self.assets_dir / "tables",
            self.assets_dir / "media",
            
            # Working directories
            self.cache_dir,
            self.temp_dir,
            self.staging_dir,
            
            # Logs
            self.logs_dir / "processing",
            self.logs_dir / "errors",
            self.logs_dir / "performance",
            
            # Metadata
            self.metadata_dir / "extracted",
            self.metadata_dir / "generated",
            self.metadata_dir / "reports",
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {directory}")
            except Exception as e:
                self.logger.error(f"Failed to create directory {directory}: {e}")
                raise AcademicAgentError(f"Directory creation failed: {e}")
    
    def get_output_path(
        self,
        category: OutputCategory,
        content_type: Optional[ContentType] = None,
        filename: Optional[str] = None,
        subdirectory: Optional[str] = None,
        create_parents: bool = True
    ) -> Path:
        """
        Get a standardized output path.
        
        Args:
            category: Output category (final, processed, etc.)
            content_type: Content type for subcategorization
            filename: Optional filename to append
            subdirectory: Optional subdirectory within the content type
            create_parents: Whether to create parent directories
            
        Returns:
            Path object for the requested location
            
        Raises:
            ValidationError: If the path configuration is invalid
        """
        # Build base path
        if category == OutputCategory.FINAL:
            base_path = self.final_dir
        elif category == OutputCategory.PROCESSED:
            base_path = self.processed_dir
        elif category == OutputCategory.ANALYSIS:
            base_path = self.analysis_dir
        elif category == OutputCategory.ASSETS:
            base_path = self.assets_dir
        elif category == OutputCategory.WORKING:
            base_path = self.working_dir
        elif category == OutputCategory.LOGS:
            base_path = self.logs_dir
        elif category == OutputCategory.METADATA:
            base_path = self.metadata_dir
        elif category == OutputCategory.CACHE:
            base_path = self.cache_dir
        else:
            raise ValidationError(f"Unknown output category: {category}")
        
        # Add content type subdirectory
        if content_type:
            base_path = base_path / content_type.value
        
        # Add custom subdirectory
        if subdirectory:
            base_path = base_path / subdirectory
        
        # Add filename
        if filename:
            final_path = base_path / filename
        else:
            final_path = base_path
        
        # Create parent directories if requested
        if create_parents and filename:
            try:
                final_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create parent directories for {final_path}: {e}")
                raise AcademicAgentError(f"Directory creation failed: {e}")
        
        # Track the location
        location_key = str(base_path)
        if location_key not in self._output_locations:
            self._output_locations[location_key] = OutputLocation(
                path=base_path,
                category=category,
                content_type=content_type
            )
        
        self.logger.debug(f"Generated output path: {final_path}")
        return final_path
    
    def get_legacy_path_mapping(self, legacy_path: str) -> Path:
        """
        Map legacy hardcoded paths to new structure.
        
        Args:
            legacy_path: Old hardcoded path
            
        Returns:
            New path in standardized structure
        """
        legacy_mappings = {
            # SRA-specific mappings
            "output/sra/ai_enhanced_study_notes": (OutputCategory.FINAL, ContentType.STUDY_NOTES),
            "output/sra/alignment_analysis": (OutputCategory.ANALYSIS, ContentType.ALIGNMENT),
            "output/sra/enhanced_integrated_notes": (OutputCategory.PROCESSED, ContentType.ENHANCED),
            "output/sra/integrated_notes": (OutputCategory.PROCESSED, ContentType.INTEGRATED),
            "output/sra/lectures": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
            "output/sra/mermaid_diagrams": (OutputCategory.ASSETS, ContentType.DIAGRAMS),
            "output/sra/notes": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
            "output/sra/resolved_content": (OutputCategory.PROCESSED, ContentType.RESOLVED),
            "output/sra/textbook": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
            "output/sra/transcripts": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
            
            # General mappings
            "output": (OutputCategory.FINAL, None),
            "processed": (OutputCategory.PROCESSED, None),
            "markdown": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
            "logs": (OutputCategory.LOGS, None),
            "metadata": (OutputCategory.METADATA, None),
        }
        
        # Normalize the legacy path
        normalized_path = legacy_path.replace(str(self.project_root), "").strip("/")
        
        # Try exact match first
        if normalized_path in legacy_mappings:
            category, content_type = legacy_mappings[normalized_path]
            return self.get_output_path(category, content_type)
        
        # Try partial matches
        for legacy_pattern, (category, content_type) in legacy_mappings.items():
            if legacy_pattern in normalized_path:
                return self.get_output_path(category, content_type)
        
        # Default fallback
        self.logger.warning(f"No mapping found for legacy path: {legacy_path}")
        return self.get_output_path(OutputCategory.FINAL)
    
    def migrate_legacy_outputs(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Migrate existing output files to new structure.
        
        Args:
            dry_run: If True, only report what would be moved
            
        Returns:
            Migration report with details of actions taken
        """
        migration_report = {
            "dry_run": dry_run,
            "files_to_migrate": [],
            "directories_created": [],
            "errors": [],
            "summary": {}
        }
        
        # Find existing output directories
        legacy_dirs = ["output", "processed", "markdown", "raw", "metadata"]
        
        for legacy_dir_name in legacy_dirs:
            legacy_dir = self.project_root / legacy_dir_name
            
            if not legacy_dir.exists():
                continue
                
            self.logger.info(f"Scanning legacy directory: {legacy_dir}")
            
            for file_path in legacy_dir.rglob("*"):
                if file_path.is_file():
                    # Determine new location
                    relative_path = str(file_path.relative_to(self.project_root))
                    new_path = self.get_legacy_path_mapping(relative_path)
                    
                    # Add to migration list
                    migration_info = {
                        "source": str(file_path),
                        "destination": str(new_path / file_path.name),
                        "size": file_path.stat().st_size if file_path.exists() else 0
                    }
                    migration_report["files_to_migrate"].append(migration_info)
                    
                    if not dry_run:
                        try:
                            # Ensure destination directory exists
                            new_path.mkdir(parents=True, exist_ok=True)
                            
                            # Move the file
                            destination_file = new_path / file_path.name
                            shutil.move(str(file_path), str(destination_file))
                            
                            self.logger.info(f"Migrated: {file_path} -> {destination_file}")
                            
                        except Exception as e:
                            error_info = {
                                "file": str(file_path),
                                "error": str(e)
                            }
                            migration_report["errors"].append(error_info)
                            self.logger.error(f"Migration failed for {file_path}: {e}")
        
        # Generate summary
        migration_report["summary"] = {
            "total_files": len(migration_report["files_to_migrate"]),
            "total_size_mb": sum(f["size"] for f in migration_report["files_to_migrate"]) / (1024 * 1024),
            "errors_count": len(migration_report["errors"])
        }
        
        return migration_report
    
    def cleanup_temporary_files(self) -> Dict[str, Any]:
        """
        Clean up old temporary files based on configuration.
        
        Returns:
            Cleanup report with details of actions taken
        """
        cleanup_report = {
            "files_removed": [],
            "directories_removed": [],
            "total_size_freed_mb": 0,
            "errors": []
        }
        
        if not self.config.cleanup_temp_files:
            return cleanup_report
        
        max_age = timedelta(hours=self.config.temp_file_max_age_hours)
        cutoff_time = datetime.now() - max_age
        
        # Clean temp directory
        if self.temp_dir.exists():
            for file_path in self.temp_dir.rglob("*"):
                try:
                    if file_path.is_file():
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        
                        if file_mtime < cutoff_time:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            
                            cleanup_report["files_removed"].append(str(file_path))
                            cleanup_report["total_size_freed_mb"] += file_size / (1024 * 1024)
                            
                except Exception as e:
                    cleanup_report["errors"].append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    self.logger.error(f"Cleanup failed for {file_path}: {e}")
        
        self.logger.info(f"Cleanup completed: {len(cleanup_report['files_removed'])} files removed, "
                        f"{cleanup_report['total_size_freed_mb']:.2f} MB freed")
        
        return cleanup_report
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current output organization.
        
        Returns:
            Summary report with directory sizes and file counts
        """
        summary = {
            "directories": {},
            "total_size_mb": 0,
            "total_files": 0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Analyze each major directory
        major_dirs = [
            ("outputs", self.outputs_dir),
            ("working", self.working_dir),
            ("logs", self.logs_dir),
            ("metadata", self.metadata_dir)
        ]
        
        for dir_name, dir_path in major_dirs:
            if dir_path.exists():
                dir_info = self._analyze_directory(dir_path)
                summary["directories"][dir_name] = dir_info
                summary["total_size_mb"] += dir_info["size_mb"]
                summary["total_files"] += dir_info["file_count"]
        
        return summary
    
    def _analyze_directory(self, directory: Path) -> Dict[str, Any]:
        """Analyze a directory for size and file count."""
        total_size = 0
        file_count = 0
        subdirs = {}
        
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    file_count += 1
                    total_size += item.stat().st_size
                elif item.is_dir() and item.parent == directory:
                    # Analyze immediate subdirectories
                    subdir_info = self._analyze_directory(item)
                    subdirs[item.name] = subdir_info
        except Exception as e:
            self.logger.error(f"Error analyzing directory {directory}: {e}")
        
        return {
            "size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "subdirectories": subdirs
        }
    
    def save_configuration(self) -> Path:
        """Save current configuration to file."""
        config_path = self.metadata_dir / "output_manager_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "project_root": str(self.project_root),
            "config": {
                "base_directory": self.config.base_directory,
                "working_directory": self.config.working_directory,
                "logs_directory": self.config.logs_directory,
                "metadata_directory": self.config.metadata_directory,
                "auto_create_dirs": self.config.auto_create_dirs,
                "cleanup_temp_files": self.config.cleanup_temp_files,
                "temp_file_max_age_hours": self.config.temp_file_max_age_hours,
                "archive_old_outputs": self.config.archive_old_outputs,
                "archive_max_age_days": self.config.archive_max_age_days,
                "max_cache_size_mb": self.config.max_cache_size_mb
            },
            "created_at": datetime.now().isoformat(),
            "directory_structure": self.get_output_summary()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_path}")
        return config_path


# Global output manager instance
_output_manager: Optional[OutputManager] = None


def get_output_manager(project_root: Optional[Union[str, Path]] = None) -> OutputManager:
    """
    Get the global output manager instance.
    
    Args:
        project_root: Optional project root (used for initialization)
        
    Returns:
        OutputManager instance
    """
    global _output_manager
    
    if _output_manager is None:
        if project_root is None:
            # Try to detect project root
            current_dir = Path.cwd()
            if (current_dir / "src").exists():
                project_root = current_dir
            else:
                # Fallback to current directory
                project_root = current_dir
        
        _output_manager = OutputManager(project_root)
    
    return _output_manager


def initialize_output_manager(project_root: Union[str, Path], config: Optional[OutputConfig] = None):
    """
    Initialize the global output manager with specific configuration.
    
    Args:
        project_root: Project root directory
        config: Optional output configuration
    """
    global _output_manager
    _output_manager = OutputManager(project_root, config)


# Convenience functions for common operations

def get_final_output_path(content_type: ContentType, filename: str = None) -> Path:
    """Get path for final output files."""
    return get_output_manager().get_output_path(OutputCategory.FINAL, content_type, filename)


def get_processed_output_path(content_type: ContentType, filename: str = None) -> Path:
    """Get path for processed output files."""
    return get_output_manager().get_output_path(OutputCategory.PROCESSED, content_type, filename)


def get_analysis_output_path(content_type: ContentType, filename: str = None) -> Path:
    """Get path for analysis output files."""
    return get_output_manager().get_output_path(OutputCategory.ANALYSIS, content_type, filename)


def get_temp_path(filename: str = None) -> Path:
    """Get path for temporary files."""
    return get_output_manager().get_output_path(OutputCategory.WORKING, ContentType.TEMP, filename)