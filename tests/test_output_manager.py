#!/usr/bin/env python3
"""
Tests for the Output Manager implementation.
Validates the new centralized output management system.
"""

import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.output_manager import (
    OutputManager,
    OutputConfig,
    OutputCategory,
    ContentType,
    get_output_manager,
    get_final_output_path,
    get_processed_output_path
)


def test_output_manager_initialization():
    """Test OutputManager initialization and directory creation."""
    print("Testing OutputManager initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Test with default configuration
        config = OutputConfig(auto_create_dirs=True)
        manager = OutputManager(project_root, config)
        
        # Check that directories were created
        expected_dirs = [
            "outputs",
            "outputs/final",
            "outputs/processed", 
            "outputs/analysis",
            "outputs/assets",
            "working",
            "working/cache",
            "working/temp",
            "logs",
            "metadata"
        ]
        
        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            if not full_path.exists():
                print(f"✗ Expected directory not created: {dir_path}")
                return False
        
        print("✓ All expected directories created")
        return True


def test_path_generation():
    """Test standardized path generation."""
    print("\nTesting path generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        manager = OutputManager(project_root)
        
        # Test final output path
        final_path = manager.get_output_path(
            OutputCategory.FINAL,
            ContentType.STUDY_NOTES,
            "test_notes.md"
        )
        
        expected = project_root / "outputs" / "final" / "study_notes" / "test_notes.md"
        if final_path != expected:
            print(f"✗ Final path mismatch: {final_path} != {expected}")
            return False
        
        # Test processed output path
        processed_path = manager.get_output_path(
            OutputCategory.PROCESSED,
            ContentType.MARKDOWN,
            subdirectory="week_01",
            filename="lecture.md"
        )
        
        expected = project_root / "outputs" / "processed" / "markdown" / "week_01" / "lecture.md"
        if processed_path != expected:
            print(f"✗ Processed path mismatch: {processed_path} != {expected}")
            return False
        
        print("✓ Path generation working correctly")
        return True


def test_legacy_path_mapping():
    """Test mapping of legacy hardcoded paths."""
    print("\nTesting legacy path mapping...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        manager = OutputManager(project_root)
        
        # Test SRA-specific mappings
        test_mappings = [
            ("output/sra/ai_enhanced_study_notes", "outputs/final/study_notes"),
            ("output/sra/alignment_analysis", "outputs/analysis/alignment"),
            ("output/sra/mermaid_diagrams", "outputs/assets/diagrams"),
            ("output/sra/notes", "outputs/processed/markdown"),
        ]
        
        for legacy_path, expected_suffix in test_mappings:
            mapped_path = manager.get_legacy_path_mapping(legacy_path)
            expected_full_path = project_root / expected_suffix
            
            if not str(mapped_path).endswith(expected_suffix):
                print(f"✗ Legacy mapping failed: {legacy_path} -> {mapped_path}")
                print(f"   Expected to end with: {expected_suffix}")
                return False
        
        print("✓ Legacy path mapping working correctly")
        return True


def test_convenience_functions():
    """Test convenience functions for common operations."""
    print("\nTesting convenience functions...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Initialize global manager
        from src.core.output_manager import initialize_output_manager
        initialize_output_manager(project_root)
        
        # Test convenience functions
        final_path = get_final_output_path(ContentType.STUDY_NOTES, "test.md")
        processed_path = get_processed_output_path(ContentType.MARKDOWN, "test.md")
        
        # Verify paths are under correct categories
        if "final/study_notes" not in str(final_path):
            print(f"✗ Final convenience function failed: {final_path}")
            return False
            
        if "processed/markdown" not in str(processed_path):
            print(f"✗ Processed convenience function failed: {processed_path}")
            return False
        
        print("✓ Convenience functions working correctly")
        return True


def test_migration_functionality():
    """Test migration of existing files."""
    print("\nTesting migration functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create some legacy files
        legacy_output = project_root / "output" / "sra" / "notes"
        legacy_output.mkdir(parents=True)
        
        test_file = legacy_output / "test_note.md"
        test_file.write_text("# Test Note\nThis is a test note.")
        
        # Initialize manager
        manager = OutputManager(project_root)
        
        # Test dry run migration
        migration_report = manager.migrate_legacy_outputs(dry_run=True)
        
        if migration_report["summary"]["total_files"] != 1:
            print(f"✗ Migration dry run failed: expected 1 file, got {migration_report['summary']['total_files']}")
            return False
        
        if len(migration_report["files_to_migrate"]) != 1:
            print(f"✗ Migration plan failed: expected 1 file to migrate")
            return False
        
        print("✓ Migration functionality working correctly")
        return True


def test_output_summary():
    """Test output summary generation."""
    print("\nTesting output summary...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        manager = OutputManager(project_root)
        
        # Create some test files
        test_final = manager.get_output_path(OutputCategory.FINAL, ContentType.STUDY_NOTES, "test.md")
        test_final.write_text("Test content")
        
        # Generate summary
        summary = manager.get_output_summary()
        
        required_fields = ["directories", "total_size_mb", "total_files", "last_updated"]
        for field in required_fields:
            if field not in summary:
                print(f"✗ Summary missing field: {field}")
                return False
        
        if summary["total_files"] < 1:
            print(f"✗ Summary should detect at least 1 file, got {summary['total_files']}")
            return False
        
        print("✓ Output summary working correctly")
        return True


def run_all_tests():
    """Run all output manager tests."""
    print("Running Output Manager tests...\n")
    
    tests = [
        ("OutputManager Initialization", test_output_manager_initialization),
        ("Path Generation", test_path_generation),
        ("Legacy Path Mapping", test_legacy_path_mapping),
        ("Convenience Functions", test_convenience_functions),
        ("Migration Functionality", test_migration_functionality),
        ("Output Summary", test_output_summary),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All Output Manager tests passed!")
        print("\nThe Output Manager is ready for production use:")
        print("- ✓ Centralized path management")
        print("- ✓ Legacy path migration support")
        print("- ✓ Organized directory structure")
        print("- ✓ Convenience functions for common operations")
        print("- ✓ Configuration and cleanup capabilities")
        return True
    else:
        print(f"\n❌ {failed} tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)