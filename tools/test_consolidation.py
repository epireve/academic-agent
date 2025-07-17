#!/usr/bin/env python
"""
Test script for the Content Consolidation Agent
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the agents directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

try:
    from academic.consolidation_agent import ContentConsolidationAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies:")
    print("pip install python-frontmatter python-magic-bin")
    sys.exit(1)


def create_test_files():
    """Create test files for consolidation testing"""
    # Create temporary directory structure
    test_dir = tempfile.mkdtemp(prefix="consolidation_test_")
    
    # Create test directories
    transcript_dir = os.path.join(test_dir, "transcripts")
    lecture_dir = os.path.join(test_dir, "lectures")
    notes_dir = os.path.join(test_dir, "notes")
    
    os.makedirs(transcript_dir)
    os.makedirs(lecture_dir)
    os.makedirs(notes_dir)
    
    # Create test files
    test_files = [
        # Transcript files
        (os.path.join(transcript_dir, "week-1-transcript.md"), "transcript", 1, 
         "# Week 1 Transcript\nThis is the transcript for week 1."),
        (os.path.join(transcript_dir, "week-2-class-notes.md"), "transcript", 2,
         "# Week 2 Class Notes\nTranscript content for week 2."),
        (os.path.join(transcript_dir, "w3-recording.md"), "transcript", 3,
         "# Week 3 Recording\nRecording transcript for week 3."),
        
        # Lecture files
        (os.path.join(lecture_dir, "lecture-1.md"), "lecture", 1,
         "# Lecture 1\nSlides and content for lecture 1."),
        (os.path.join(lecture_dir, "lecture-2.md"), "lecture", 2,
         "# Lecture 2\nSlides and content for lecture 2."),
        
        # Notes files
        (os.path.join(notes_dir, "week1-notes.md"), "notes", 1,
         "# Week 1 Notes\nStudent notes for week 1."),
        (os.path.join(notes_dir, "week2-summary.md"), "notes", 2,
         "# Week 2 Summary\nSummary notes for week 2."),
        
        # Duplicate files (should be resolved)
        (os.path.join(transcript_dir, "week-1-duplicate.md"), "transcript", 1,
         "# Week 1 Duplicate\nDuplicate transcript for week 1."),
        
        # Files without clear week numbers
        (os.path.join(notes_dir, "general-notes.md"), "notes", None,
         "# General Notes\nGeneral course notes."),
    ]
    
    # Create the test files
    for file_path, content_type, week, content in test_files:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return test_dir, [transcript_dir, lecture_dir, notes_dir]


def test_file_scanning():
    """Test file scanning functionality"""
    print("Testing file scanning...")
    
    # Create test files
    test_dir, search_paths = create_test_files()
    
    try:
        # Create agent
        agent = ContentConsolidationAgent()
        
        # Test file scanning
        discovered_files = agent.scan_locations(search_paths)
        
        print(f"Discovered {len(discovered_files)} files")
        
        # Check if files were discovered correctly
        assert len(discovered_files) > 0, "No files discovered"
        
        # Check content types
        content_types = set(f["content_type"] for f in discovered_files)
        expected_types = {"transcript", "lecture", "notes"}
        assert expected_types.issubset(content_types), f"Missing content types: {expected_types - content_types}"
        
        # Check week number extraction
        week_files = [f for f in discovered_files if f["week_number"] is not None]
        assert len(week_files) > 0, "No week numbers extracted"
        
        print("✅ File scanning test passed")
        
    finally:
        shutil.rmtree(test_dir)


def test_naming_resolution():
    """Test naming conflict resolution"""
    print("Testing naming conflict resolution...")
    
    # Create test files
    test_dir, search_paths = create_test_files()
    
    try:
        # Create agent
        agent = ContentConsolidationAgent()
        
        # Discover files
        discovered_files = agent.scan_locations(search_paths)
        
        # Test naming resolution
        file_mappings = agent.resolve_naming_conflicts(discovered_files)
        
        print(f"Created {len(file_mappings)} file mappings")
        
        # Check if mappings were created
        assert len(file_mappings) > 0, "No file mappings created"
        
        # Check for duplicate resolution
        week_1_transcripts = [m for m in file_mappings if m.week_number == 1 and m.content_type == "transcript"]
        
        # Should have resolved to single mapping for week 1 transcripts
        assert len(week_1_transcripts) == 1, f"Expected 1 week 1 transcript mapping, got {len(week_1_transcripts)}"
        
        # Check if duplicates were noted
        mapping = week_1_transcripts[0]
        assert mapping.metadata and "duplicates" in mapping.metadata, "Duplicates not recorded in metadata"
        
        print("✅ Naming conflict resolution test passed")
        
    finally:
        shutil.rmtree(test_dir)


def test_content_merging():
    """Test content merging functionality"""
    print("Testing content merging...")
    
    # Create test files
    test_dir, search_paths = create_test_files()
    
    try:
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="consolidation_output_")
        
        # Create agent
        agent = ContentConsolidationAgent()
        
        # Run full consolidation workflow
        result = agent.consolidate_workflow(search_paths, output_dir)
        
        print(f"Consolidation result: Success={result.success}")
        print(f"Processed files: {len(result.processed_files)}")
        print(f"Skipped files: {len(result.skipped_files)}")
        print(f"Errors: {len(result.errors)}")
        
        # Check if consolidation was successful
        assert result.success, f"Consolidation failed: {result.errors}"
        
        # Check if files were processed
        assert len(result.processed_files) > 0, "No files were processed"
        
        # Check if unified structure was created
        assert result.unified_structure, "Unified structure not created"
        
        # Check if files exist in output
        for mapping in result.processed_files:
            target_path = os.path.join(output_dir, mapping.target_path)
            assert os.path.exists(target_path), f"Target file not created: {target_path}"
        
        # Check if report was generated
        report_path = os.path.join(output_dir, "consolidation_report.json")
        assert os.path.exists(report_path), "Consolidation report not generated"
        
        print("✅ Content merging test passed")
        
        # Clean up output directory
        shutil.rmtree(output_dir)
        
    finally:
        shutil.rmtree(test_dir)


def test_quality_scoring():
    """Test quality scoring functionality"""
    print("Testing quality scoring...")
    
    # Create test files
    test_dir, search_paths = create_test_files()
    
    try:
        # Create output directory
        output_dir = tempfile.mkdtemp(prefix="consolidation_output_")
        
        # Create agent
        agent = ContentConsolidationAgent()
        
        # Run consolidation
        result = agent.consolidate_workflow(search_paths, output_dir)
        
        # Test quality scoring
        quality_score = agent.check_quality({"consolidation_result": result})
        
        print(f"Quality score: {quality_score:.2f}")
        
        # Check if quality score is reasonable
        assert 0.0 <= quality_score <= 1.0, f"Quality score out of range: {quality_score}"
        
        # For successful consolidation, quality should be decent
        if result.success and not result.errors:
            assert quality_score > 0.5, f"Quality score too low for successful consolidation: {quality_score}"
        
        print("✅ Quality scoring test passed")
        
        # Clean up
        shutil.rmtree(output_dir)
        
    finally:
        shutil.rmtree(test_dir)


def run_all_tests():
    """Run all tests"""
    print("Running Content Consolidation Agent Tests")
    print("=" * 50)
    
    tests = [
        test_file_scanning,
        test_naming_resolution,
        test_content_merging,
        test_quality_scoring
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {str(e)}")
            failed += 1
        
        print()
    
    print("Test Results:")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tests)}")
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)