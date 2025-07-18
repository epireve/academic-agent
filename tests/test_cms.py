#!/usr/bin/env python
"""
Test suite for the Content Management System

This test suite verifies the functionality of the CMS implementation,
including course management, content operations, relationships, versioning,
search, and analytics.
"""

import os
import sys
import json
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the agents directory to the path
sys.path.append(str(Path(__file__).parent.parent / "agents"))

from academic.content_management_system import (
    ContentManagementSystem, CourseInfo, ContentItem, ContentType,
    ProcessingStatus, ChangeType, ContentRelationship, ProcessingRecord,
    ContentVersion
)
from academic.cms_integration import CMSIntegration


class TestContentManagementSystem(unittest.TestCase):
    """Test cases for the Content Management System"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.cms = ContentManagementSystem(self.test_dir)
        
        # Create test course
        self.test_course = CourseInfo(
            course_id="test_course_001",
            course_name="Test Course",
            course_code="TEST101",
            academic_year="2023",
            semester="Fall",
            instructor="Test Instructor",
            department="Computer Science",
            description="A test course for CMS testing"
        )
        
        # Create test content item
        self.test_content = ContentItem(
            content_id="test_content_001",
            title="Test Content",
            content_type=ContentType.LECTURE,
            course_id="test_course_001",
            file_path=str(Path(self.test_dir) / "test_file.md"),
            original_filename="test_file.md",
            file_size=1024,
            file_hash="abc123def456",
            mime_type="text/markdown",
            week_number=1,
            tags=["test", "lecture"],
            keywords=["testing", "cms", "content"],
            description="Test content for CMS testing"
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.cms.shutdown()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_course_management(self):
        """Test course creation and retrieval"""
        # Test course creation
        success = self.cms.create_course(self.test_course)
        self.assertTrue(success, "Course creation should succeed")
        
        # Test course retrieval
        retrieved_course = self.cms.get_course(self.test_course.course_id)
        self.assertIsNotNone(retrieved_course, "Retrieved course should not be None")
        self.assertEqual(retrieved_course.course_id, self.test_course.course_id)
        self.assertEqual(retrieved_course.course_name, self.test_course.course_name)
        
        # Test course listing
        courses = self.cms.list_courses()
        self.assertGreater(len(courses), 0, "Should have at least one course")
        
        course_ids = [course.course_id for course in courses]
        self.assertIn(self.test_course.course_id, course_ids)
    
    def test_content_management(self):
        """Test content creation and retrieval"""
        # First create the course
        self.cms.create_course(self.test_course)
        
        # Create test file
        test_file_content = "# Test Content\\n\\nThis is test content for CMS testing."
        with open(self.test_content.file_path, 'w', encoding='utf-8') as f:
            f.write(test_file_content)
        
        # Test content creation
        success = self.cms.add_content(self.test_content)
        self.assertTrue(success, "Content creation should succeed")
        
        # Test content retrieval
        retrieved_content = self.cms.get_content(self.test_content.content_id)
        self.assertIsNotNone(retrieved_content, "Retrieved content should not be None")
        self.assertEqual(retrieved_content.content_id, self.test_content.content_id)
        self.assertEqual(retrieved_content.title, self.test_content.title)
        self.assertEqual(retrieved_content.content_type, self.test_content.content_type)
        
        # Test content listing
        content_list = self.cms.list_content(course_id=self.test_course.course_id)
        self.assertGreater(len(content_list), 0, "Should have at least one content item")
        
        content_ids = [item.content_id for item in content_list]
        self.assertIn(self.test_content.content_id, content_ids)
        
        # Test content filtering
        lecture_content = self.cms.list_content(
            course_id=self.test_course.course_id,
            content_type=ContentType.LECTURE
        )
        self.assertGreater(len(lecture_content), 0, "Should have lecture content")
        
        week1_content = self.cms.list_content(
            course_id=self.test_course.course_id,
            week_number=1
        )
        self.assertGreater(len(week1_content), 0, "Should have week 1 content")
    
    def test_search_functionality(self):
        """Test search and indexing functionality"""
        # Set up content for search
        self.cms.create_course(self.test_course)
        
        # Create test file with searchable content
        test_file_content = "# Algorithms and Data Structures\\n\\nThis lecture covers sorting algorithms, binary trees, and hash tables."
        with open(self.test_content.file_path, 'w', encoding='utf-8') as f:
            f.write(test_file_content)
        
        self.test_content.keywords = ["algorithms", "data structures", "sorting", "trees"]
        self.cms.add_content(self.test_content)
        
        # Build search index
        success = self.cms.build_search_index()
        self.assertTrue(success, "Search index building should succeed")
        
        # Test search functionality
        search_results = self.cms.search_content(
            query="algorithms",
            course_id=self.test_course.course_id
        )
        
        self.assertGreater(len(search_results), 0, "Should find search results")
        
        # Check if our content is in the results
        found_content_ids = [item[0].content_id for item in search_results]
        self.assertIn(self.test_content.content_id, found_content_ids)
        
        # Test search with filters
        filtered_results = self.cms.search_content(
            query="algorithms",
            course_id=self.test_course.course_id,
            content_type=ContentType.LECTURE,
            week_number=1
        )
        
        self.assertGreater(len(filtered_results), 0, "Should find filtered results")
    
    def test_relationship_detection(self):
        """Test relationship detection and management"""
        # Set up content for relationship testing
        self.cms.create_course(self.test_course)
        
        # Create first content item (lecture)
        lecture_content = "# Week 1 Lecture: Introduction to Algorithms\\n\\nBasic concepts and sorting algorithms."
        with open(self.test_content.file_path, 'w', encoding='utf-8') as f:
            f.write(lecture_content)
        
        self.cms.add_content(self.test_content)
        
        # Create second content item (notes)
        notes_content = ContentItem(
            content_id="test_content_002",
            title="Week 1 Notes",
            content_type=ContentType.NOTES,
            course_id="test_course_001",
            file_path=str(Path(self.test_dir) / "test_notes.md"),
            original_filename="test_notes.md",
            file_size=512,
            file_hash="def456abc789",
            mime_type="text/markdown",
            week_number=1,
            tags=["notes", "week1"],
            keywords=["algorithms", "sorting", "notes"],
            description="Notes for Week 1 lecture"
        )
        
        notes_file_content = "# Week 1 Notes\\n\\nKey points from the algorithms lecture:\\n- Sorting algorithms\\n- Big O notation"
        with open(notes_content.file_path, 'w', encoding='utf-8') as f:
            f.write(notes_file_content)
        
        self.cms.add_content(notes_content)
        
        # Test automatic relationship detection
        relationships = self.cms.detect_relationships(self.test_content.content_id)
        self.assertGreater(len(relationships), 0, "Should detect relationships")
        
        # Test manual relationship addition
        manual_relationship = ContentRelationship(
            relationship_id="test_rel_001",
            source_content_id=self.test_content.content_id,
            target_content_id=notes_content.content_id,
            relationship_type="notes_for",
            strength=0.9,
            description="Notes for the lecture",
            auto_detected=False
        )
        
        success = self.cms.add_relationship(manual_relationship)
        self.assertTrue(success, "Manual relationship addition should succeed")
        
        # Test relationship retrieval
        retrieved_relationships = self.cms.get_relationships(self.test_content.content_id)
        self.assertGreater(len(retrieved_relationships), 0, "Should retrieve relationships")
        
        relationship_ids = [rel.relationship_id for rel in retrieved_relationships]
        self.assertIn(manual_relationship.relationship_id, relationship_ids)
    
    def test_processing_history(self):
        """Test processing history and audit trails"""
        # Set up content
        self.cms.create_course(self.test_course)
        self.cms.add_content(self.test_content)
        
        # Create processing record
        processing_record = ProcessingRecord(
            record_id="test_record_001",
            content_id=self.test_content.content_id,
            operation="content_import",
            agent_id="test_agent",
            input_data={"source_path": "/test/path"},
            output_data={"success": True, "content_id": self.test_content.content_id},
            processing_time=1.5,
            success=True,
            quality_score=0.85,
            metadata={"test": True}
        )
        
        # Test recording processing history
        success = self.cms.record_processing(processing_record)
        self.assertTrue(success, "Processing record creation should succeed")
        
        # Test retrieving processing history
        history = self.cms.get_processing_history(content_id=self.test_content.content_id)
        self.assertGreater(len(history), 0, "Should have processing history")
        
        # Check if our record is in the history
        record_ids = [record.record_id for record in history]
        self.assertIn(processing_record.record_id, record_ids)
        
        # Test filtering by agent
        agent_history = self.cms.get_processing_history(agent_id="test_agent")
        self.assertGreater(len(agent_history), 0, "Should have agent-specific history")
        
        # Test filtering by operation
        operation_history = self.cms.get_processing_history(operation="content_import")
        self.assertGreater(len(operation_history), 0, "Should have operation-specific history")
    
    def test_version_control(self):
        """Test version control functionality"""
        # Set up content
        self.cms.create_course(self.test_course)
        
        # Create test file
        original_content = "# Original Content\\n\\nThis is the original version."
        with open(self.test_content.file_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        self.cms.add_content(self.test_content)
        
        # Create first version
        version_id = self.cms.create_version(
            self.test_content.content_id,
            ChangeType.CREATE,
            "Initial version",
            "test_author"
        )
        
        self.assertIsNotNone(version_id, "Version creation should succeed")
        
        # Update content and create new version
        updated_content = "# Updated Content\\n\\nThis is the updated version with new information."
        with open(self.test_content.file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        version_id_2 = self.cms.create_version(
            self.test_content.content_id,
            ChangeType.UPDATE,
            "Added new information",
            "test_author"
        )
        
        self.assertIsNotNone(version_id_2, "Second version creation should succeed")
        
        # Test version retrieval
        versions = self.cms.get_versions(self.test_content.content_id)
        self.assertGreaterEqual(len(versions), 2, "Should have at least 2 versions")
        
        # Check version numbers
        version_numbers = [v.version_number for v in versions]
        self.assertIn(1, version_numbers)
        self.assertIn(2, version_numbers)
    
    def test_quality_assessment_integration(self):
        """Test quality assessment integration"""
        # Set up content
        self.cms.create_course(self.test_course)
        
        # Create content with quality-assessable text
        quality_content = "# High Quality Content\\n\\nThis is well-structured content with clear headings, detailed explanations, and proper formatting.\\n\\n## Key Points\\n\\n- Point 1\\n- Point 2\\n- Point 3"
        with open(self.test_content.file_path, 'w', encoding='utf-8') as f:
            f.write(quality_content)
        
        self.cms.add_content(self.test_content)
        
        # Test quality assessment
        evaluation = self.cms.quality_assessment(self.test_content.content_id)
        
        self.assertIsNotNone(evaluation, "Quality evaluation should not be None")
        self.assertIsInstance(evaluation.quality_score, float)
        self.assertGreaterEqual(evaluation.quality_score, 0.0)
        self.assertLessEqual(evaluation.quality_score, 1.0)
        
        # Check if quality score was updated in content
        updated_content = self.cms.get_content(self.test_content.content_id)
        self.assertIsNotNone(updated_content.quality_score)
        self.assertEqual(updated_content.quality_score, evaluation.quality_score)
    
    def test_analytics_generation(self):
        """Test analytics report generation"""
        # Set up test data
        self.cms.create_course(self.test_course)
        self.cms.add_content(self.test_content)
        
        # Add some processing history
        processing_record = ProcessingRecord(
            record_id="analytics_test_record",
            content_id=self.test_content.content_id,
            operation="test_operation",
            agent_id="test_agent",
            input_data={"test": "data"},
            output_data={"success": True},
            processing_time=1.0,
            success=True,
            quality_score=0.8
        )
        self.cms.record_processing(processing_record)
        
        # Generate analytics report
        report = self.cms.generate_analytics_report(course_id=self.test_course.course_id)
        
        self.assertIsNotNone(report, "Analytics report should not be None")
        self.assertIn('report_id', report)
        self.assertIn('content_statistics', report)
        self.assertIn('processing_statistics', report)
        
        # Check content statistics
        content_stats = report['content_statistics']
        self.assertGreater(content_stats['total_content'], 0)
        self.assertIn('content_by_type', content_stats)
        
        # Check processing statistics
        processing_stats = report['processing_statistics']
        self.assertGreaterEqual(processing_stats['total_operations'], 0)


class TestCMSIntegration(unittest.TestCase):
    """Test cases for CMS Integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.integration = CMSIntegration(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        self.integration.shutdown()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_content_import_request(self):
        """Test content import request processing"""
        # Create test content directory
        test_content_dir = Path(self.test_dir) / "test_content"
        test_content_dir.mkdir(exist_ok=True)
        
        # Create test markdown file
        test_file = test_content_dir / "week_01_lecture.md"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("# Week 1 Lecture\\n\\nIntroduction to the course.")
        
        # Process content import request
        request = {
            'course_id': 'integration_test_course',
            'course_name': 'Integration Test Course',
            'source_paths': [str(test_content_dir)],
            'content_type': 'lecture'
        }
        
        result = self.integration.process_content_import_request(request)
        
        self.assertTrue(result['success'], "Content import should succeed")
        self.assertIn('course_id', result)
        self.assertEqual(result['course_id'], request['course_id'])
    
    def test_search_request(self):
        """Test content search request processing"""
        # Set up test content first
        course_id = 'search_test_course'
        course = CourseInfo(
            course_id=course_id,
            course_name="Search Test Course",
            course_code="SEARCH101",
            academic_year="2023",
            semester="Fall",
            instructor="Test Instructor",
            department="Computer Science"
        )
        
        self.integration.cms.create_course(course)
        
        # Add test content
        content_item = ContentItem(
            content_id="search_test_content",
            title="Searchable Content",
            content_type=ContentType.LECTURE,
            course_id=course_id,
            file_path=str(Path(self.test_dir) / "search_test.md"),
            original_filename="search_test.md",
            file_size=100,
            file_hash="search123",
            mime_type="text/markdown",
            keywords=["search", "test", "content"]
        )
        
        # Create test file
        with open(content_item.file_path, 'w', encoding='utf-8') as f:
            f.write("# Searchable Content\\n\\nThis content can be searched.")
        
        self.integration.cms.add_content(content_item)
        self.integration.cms.build_search_index()
        
        # Process search request
        search_request = {
            'query': 'searchable',
            'course_id': course_id,
            'limit': 10
        }
        
        result = self.integration.process_content_search_request(search_request)
        
        self.assertTrue(result['success'], "Search request should succeed")
        self.assertIn('results', result)
        self.assertIn('query', result)
        self.assertEqual(result['query'], search_request['query'])
    
    def test_cms_status(self):
        """Test CMS status retrieval"""
        status = self.integration.get_cms_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('total_courses', status)
        self.assertIn('total_content', status)
        self.assertIn('content_by_type', status)
        self.assertIn('cms_agent_status', status)


def run_performance_tests():
    """Run performance tests for CMS operations"""
    print("\\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    test_dir = tempfile.mkdtemp()
    try:
        cms = ContentManagementSystem(test_dir)
        
        # Test course creation performance
        start_time = datetime.now()
        for i in range(100):
            course = CourseInfo(
                course_id=f"perf_course_{i}",
                course_name=f"Performance Test Course {i}",
                course_code=f"PERF{i:03d}",
                academic_year="2023",
                semester="Fall",
                instructor="Test Instructor",
                department="Computer Science"
            )
            cms.create_course(course)
        
        course_creation_time = (datetime.now() - start_time).total_seconds()
        print(f"Course creation (100 courses): {course_creation_time:.2f} seconds")
        
        # Test content creation performance
        start_time = datetime.now()
        for i in range(50):
            content = ContentItem(
                content_id=f"perf_content_{i}",
                title=f"Performance Test Content {i}",
                content_type=ContentType.LECTURE,
                course_id="perf_course_0",
                file_path=str(Path(test_dir) / f"perf_test_{i}.md"),
                original_filename=f"perf_test_{i}.md",
                file_size=1024,
                file_hash=f"perf{i:03d}",
                mime_type="text/markdown",
                week_number=(i % 13) + 1
            )
            
            # Create test file
            with open(content.file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Performance Test {i}\\n\\nContent for performance testing.")
            
            cms.add_content(content)
        
        content_creation_time = (datetime.now() - start_time).total_seconds()
        print(f"Content creation (50 items): {content_creation_time:.2f} seconds")
        
        # Test search index building performance
        start_time = datetime.now()
        cms.build_search_index()
        index_build_time = (datetime.now() - start_time).total_seconds()
        print(f"Search index building: {index_build_time:.2f} seconds")
        
        # Test search performance
        start_time = datetime.now()
        for i in range(20):
            results = cms.search_content("performance test", limit=10)
        search_time = (datetime.now() - start_time).total_seconds()
        print(f"Search operations (20 queries): {search_time:.2f} seconds")
        
        # Test relationship detection performance
        start_time = datetime.now()
        for i in range(10):
            cms.detect_relationships(f"perf_content_{i}")
        relationship_time = (datetime.now() - start_time).total_seconds()
        print(f"Relationship detection (10 items): {relationship_time:.2f} seconds")
        
        cms.shutdown()
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def main():
    """Main test runner"""
    print("Starting Content Management System Tests")
    print("="*50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add CMS tests
    cms_tests = unittest.TestLoader().loadTestsFromTestCase(TestContentManagementSystem)
    test_suite.addTests(cms_tests)
    
    # Add integration tests
    integration_tests = unittest.TestLoader().loadTestsFromTestCase(TestCMSIntegration)
    test_suite.addTests(integration_tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run performance tests
    if result.wasSuccessful():
        run_performance_tests()
    
    # Print summary
    print("\\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        
        if result.failures:
            print("\\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())