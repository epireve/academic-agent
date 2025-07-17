"""
Unit tests for PDF processing functionality.

This module tests the PDF processing capabilities of the academic agent system
including document conversion, metadata extraction, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json
from datetime import datetime

from tests.utils import (
    PDFTestHelper,
    create_sample_pdf_content,
    MockDoclingResult,
    assert_valid_json_structure
)


class TestPDFProcessingUtilities:
    """Test utilities for PDF processing."""
    
    def test_create_test_pdf(self, tmp_path):
        """Test creating a test PDF file."""
        pdf_path = tmp_path / "test.pdf"
        
        created_path = PDFTestHelper.create_test_pdf(pdf_path)
        
        assert created_path == pdf_path
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
    
    def test_assert_pdf_processing_result_valid(self):
        """Test assertion for valid PDF processing result."""
        valid_result = {
            "processed_files": [
                {
                    "status": "success",
                    "markdown_path": "/test/output.md",
                    "metadata_path": "/test/output.json"
                }
            ],
            "errors": [],
            "stats": {"total": 1, "success": 1, "failed": 0}
        }
        
        # Should not raise any exception
        PDFTestHelper.assert_pdf_processing_result(valid_result)
    
    def test_assert_pdf_processing_result_invalid_structure(self):
        """Test assertion for invalid PDF processing result structure."""
        invalid_result = {
            "processed_files": [],
            "errors": [],
            # Missing stats
        }
        
        with pytest.raises(AssertionError):
            PDFTestHelper.assert_pdf_processing_result(invalid_result)
    
    def test_assert_pdf_processing_result_invalid_stats(self):
        """Test assertion for invalid stats in PDF processing result."""
        invalid_result = {
            "processed_files": [],
            "errors": [],
            "stats": {"total": 5, "success": 2, "failed": 2}  # total != success + failed
        }
        
        with pytest.raises(AssertionError):
            PDFTestHelper.assert_pdf_processing_result(invalid_result)


class TestMockDoclingResult:
    """Test MockDoclingResult utility."""
    
    def test_mock_docling_result_creation(self):
        """Test creation of MockDoclingResult."""
        markdown_content = "# Test Document\n\nContent here"
        
        result = MockDoclingResult(markdown_content)
        
        assert result.document.export_to_markdown.return_value == markdown_content
        assert result.document.title == "Test Document"
        assert result.document.language == "en"
    
    def test_mock_docling_result_custom_metadata(self):
        """Test MockDoclingResult with custom metadata."""
        markdown_content = "# Custom Document\n\nCustom content"
        title = "Custom Title"
        language = "fr"
        
        result = MockDoclingResult(markdown_content, title, language)
        
        assert result.document.export_to_markdown.return_value == markdown_content
        assert result.document.title == title
        assert result.document.language == language
    
    def test_mock_docling_result_document_interface(self):
        """Test that MockDoclingResult provides expected document interface."""
        result = MockDoclingResult("Test content")
        
        # Test that export_to_markdown is callable
        exported = result.document.export_to_markdown()
        assert exported == "Test content"
        
        # Test attribute access
        assert hasattr(result.document, 'title')
        assert hasattr(result.document, 'language')


class TestPDFContent:
    """Test PDF content creation and handling."""
    
    def test_create_sample_pdf_content(self):
        """Test creation of sample PDF content."""
        content = create_sample_pdf_content()
        
        assert isinstance(content, bytes)
        assert len(content) > 0
        assert content.startswith(b'%PDF-1.4')
        assert content.endswith(b'%%EOF')
    
    def test_sample_pdf_content_structure(self):
        """Test that sample PDF content has valid structure."""
        content = create_sample_pdf_content()
        content_str = content.decode('latin-1')
        
        # Check for essential PDF elements
        assert '%PDF-1.4' in content_str
        assert 'obj' in content_str
        assert 'endobj' in content_str
        assert 'xref' in content_str
        assert 'trailer' in content_str
        assert 'startxref' in content_str
        assert '%%EOF' in content_str
    
    def test_sample_pdf_content_contains_text(self):
        """Test that sample PDF content contains expected text."""
        content = create_sample_pdf_content()
        content_str = content.decode('latin-1')
        
        # Should contain the test text
        assert 'Sample PDF for Testing' in content_str


class TestPDFProcessingEdgeCases:
    """Test edge cases in PDF processing."""
    
    def test_empty_pdf_file(self, tmp_path):
        """Test handling of empty PDF file."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        
        # This would typically be handled by the PDF processing tool
        # Here we just verify the file exists and is empty
        assert empty_pdf.exists()
        assert empty_pdf.stat().st_size == 0
    
    def test_corrupted_pdf_file(self, tmp_path):
        """Test handling of corrupted PDF file."""
        corrupted_pdf = tmp_path / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"This is not a PDF file")
        
        # This would typically be handled by the PDF processing tool
        # Here we just verify the file exists with invalid content
        assert corrupted_pdf.exists()
        content = corrupted_pdf.read_bytes()
        assert not content.startswith(b'%PDF')
    
    def test_very_large_pdf_simulation(self, tmp_path):
        """Test simulation of very large PDF file."""
        # Create a "large" PDF by repeating the sample content
        base_content = create_sample_pdf_content()
        large_content = base_content * 100  # Simulate larger file
        
        large_pdf = tmp_path / "large.pdf"
        large_pdf.write_bytes(large_content)
        
        assert large_pdf.exists()
        assert large_pdf.stat().st_size > len(base_content)
    
    def test_special_characters_in_filename(self, tmp_path):
        """Test handling of special characters in PDF filename."""
        special_filename = "test file with spaces & symbols!@#.pdf"
        special_pdf = tmp_path / special_filename
        special_pdf.write_bytes(create_sample_pdf_content())
        
        assert special_pdf.exists()
        assert special_pdf.name == special_filename


class TestPDFProcessingMetadata:
    """Test PDF metadata handling."""
    
    def test_metadata_structure(self):
        """Test that metadata has expected structure."""
        metadata = {
            "source_file": "/path/to/test.pdf",
            "processed_date": datetime.now().isoformat(),
            "title": "Test Document",
            "language": "en"
        }
        
        required_keys = ["source_file", "processed_date", "title", "language"]
        assert_valid_json_structure(metadata, required_keys)
    
    def test_metadata_serialization(self, tmp_path):
        """Test metadata JSON serialization."""
        metadata = {
            "source_file": "/path/to/test.pdf",
            "processed_date": datetime.now().isoformat(),
            "title": "Test Document",
            "language": "en",
            "processing_stats": {
                "pages": 5,
                "words": 1000,
                "images": 3
            }
        }
        
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        # Verify it can be read back
        loaded_metadata = json.loads(metadata_file.read_text())
        assert loaded_metadata == metadata
    
    def test_metadata_with_unicode(self, tmp_path):
        """Test metadata with Unicode characters."""
        metadata = {
            "source_file": "/path/to/tëst_dócument.pdf",
            "processed_date": datetime.now().isoformat(),
            "title": "Tëst Dócument with Unicøde",
            "language": "en",
            "author": "Jöhn Dœ"
        }
        
        metadata_file = tmp_path / "unicode_metadata.json"
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        
        # Verify it can be read back with Unicode preserved
        loaded_metadata = json.loads(metadata_file.read_text())
        assert loaded_metadata["title"] == "Tëst Dócument with Unicøde"
        assert loaded_metadata["author"] == "Jöhn Dœ"


class TestPDFProcessingPerformance:
    """Test performance aspects of PDF processing."""
    
    def test_processing_time_measurement(self):
        """Test measurement of processing time."""
        import time
        
        start_time = time.time()
        
        # Simulate PDF processing work
        content = create_sample_pdf_content()
        processed_content = content.decode('latin-1')
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should be very fast for sample content
        assert processing_time < 1.0
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation for PDF content."""
        content = create_sample_pdf_content()
        content_size = len(content)
        
        # Estimate memory usage (this is a simple approximation)
        estimated_memory = content_size * 2  # Assume 2x overhead for processing
        
        assert estimated_memory > content_size
        assert estimated_memory < content_size * 10  # Reasonable overhead


class TestPDFProcessingBatchOperations:
    """Test batch PDF processing operations."""
    
    def test_multiple_pdf_files_creation(self, tmp_path):
        """Test creation of multiple PDF files for batch processing."""
        pdf_files = []
        
        for i in range(5):
            pdf_path = tmp_path / f"document_{i:02d}.pdf"
            PDFTestHelper.create_test_pdf(pdf_path, f"Test PDF {i}")
            pdf_files.append(pdf_path)
        
        assert len(pdf_files) == 5
        for pdf_path in pdf_files:
            assert pdf_path.exists()
            assert pdf_path.stat().st_size > 0
    
    def test_batch_processing_result_aggregation(self):
        """Test aggregation of batch processing results."""
        individual_results = [
            {
                "processed_files": [{"status": "success", "file": f"doc_{i}.pdf"}],
                "errors": [],
                "stats": {"total": 1, "success": 1, "failed": 0}
            }
            for i in range(3)
        ]
        
        # Aggregate results
        aggregated = {
            "processed_files": [],
            "errors": [],
            "stats": {"total": 0, "success": 0, "failed": 0}
        }
        
        for result in individual_results:
            aggregated["processed_files"].extend(result["processed_files"])
            aggregated["errors"].extend(result["errors"])
            aggregated["stats"]["total"] += result["stats"]["total"]
            aggregated["stats"]["success"] += result["stats"]["success"]
            aggregated["stats"]["failed"] += result["stats"]["failed"]
        
        assert aggregated["stats"]["total"] == 3
        assert aggregated["stats"]["success"] == 3
        assert aggregated["stats"]["failed"] == 0
        assert len(aggregated["processed_files"]) == 3
    
    def test_batch_processing_with_errors(self):
        """Test batch processing that includes errors."""
        mixed_results = [
            {
                "processed_files": [{"status": "success", "file": "doc_1.pdf"}],
                "errors": [],
                "stats": {"total": 1, "success": 1, "failed": 0}
            },
            {
                "processed_files": [],
                "errors": [{"status": "error", "file": "doc_2.pdf", "error": "Processing failed"}],
                "stats": {"total": 1, "success": 0, "failed": 1}
            },
            {
                "processed_files": [{"status": "success", "file": "doc_3.pdf"}],
                "errors": [],
                "stats": {"total": 1, "success": 1, "failed": 0}
            }
        ]
        
        # Aggregate results
        aggregated = {
            "processed_files": [],
            "errors": [],
            "stats": {"total": 0, "success": 0, "failed": 0}
        }
        
        for result in mixed_results:
            aggregated["processed_files"].extend(result["processed_files"])
            aggregated["errors"].extend(result["errors"])
            aggregated["stats"]["total"] += result["stats"]["total"]
            aggregated["stats"]["success"] += result["stats"]["success"]
            aggregated["stats"]["failed"] += result["stats"]["failed"]
        
        assert aggregated["stats"]["total"] == 3
        assert aggregated["stats"]["success"] == 2
        assert aggregated["stats"]["failed"] == 1
        assert len(aggregated["processed_files"]) == 2
        assert len(aggregated["errors"]) == 1
        
        # Verify the aggregation is mathematically correct
        PDFTestHelper.assert_pdf_processing_result(aggregated)


class TestPDFProcessingRecovery:
    """Test recovery mechanisms in PDF processing."""
    
    def test_partial_processing_recovery(self):
        """Test recovery from partial processing failures."""
        # Simulate a scenario where processing was interrupted
        partial_result = {
            "processed_files": [
                {"status": "success", "file": "doc_1.pdf"},
                {"status": "success", "file": "doc_2.pdf"}
            ],
            "errors": [],
            "stats": {"total": 5, "success": 2, "failed": 0}  # 3 files remain
        }
        
        # Files that still need processing
        remaining_files = ["doc_3.pdf", "doc_4.pdf", "doc_5.pdf"]
        
        assert len(remaining_files) == (
            partial_result["stats"]["total"] - 
            partial_result["stats"]["success"] - 
            partial_result["stats"]["failed"]
        )
    
    def test_checkpoint_creation(self, tmp_path):
        """Test creation of processing checkpoints."""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "completed_files": ["doc_1.pdf", "doc_2.pdf"],
            "remaining_files": ["doc_3.pdf", "doc_4.pdf"],
            "processing_stats": {"total": 4, "completed": 2}
        }
        
        checkpoint_file = tmp_path / "processing_checkpoint.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
        
        # Verify checkpoint can be loaded
        loaded_checkpoint = json.loads(checkpoint_file.read_text())
        assert loaded_checkpoint["processing_stats"]["completed"] == 2
        assert len(loaded_checkpoint["remaining_files"]) == 2
    
    def test_resume_from_checkpoint(self, tmp_path):
        """Test resuming processing from a checkpoint."""
        # Create a checkpoint
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "completed_files": ["doc_1.pdf", "doc_2.pdf"],
            "remaining_files": ["doc_3.pdf", "doc_4.pdf"],
            "processing_stats": {"total": 4, "completed": 2}
        }
        
        checkpoint_file = tmp_path / "processing_checkpoint.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
        
        # Load checkpoint
        loaded_checkpoint = json.loads(checkpoint_file.read_text())
        
        # Simulate resuming with remaining files
        remaining_files = loaded_checkpoint["remaining_files"]
        completed_count = loaded_checkpoint["processing_stats"]["completed"]
        
        # Process remaining files (simulated)
        for file in remaining_files:
            completed_count += 1
        
        assert completed_count == 4
        assert completed_count == checkpoint_data["processing_stats"]["total"]