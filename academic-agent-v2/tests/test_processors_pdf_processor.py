"""Tests for PDF processor module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.core.config import MarkerConfig
from src.core.exceptions import ProcessingError, ValidationError
from src.processors.pdf_processor import PDFProcessor


def create_dummy_pdf(file_path: Path) -> None:
    """Create a dummy PDF file for testing."""
    content = (
        b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n174\n%%EOF"
    )
    file_path.write_bytes(content)


class TestPDFProcessor:
    """Test PDFProcessor class."""

    def test_initialization_with_default_config(self):
        """Test PDFProcessor initialization with default config."""
        processor = PDFProcessor()

        assert processor.config is not None
        assert processor.config.use_gpu is True
        assert processor.executor is not None
        assert processor.config.device in ["cpu", "mps", "cuda"]

    def test_initialization_with_custom_config(self):
        """Test PDFProcessor initialization with custom config."""
        config = MarkerConfig(use_gpu=False, batch_size=4)
        processor = PDFProcessor(config)

        assert processor.config == config
        assert processor.config.use_gpu is False
        assert processor.config.batch_size == 4
        assert processor.config.device == "cpu"

    def test_validate_pdf_file_success(self):
        """Test PDF file validation with valid file."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            # Should not raise any exception
            processor._validate_pdf_file(pdf_path)

    def test_validate_pdf_file_not_exists(self):
        """Test PDF file validation with non-existent file."""
        processor = PDFProcessor()

        with pytest.raises(ValidationError, match="PDF file does not exist"):
            processor._validate_pdf_file(Path("/nonexistent/file.pdf"))

    def test_validate_pdf_file_not_file(self):
        """Test PDF file validation with directory."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)

            with pytest.raises(ValidationError, match="Path is not a file"):
                processor._validate_pdf_file(dir_path)

    def test_validate_pdf_file_wrong_extension(self):
        """Test PDF file validation with wrong extension."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = Path(tmp_dir) / "test.txt"
            txt_path.write_text("Not a PDF")

            with pytest.raises(ValidationError, match="File is not a PDF"):
                processor._validate_pdf_file(txt_path)

    def test_validate_pdf_file_large_size_warning(self, caplog):
        """Test PDF file validation with large file size warning."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "large.pdf"
            create_dummy_pdf(pdf_path)

            # Mock get_file_size_mb to return large size
            with patch("src.processors.pdf_processor.get_file_size_mb", return_value=150.0):
                processor._validate_pdf_file(pdf_path)

                # Check that warning was logged
                assert "Large PDF file detected" in caplog.text

    def test_simulate_marker_processing_cpu(self):
        """Test marker processing simulation with CPU device."""
        config = MarkerConfig(device="cpu")
        processor = PDFProcessor(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            result = processor._simulate_marker_processing(pdf_path)

            assert result["success"] is True
            assert result["device_used"] == "cpu"
            assert result["file_path"] == str(pdf_path)
            assert "processing_time" in result
            assert "content_extracted" in result
            assert "metadata" in result

    def test_simulate_marker_processing_mps(self):
        """Test marker processing simulation with MPS device."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        config = MarkerConfig(device="mps")
        processor = PDFProcessor(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            result = processor._simulate_marker_processing(pdf_path)

            assert result["success"] is True
            assert result["device_used"] == "mps"
            assert result["file_path"] == str(pdf_path)
            assert "processing_time" in result

    def test_process_pdf_success(self):
        """Test successful PDF processing."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            result = processor.process_pdf(pdf_path)

            assert result["success"] is True
            assert result["file_path"] == str(pdf_path)
            assert "processing_time" in result
            assert "content_extracted" in result
            assert "metadata" in result

    def test_process_pdf_validation_error(self):
        """Test PDF processing with validation error."""
        processor = PDFProcessor()

        with pytest.raises(ValidationError):
            processor.process_pdf(Path("/nonexistent/file.pdf"))

    def test_process_pdf_processing_error(self):
        """Test PDF processing with processing error."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            # Mock _simulate_marker_processing to raise an exception
            with patch.object(
                processor, "_simulate_marker_processing", side_effect=Exception("Processing failed")
            ):
                with pytest.raises(ProcessingError, match="Failed to process PDF"):
                    processor.process_pdf(pdf_path)

    @pytest.mark.asyncio
    async def test_process_pdf_async(self):
        """Test asynchronous PDF processing."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            result = await processor.process_pdf_async(pdf_path)

            assert result["success"] is True
            assert result["file_path"] == str(pdf_path)

    @pytest.mark.asyncio
    async def test_process_batch_success(self):
        """Test successful batch processing."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_paths = []
            for i in range(3):
                pdf_path = Path(tmp_dir) / f"test_{i}.pdf"
                create_dummy_pdf(pdf_path)
                pdf_paths.append(pdf_path)

            results = await processor.process_batch(pdf_paths)

            assert len(results) == 3
            assert all(result["success"] for result in results)
            assert all(result["file_path"] == str(pdf_paths[i]) for i, result in enumerate(results))

    @pytest.mark.asyncio
    async def test_process_batch_with_errors(self):
        """Test batch processing with some errors."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_paths = []
            # Create one valid PDF
            pdf_path = Path(tmp_dir) / "test_0.pdf"
            create_dummy_pdf(pdf_path)
            pdf_paths.append(pdf_path)

            # Add one invalid path
            pdf_paths.append(Path("/nonexistent/file.pdf"))

            results = await processor.process_batch(pdf_paths)

            assert len(results) == 2
            assert results[0]["success"] is True
            assert results[1]["success"] is False
            assert "error" in results[1]

    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        config = MarkerConfig(batch_size=4)
        processor = PDFProcessor(config)

        stats = processor.get_processing_stats()

        assert "config" in stats
        assert "device_info" in stats
        assert "executor_info" in stats
        assert stats["config"]["batch_size"] == 4
        assert stats["device_info"]["device"] == config.device
        assert stats["executor_info"]["max_workers"] == 4

    def test_destructor(self):
        """Test processor destructor."""
        processor = PDFProcessor()

        # Mock the executor shutdown method
        with patch.object(processor.executor, "shutdown") as mock_shutdown:
            processor.__del__()
            mock_shutdown.assert_called_once_with(wait=True)

    def test_different_configurations(self):
        """Test processor with different configurations."""
        # Test with different batch sizes
        config1 = MarkerConfig(batch_size=1)
        processor1 = PDFProcessor(config1)
        assert processor1.executor._max_workers == 1

        config2 = MarkerConfig(batch_size=8)
        processor2 = PDFProcessor(config2)
        assert processor2.executor._max_workers == 8

        # Test with different devices
        config3 = MarkerConfig(device="cpu")
        processor3 = PDFProcessor(config3)
        assert processor3.config.device == "cpu"

    def test_content_extraction_format(self):
        """Test that content extraction follows expected format."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "academic_paper.pdf"
            create_dummy_pdf(pdf_path)

            result = processor.process_pdf(pdf_path)

            content = result["content_extracted"]
            assert content.startswith("# academic_paper")
            assert "File: academic_paper.pdf" in content
            assert "Device:" in content
            assert "Processing time:" in content

    def test_metadata_structure(self):
        """Test that metadata has expected structure."""
        processor = PDFProcessor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "test.pdf"
            create_dummy_pdf(pdf_path)

            result = processor.process_pdf(pdf_path)

            metadata = result["metadata"]
            required_fields = ["title", "author", "subject", "creator", "producer"]

            for field in required_fields:
                assert field in metadata

            assert metadata["title"] == "test"
            assert metadata["creator"] == "Academic Agent v2"
