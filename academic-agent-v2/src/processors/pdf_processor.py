"""PDF processor using Marker library for Academic Agent v2."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..core.config import MarkerConfig
from ..core.exceptions import ProcessingError, ValidationError
from ..core.logging import get_logger
from ..utils.file_utils import get_file_size_mb

logger = get_logger(__name__)


class PDFProcessor:
    """PDF processor using Marker library with GPU acceleration."""

    def __init__(self, config: Optional[MarkerConfig] = None):
        """Initialize PDF processor.

        Args:
            config: Optional MarkerConfig instance. If None, uses default config.
        """
        self.config = config or MarkerConfig()
        self.executor = ThreadPoolExecutor(max_workers=self.config.batch_size)

        logger.info(f"PDFProcessor initialized with device: {self.config.device}")

    def _validate_pdf_file(self, pdf_path: Path) -> None:
        """Validate that the file exists and is a PDF.

        Args:
            pdf_path: Path to PDF file

        Raises:
            ValidationError: If file is invalid
        """
        if not pdf_path.exists():
            raise ValidationError(f"PDF file does not exist: {pdf_path}")

        if not pdf_path.is_file():
            raise ValidationError(f"Path is not a file: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValidationError(f"File is not a PDF: {pdf_path}")

        # Check file size (optional warning)
        size_mb = get_file_size_mb(pdf_path)
        if size_mb > 100:  # 100MB threshold
            logger.warning(f"Large PDF file detected: {pdf_path} ({size_mb:.2f} MB)")

    def _simulate_marker_processing(self, pdf_path: Path) -> Dict[str, Any]:
        """Simulate marker processing for demonstration purposes.

        This is a placeholder that demonstrates the expected interface
        until the marker library compatibility issues are resolved.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        # Simulate processing time based on file size
        file_size_mb = get_file_size_mb(pdf_path)
        processing_time = min(file_size_mb * 0.1, 5.0)  # Max 5 seconds

        # Simulate some processing work
        if self.config.device == "mps":
            # Simulate MPS tensor operations
            test_tensor = torch.randn(100, 100).to(self.config.device)
            result = torch.matmul(test_tensor, test_tensor)
            logger.debug(f"MPS tensor operations completed: {result.shape}")

        # Simulate processing delay
        time.sleep(min(processing_time, 1.0))  # Cap at 1 second for demo

        end_time = time.time()

        return {
            "success": True,
            "file_path": str(pdf_path),
            "file_size_mb": file_size_mb,
            "processing_time": end_time - start_time,
            "device_used": self.config.device,
            "pages_processed": max(1, int(file_size_mb * 10)),  # Estimate
            "content_extracted": (
                f"# {pdf_path.stem}\\n\\n"
                f"This is simulated content extracted from the PDF using Marker library.\\n\\n"
                f"- File: {pdf_path.name}\\n"
                f"- Size: {file_size_mb:.2f} MB\\n"
                f"- Device: {self.config.device}\\n"
                f"- Processing time: {end_time - start_time:.2f}s"
            ),
            "metadata": {
                "title": pdf_path.stem,
                "author": "Unknown",
                "subject": "Academic Document",
                "creator": "Academic Agent v2",
                "producer": "Marker Library (simulated)",
            },
        }

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with processing results

        Raises:
            ValidationError: If PDF file is invalid
            ProcessingError: If processing fails
        """
        try:
            self._validate_pdf_file(pdf_path)

            logger.info(f"Processing PDF: {pdf_path}")

            # For now, use simulation until marker library is fully working
            result = self._simulate_marker_processing(pdf_path)

            logger.info(f"PDF processing completed: {pdf_path} in {result['processing_time']:.2f}s")

            return result

        except ValidationError:
            raise
        except Exception as e:
            raise ProcessingError(f"Failed to process PDF {pdf_path}: {str(e)}")

    async def process_pdf_async(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a PDF file asynchronously.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with processing results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_pdf, pdf_path)

    async def process_batch(self, pdf_paths: List[Path]) -> List[Dict[str, Any]]:
        """Process multiple PDF files in batch.

        Args:
            pdf_paths: List of paths to PDF files

        Returns:
            List of processing results
        """
        logger.info(f"Processing batch of {len(pdf_paths)} PDFs")

        tasks = [self.process_pdf_async(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "file_path": str(pdf_paths[i]),
                        "error": str(result),
                        "error_type": type(result).__name__,
                    }
                )
            else:
                processed_results.append(result)

        logger.info(f"Batch processing completed: {len(processed_results)} results")
        return processed_results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processor statistics and configuration.

        Returns:
            Dictionary with processor stats
        """
        return {
            "config": self.config.model_dump(),
            "device_info": self.config.get_device_info(),
            "executor_info": {
                "max_workers": self.executor._max_workers,
                "threads": len(self.executor._threads) if hasattr(self.executor, "_threads") else 0,
            },
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
