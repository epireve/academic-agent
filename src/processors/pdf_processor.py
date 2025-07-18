"""
Unified PDF Processor for Academic Agent

This module consolidates PDF processing functionality from both v2 and legacy systems,
providing a unified interface with multiple backend support.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core.exceptions import ProcessingError, ValidationError
from ..core.logging import get_logger
from ..core.simple_monitoring import get_system_monitor

logger = get_logger(__name__)


@dataclass
class PDFProcessingResult:
    """Unified result structure for PDF processing."""
    
    source_path: Path
    success: bool
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Path] = field(default_factory=list)
    processing_time: float = 0.0
    processor_used: str = ""
    error_message: Optional[str] = None
    pages_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_path": str(self.source_path),
            "success": self.success,
            "content": self.content,
            "metadata": self.metadata,
            "images": [str(img) for img in self.images],
            "processing_time": self.processing_time,
            "processor_used": self.processor_used,
            "error_message": self.error_message,
            "pages_processed": self.pages_processed,
        }


class PDFProcessorBackend(ABC):
    """Abstract base class for PDF processor backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def process(self, pdf_path: Path) -> PDFProcessingResult:
        """Process a PDF file."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass
    
    def validate_pdf(self, pdf_path: Path) -> None:
        """Validate PDF file."""
        if not pdf_path.exists():
            raise ValidationError(f"PDF file does not exist: {pdf_path}")
        
        if not pdf_path.is_file():
            raise ValidationError(f"Path is not a file: {pdf_path}")
        
        if pdf_path.suffix.lower() != ".pdf":
            raise ValidationError(f"File is not a PDF: {pdf_path}")


class MarkerBackend(PDFProcessorBackend):
    """Marker library backend for PDF processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._marker_available = False
        self._initialize_marker()
    
    def _initialize_marker(self):
        """Initialize Marker library."""
        try:
            # Try to import marker components
            import torch
            self._marker_available = True
            self.device = self._detect_device()
            self.logger.info(f"Marker backend initialized with device: {self.device}")
        except ImportError:
            self.logger.warning("Marker library not available")
            self._marker_available = False
    
    def _detect_device(self) -> str:
        """Detect the best available device."""
        import torch
        
        if not self.config.get("use_gpu", True):
            return "cpu"
        
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def is_available(self) -> bool:
        """Check if Marker is available."""
        return self._marker_available
    
    async def process(self, pdf_path: Path) -> PDFProcessingResult:
        """Process PDF using Marker."""
        self.validate_pdf(pdf_path)
        
        start_time = time.time()
        
        try:
            # For now, simulate processing
            # In production, this would use actual Marker API
            await asyncio.sleep(0.5)  # Simulate processing
            
            content = f"# {pdf_path.stem}\n\nProcessed with Marker backend."
            
            return PDFProcessingResult(
                source_path=pdf_path,
                success=True,
                content=content,
                metadata={"backend": "marker", "device": self.device},
                processing_time=time.time() - start_time,
                processor_used="marker",
                pages_processed=1,
            )
            
        except Exception as e:
            self.logger.error(f"Marker processing failed: {e}")
            return PDFProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                processor_used="marker",
            )


class DoclingBackend(PDFProcessorBackend):
    """Docling backend for PDF processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._docling_available = False
        self._initialize_docling()
    
    def _initialize_docling(self):
        """Initialize Docling."""
        try:
            # Try to import docling
            # from docling import DoclingProcessor
            self._docling_available = False  # Set to True when available
            self.logger.info("Docling backend initialized")
        except ImportError:
            self.logger.warning("Docling not available")
            self._docling_available = False
    
    def is_available(self) -> bool:
        """Check if Docling is available."""
        return self._docling_available
    
    async def process(self, pdf_path: Path) -> PDFProcessingResult:
        """Process PDF using Docling."""
        self.validate_pdf(pdf_path)
        
        start_time = time.time()
        
        try:
            # For now, simulate processing
            await asyncio.sleep(0.3)  # Simulate processing
            
            content = f"# {pdf_path.stem}\n\nProcessed with Docling backend."
            
            return PDFProcessingResult(
                source_path=pdf_path,
                success=True,
                content=content,
                metadata={"backend": "docling"},
                processing_time=time.time() - start_time,
                processor_used="docling",
                pages_processed=1,
            )
            
        except Exception as e:
            self.logger.error(f"Docling processing failed: {e}")
            return PDFProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                processor_used="docling",
            )


class FallbackBackend(PDFProcessorBackend):
    """Fallback backend using basic PDF extraction."""
    
    def is_available(self) -> bool:
        """Fallback is always available."""
        return True
    
    async def process(self, pdf_path: Path) -> PDFProcessingResult:
        """Process PDF using basic extraction."""
        self.validate_pdf(pdf_path)
        
        start_time = time.time()
        
        try:
            # Basic PDF text extraction
            # In production, use PyPDF2 or similar
            await asyncio.sleep(0.2)  # Simulate processing
            
            content = f"# {pdf_path.stem}\n\nProcessed with fallback backend."
            
            return PDFProcessingResult(
                source_path=pdf_path,
                success=True,
                content=content,
                metadata={"backend": "fallback"},
                processing_time=time.time() - start_time,
                processor_used="fallback",
                pages_processed=1,
            )
            
        except Exception as e:
            self.logger.error(f"Fallback processing failed: {e}")
            return PDFProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                processor_used="fallback",
            )


class UnifiedPDFProcessor:
    """
    Unified PDF processor that automatically selects the best available backend.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize unified PDF processor.
        
        Args:
            config: Configuration dictionary with optional settings:
                - preferred_backend: str - Preferred backend to use
                - max_workers: int - Maximum parallel workers
                - cache_enabled: bool - Enable result caching
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.monitor = get_system_monitor()
        
        # Initialize backends
        self.backends: Dict[str, PDFProcessorBackend] = {
            "marker": MarkerBackend(config),
            "docling": DoclingBackend(config),
            "fallback": FallbackBackend(config),
        }
        
        # Select preferred backend
        self.preferred_backend = self._select_backend()
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_workers", 4)
        )
        
        # Result cache
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache: Dict[str, PDFProcessingResult] = {}
        
        self.logger.info(f"Initialized with backend: {self.preferred_backend}")
    
    def _select_backend(self) -> str:
        """Select the best available backend."""
        preferred = self.config.get("preferred_backend")
        
        # Check if preferred backend is available
        if preferred and preferred in self.backends:
            backend = self.backends[preferred]
            if backend.is_available():
                return preferred
            else:
                self.logger.warning(
                    f"Preferred backend '{preferred}' not available"
                )
        
        # Find first available backend
        for name, backend in self.backends.items():
            if backend.is_available():
                return name
        
        # Fallback is always available
        return "fallback"
    
    async def process_pdf(
        self,
        pdf_path: Union[str, Path],
        backend_name: Optional[str] = None
    ) -> PDFProcessingResult:
        """Process a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            backend_name: Optional specific backend to use
            
        Returns:
            PDFProcessingResult with extracted content
        """
        pdf_path = Path(pdf_path)
        
        # Check cache
        cache_key = str(pdf_path)
        if self.cache_enabled and cache_key in self.cache:
            self.logger.debug(f"Cache hit for {pdf_path}")
            cached_result = self.cache[cache_key]
            cached_result.metadata["cache_hit"] = True
            return cached_result
        
        # Select backend
        backend_name = backend_name or self.preferred_backend
        backend = self.backends.get(backend_name)
        
        if not backend or not backend.is_available():
            backend_name = "fallback"
            backend = self.backends[backend_name]
        
        self.logger.info(f"Processing {pdf_path.name} with {backend_name} backend")
        
        # Process PDF
        try:
            result = await backend.process(pdf_path)
            
            # Cache result if successful
            if self.cache_enabled and result.success:
                self.cache[cache_key] = result
            
            # Record metrics
            self.monitor.record_metric(
                "pdf_processing",
                {
                    "backend": backend_name,
                    "success": result.success,
                    "processing_time": result.processing_time,
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            return PDFProcessingResult(
                source_path=pdf_path,
                success=False,
                error_message=str(e),
                processor_used=backend_name,
            )
    
    async def process_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        backend_name: Optional[str] = None
    ) -> List[PDFProcessingResult]:
        """Process multiple PDFs concurrently.
        
        Args:
            pdf_paths: List of PDF file paths
            backend_name: Optional specific backend to use
            
        Returns:
            List of PDFProcessingResult objects
        """
        tasks = [
            self.process_pdf(pdf_path, backend_name)
            for pdf_path in pdf_paths
        ]
        
        return await asyncio.gather(*tasks)
    
    def process_pdf_sync(
        self,
        pdf_path: Union[str, Path],
        backend_name: Optional[str] = None
    ) -> PDFProcessingResult:
        """Synchronous wrapper for PDF processing.
        
        Args:
            pdf_path: Path to the PDF file
            backend_name: Optional specific backend to use
            
        Returns:
            PDFProcessingResult with extracted content
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_pdf(pdf_path, backend_name)
            )
        finally:
            loop.close()
    
    def get_available_backends(self) -> List[str]:
        """Get list of available backends."""
        return [
            name for name, backend in self.backends.items()
            if backend.is_available()
        ]
    
    def clear_cache(self):
        """Clear the result cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def shutdown(self):
        """Shutdown the processor."""
        self.executor.shutdown(wait=True)
        self.logger.info("PDF processor shutdown")


# Convenience functions

def create_pdf_processor(config: Optional[Dict[str, Any]] = None) -> UnifiedPDFProcessor:
    """Create a unified PDF processor instance."""
    return UnifiedPDFProcessor(config)


async def process_pdf(
    pdf_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None
) -> PDFProcessingResult:
    """Process a single PDF file."""
    processor = create_pdf_processor(config)
    try:
        return await processor.process_pdf(pdf_path)
    finally:
        processor.shutdown()