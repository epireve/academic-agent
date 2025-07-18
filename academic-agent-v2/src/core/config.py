"""Configuration classes for Academic Agent v2."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .exceptions import ConfigurationError

# Note: We'll use basic logging here to avoid circular imports
import logging
logger = logging.getLogger(__name__)


class MarkerConfig(BaseModel):
    """Configuration for Marker PDF processing library."""

    # Device configuration
    use_gpu: bool = Field(default=True, description="Whether to use GPU acceleration if available")
    device: Optional[str] = Field(
        default=None, description="Specific device to use (e.g., 'cuda', 'mps', 'cpu')"
    )

    # Processing configuration
    max_pages: Optional[int] = Field(
        default=None, description="Maximum number of pages to process (None for all)"
    )
    extract_images: bool = Field(default=True, description="Whether to extract images from PDFs")

    # Output configuration
    output_format: str = Field(
        default="markdown", description="Output format for processed content"
    )

    # Performance configuration
    batch_size: int = Field(default=1, description="Batch size for processing multiple documents")

    # Model configuration
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "marker_models",
        description="Directory to cache ML models",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("output_format")
    def validate_output_format(cls, v):
        """Validate output format."""
        valid_formats = ["markdown", "html", "json"]
        if v not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")
        return v

    @field_validator("batch_size")
    def validate_batch_size(cls, v):
        """Validate batch size."""
        if v < 1:
            raise ValueError("batch_size must be at least 1")
        return v

    def __init__(self, **data):
        """Initialize MarkerConfig with device detection."""
        super().__init__(**data)

        # Auto-detect device if not specified
        if self.device is None:
            self.device = self._detect_device()

        # Ensure model cache directory exists
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"MarkerConfig initialized with device: {self.device}")

    def _detect_device(self) -> str:
        """Detect the best available device for processing."""
        if not self.use_gpu:
            return "cpu"

        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) for GPU acceleration")
            return "mps"

        # Check for CUDA
        if torch.cuda.is_available():
            logger.info("Using CUDA for GPU acceleration")
            return "cuda"

        # Fall back to CPU
        logger.info("GPU not available, using CPU")
        return "cpu"

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about the configured device."""
        info = {
            "device": self.device,
            "use_gpu": self.use_gpu,
            "torch_version": torch.__version__,
        }

        if self.device == "mps":
            info.update(
                {
                    "mps_available": torch.backends.mps.is_available(),
                    "mps_built": torch.backends.mps.is_built(),
                }
            )
        elif self.device == "cuda":
            info.update(
                {
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                }
            )

        return info

    def to_marker_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs for marker functions."""
        # This will be implemented based on the actual marker API
        return {
            "device": self.device,
            "max_pages": self.max_pages,
            "extract_images": self.extract_images,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_env(cls) -> "MarkerConfig":
        """Create configuration from environment variables."""
        env_config = {}

        # Map environment variables to config fields
        env_mapping = {
            "MARKER_USE_GPU": ("use_gpu", lambda x: x.lower() == "true"),
            "MARKER_DEVICE": ("device", str),
            "MARKER_MAX_PAGES": ("max_pages", int),
            "MARKER_EXTRACT_IMAGES": ("extract_images", lambda x: x.lower() == "true"),
            "MARKER_OUTPUT_FORMAT": ("output_format", str),
            "MARKER_BATCH_SIZE": ("batch_size", int),
            "MARKER_MODEL_CACHE_DIR": ("model_cache_dir", Path),
        }

        for env_var, (field_name, converter) in env_mapping.items():
            if value := os.getenv(env_var):
                try:
                    env_config[field_name] = converter(value)
                except (ValueError, TypeError) as e:
                    raise ConfigurationError(f"Invalid value for {env_var}: {value}. Error: {e}")

        return cls(**env_config)
