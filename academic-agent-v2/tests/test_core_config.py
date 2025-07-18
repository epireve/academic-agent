"""Tests for core configuration module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.core.config import MarkerConfig
from src.core.exceptions import ConfigurationError


class TestMarkerConfig:
    """Test MarkerConfig class."""

    def test_default_initialization(self):
        """Test MarkerConfig with default values."""
        config = MarkerConfig()

        assert config.use_gpu is True
        assert config.extract_images is True
        assert config.output_format == "markdown"
        assert config.batch_size == 1
        assert config.max_pages is None
        assert config.model_cache_dir.exists()
        assert config.device in ["cpu", "mps", "cuda"]

    def test_custom_initialization(self):
        """Test MarkerConfig with custom values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "custom_cache"

            config = MarkerConfig(
                use_gpu=False,
                max_pages=100,
                extract_images=False,
                output_format="html",
                batch_size=5,
                model_cache_dir=cache_dir,
            )

            assert config.use_gpu is False
            assert config.max_pages == 100
            assert config.extract_images is False
            assert config.output_format == "html"
            assert config.batch_size == 5
            assert config.model_cache_dir == cache_dir
            assert config.device == "cpu"  # Should be CPU when use_gpu=False

    def test_device_detection_mps(self):
        """Test device detection when MPS is available."""
        with patch.object(torch.backends.mps, "is_available", return_value=True):
            config = MarkerConfig()
            assert config.device == "mps"

    def test_device_detection_cuda(self):
        """Test device detection when CUDA is available."""
        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            config = MarkerConfig()
            assert config.device == "cuda"

    def test_device_detection_cpu_fallback(self):
        """Test device detection falls back to CPU."""
        with (
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            config = MarkerConfig()
            assert config.device == "cpu"

    def test_explicit_device_override(self):
        """Test that explicit device setting overrides detection."""
        config = MarkerConfig(device="cpu")
        assert config.device == "cpu"

    def test_invalid_output_format(self):
        """Test that invalid output format raises error."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            MarkerConfig(output_format="invalid")

    def test_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            MarkerConfig(batch_size=0)

    def test_get_device_info_mps(self):
        """Test get_device_info for MPS device."""
        with (
            patch.object(torch.backends.mps, "is_available", return_value=True),
            patch.object(torch.backends.mps, "is_built", return_value=True),
        ):
            config = MarkerConfig(device="mps")
            info = config.get_device_info()

            assert info["device"] == "mps"
            assert info["use_gpu"] is True
            assert "torch_version" in info
            assert info["mps_available"] is True
            assert info["mps_built"] is True

    def test_get_device_info_cuda(self):
        """Test get_device_info for CUDA device."""
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(torch.version, "cuda", "11.8"),
        ):
            config = MarkerConfig(device="cuda")
            info = config.get_device_info()

            assert info["device"] == "cuda"
            assert info["use_gpu"] is True
            assert "torch_version" in info
            assert info["cuda_available"] is True
            assert info["cuda_version"] == "11.8"
            assert info["gpu_count"] == 2

    def test_get_device_info_cpu(self):
        """Test get_device_info for CPU device."""
        config = MarkerConfig(device="cpu")
        info = config.get_device_info()

        assert info["device"] == "cpu"
        assert "torch_version" in info
        # CPU device info should not have MPS or CUDA specific keys
        assert "mps_available" not in info
        assert "cuda_available" not in info

    def test_to_marker_kwargs(self):
        """Test conversion to marker kwargs."""
        config = MarkerConfig(device="mps", max_pages=50, extract_images=False, batch_size=3)

        kwargs = config.to_marker_kwargs()

        assert kwargs["device"] == "mps"
        assert kwargs["max_pages"] == 50
        assert kwargs["extract_images"] is False
        assert kwargs["batch_size"] == 3

    def test_from_env_with_valid_values(self):
        """Test creating config from environment variables."""
        env_vars = {
            "MARKER_USE_GPU": "false",
            "MARKER_DEVICE": "cpu",
            "MARKER_MAX_PAGES": "100",
            "MARKER_EXTRACT_IMAGES": "false",
            "MARKER_OUTPUT_FORMAT": "html",
            "MARKER_BATCH_SIZE": "5",
        }

        with patch.dict(os.environ, env_vars):
            config = MarkerConfig.from_env()

            assert config.use_gpu is False
            assert config.device == "cpu"
            assert config.max_pages == 100
            assert config.extract_images is False
            assert config.output_format == "html"
            assert config.batch_size == 5

    def test_from_env_with_invalid_values(self):
        """Test that invalid environment values raise ConfigurationError."""
        with patch.dict(os.environ, {"MARKER_BATCH_SIZE": "invalid"}):
            with pytest.raises(ConfigurationError, match="Invalid value for MARKER_BATCH_SIZE"):
                MarkerConfig.from_env()

    def test_from_env_with_missing_values(self):
        """Test creating config from env with missing values uses defaults."""
        # Clear any existing marker env vars
        marker_env_vars = [k for k in os.environ if k.startswith("MARKER_")]
        with patch.dict(os.environ, {k: "" for k in marker_env_vars}, clear=False):
            config = MarkerConfig.from_env()

            # Should use default values
            assert config.use_gpu is True
            assert config.extract_images is True
            assert config.output_format == "markdown"
            assert config.batch_size == 1

    def test_model_cache_dir_creation(self):
        """Test that model cache directory is created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "test_cache"

            # Directory shouldn't exist initially
            assert not cache_dir.exists()

            # Creating config should create the directory
            config = MarkerConfig(model_cache_dir=cache_dir)
            assert cache_dir.exists()
            assert config.model_cache_dir == cache_dir

    def test_pydantic_validation(self):
        """Test Pydantic validation features."""
        # Test that the model validates correctly
        config_data = {
            "use_gpu": True,
            "max_pages": 50,
            "extract_images": True,
            "output_format": "markdown",
            "batch_size": 2,
        }

        config = MarkerConfig(**config_data)

        # Test dict conversion
        config_dict = config.model_dump()
        assert config_dict["use_gpu"] is True
        assert config_dict["max_pages"] == 50

        # Test JSON serialization
        json_str = config.model_dump_json()
        assert "use_gpu" in json_str
        assert "max_pages" in json_str
