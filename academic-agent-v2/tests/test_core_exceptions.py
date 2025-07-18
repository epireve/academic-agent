"""Tests for core exceptions module."""

import pytest

from src.core.exceptions import (
    AcademicAgentError,
    ConfigurationError,
    ContentError,
    MarkerError,
    ProcessingError,
    ValidationError,
)


def test_academic_agent_error_inheritance():
    """Test that AcademicAgentError inherits from Exception."""
    error = AcademicAgentError("test message")
    assert isinstance(error, Exception)
    assert str(error) == "test message"


def test_configuration_error_inheritance():
    """Test that ConfigurationError inherits from AcademicAgentError."""
    error = ConfigurationError("config error")
    assert isinstance(error, AcademicAgentError)
    assert isinstance(error, Exception)
    assert str(error) == "config error"


def test_processing_error_inheritance():
    """Test that ProcessingError inherits from AcademicAgentError."""
    error = ProcessingError("processing error")
    assert isinstance(error, AcademicAgentError)
    assert isinstance(error, Exception)
    assert str(error) == "processing error"


def test_validation_error_inheritance():
    """Test that ValidationError inherits from AcademicAgentError."""
    error = ValidationError("validation error")
    assert isinstance(error, AcademicAgentError)
    assert isinstance(error, Exception)
    assert str(error) == "validation error"


def test_marker_error_inheritance():
    """Test that MarkerError inherits from ProcessingError."""
    error = MarkerError("marker error")
    assert isinstance(error, ProcessingError)
    assert isinstance(error, AcademicAgentError)
    assert isinstance(error, Exception)
    assert str(error) == "marker error"


def test_content_error_inheritance():
    """Test that ContentError inherits from ProcessingError."""
    error = ContentError("content error")
    assert isinstance(error, ProcessingError)
    assert isinstance(error, AcademicAgentError)
    assert isinstance(error, Exception)
    assert str(error) == "content error"


def test_exception_can_be_raised():
    """Test that custom exceptions can be raised and caught."""
    with pytest.raises(AcademicAgentError):
        raise AcademicAgentError("test")

    with pytest.raises(ConfigurationError):
        raise ConfigurationError("test")

    with pytest.raises(ProcessingError):
        raise ProcessingError("test")

    with pytest.raises(ValidationError):
        raise ValidationError("test")

    with pytest.raises(MarkerError):
        raise MarkerError("test")

    with pytest.raises(ContentError):
        raise ContentError("test")


def test_exception_hierarchy():
    """Test that exception hierarchy works correctly."""
    # MarkerError should be catchable as ProcessingError
    with pytest.raises(ProcessingError):
        raise MarkerError("marker error")

    # ContentError should be catchable as ProcessingError
    with pytest.raises(ProcessingError):
        raise ContentError("content error")

    # All custom exceptions should be catchable as AcademicAgentError
    with pytest.raises(AcademicAgentError):
        raise ConfigurationError("config error")

    with pytest.raises(AcademicAgentError):
        raise ProcessingError("processing error")

    with pytest.raises(AcademicAgentError):
        raise ValidationError("validation error")

    with pytest.raises(AcademicAgentError):
        raise MarkerError("marker error")

    with pytest.raises(AcademicAgentError):
        raise ContentError("content error")
