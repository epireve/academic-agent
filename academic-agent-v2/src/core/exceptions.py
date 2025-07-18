"""Comprehensive exception hierarchy for Academic Agent v2."""

import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class AcademicAgentError(Exception):
    """Base exception for all Academic Agent errors.
    
    Provides structured error information with context, suggestions,
    and recovery options.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None,
        recoverable: bool = True,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.suggestions = suggestions or []
        self.recoverable = recoverable
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        self.traceback_info = traceback.format_exc() if original_exception else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback_info,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }
    
    def add_context(self, key: str, value: Any) -> 'AcademicAgentError':
        """Add context information to the exception."""
        self.context[key] = value
        return self
    
    def add_suggestion(self, suggestion: str) -> 'AcademicAgentError':
        """Add a recovery suggestion."""
        self.suggestions.append(suggestion)
        return self
    
    def is_recoverable(self) -> bool:
        """Check if the error is recoverable."""
        return self.recoverable
    
    def get_recovery_suggestions(self) -> List[str]:
        """Get list of recovery suggestions."""
        return self.suggestions


class ConfigurationError(AcademicAgentError):
    """Raised when there's an issue with configuration."""
    
    def __init__(
        self,
        message: str,
        config_section: Optional[str] = None,
        config_key: Optional[str] = None,
        expected_type: Optional[type] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'config_section': config_section,
            'config_key': config_key,
            'expected_type': expected_type.__name__ if expected_type else None,
            'actual_value': actual_value
        })
        
        suggestions = kwargs.get('suggestions', [])
        if config_section and config_key:
            suggestions.append(f"Check the '{config_key}' setting in the '{config_section}' section")
        if expected_type:
            suggestions.append(f"Ensure the value is of type {expected_type.__name__}")
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class ProcessingError(AcademicAgentError):
    """Raised when document processing fails."""
    
    def __init__(
        self,
        message: str,
        stage: Optional[str] = None,
        file_path: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'processing_stage': stage,
            'file_path': file_path,
            'agent_id': agent_id
        })
        
        suggestions = kwargs.get('suggestions', [])
        if file_path:
            suggestions.append(f"Verify that the file exists and is readable: {file_path}")
        if stage:
            suggestions.append(f"Check the {stage} processing stage for issues")
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class ValidationError(AcademicAgentError):
    """Raised when validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        field_name: Optional[str] = None,
        expected_value: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'validation_type': validation_type,
            'field_name': field_name,
            'expected_value': expected_value,
            'actual_value': actual_value
        })
        
        suggestions = kwargs.get('suggestions', [])
        if field_name:
            suggestions.append(f"Check the '{field_name}' field value")
        if expected_value is not None:
            suggestions.append(f"Expected value: {expected_value}")
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class MarkerError(ProcessingError):
    """Raised when Marker library encounters an error."""
    
    def __init__(
        self,
        message: str,
        marker_stage: Optional[str] = None,
        device_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'marker_stage': marker_stage,
            'device_info': device_info
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if the PDF file is corrupted or password-protected",
            "Verify that the marker library is properly installed",
            "Try with a different device configuration (CPU vs GPU)"
        ])
        
        super().__init__(message, stage=marker_stage, context=context, suggestions=suggestions, **kwargs)


class ContentError(ProcessingError):
    """Raised when content processing fails."""
    
    def __init__(
        self,
        message: str,
        content_type: Optional[str] = None,
        quality_score: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'content_type': content_type,
            'quality_score': quality_score
        })
        
        suggestions = kwargs.get('suggestions', [])
        if quality_score is not None and quality_score < 0.7:
            suggestions.append(f"Quality score ({quality_score:.2f}) is below threshold")
        suggestions.extend([
            "Review the input content for clarity and structure",
            "Check if the content processing parameters are appropriate"
        ])
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class QualityError(AcademicAgentError):
    """Raised when content quality checks fail."""
    
    def __init__(
        self,
        message: str,
        quality_score: Optional[float] = None,
        threshold: Optional[float] = None,
        quality_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'quality_score': quality_score,
            'threshold': threshold,
            'quality_metrics': quality_metrics
        })
        
        suggestions = kwargs.get('suggestions', [])
        if quality_score is not None and threshold is not None:
            suggestions.append(f"Quality score ({quality_score:.2f}) is below threshold ({threshold:.2f})")
        suggestions.extend([
            "Review the content for completeness and accuracy",
            "Check if the quality evaluation criteria are appropriate",
            "Consider regenerating the content with different parameters"
        ])
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class CommunicationError(AcademicAgentError):
    """Raised when inter-agent communication fails."""
    
    def __init__(
        self,
        message: str,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        message_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'sender': sender,
            'recipient': recipient,
            'message_type': message_type
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check if the recipient agent is running and accessible",
            "Verify the message format and content",
            "Review the communication configuration"
        ])
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class TimeoutError(AcademicAgentError):
    """Raised when operations exceed timeout limits."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'operation': operation,
            'timeout_seconds': timeout_seconds,
            'elapsed_seconds': elapsed_seconds
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Consider increasing the timeout limit for this operation",
            "Check if the operation can be optimized or broken down",
            "Verify that the system has sufficient resources"
        ])
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class ModelError(AcademicAgentError):
    """Raised when AI model operations fail."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'model_name': model_name,
            'api_endpoint': api_endpoint,
            'status_code': status_code
        })
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check API key and authentication credentials",
            "Verify the model endpoint is accessible",
            "Review the request parameters and payload"
        ])
        
        if status_code:
            if status_code == 401:
                suggestions.append("Authentication failed - check API key")
            elif status_code == 429:
                suggestions.append("Rate limit exceeded - implement retry with backoff")
            elif status_code >= 500:
                suggestions.append("Server error - try again later")
        
        super().__init__(message, context=context, suggestions=suggestions, **kwargs)


class RetryableError(AcademicAgentError):
    """Base class for errors that can be retried."""
    
    def __init__(self, message: str, max_retries: int = 3, **kwargs):
        kwargs['recoverable'] = True
        context = kwargs.get('context', {})
        context['max_retries'] = max_retries
        super().__init__(message, context=context, **kwargs)


class NonRetryableError(AcademicAgentError):
    """Base class for errors that should not be retried."""
    
    def __init__(self, message: str, **kwargs):
        kwargs['recoverable'] = False
        super().__init__(message, **kwargs)


# Exception handling utilities
def handle_exception(
    exception: Exception,
    logger,
    operation: str = "unknown",
    context: Optional[Dict[str, Any]] = None
) -> AcademicAgentError:
    """Convert generic exceptions to AcademicAgentError with context.
    
    Args:
        exception: The original exception
        logger: Logger instance for recording the exception
        operation: Name of the operation that failed
        context: Additional context information
    
    Returns:
        AcademicAgentError: Structured exception with context
    """
    context = context or {}
    context['operation'] = operation
    
    # Map common exceptions to specific AcademicAgentError types
    if isinstance(exception, FileNotFoundError):
        return ProcessingError(
            f"File not found during {operation}: {exception}",
            original_exception=exception,
            context=context
        ).add_suggestion("Check if the file path is correct and the file exists")
    
    elif isinstance(exception, PermissionError):
        return ProcessingError(
            f"Permission denied during {operation}: {exception}",
            original_exception=exception,
            context=context
        ).add_suggestion("Check file permissions and access rights")
    
    elif isinstance(exception, ValueError):
        return ValidationError(
            f"Invalid value during {operation}: {exception}",
            original_exception=exception,
            context=context
        ).add_suggestion("Check input parameters and data types")
    
    elif isinstance(exception, ConnectionError):
        return CommunicationError(
            f"Connection error during {operation}: {exception}",
            original_exception=exception,
            context=context
        ).add_suggestion("Check network connectivity and service availability")
    
    elif isinstance(exception, TimeoutError):
        return TimeoutError(
            f"Operation timed out during {operation}: {exception}",
            operation=operation,
            original_exception=exception,
            context=context
        )
    
    else:
        # Generic error handling
        return AcademicAgentError(
            f"Unexpected error during {operation}: {exception}",
            original_exception=exception,
            context=context
        ).add_suggestion("Check the logs for more detailed error information")


def create_error_handler(logger, operation_name: str):
    """Create a decorator for automatic error handling.
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation for context
    
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AcademicAgentError:
                # Re-raise already structured exceptions
                raise
            except Exception as e:
                # Convert to structured exception
                structured_error = handle_exception(e, logger, operation_name)
                logger.exception(f"Error in {operation_name}", extra_context=structured_error.context)
                raise structured_error
        return wrapper
    return decorator
