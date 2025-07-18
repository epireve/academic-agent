"""Error handling middleware and decorators for Academic Agent v2."""

import asyncio
import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .exceptions import (
    AcademicAgentError,
    RetryableError,
    NonRetryableError,
    TimeoutError,
    handle_exception
)
from .logging import get_logger


class ErrorHandler:
    """Comprehensive error handling system with retry logic and monitoring."""
    
    def __init__(self, logger_name: str = "error_handler"):
        self.logger = get_logger(logger_name)
        self.error_stats = {
            'total_errors': 0,
            'retryable_errors': 0,
            'non_retryable_errors': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'error_types': {}
        }
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ) -> AcademicAgentError:
        """Handle an error with optional retry logic.
        
        Args:
            error: The exception to handle
            operation: Name of the operation that failed
            context: Additional context information
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
            exponential_backoff: Whether to use exponential backoff
        
        Returns:
            AcademicAgentError: Structured error with context
        """
        # Update error statistics
        self.error_stats['total_errors'] += 1
        error_type = type(error).__name__
        self.error_stats['error_types'][error_type] = self.error_stats['error_types'].get(error_type, 0) + 1
        
        # Convert to structured error
        if isinstance(error, AcademicAgentError):
            structured_error = error
        else:
            structured_error = handle_exception(error, self.logger, operation, context)
        
        # Log the error
        self.logger.error(
            f"Error in {operation}: {structured_error.message}",
            extra_context=structured_error.context
        )
        
        # Update retry statistics
        if structured_error.is_recoverable():
            self.error_stats['retryable_errors'] += 1
        else:
            self.error_stats['non_retryable_errors'] += 1
        
        return structured_error
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return self.error_stats.copy()
    
    def reset_error_stats(self):
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'retryable_errors': 0,
            'non_retryable_errors': 0,
            'successful_retries': 0,
            'failed_retries': 0,
            'error_types': {}
        }


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, logger_name: str = "retry_handler"):
        self.logger = get_logger(logger_name)
    
    def retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ) -> Callable:
        """Decorator for retrying functions with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add jitter to delay
            retryable_exceptions: List of exception types that should trigger retries
        
        Returns:
            Decorated function with retry logic
        """
        if retryable_exceptions is None:
            retryable_exceptions = [RetryableError]
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception type should be retried
                    should_retry = (
                        isinstance(e, tuple(retryable_exceptions)) or
                        (isinstance(e, AcademicAgentError) and e.is_recoverable())
                    )
                    
                    if attempt == max_retries or not should_retry:
                        self.logger.error(
                            f"Final attempt failed for {func.__name__}: {e}",
                            extra_context={'attempt': attempt + 1, 'max_retries': max_retries}
                        )
                        raise
                    
                    # Calculate delay
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    
                    if jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s",
                        extra_context={'attempt': attempt + 1, 'delay': delay}
                    )
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    
    async def async_retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ) -> Callable:
        """Async version of retry_with_backoff."""
        if retryable_exceptions is None:
            retryable_exceptions = [RetryableError]
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this exception type should be retried
                    should_retry = (
                        isinstance(e, tuple(retryable_exceptions)) or
                        (isinstance(e, AcademicAgentError) and e.is_recoverable())
                    )
                    
                    if attempt == max_retries or not should_retry:
                        self.logger.error(
                            f"Final attempt failed for {func.__name__}: {e}",
                            extra_context={'attempt': attempt + 1, 'max_retries': max_retries}
                        )
                        raise
                    
                    # Calculate delay
                    delay = min(initial_delay * (exponential_base ** attempt), max_delay)
                    
                    if jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s",
                        extra_context={'attempt': attempt + 1, 'delay': delay}
                    )
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper


class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        logger_name: str = "circuit_breaker"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.logger = get_logger(logger_name)
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator that implements circuit breaker pattern."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.logger.info(f"Circuit breaker for {func.__name__} is now HALF_OPEN")
                else:
                    raise NonRetryableError(
                        f"Circuit breaker is OPEN for {func.__name__}",
                        context={'failure_count': self.failure_count, 'last_failure_time': self.last_failure_time}
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker is now CLOSED")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker is now OPEN (failures: {self.failure_count})")


class ErrorMonitor:
    """Monitors error patterns and provides alerts."""
    
    def __init__(self, logger_name: str = "error_monitor"):
        self.logger = get_logger(logger_name)
        self.error_history = []
        self.alert_thresholds = {
            'error_rate': 0.1,  # 10% error rate
            'consecutive_errors': 5,
            'error_spike': 10  # 10 errors in short time
        }
    
    def record_error(self, error: AcademicAgentError, operation: str):
        """Record an error for monitoring."""
        error_record = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'operation': operation,
            'recoverable': error.is_recoverable(),
            'context': error.context
        }
        
        self.error_history.append(error_record)
        
        # Keep only recent errors (last hour)
        current_time = time.time()
        self.error_history = [
            record for record in self.error_history
            if current_time - record['timestamp'] < 3600
        ]
        
        # Check for alert conditions
        self._check_alerts()
    
    def _check_alerts(self):
        """Check if any alert conditions are met."""
        if not self.error_history:
            return
        
        current_time = time.time()
        recent_errors = [
            record for record in self.error_history
            if current_time - record['timestamp'] < 300  # Last 5 minutes
        ]
        
        # Check for error spike
        if len(recent_errors) >= self.alert_thresholds['error_spike']:
            self.logger.critical(
                f"Error spike detected: {len(recent_errors)} errors in the last 5 minutes",
                extra_context={'recent_errors': len(recent_errors)}
            )
        
        # Check for consecutive errors
        consecutive_count = 0
        for record in reversed(self.error_history):
            if not record['recoverable']:
                consecutive_count += 1
            else:
                break
        
        if consecutive_count >= self.alert_thresholds['consecutive_errors']:
            self.logger.critical(
                f"Consecutive error threshold exceeded: {consecutive_count} consecutive non-recoverable errors",
                extra_context={'consecutive_errors': consecutive_count}
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of recent errors."""
        if not self.error_history:
            return {'total_errors': 0, 'error_types': {}, 'recent_errors': 0}
        
        current_time = time.time()
        recent_errors = [
            record for record in self.error_history
            if current_time - record['timestamp'] < 300  # Last 5 minutes
        ]
        
        error_types = {}
        for record in self.error_history:
            error_type = record['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_types': error_types,
            'recoverable_errors': sum(1 for r in self.error_history if r['recoverable']),
            'non_recoverable_errors': sum(1 for r in self.error_history if not r['recoverable'])
        }


# Global error handling instances
_error_handler = ErrorHandler()
_retry_handler = RetryHandler()
_error_monitor = ErrorMonitor()


def with_error_handling(
    operation_name: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    circuit_breaker: bool = False,
    monitor_errors: bool = True
):
    """Decorator that provides comprehensive error handling.
    
    Args:
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries
        circuit_breaker: Whether to use circuit breaker pattern
        monitor_errors: Whether to monitor errors for alerts
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Apply circuit breaker if requested
        if circuit_breaker:
            func = CircuitBreaker()(func)
        
        # Apply retry logic
        func = _retry_handler.retry_with_backoff(
            func,
            max_retries=max_retries,
            initial_delay=retry_delay
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the error
                structured_error = _error_handler.handle_error(
                    e,
                    operation_name,
                    context={'function': func.__name__, 'args': str(args)[:100]}
                )
                
                # Monitor the error if requested
                if monitor_errors:
                    _error_monitor.record_error(structured_error, operation_name)
                
                raise structured_error
        
        return wrapper
    return decorator


@contextmanager
def error_context(operation_name: str, context: Optional[Dict[str, Any]] = None):
    """Context manager for handling errors with context.
    
    Args:
        operation_name: Name of the operation
        context: Additional context information
    
    Yields:
        Logger instance with operation context
    """
    logger = get_logger("error_context")
    
    try:
        with logger.operation(operation_name, **(context or {})) as operation_logger:
            yield operation_logger
    except Exception as e:
        structured_error = _error_handler.handle_error(e, operation_name, context)
        _error_monitor.record_error(structured_error, operation_name)
        raise structured_error


def get_error_stats() -> Dict[str, Any]:
    """Get comprehensive error statistics."""
    return {
        'handler_stats': _error_handler.get_error_stats(),
        'monitor_summary': _error_monitor.get_error_summary()
    }


def reset_error_stats():
    """Reset all error statistics."""
    _error_handler.reset_error_stats()
    _error_monitor.error_history.clear()