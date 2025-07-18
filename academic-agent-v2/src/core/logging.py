"""Comprehensive logging system for Academic Agent v2."""

import json
import logging
import logging.handlers
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class StructuredFormatter(logging.Formatter):
    """Custom formatter that supports structured logging with JSON output."""
    
    def __init__(self, json_format: bool = False, include_context: bool = True):
        self.json_format = json_format
        self.include_context = include_context
        
        if json_format:
            super().__init__()
        else:
            super().__init__(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
    
    def format(self, record: logging.LogRecord) -> str:
        if self.json_format:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
            
            # Add exception information if present
            if record.exc_info:
                log_entry["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": traceback.format_exception(*record.exc_info)
                }
            
            # Add extra context if available
            if self.include_context and hasattr(record, 'extra_context'):
                log_entry["context"] = record.extra_context
            
            return json.dumps(log_entry)
        else:
            return super().format(record)


class AcademicAgentLogger:
    """Enhanced logger with structured logging and context management."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"academic_agent.{name}")
        self.context_stack = []
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger with handlers and formatters."""
        level = self.config.get("level", "INFO")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = StructuredFormatter(
            json_format=self.config.get("json_console", False)
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}.log",
            maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),  # 10MB
            backupCount=self.config.get("backup_count", 5)
        )
        file_formatter = StructuredFormatter(
            json_format=self.config.get("json_file", True)
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}_errors.log",
            maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
            backupCount=self.config.get("backup_count", 5)
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance handler for metrics
        if self.config.get("enable_metrics", True):
            metrics_handler = logging.handlers.RotatingFileHandler(
                log_dir / f"{self.name}_metrics.log",
                maxBytes=self.config.get("max_bytes", 10 * 1024 * 1024),
                backupCount=self.config.get("backup_count", 5)
            )
            metrics_handler.setLevel(logging.INFO)
            metrics_handler.setFormatter(file_formatter)
            metrics_handler.addFilter(lambda record: hasattr(record, 'metric_type'))
            self.logger.addHandler(metrics_handler)
    
    def with_context(self, **context) -> 'AcademicAgentLogger':
        """Add context to all subsequent log messages."""
        new_logger = AcademicAgentLogger(self.name, self.config)
        new_logger.context_stack = self.context_stack + [context]
        return new_logger
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with accumulated context."""
        # Merge all context from the stack
        extra_context = {}
        for ctx in self.context_stack:
            extra_context.update(ctx)
        
        # Add any extra context from kwargs
        if 'extra_context' in kwargs:
            extra_context.update(kwargs.pop('extra_context'))
        
        # Create log record with context
        extra = kwargs.get('extra', {})
        if extra_context:
            extra['extra_context'] = extra_context
        kwargs['extra'] = extra
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with full traceback."""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def metric(self, metric_name: str, value: Union[int, float], unit: str = None, **context):
        """Log performance metrics."""
        metric_data = {
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat(),
            **context
        }
        
        extra = {'metric_type': 'performance', 'metric_data': metric_data}
        self._log_with_context(logging.INFO, f"Metric: {metric_name}={value}{unit or ''}", extra=extra)
    
    def log_operation_start(self, operation: str, **context):
        """Log the start of an operation."""
        self.info(f"Starting operation: {operation}", extra_context={'operation': operation, 'stage': 'start', **context})
    
    def log_operation_end(self, operation: str, success: bool = True, **context):
        """Log the end of an operation."""
        status = 'success' if success else 'failure'
        level = logging.INFO if success else logging.ERROR
        self._log_with_context(level, f"Operation {status}: {operation}", extra_context={'operation': operation, 'stage': 'end', 'success': success, **context})
    
    @contextmanager
    def operation(self, operation_name: str, **context):
        """Context manager for logging operations."""
        start_time = datetime.now()
        self.log_operation_start(operation_name, **context)
        
        try:
            yield self.with_context(operation=operation_name, **context)
            duration = (datetime.now() - start_time).total_seconds()
            self.metric(f"{operation_name}_duration", duration, "seconds")
            self.log_operation_end(operation_name, success=True, duration=duration)
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.metric(f"{operation_name}_duration", duration, "seconds")
            self.log_operation_end(operation_name, success=False, duration=duration, error=str(e))
            raise


class LoggingConfig:
    """Configuration manager for logging system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration from file or defaults."""
        default_config = {
            "level": "INFO",
            "log_dir": "logs",
            "max_bytes": 10 * 1024 * 1024,  # 10MB
            "backup_count": 5,
            "json_console": False,
            "json_file": True,
            "enable_metrics": True,
            "loggers": {
                "academic_agent": {
                    "level": "INFO",
                    "handlers": ["console", "file", "error"]
                },
                "academic_agent.ingestion": {
                    "level": "DEBUG"
                },
                "academic_agent.analysis": {
                    "level": "INFO"
                },
                "academic_agent.outline": {
                    "level": "INFO"
                },
                "academic_agent.notes": {
                    "level": "INFO"
                },
                "academic_agent.quality": {
                    "level": "INFO"
                },
                "academic_agent.update": {
                    "level": "INFO"
                }
            }
        }
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    default_config.update(file_config.get("logging", {}))
            except Exception as e:
                print(f"Warning: Could not load logging config from {self.config_path}: {e}")
        
        return default_config
    
    def get_logger_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for a specific logger."""
        base_config = self.config.copy()
        logger_config = self.config.get("loggers", {}).get(name, {})
        base_config.update(logger_config)
        return base_config


# Global logging configuration
_logging_config = LoggingConfig()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    config_path: Optional[Path] = None
) -> logging.Logger:
    """Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
        config_path: Optional path to logging configuration file
    
    Returns:
        Configured logger instance
    """
    global _logging_config
    
    if config_path:
        _logging_config = LoggingConfig(config_path)
    
    # Set up root logger
    root_logger = logging.getLogger("academic_agent")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Set up with structured logging
    structured_logger = AcademicAgentLogger("root", _logging_config.get_logger_config("academic_agent"))
    
    return structured_logger.logger


def get_logger(name: str) -> AcademicAgentLogger:
    """Get a structured logger with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        AcademicAgentLogger instance
    """
    return AcademicAgentLogger(name, _logging_config.get_logger_config(f"academic_agent.{name}"))


def configure_logging_from_yaml(config: Dict[str, Any]):
    """Configure logging from YAML configuration.
    
    Args:
        config: Configuration dictionary from YAML
    """
    global _logging_config
    
    if "logging" in config:
        _logging_config.config.update(config["logging"])
    
    # Set up root logger with new configuration
    setup_logging(config_path=None)
