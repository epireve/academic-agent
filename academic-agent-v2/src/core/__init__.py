"""
Academic Agent v2 Core Module

This module provides comprehensive logging, error handling, monitoring, and configuration
management for the Academic Agent system.
"""

__version__ = "2.0.0"

# Import key components for easy access
try:
    from .logging import get_logger, setup_logging, configure_logging_from_yaml
    from .exceptions import (
        AcademicAgentError,
        ConfigurationError,
        ProcessingError,
        ValidationError,
        MarkerError,
        ContentError,
        QualityError,
        CommunicationError,
        TimeoutError,
        ModelError,
        RetryableError,
        NonRetryableError,
        handle_exception,
        create_error_handler
    )
    from .error_handling import (
        with_error_handling,
        error_context,
        get_error_stats,
        reset_error_stats
    )
    from .config_manager import (
        get_config,
        get_config_manager,
        reload_config
    )
    from .monitoring import (
        get_system_monitor,
        start_monitoring,
        get_monitoring_summary
    )
    from .memory_manager import (
        MemoryManager,
        MemoryPool,
        LRUCache,
        MemoryStats,
        MemoryThresholds,
        get_memory_manager,
        shutdown_memory_manager,
        memory_monitor,
        memory_profile
    )
    
    # Make key components available at package level
    __all__ = [
        # Logging
        'get_logger',
        'setup_logging',
        'configure_logging_from_yaml',
        
        # Exceptions
        'AcademicAgentError',
        'ConfigurationError',
        'ProcessingError',
        'ValidationError',
        'MarkerError',
        'ContentError',
        'QualityError',
        'CommunicationError',
        'TimeoutError',
        'ModelError',
        'RetryableError',
        'NonRetryableError',
        'handle_exception',
        'create_error_handler',
        
        # Error Handling
        'with_error_handling',
        'error_context',
        'get_error_stats',
        'reset_error_stats',
        
        # Configuration
        'get_config',
        'get_config_manager',
        'reload_config',
        
        # Monitoring
        'get_system_monitor',
        'start_monitoring',
        'get_monitoring_summary',
        
        # Memory Management
        'MemoryManager',
        'MemoryPool',
        'LRUCache',
        'MemoryStats',
        'MemoryThresholds',
        'get_memory_manager',
        'shutdown_memory_manager',
        'memory_monitor',
        'memory_profile',
    ]

except ImportError as e:
    # Graceful degradation if dependencies are not available
    import warnings
    warnings.warn(f"Some Academic Agent v2 core features may not be available: {e}")
    
    # Provide minimal functionality
    import logging
    
    def get_logger(name):
        return logging.getLogger(name)
    
    class AcademicAgentError(Exception):
        pass
    
    def with_error_handling(operation_name, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    __all__ = ['get_logger', 'AcademicAgentError', 'with_error_handling']

# Version information
VERSION_INFO = {
    'version': __version__,
    'features': [
        'Structured JSON logging',
        'Context-aware error handling',
        'Retry mechanisms with exponential backoff',
        'Circuit breaker pattern',
        'Real-time monitoring and alerting',
        'YAML configuration with JSON integration',
        'Performance metrics collection',
        'Health checking',
        'Enhanced agent communication'
    ],
    'dependencies': [
        'pydantic',
        'pyyaml',
        'psutil (optional, for system metrics)'
    ]
}

def get_version_info():
    """Get version and feature information."""
    return VERSION_INFO.copy()

def check_dependencies():
    """Check if all optional dependencies are available."""
    dependencies = {
        'pydantic': False,
        'yaml': False,
        'psutil': False
    }
    
    try:
        import pydantic
        dependencies['pydantic'] = True
    except ImportError:
        pass
    
    try:
        import yaml
        dependencies['yaml'] = True
    except ImportError:
        pass
    
    try:
        import psutil
        dependencies['psutil'] = True
    except ImportError:
        pass
    
    return dependencies

def setup_academic_agent_v2(config_path=None, start_monitoring_system=True):
    """
    Complete setup for Academic Agent v2 logging and error handling system.
    
    Args:
        config_path: Optional path to YAML configuration file
        start_monitoring_system: Whether to start the monitoring system
    
    Returns:
        Dict with setup status and components
    """
    setup_status = {
        'success': False,
        'components': {},
        'errors': []
    }
    
    try:
        # Setup logging
        if config_path:
            configure_logging_from_yaml({'logging_config_path': config_path})
        
        logger = get_logger('setup')
        setup_status['components']['logger'] = logger
        logger.info("Academic Agent v2 logging system initialized")
        
        # Setup configuration
        try:
            config_manager = get_config_manager(config_path)
            setup_status['components']['config_manager'] = config_manager
            logger.info("Configuration manager initialized")
        except Exception as e:
            setup_status['errors'].append(f"Configuration setup failed: {e}")
            logger.warning(f"Configuration setup failed: {e}")
        
        # Setup monitoring
        if start_monitoring_system:
            try:
                monitor = get_system_monitor()
                start_monitoring()
                setup_status['components']['monitor'] = monitor
                logger.info("Monitoring system started")
            except Exception as e:
                setup_status['errors'].append(f"Monitoring setup failed: {e}")
                logger.warning(f"Monitoring setup failed: {e}")
        
        setup_status['success'] = True
        logger.info("Academic Agent v2 setup completed successfully")
        
    except Exception as e:
        setup_status['errors'].append(f"Setup failed: {e}")
        
        # Try to log with basic logging if enhanced logging failed
        try:
            logger.error(f"Academic Agent v2 setup failed: {e}")
        except:
            import logging
            logging.getLogger('setup').error(f"Academic Agent v2 setup failed: {e}")
    
    return setup_status
