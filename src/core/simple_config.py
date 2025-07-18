"""
Simple configuration module for unified architecture.
Provides basic configuration functionality without complex dependencies.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import ConfigurationError


class SimpleConfig:
    """Simple configuration container."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "quality_threshold": 0.7,
            "max_improvement_cycles": 3,
            "communication_interval": 30,
            "max_retries": 3,
            "logging": {
                "level": "INFO",
                "format": "detailed",
            },
            "error_handling": {
                "max_retries": 3,
                "retry_delay": 1.0,
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value as attribute."""
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration has no attribute '{name}'")


# Global configuration instance
_simple_config: Optional[SimpleConfig] = None


def get_config() -> SimpleConfig:
    """Get the global configuration instance."""
    global _simple_config
    if _simple_config is None:
        _simple_config = SimpleConfig()
    return _simple_config