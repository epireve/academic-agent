"""Configuration management system that integrates YAML and JSON configurations."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger("config_manager")


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    
    level: str = Field(default="INFO", description="Default logging level")
    log_dir: str = Field(default="logs", description="Directory for log files")
    max_bytes: int = Field(default=10485760, description="Maximum log file size")
    backup_count: int = Field(default=5, description="Number of backup files to keep")
    json_console: bool = Field(default=False, description="Use JSON format for console output")
    json_file: bool = Field(default=True, description="Use JSON format for file output")
    enable_metrics: bool = Field(default=True, description="Enable metrics logging")
    
    formatters: Dict[str, Any] = Field(default_factory=dict)
    handlers: Dict[str, Any] = Field(default_factory=dict)
    loggers: Dict[str, Any] = Field(default_factory=dict)


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling system."""
    
    max_retries: int = Field(default=3, description="Default maximum retry attempts")
    retry_delay: float = Field(default=1.0, description="Default retry delay in seconds")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker pattern")
    error_monitoring_enabled: bool = Field(default=True, description="Enable error monitoring")
    
    retry: Dict[str, Any] = Field(default_factory=dict)
    circuit_breaker: Dict[str, Any] = Field(default_factory=dict)
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    operations: Dict[str, Any] = Field(default_factory=dict)


class MetricsConfig(BaseModel):
    """Configuration for metrics collection."""
    
    enabled: bool = Field(default=True, description="Enable metrics collection")
    collection_interval: int = Field(default=60, description="Metrics collection interval in seconds")
    
    performance: list = Field(default_factory=list)
    quality: list = Field(default_factory=list)
    communication: list = Field(default_factory=list)
    export: Dict[str, Any] = Field(default_factory=dict)


class IntegrationConfig(BaseModel):
    """Configuration for integrating with existing JSON config."""
    
    json_config_path: str = Field(default="config/academic_agent_config.json")
    override_from_json: list = Field(default_factory=list)
    merge_agent_prompts: bool = Field(default=True)
    merge_feedback_loops: bool = Field(default=True)


class AcademicAgentConfig(BaseModel):
    """Main configuration class for Academic Agent."""
    
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    
    # Additional configuration sections
    quality_threshold: float = Field(default=0.75, description="Quality threshold for content")
    improvement_threshold: float = Field(default=0.3, description="Improvement threshold")
    max_improvement_cycles: int = Field(default=3, description="Maximum improvement cycles")
    communication_interval: int = Field(default=30, description="Communication interval in seconds")
    
    # Agent-specific configurations
    agent_specialized_prompts: Dict[str, str] = Field(default_factory=dict)
    feedback_loops: list = Field(default_factory=list)
    inter_agent_communication: Dict[str, Any] = Field(default_factory=dict)
    improvement_criteria: Dict[str, Any] = Field(default_factory=dict)
    processing: Dict[str, Any] = Field(default_factory=dict)


class ConfigManager:
    """Manages configuration loading, merging, and validation."""
    
    def __init__(self, yaml_config_path: Optional[Union[str, Path]] = None):
        self.yaml_config_path = Path(yaml_config_path) if yaml_config_path else None
        self.config: Optional[AcademicAgentConfig] = None
        self.json_config: Optional[Dict[str, Any]] = None
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load and merge configurations from YAML and JSON sources."""
        try:
            # Load YAML configuration
            yaml_config = self._load_yaml_config()
            
            # Load JSON configuration
            json_config = self._load_json_config(yaml_config.get("integration", {}).get("json_config_path"))
            
            # Merge configurations
            merged_config = self._merge_configurations(yaml_config, json_config)
            
            # Validate configuration
            self.config = AcademicAgentConfig(**merged_config)
            
            logger.info("Configuration loaded and validated successfully")
            
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg, original_exception=e)
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.yaml_config_path or not self.yaml_config_path.exists():
            logger.warning(f"YAML config file not found at {self.yaml_config_path}, using defaults")
            return {}
        
        try:
            with open(self.yaml_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"YAML configuration loaded from {self.yaml_config_path}")
            return config or {}
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration: {e}",
                config_section="yaml_parsing",
                original_exception=e
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load YAML configuration: {e}",
                config_section="yaml_loading",
                original_exception=e
            )
    
    def _load_json_config(self, json_path: Optional[str]) -> Dict[str, Any]:
        """Load JSON configuration file."""
        if not json_path:
            return {}
        
        json_config_path = Path(json_path)
        if not json_config_path.exists():
            logger.warning(f"JSON config file not found at {json_config_path}")
            return {}
        
        try:
            with open(json_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"JSON configuration loaded from {json_config_path}")
            self.json_config = config
            return config
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Failed to parse JSON configuration: {e}",
                config_section="json_parsing",
                original_exception=e
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load JSON configuration: {e}",
                config_section="json_loading",
                original_exception=e
            )
    
    def _merge_configurations(self, yaml_config: Dict[str, Any], json_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge YAML and JSON configurations."""
        merged_config = yaml_config.copy()
        
        if not json_config:
            return merged_config
        
        integration_config = yaml_config.get("integration", {})
        
        # Override specific settings from JSON
        override_paths = integration_config.get("override_from_json", [])
        for path in override_paths:
            value = self._get_nested_value(json_config, path)
            if value is not None:
                self._set_nested_value(merged_config, path, value)
        
        # Merge agent prompts
        if integration_config.get("merge_agent_prompts", True):
            agent_prompts = json_config.get("agent_specialized_prompts", {})
            if agent_prompts:
                merged_config.setdefault("agent_specialized_prompts", {}).update(agent_prompts)
        
        # Merge feedback loops
        if integration_config.get("merge_feedback_loops", True):
            feedback_loops = json_config.get("feedback_loops", [])
            if feedback_loops:
                merged_config.setdefault("feedback_loops", []).extend(feedback_loops)
        
        # Merge other JSON sections
        for key in ["inter_agent_communication", "improvement_criteria", "processing"]:
            if key in json_config:
                merged_config.setdefault(key, {}).update(json_config[key])
        
        # Override top-level settings
        for key in ["quality_threshold", "improvement_threshold", "max_improvement_cycles", "communication_interval"]:
            if key in json_config:
                merged_config[key] = json_config[key]
        
        return merged_config
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value\n    \n    def get_config(self) -> AcademicAgentConfig:\n        """Get the current configuration."""\n        if self.config is None:\n            raise ConfigurationError("Configuration not loaded")\n        return self.config\n    \n    def get_logging_config(self) -> LoggingConfig:\n        """Get logging configuration."""\n        return self.get_config().logging\n    \n    def get_error_handling_config(self) -> ErrorHandlingConfig:\n        """Get error handling configuration."""\n        return self.get_config().error_handling\n    \n    def get_metrics_config(self) -> MetricsConfig:\n        """Get metrics configuration."""\n        return self.get_config().metrics\n    \n    def get_operation_config(self, operation_name: str) -> Dict[str, Any]:\n        """Get configuration for a specific operation."""\n        operations_config = self.get_config().error_handling.operations\n        return operations_config.get(operation_name, {})\n    \n    def get_agent_prompt(self, agent_name: str) -> Optional[str]:\n        """Get specialized prompt for an agent."""\n        return self.get_config().agent_specialized_prompts.get(agent_name)\n    \n    def reload_configuration(self):\n        """Reload configuration from files."""\n        logger.info("Reloading configuration")\n        self._load_configuration()\n    \n    def validate_configuration(self) -> bool:\n        """Validate the current configuration."""\n        try:\n            if self.config is None:\n                return False\n            \n            # Validate logging configuration\n            log_dir = Path(self.config.logging.log_dir)\n            if not log_dir.exists():\n                log_dir.mkdir(parents=True, exist_ok=True)\n            \n            # Validate error handling configuration\n            if self.config.error_handling.max_retries < 0:\n                raise ConfigurationError("max_retries must be non-negative")\n            \n            if self.config.error_handling.retry_delay < 0:\n                raise ConfigurationError("retry_delay must be non-negative")\n            \n            # Validate quality thresholds\n            if not 0 <= self.config.quality_threshold <= 1:\n                raise ConfigurationError("quality_threshold must be between 0 and 1")\n            \n            logger.info("Configuration validation successful")\n            return True\n            \n        except Exception as e:\n            logger.error(f"Configuration validation failed: {e}")\n            return False\n    \n    def export_merged_config(self, output_path: Union[str, Path], format: str = "yaml") -> None:\n        """Export the merged configuration to a file.\n        \n        Args:\n            output_path: Path to save the configuration\n            format: Output format ('yaml' or 'json')\n        \"\"\"\n        if self.config is None:\n            raise ConfigurationError("Configuration not loaded")\n        \n        output_path = Path(output_path)\n        output_path.parent.mkdir(parents=True, exist_ok=True)\n        \n        config_dict = self.config.dict()\n        \n        if format.lower() == "yaml":\n            with open(output_path, 'w', encoding='utf-8') as f:\n                yaml.dump(config_dict, f, default_flow_style=False, indent=2)\n        elif format.lower() == "json":\n            with open(output_path, 'w', encoding='utf-8') as f:\n                json.dump(config_dict, f, indent=2, ensure_ascii=False)\n        else:\n            raise ConfigurationError(f"Unsupported format: {format}")\n        \n        logger.info(f"Configuration exported to {output_path}")\n    \n    def get_environment_overrides(self) -> Dict[str, Any]:\n        """Get configuration overrides from environment variables."""\n        overrides = {}\n        \n        # Environment variable mappings\n        env_mappings = {\n            "ACADEMIC_AGENT_LOG_LEVEL": "logging.level",\n            "ACADEMIC_AGENT_LOG_DIR": "logging.log_dir",\n            "ACADEMIC_AGENT_MAX_RETRIES": ("error_handling.max_retries", int),\n            "ACADEMIC_AGENT_RETRY_DELAY": ("error_handling.retry_delay", float),\n            "ACADEMIC_AGENT_QUALITY_THRESHOLD": ("quality_threshold", float),\n            "ACADEMIC_AGENT_IMPROVEMENT_THRESHOLD": ("improvement_threshold", float),\n            "ACADEMIC_AGENT_MAX_IMPROVEMENT_CYCLES": ("max_improvement_cycles", int),\n            "ACADEMIC_AGENT_COMMUNICATION_INTERVAL": ("communication_interval", int),\n        }\n        \n        for env_var, config_path in env_mappings.items():\n            value = os.getenv(env_var)\n            if value is not None:\n                if isinstance(config_path, tuple):\n                    path, converter = config_path\n                    try:\n                        value = converter(value)\n                    except (ValueError, TypeError) as e:\n                        logger.warning(f"Invalid value for {env_var}: {value}. Error: {e}")\n                        continue\n                else:\n                    path = config_path\n                \n                self._set_nested_value(overrides, path, value)\n        \n        return overrides\n    \n    def apply_environment_overrides(self):\n        """Apply environment variable overrides to the configuration."""\n        if self.config is None:\n            raise ConfigurationError("Configuration not loaded")\n        \n        overrides = self.get_environment_overrides()\n        if overrides:\n            logger.info(f"Applying environment overrides: {overrides}")\n            \n            # Update configuration with overrides\n            config_dict = self.config.dict()\n            for path, value in overrides.items():\n                self._set_nested_value(config_dict, path, value)\n            \n            # Recreate configuration with overrides\n            self.config = AcademicAgentConfig(**config_dict)\n\n\n# Global configuration manager instance\n_config_manager: Optional[ConfigManager] = None\n\n\ndef get_config_manager(\n    yaml_config_path: Optional[Union[str, Path]] = None,\n    reload: bool = False\n) -> ConfigManager:\n    """Get the global configuration manager instance.\n    \n    Args:\n        yaml_config_path: Path to YAML configuration file\n        reload: Whether to reload the configuration\n    \n    Returns:\n        ConfigManager instance\n    \"\"\"\n    global _config_manager\n    \n    if _config_manager is None or reload:\n        if yaml_config_path is None:\n            # Default path\n            yaml_config_path = Path("config/logging_config.yaml")\n        \n        _config_manager = ConfigManager(yaml_config_path)\n        _config_manager.apply_environment_overrides()\n    \n    return _config_manager\n\n\ndef get_config() -> AcademicAgentConfig:\n    """Get the current configuration.\n    \n    Returns:\n        AcademicAgentConfig instance\n    \"\"\"\n    return get_config_manager().get_config()\n\n\ndef reload_config():\n    """Reload the configuration from files.\"\"\"\n    get_config_manager(reload=True)