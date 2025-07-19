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
        
        current[keys[-1]] = value

    def get_config(self) -> AcademicAgentConfig:
        """Get the current configuration."""
        if self.config is None:
            raise ConfigurationError("Configuration not loaded")
        return self.config

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.get_config().logging

    def get_error_handling_config(self) -> ErrorHandlingConfig:
        """Get error handling configuration."""
        return self.get_config().error_handling

    def get_metrics_config(self) -> MetricsConfig:
        """Get metrics configuration."""
        return self.get_config().metrics

    def get_operation_config(self, operation_name: str) -> Dict[str, Any]:
        """Get configuration for a specific operation."""
        operations_config = self.get_config().error_handling.operations
        return operations_config.get(operation_name, {})

    def get_agent_prompt(self, agent_name: str) -> Optional[str]:
        """Get specialized prompt for an agent."""
        return self.get_config().agent_specialized_prompts.get(agent_name)

    def reload_configuration(self):
        """Reload configuration from files."""
        logger.info("Reloading configuration")
        self._load_configuration()

    def validate_configuration(self) -> bool:
        """Validate the current configuration."""
        try:
            if self.config is None:
                return False

            # Validate logging configuration
            log_dir = Path(self.config.logging.log_dir)
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)

            # Validate error handling configuration
            if self.config.error_handling.max_retries < 0:
                raise ConfigurationError("max_retries must be non-negative")

            if self.config.error_handling.retry_delay < 0:
                raise ConfigurationError("retry_delay must be non-negative")

            # Validate quality thresholds
            if not 0 <= self.config.quality_threshold <= 1:
                raise ConfigurationError("quality_threshold must be between 0 and 1")

            logger.info("Configuration validation successful")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def export_merged_config(self, output_path: Union[str, Path], format: str = "yaml") -> None:
        """Export the merged configuration to a file.
        
        Args:
            output_path: Path to save the configuration
            format: Output format ('yaml' or 'json')
        """
        if self.config is None:
            raise ConfigurationError("Configuration not loaded")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.config.dict()

        if format.lower() == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ConfigurationError(f"Unsupported format: {format}")

        logger.info(f"Configuration exported to {output_path}")

    def get_environment_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}

        # Environment variable mappings
        env_mappings = {
            "ACADEMIC_AGENT_LOG_LEVEL": "logging.level",
            "ACADEMIC_AGENT_LOG_DIR": "logging.log_dir",
            "ACADEMIC_AGENT_MAX_RETRIES": ("error_handling.max_retries", int),
            "ACADEMIC_AGENT_RETRY_DELAY": ("error_handling.retry_delay", float),
            "ACADEMIC_AGENT_QUALITY_THRESHOLD": ("quality_threshold", float),
            "ACADEMIC_AGENT_IMPROVEMENT_THRESHOLD": ("improvement_threshold", float),
            "ACADEMIC_AGENT_MAX_IMPROVEMENT_CYCLES": ("max_improvement_cycles", int),
            "ACADEMIC_AGENT_COMMUNICATION_INTERVAL": ("communication_interval", int),
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_path, tuple):
                    path, converter = config_path
                    try:
                        value = converter(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid value for {env_var}: {value}. Error: {e}")
                        continue
                else:
                    path = config_path

                self._set_nested_value(overrides, path, value)

        return overrides

    def apply_environment_overrides(self):
        """Apply environment variable overrides to the configuration."""
        if self.config is None:
            raise ConfigurationError("Configuration not loaded")

        overrides = self.get_environment_overrides()
        if overrides:
            logger.info(f"Applying environment overrides: {overrides}")

            # Update configuration with overrides
            config_dict = self.config.dict()
            for path, value in overrides.items():
                self._set_nested_value(config_dict, path, value)

            # Recreate configuration with overrides
            self.config = AcademicAgentConfig(**config_dict)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    yaml_config_path: Optional[Union[str, Path]] = None,
    reload: bool = False
) -> ConfigManager:
    """Get the global configuration manager instance.
    
    Args:
        yaml_config_path: Path to YAML configuration file
        reload: Whether to reload the configuration
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None or reload:
        if yaml_config_path is None:
            # Default path
            yaml_config_path = Path("config/logging_config.yaml")
        
        _config_manager = ConfigManager(yaml_config_path)
        _config_manager.apply_environment_overrides()
    
    return _config_manager


def get_config() -> AcademicAgentConfig:
    """Get the current configuration.
    
    Returns:
        AcademicAgentConfig instance
    """
    return get_config_manager().get_config()


def reload_config():
    """Reload the configuration from files."""
    get_config_manager(reload=True)