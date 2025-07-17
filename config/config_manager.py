"""
Configuration management system for the academic-agent project.

This module provides a robust YAML-based configuration system with Pydantic validation,
environment-specific configurations, and proper error handling.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    enabled: bool = Field(default=True, description="Whether the agent is enabled")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum number of retries")
    timeout: int = Field(default=300, ge=30, le=3600, description="Processing timeout in seconds")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Quality threshold")
    specialized_prompt: Optional[str] = Field(default=None, description="Specialized prompt for the agent")
    
    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent ID cannot be empty")
        return v.strip()


class FeedbackLoopConfig(BaseModel):
    """Configuration for feedback loops between agents."""
    
    source: str = Field(..., description="Source agent ID")
    target: str = Field(..., description="Target agent ID")
    type: str = Field(..., description="Type of feedback")
    interval: int = Field(default=300, ge=30, le=3600, description="Interval in seconds")
    enabled: bool = Field(default=True, description="Whether the feedback loop is enabled")
    
    @field_validator('source', 'target')
    @classmethod
    def validate_agent_refs(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Agent reference cannot be empty")
        return v.strip()


class InterAgentCommunicationConfig(BaseModel):
    """Configuration for inter-agent communication."""
    
    enabled: bool = Field(default=True, description="Whether inter-agent communication is enabled")
    message_timeout: int = Field(default=120, ge=30, le=600, description="Message timeout in seconds")
    retry_count: int = Field(default=3, ge=1, le=10, description="Number of message retries")
    max_log_size: int = Field(default=1000, ge=100, le=10000, description="Maximum log size")
    compression_enabled: bool = Field(default=False, description="Whether message compression is enabled")


class QualityCriterionConfig(BaseModel):
    """Configuration for quality criteria."""
    
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight of the criterion")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Threshold for the criterion")
    enabled: bool = Field(default=True, description="Whether the criterion is enabled")


class ImprovementCriteriaConfig(BaseModel):
    """Configuration for improvement criteria."""
    
    content_quality: QualityCriterionConfig = Field(default_factory=lambda: QualityCriterionConfig(weight=0.4, threshold=0.75))
    clarity: QualityCriterionConfig = Field(default_factory=lambda: QualityCriterionConfig(weight=0.3, threshold=0.7))
    structure: QualityCriterionConfig = Field(default_factory=lambda: QualityCriterionConfig(weight=0.2, threshold=0.8))
    citations: QualityCriterionConfig = Field(default_factory=lambda: QualityCriterionConfig(weight=0.1, threshold=0.9))
    
    def get_total_weight(self) -> float:
        """Calculate total weight of all criteria."""
        return sum(criterion.weight for criterion in [self.content_quality, self.clarity, self.structure, self.citations])
    
    def validate_weights(self) -> bool:
        """Validate that weights sum to 1.0."""
        total_weight = self.get_total_weight()
        return abs(total_weight - 1.0) < 0.01  # Allow small floating point errors


class ProcessingConfig(BaseModel):
    """Configuration for processing settings."""
    
    max_concurrent_agents: int = Field(default=2, ge=1, le=10, description="Maximum concurrent agents")
    processing_timeout: int = Field(default=3600, ge=600, le=7200, description="Processing timeout in seconds")
    retry_on_failure: bool = Field(default=True, description="Whether to retry on failure")
    preserve_intermediate_results: bool = Field(default=True, description="Whether to preserve intermediate results")
    batch_size: int = Field(default=1, ge=1, le=100, description="Batch size for processing")
    enable_checkpointing: bool = Field(default=True, description="Whether to enable checkpointing")


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_enabled: bool = Field(default=True, description="Whether file logging is enabled")
    console_enabled: bool = Field(default=True, description="Whether console logging is enabled")
    log_dir: Path = Field(default=Path("logs"), description="Directory for log files")
    max_file_size: int = Field(default=10485760, ge=1048576, description="Maximum log file size in bytes")  # 10MB
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class PathConfig(BaseModel):
    """Configuration for file paths."""
    
    input_dir: Path = Field(default=Path("input"), description="Input directory")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    processed_dir: Path = Field(default=Path("processed"), description="Processed files directory")
    analysis_dir: Path = Field(default=Path("processed/analysis"), description="Analysis directory")
    outlines_dir: Path = Field(default=Path("processed/outlines"), description="Outlines directory")
    notes_dir: Path = Field(default=Path("processed/notes"), description="Notes directory")
    metadata_dir: Path = Field(default=Path("metadata"), description="Metadata directory")
    temp_dir: Path = Field(default=Path("temp"), description="Temporary files directory")
    
    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        directories = [
            self.input_dir, self.output_dir, self.processed_dir,
            self.analysis_dir, self.outlines_dir, self.notes_dir,
            self.metadata_dir, self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


class DatabaseConfig(BaseModel):
    """Configuration for database settings."""
    
    enabled: bool = Field(default=False, description="Whether database is enabled")
    type: str = Field(default="sqlite", description="Database type")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="academic_agent", description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, description="Database password")
    connection_pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    
    @field_validator('type')
    @classmethod
    def validate_db_type(cls, v: str) -> str:
        valid_types = ['sqlite', 'postgresql', 'mysql']
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid database type: {v}. Must be one of {valid_types}")
        return v.lower()


class AcademicAgentConfig(BaseModel):
    """Main configuration model for the academic agent system."""
    
    # Core settings
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    version: str = Field(default="1.0.0", description="Configuration version")
    
    # Quality and improvement settings
    quality_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Global quality threshold")
    improvement_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Improvement threshold")
    max_improvement_cycles: int = Field(default=3, ge=1, le=10, description="Maximum improvement cycles")
    communication_interval: int = Field(default=30, ge=10, le=300, description="Communication interval in seconds")
    
    # Component configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=dict, description="Agent configurations")
    feedback_loops: List[FeedbackLoopConfig] = Field(default_factory=list, description="Feedback loop configurations")
    inter_agent_communication: InterAgentCommunicationConfig = Field(default_factory=InterAgentCommunicationConfig, description="Inter-agent communication settings")
    improvement_criteria: ImprovementCriteriaConfig = Field(default_factory=ImprovementCriteriaConfig, description="Improvement criteria")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="Processing configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    paths: PathConfig = Field(default_factory=PathConfig, description="Path configuration")
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Database configuration")
    
    # Extension point for custom configurations
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration values")
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation and setup."""
        # Validate improvement criteria weights
        if not self.improvement_criteria.validate_weights():
            logger.warning("Improvement criteria weights do not sum to 1.0")
        
        # Ensure required directories exist
        self.paths.ensure_directories()
        
        # Validate feedback loop references
        self._validate_feedback_loops()
    
    def _validate_feedback_loops(self) -> None:
        """Validate that feedback loops reference existing agents."""
        agent_ids = set(self.agents.keys())
        
        for loop in self.feedback_loops:
            if loop.source not in agent_ids:
                logger.warning(f"Feedback loop references unknown source agent: {loop.source}")
            if loop.target not in agent_ids:
                logger.warning(f"Feedback loop references unknown target agent: {loop.target}")
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_id)
    
    def add_agent_config(self, agent_config: AgentConfig) -> None:
        """Add or update agent configuration."""
        self.agents[agent_config.agent_id] = agent_config
    
    def get_feedback_loops_for_agent(self, agent_id: str) -> List[FeedbackLoopConfig]:
        """Get all feedback loops involving a specific agent."""
        return [loop for loop in self.feedback_loops if loop.source == agent_id or loop.target == agent_id]
    
    def is_agent_enabled(self, agent_id: str) -> bool:
        """Check if an agent is enabled."""
        agent_config = self.get_agent_config(agent_id)
        return agent_config.enabled if agent_config else False


class EnvironmentSettings(BaseSettings):
    """Environment-specific settings using Pydantic BaseSettings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    hf_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    # Environment settings
    environment: str = Field(default="development", description="Environment name")
    debug_mode: bool = Field(default=False, description="Debug mode")
    
    # Processing settings
    pdf_processing_device: str = Field(default="cpu", description="Device for PDF processing")
    default_output_dir: Path = Field(default=Path("./output"), description="Default output directory")
    
    # Database settings
    database_url: Optional[str] = Field(default=None, description="Database URL")
    
    @field_validator('pdf_processing_device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices = ['cpu', 'cuda', 'mps']
        if v.lower() not in valid_devices:
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v.lower()


class ConfigurationManager:
    """
    Configuration manager for loading and managing YAML-based configurations.
    
    This class provides methods to load configurations from YAML files,
    validate them using Pydantic models, and manage environment-specific settings.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Optional[AcademicAgentConfig] = None
        self._env_settings: Optional[EnvironmentSettings] = None
        
        logger.info(f"Configuration manager initialized with config directory: {self.config_dir}")
    
    def load_config(self, environment: str = "development") -> AcademicAgentConfig:
        """
        Load configuration for the specified environment.
        
        Args:
            environment: Environment name (development, production, test)
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If configuration cannot be loaded or validated
        """
        try:
            # Load environment settings first
            self._env_settings = EnvironmentSettings()
            
            # Load base configuration
            base_config = self._load_yaml_config("base.yaml")
            
            # Load environment-specific configuration
            env_config_file = f"{environment}.yaml"
            env_config = self._load_yaml_config(env_config_file)
            
            # Merge configurations (environment overrides base)
            merged_config = self._merge_configs(base_config, env_config)
            
            # Override with environment variables
            merged_config = self._apply_env_overrides(merged_config)
            
            # Validate and create configuration object
            self._config = AcademicAgentConfig(**merged_config)
            
            logger.info(f"Successfully loaded configuration for environment: {environment}")
            return self._config
            
        except Exception as e:
            error_msg = f"Failed to load configuration for environment '{environment}': {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            filename: Name of the YAML file
            
        Returns:
            Parsed YAML configuration
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            if filename == "base.yaml":
                # Create default base configuration if it doesn't exist
                return self._create_default_base_config()
            else:
                # For environment-specific configs, return empty dict
                logger.warning(f"Configuration file not found: {config_path}")
                return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            logger.debug(f"Loaded configuration from: {config_path}")
            return config_data
            
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse YAML file '{config_path}': {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load configuration file '{config_path}': {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _create_default_base_config(self) -> Dict[str, Any]:
        """Create default base configuration."""
        return {
            "environment": "development",
            "debug": False,
            "version": "1.0.0",
            "quality_threshold": 0.75,
            "improvement_threshold": 0.3,
            "max_improvement_cycles": 3,
            "communication_interval": 30,
            "agents": {},
            "feedback_loops": [],
            "inter_agent_communication": {
                "enabled": True,
                "message_timeout": 120,
                "retry_count": 3,
                "max_log_size": 1000
            },
            "improvement_criteria": {
                "content_quality": {"weight": 0.4, "threshold": 0.75},
                "clarity": {"weight": 0.3, "threshold": 0.7},
                "structure": {"weight": 0.2, "threshold": 0.8},
                "citations": {"weight": 0.1, "threshold": 0.9}
            },
            "processing": {
                "max_concurrent_agents": 2,
                "processing_timeout": 3600,
                "retry_on_failure": True,
                "preserve_intermediate_results": True
            },
            "logging": {
                "level": "INFO",
                "file_enabled": True,
                "console_enabled": True,
                "log_dir": "logs"
            },
            "paths": {
                "input_dir": "input",
                "output_dir": "output",
                "processed_dir": "processed",
                "analysis_dir": "processed/analysis",
                "outlines_dir": "processed/outlines",
                "notes_dir": "processed/notes",
                "metadata_dir": "metadata",
                "temp_dir": "temp"
            },
            "database": {
                "enabled": False,
                "type": "sqlite"
            }
        }
    
    def _merge_configs(self, base_config: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base and environment-specific configurations.
        
        Args:
            base_config: Base configuration
            env_config: Environment-specific configuration
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        for key, value in env_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override with environment value
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        if not self._env_settings:
            return config
        
        # Apply environment-specific overrides
        if self._env_settings.debug_mode:
            config["debug"] = True
            config["logging"]["level"] = "DEBUG"
        
        if self._env_settings.environment:
            config["environment"] = self._env_settings.environment
        
        if self._env_settings.default_output_dir:
            config["paths"]["output_dir"] = str(self._env_settings.default_output_dir)
        
        return config
    
    def save_config(self, config: AcademicAgentConfig, filename: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration to save
            filename: Name of the YAML file
            
        Raises:
            ConfigurationError: If configuration cannot be saved
        """
        try:
            config_path = self.config_dir / filename
            
            # Convert Pydantic model to dict, excluding default values
            config_dict = config.model_dump(exclude_defaults=True, exclude_none=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            error_msg = f"Failed to save configuration to '{filename}': {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def get_config(self) -> AcademicAgentConfig:
        """
        Get the current configuration.
        
        Returns:
            Current configuration
            
        Raises:
            ConfigurationError: If no configuration is loaded
        """
        if self._config is None:
            raise ConfigurationError("No configuration loaded. Call load_config() first.")
        return self._config
    
    def get_env_settings(self) -> EnvironmentSettings:
        """
        Get the current environment settings.
        
        Returns:
            Current environment settings
            
        Raises:
            ConfigurationError: If no environment settings are loaded
        """
        if self._env_settings is None:
            raise ConfigurationError("No environment settings loaded. Call load_config() first.")
        return self._env_settings
    
    def reload_config(self) -> AcademicAgentConfig:
        """
        Reload the current configuration.
        
        Returns:
            Reloaded configuration
        """
        if self._config is None:
            raise ConfigurationError("No configuration to reload. Call load_config() first.")
        
        environment = self._config.environment
        return self.load_config(environment)
    
    def validate_config(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            AcademicAgentConfig(**config_dict)
            return []
        except ValidationError as e:
            return [str(error) for error in e.errors()]
    
    def create_sample_configs(self) -> None:
        """Create sample configuration files for different environments."""
        # Create base configuration
        base_config = self._create_default_base_config()
        base_config_path = self.config_dir / "base.yaml"
        
        with open(base_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(base_config, f, default_flow_style=False, sort_keys=False)
        
        # Create development configuration
        dev_config = {
            "debug": True,
            "logging": {
                "level": "DEBUG"
            },
            "processing": {
                "max_concurrent_agents": 1
            }
        }
        dev_config_path = self.config_dir / "development.yaml"
        
        with open(dev_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(dev_config, f, default_flow_style=False, sort_keys=False)
        
        # Create production configuration
        prod_config = {
            "debug": False,
            "logging": {
                "level": "INFO"
            },
            "processing": {
                "max_concurrent_agents": 4,
                "processing_timeout": 7200
            }
        }
        prod_config_path = self.config_dir / "production.yaml"
        
        with open(prod_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(prod_config, f, default_flow_style=False, sort_keys=False)
        
        # Create test configuration
        test_config = {
            "debug": True,
            "logging": {
                "level": "DEBUG",
                "file_enabled": False
            },
            "processing": {
                "max_concurrent_agents": 1,
                "processing_timeout": 60
            }
        }
        test_config_path = self.config_dir / "test.yaml"
        
        with open(test_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(test_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info("Sample configuration files created")


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> AcademicAgentConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def get_env_settings() -> EnvironmentSettings:
    """Get the global environment settings instance."""
    return config_manager.get_env_settings()