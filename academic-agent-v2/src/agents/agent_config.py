"""
Agent Configuration System for Academic Agent v2.

This module provides configuration management specifically for academic agents,
including plugin configurations, task planning settings, and quality thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, validator

from ..core.config_manager import ConfigManager
from ..core.exceptions import ConfigurationError
from ..core.logging import get_logger

logger = get_logger(__name__)


class PluginConfig(BaseModel):
    """Configuration for individual plugins."""
    
    name: str = Field(description="Plugin name")
    enabled: bool = Field(default=True, description="Whether plugin is enabled")
    priority: int = Field(default=0, description="Plugin execution priority")
    config: Dict[str, Any] = Field(default_factory=dict, description="Plugin-specific configuration")
    timeout: Optional[float] = Field(default=None, description="Plugin execution timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    
    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority is non-negative."""
        if v < 0:
            raise ValueError("Priority must be non-negative")
        return v
    
    @validator("max_retries")
    def validate_max_retries(cls, v):
        """Validate max_retries is non-negative."""
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v


class TaskPlanningConfig(BaseModel):
    """Configuration for task planning."""
    
    max_concurrent_tasks: int = Field(default=3, description="Maximum concurrent tasks")
    task_timeout: float = Field(default=3600.0, description="Default task timeout in seconds")
    enable_dependency_resolution: bool = Field(default=True, description="Enable automatic dependency resolution")
    priority_scheduling: bool = Field(default=True, description="Enable priority-based task scheduling")
    retry_failed_tasks: bool = Field(default=True, description="Automatically retry failed tasks")
    max_task_retries: int = Field(default=2, description="Maximum retries per task")
    
    @validator("max_concurrent_tasks")
    def validate_max_concurrent_tasks(cls, v):
        """Validate max_concurrent_tasks is positive."""
        if v <= 0:
            raise ValueError("max_concurrent_tasks must be positive")
        return v
    
    @validator("task_timeout")
    def validate_task_timeout(cls, v):
        """Validate task_timeout is positive."""
        if v <= 0:
            raise ValueError("task_timeout must be positive")
        return v


class QualityConfig(BaseModel):
    """Configuration for quality management."""
    
    quality_threshold: float = Field(default=0.7, description="Minimum quality threshold")
    enable_quality_checks: bool = Field(default=True, description="Enable quality checking")
    quality_improvement_cycles: int = Field(default=3, description="Maximum quality improvement cycles")
    auto_reject_low_quality: bool = Field(default=False, description="Automatically reject low-quality results")
    quality_metrics: List[str] = Field(
        default_factory=lambda: ["completeness", "accuracy", "clarity"],
        description="Quality metrics to evaluate"
    )
    
    @validator("quality_threshold")
    def validate_quality_threshold(cls, v):
        """Validate quality threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("quality_threshold must be between 0 and 1")
        return v
    
    @validator("quality_improvement_cycles")
    def validate_quality_improvement_cycles(cls, v):
        """Validate quality improvement cycles is non-negative."""
        if v < 0:
            raise ValueError("quality_improvement_cycles must be non-negative")
        return v


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and metrics."""
    
    enable_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    metrics_collection_interval: float = Field(default=60.0, description="Metrics collection interval in seconds")
    enable_detailed_logging: bool = Field(default=True, description="Enable detailed operation logging")
    performance_alerts: bool = Field(default=True, description="Enable performance alerts")
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "task_failure_rate": 0.1,
            "average_processing_time": 300.0,
            "memory_usage": 0.8,
            "cpu_usage": 0.9
        },
        description="Alert thresholds for various metrics"
    )
    
    @validator("metrics_collection_interval")
    def validate_metrics_interval(cls, v):
        """Validate metrics collection interval is positive."""
        if v <= 0:
            raise ValueError("metrics_collection_interval must be positive")
        return v


class AgentConfig(BaseModel):
    """Main configuration for academic agents."""
    
    agent_id: str = Field(description="Unique agent identifier")
    agent_type: str = Field(default="academic", description="Type of agent")
    version: str = Field(default="2.0", description="Agent version")
    
    # Core configurations
    plugins: Dict[str, PluginConfig] = Field(default_factory=dict, description="Plugin configurations")
    task_planning: TaskPlanningConfig = Field(default_factory=TaskPlanningConfig, description="Task planning configuration")
    quality: QualityConfig = Field(default_factory=QualityConfig, description="Quality management configuration")
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig, description="Monitoring configuration")
    
    # File paths
    working_directory: Path = Field(default=Path.cwd(), description="Agent working directory")
    output_directory: Path = Field(default=Path("output"), description="Default output directory")
    temp_directory: Path = Field(default=Path("temp"), description="Temporary files directory")
    state_file: Optional[Path] = Field(default=None, description="Agent state persistence file")
    
    # Advanced settings
    enable_state_persistence: bool = Field(default=True, description="Enable agent state persistence")
    state_save_interval: float = Field(default=300.0, description="State save interval in seconds")
    enable_plugin_hot_reload: bool = Field(default=False, description="Enable plugin hot reloading")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    @validator("working_directory", "output_directory", "temp_directory")
    def ensure_path_exists(cls, v):
        """Ensure directories exist."""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("state_save_interval")
    def validate_state_save_interval(cls, v):
        """Validate state save interval is positive."""
        if v <= 0:
            raise ValueError("state_save_interval must be positive")
        return v
    
    def add_plugin_config(self, plugin_name: str, plugin_config: PluginConfig):
        """Add plugin configuration."""
        self.plugins[plugin_name] = plugin_config
    
    def remove_plugin_config(self, plugin_name: str):
        """Remove plugin configuration."""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
    
    def get_plugin_config(self, plugin_name: str) -> Optional[PluginConfig]:
        """Get plugin configuration."""
        return self.plugins.get(plugin_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = self.dict()
        
        # Convert Path objects to strings
        for key in ["working_directory", "output_directory", "temp_directory", "state_file"]:
            if config_dict[key] is not None:
                config_dict[key] = str(config_dict[key])
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary."""
        # Convert plugin configurations
        if "plugins" in config_dict:
            plugins = {}
            for name, plugin_data in config_dict["plugins"].items():
                if isinstance(plugin_data, dict):
                    plugins[name] = PluginConfig(**plugin_data)
                else:
                    plugins[name] = plugin_data
            config_dict["plugins"] = plugins
        
        # Convert nested configurations
        for nested_config, config_class in [
            ("task_planning", TaskPlanningConfig),
            ("quality", QualityConfig),
            ("monitoring", MonitoringConfig),
        ]:
            if nested_config in config_dict and isinstance(config_dict[nested_config], dict):
                config_dict[nested_config] = config_class(**config_dict[nested_config])
        
        return cls(**config_dict)


class AgentConfigManager:
    """Manager for academic agent configurations."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize agent config manager."""
        self.config_manager = config_manager or ConfigManager()
        self.logger = get_logger(__name__)
        self._configs: Dict[str, AgentConfig] = {}
    
    def create_default_config(self, agent_id: str) -> AgentConfig:
        """Create default configuration for an agent."""
        # Create default plugin configurations
        default_plugins = {
            "PDFProcessorPlugin": PluginConfig(
                name="PDFProcessorPlugin",
                enabled=True,
                priority=1,
                config={
                    "use_gpu": True,
                    "extract_images": True,
                    "max_pages": None,
                },
                timeout=600.0,
                max_retries=3,
            ),
            "ContentAnalysisPlugin": PluginConfig(
                name="ContentAnalysisPlugin",
                enabled=True,
                priority=2,
                config={
                    "enable_advanced_analysis": True,
                    "quality_threshold": 0.7,
                },
                timeout=300.0,
                max_retries=2,
            ),
        }
        
        config = AgentConfig(
            agent_id=agent_id,
            plugins=default_plugins,
            state_file=Path(f"state/{agent_id}_state.json"),
        )
        
        self.logger.info(f"Created default configuration for agent {agent_id}")
        return config
    
    def load_config_from_file(self, config_file: Path) -> AgentConfig:
        """Load agent configuration from YAML file."""
        try:
            if not config_file.exists():
                raise ConfigurationError(f"Configuration file not found: {config_file}")
            
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ConfigurationError(f"Empty configuration file: {config_file}")
            
            config = AgentConfig.from_dict(config_data)
            self._configs[config.agent_id] = config
            
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def save_config_to_file(self, config: AgentConfig, config_file: Path):
        """Save agent configuration to YAML file."""
        try:
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(
                    config.to_dict(),
                    f,
                    default_flow_style=False,
                    indent=2,
                    allow_unicode=True,
                )
            
            self.logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def get_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for an agent."""
        return self._configs.get(agent_id)
    
    def set_config(self, agent_id: str, config: AgentConfig):
        """Set configuration for an agent."""
        self._configs[agent_id] = config
        self.logger.info(f"Set configuration for agent {agent_id}")
    
    def list_configs(self) -> List[str]:
        """List all agent IDs with configurations."""
        return list(self._configs.keys())
    
    def validate_config(self, config: AgentConfig) -> List[str]:
        """Validate agent configuration and return any issues."""
        issues = []
        
        # Validate plugin configurations
        for plugin_name, plugin_config in config.plugins.items():
            try:
                # Re-validate using Pydantic
                PluginConfig(**plugin_config.dict())
            except Exception as e:
                issues.append(f"Plugin {plugin_name}: {str(e)}")
        
        # Validate paths
        for path_name, path_value in [
            ("working_directory", config.working_directory),
            ("output_directory", config.output_directory),
            ("temp_directory", config.temp_directory),
        ]:
            if not isinstance(path_value, (str, Path)):
                issues.append(f"{path_name} must be a string or Path object")
            elif not Path(path_value).exists():
                issues.append(f"{path_name} directory does not exist: {path_value}")
        
        # Validate quality thresholds
        if not 0 <= config.quality.quality_threshold <= 1:
            issues.append("Quality threshold must be between 0 and 1")
        
        return issues
    
    def create_template_config(self, template_file: Path):
        """Create a template configuration file."""
        template_config = self.create_default_config("template_agent")
        self.save_config_to_file(template_config, template_file)
        self.logger.info(f"Created template configuration at {template_file}")


# Factory functions for common configurations

def create_pdf_processing_config(agent_id: str, **kwargs) -> AgentConfig:
    """Create configuration optimized for PDF processing."""
    config = AgentConfig(
        agent_id=agent_id,
        plugins={
            "PDFProcessorPlugin": PluginConfig(
                name="PDFProcessorPlugin",
                enabled=True,
                priority=1,
                config={
                    "use_gpu": kwargs.get("use_gpu", True),
                    "extract_images": kwargs.get("extract_images", True),
                    "max_pages": kwargs.get("max_pages"),
                    "batch_size": kwargs.get("batch_size", 1),
                },
                timeout=kwargs.get("pdf_timeout", 600.0),
                max_retries=kwargs.get("pdf_retries", 3),
            ),
        },
        task_planning=TaskPlanningConfig(
            max_concurrent_tasks=kwargs.get("max_concurrent", 2),
            task_timeout=kwargs.get("task_timeout", 3600.0),
        ),
        quality=QualityConfig(
            quality_threshold=kwargs.get("quality_threshold", 0.75),
            enable_quality_checks=kwargs.get("enable_quality", True),
        ),
    )
    
    return config


def create_content_analysis_config(agent_id: str, **kwargs) -> AgentConfig:
    """Create configuration optimized for content analysis."""
    config = AgentConfig(
        agent_id=agent_id,
        plugins={
            "ContentAnalysisPlugin": PluginConfig(
                name="ContentAnalysisPlugin",
                enabled=True,
                priority=1,
                config={
                    "enable_advanced_analysis": kwargs.get("advanced_analysis", True),
                    "quality_threshold": kwargs.get("analysis_quality", 0.7),
                    "enable_semantic_analysis": kwargs.get("semantic_analysis", False),
                },
                timeout=kwargs.get("analysis_timeout", 300.0),
                max_retries=kwargs.get("analysis_retries", 2),
            ),
        },
        task_planning=TaskPlanningConfig(
            max_concurrent_tasks=kwargs.get("max_concurrent", 3),
            enable_dependency_resolution=True,
        ),
        quality=QualityConfig(
            quality_threshold=kwargs.get("quality_threshold", 0.8),
            quality_improvement_cycles=kwargs.get("improvement_cycles", 2),
        ),
    )
    
    return config