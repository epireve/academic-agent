"""
Unified Configuration System for Academic Agent

This module provides a unified configuration system that supports both
v2 and legacy configuration formats, with automatic migration.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from pydantic import BaseModel, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create minimal base model
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    def Field(**kwargs):
        return kwargs.get('default', None)
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..core.exceptions import ConfigurationError
from ..core.logging import get_logger

logger = get_logger(__name__)


class AgentConfig(BaseModel):
    """Configuration for an individual agent."""
    
    agent_id: str
    agent_type: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
    
    @field_validator("agent_type")
    def validate_agent_type(cls, v):
        """Validate agent type."""
        valid_types = ["unified", "v2", "legacy", "custom"]
        if v not in valid_types:
            raise ValueError(f"agent_type must be one of {valid_types}")
        return v


class ProcessorConfig(BaseModel):
    """Configuration for processors."""
    
    pdf_processor: Dict[str, Any] = Field(default_factory=lambda: {
        "preferred_backend": "marker",
        "use_gpu": True,
        "cache_enabled": True,
        "max_workers": 4,
    })
    
    content_processor: Dict[str, Any] = Field(default_factory=dict)
    
    week_resolver: Dict[str, Any] = Field(default_factory=dict)


class SystemConfig(BaseModel):
    """System-wide configuration."""
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "detailed"
    log_file: Optional[Path] = None
    
    # Monitoring configuration
    metrics_enabled: bool = True
    metrics_export_interval: int = 60
    
    # Performance settings
    max_concurrent_tasks: int = 10
    task_timeout: int = 300
    
    # Storage paths
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    output_dir: Path = Field(default_factory=lambda: Path("output"))
    cache_dir: Path = Field(default_factory=lambda: Path(".cache"))
    
    @field_validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class UnifiedConfig(BaseModel):
    """Unified configuration for the entire Academic Agent system."""
    
    version: str = "3.0.0"
    system: SystemConfig = Field(default_factory=SystemConfig)
    processors: ProcessorConfig = Field(default_factory=ProcessorConfig)
    agents: List[AgentConfig] = Field(default_factory=list)
    
    # Legacy compatibility
    legacy_mode: bool = False
    v2_compatibility: bool = True
    
    def get_agent_config(self, agent_id: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def add_agent(self, agent_config: AgentConfig):
        """Add an agent configuration."""
        # Remove existing config if present
        if self.agents is None:
            self.agents = []
        self.agents = [a for a in self.agents if a.agent_id != agent_config.agent_id]
        self.agents.append(agent_config)
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "UnifiedConfig":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ConfigurationError("PyYAML not available for YAML config files")
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                data = json.load(f)
        else:
            raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**data)
    
    def to_file(self, config_path: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if config_path.suffix in [".yaml", ".yml"]:
            if not YAML_AVAILABLE:
                raise ConfigurationError("PyYAML not available for YAML config files")
            with open(config_path, "w") as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False)
        elif config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(self.model_dump(), f, indent=2)
        else:
            raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
    
    @classmethod
    def from_legacy_config(cls, legacy_config: Dict[str, Any]) -> "UnifiedConfig":
        """Create unified config from legacy configuration."""
        unified = cls(legacy_mode=True)
        
        # Map legacy settings
        if "agents" in legacy_config:
            for agent_id, agent_data in legacy_config["agents"].items():
                agent_config = AgentConfig(
                    agent_id=agent_id,
                    agent_type="legacy",
                    enabled=agent_data.get("enabled", True),
                    config=agent_data,
                )
                unified.add_agent(agent_config)
        
        # Map system settings
        if "system" in legacy_config:
            system = legacy_config["system"]
            unified.system.log_level = system.get("log_level", "INFO")
            unified.system.max_concurrent_tasks = system.get("max_tasks", 10)
        
        return unified
    
    @classmethod
    def from_v2_config(cls, v2_config: Dict[str, Any]) -> "UnifiedConfig":
        """Create unified config from v2 configuration."""
        unified = cls(v2_compatibility=True)
        
        # Map v2 settings
        if "agents" in v2_config:
            for agent_data in v2_config["agents"]:
                agent_config = AgentConfig(
                    agent_id=agent_data.get("agent_id", "v2_agent"),
                    agent_type="v2",
                    enabled=agent_data.get("enabled", True),
                    config=agent_data.get("config", {}),
                    capabilities=agent_data.get("plugins", []),
                )
                unified.add_agent(agent_config)
        
        # Map processor settings
        if "pdf_processing" in v2_config:
            unified.processors.pdf_processor.update(v2_config["pdf_processing"])
        
        return unified
    
    def to_v2_format(self) -> Dict[str, Any]:
        """Convert to v2 configuration format."""
        v2_config = {
            "version": "2.0",
            "agents": [],
            "pdf_processing": self.processors.pdf_processor,
        }
        
        for agent in self.agents:
            v2_config["agents"].append({
                "agent_id": agent.agent_id,
                "enabled": agent.enabled,
                "config": agent.config,
                "plugins": agent.capabilities,
            })
        
        return v2_config
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy configuration format."""
        legacy_config = {
            "agents": {},
            "system": {
                "log_level": self.system.log_level,
                "max_tasks": self.system.max_concurrent_tasks,
            }
        }
        
        for agent in self.agents:
            legacy_config["agents"][agent.agent_id] = {
                "enabled": agent.enabled,
                **agent.config
            }
        
        return legacy_config


class ConfigMigrator:
    """Handles configuration migration between formats."""
    
    @staticmethod
    def detect_format(config_data: Dict[str, Any]) -> str:
        """Detect configuration format."""
        # Check for version markers
        if "version" in config_data:
            version = config_data["version"]
            if version.startswith("3."):
                return "unified"
            elif version.startswith("2."):
                return "v2"
        
        # Check for v2 structure
        if "agents" in config_data and isinstance(config_data["agents"], list):
            return "v2"
        
        # Default to legacy
        return "legacy"
    
    @staticmethod
    def migrate(config_data: Dict[str, Any]) -> UnifiedConfig:
        """Migrate configuration to unified format."""
        format_type = ConfigMigrator.detect_format(config_data)
        
        if format_type == "unified":
            return UnifiedConfig(**config_data)
        elif format_type == "v2":
            logger.info("Migrating v2 configuration to unified format")
            return UnifiedConfig.from_v2_config(config_data)
        else:
            logger.info("Migrating legacy configuration to unified format")
            return UnifiedConfig.from_legacy_config(config_data)
    
    @staticmethod
    def migrate_file(
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> UnifiedConfig:
        """Migrate configuration file to unified format."""
        input_path = Path(input_path)
        
        # Load configuration
        if input_path.suffix in [".yaml", ".yml"]:
            with open(input_path, "r") as f:
                config_data = yaml.safe_load(f)
        elif input_path.suffix == ".json":
            with open(input_path, "r") as f:
                config_data = json.load(f)
        else:
            raise ConfigurationError(f"Unsupported config format: {input_path.suffix}")
        
        # Migrate to unified format
        unified_config = ConfigMigrator.migrate(config_data)
        
        # Save if output path provided
        if output_path:
            unified_config.to_file(output_path)
            logger.info(f"Migrated configuration saved to {output_path}")
        
        return unified_config


# Global configuration instance
_global_config: Optional[UnifiedConfig] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> UnifiedConfig:
    """Load and return the global configuration."""
    global _global_config
    
    if config_path:
        _global_config = UnifiedConfig.from_file(config_path)
    elif _global_config is None:
        # Load default configuration
        _global_config = UnifiedConfig()
    
    return _global_config


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    if _global_config is None:
        return load_config()
    return _global_config


def save_config(config_path: Union[str, Path]):
    """Save the global configuration."""
    config = get_config()
    config.to_file(config_path)


# Environment variable support

def load_from_env() -> UnifiedConfig:
    """Load configuration from environment variables."""
    config = UnifiedConfig()
    
    # System settings
    if log_level := os.getenv("ACADEMIC_AGENT_LOG_LEVEL"):
        config.system.log_level = log_level
    
    if metrics := os.getenv("ACADEMIC_AGENT_METRICS_ENABLED"):
        config.system.metrics_enabled = metrics.lower() == "true"
    
    # Processor settings
    if backend := os.getenv("ACADEMIC_AGENT_PDF_BACKEND"):
        config.processors.pdf_processor["preferred_backend"] = backend
    
    if use_gpu := os.getenv("ACADEMIC_AGENT_USE_GPU"):
        config.processors.pdf_processor["use_gpu"] = use_gpu.lower() == "true"
    
    return config