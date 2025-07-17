#!/usr/bin/env python3
"""
Configuration migration script for academic-agent.

This script migrates the existing JSON configuration to the new YAML-based
configuration system with proper validation and error handling.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json_config(config_path: Path) -> Dict[str, Any]:
    """
    Load the existing JSON configuration.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Parsed JSON configuration
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        json.JSONDecodeError: If configuration file is invalid JSON
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    logger.info(f"Loaded JSON configuration from: {config_path}")
    return config_data


def migrate_agent_prompts(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate agent specialized prompts to agent configurations.
    
    Args:
        json_config: Original JSON configuration
        
    Returns:
        Migrated agent configurations
    """
    agents = {}
    
    if "agent_specialized_prompts" in json_config:
        for agent_id, prompt in json_config["agent_specialized_prompts"].items():
            agents[agent_id] = {
                "agent_id": agent_id,
                "enabled": True,
                "specialized_prompt": prompt.strip() if prompt else None
            }
    
    logger.info(f"Migrated {len(agents)} agent configurations")
    return agents


def migrate_feedback_loops(json_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Migrate feedback loop configurations.
    
    Args:
        json_config: Original JSON configuration
        
    Returns:
        Migrated feedback loop configurations
    """
    feedback_loops = []
    
    if "feedback_loops" in json_config:
        for loop in json_config["feedback_loops"]:
            migrated_loop = {
                "source": loop.get("source"),
                "target": loop.get("target"),
                "type": loop.get("type"),
                "interval": loop.get("interval", 300),
                "enabled": True
            }
            feedback_loops.append(migrated_loop)
    
    logger.info(f"Migrated {len(feedback_loops)} feedback loops")
    return feedback_loops


def migrate_improvement_criteria(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate improvement criteria configurations.
    
    Args:
        json_config: Original JSON configuration
        
    Returns:
        Migrated improvement criteria configuration
    """
    improvement_criteria = {}
    
    if "improvement_criteria" in json_config:
        original_criteria = json_config["improvement_criteria"]
        
        for criterion_name, criterion_config in original_criteria.items():
            improvement_criteria[criterion_name] = {
                "weight": criterion_config.get("weight", 0.25),
                "threshold": criterion_config.get("threshold", 0.75),
                "enabled": True
            }
    
    logger.info(f"Migrated {len(improvement_criteria)} improvement criteria")
    return improvement_criteria


def migrate_inter_agent_communication(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate inter-agent communication configuration.
    
    Args:
        json_config: Original JSON configuration
        
    Returns:
        Migrated inter-agent communication configuration
    """
    inter_agent_comm = {}
    
    if "inter_agent_communication" in json_config:
        original_comm = json_config["inter_agent_communication"]
        
        inter_agent_comm = {
            "enabled": original_comm.get("enabled", True),
            "message_timeout": original_comm.get("message_timeout", 120),
            "retry_count": original_comm.get("retry_count", 3),
            "max_log_size": original_comm.get("max_log_size", 1000),
            "compression_enabled": False  # New feature
        }
    
    logger.info("Migrated inter-agent communication configuration")
    return inter_agent_comm


def migrate_processing_config(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate processing configuration.
    
    Args:
        json_config: Original JSON configuration
        
    Returns:
        Migrated processing configuration
    """
    processing = {}
    
    if "processing" in json_config:
        original_processing = json_config["processing"]
        
        processing = {
            "max_concurrent_agents": original_processing.get("max_concurrent_agents", 2),
            "processing_timeout": original_processing.get("processing_timeout", 3600),
            "retry_on_failure": original_processing.get("retry_on_failure", True),
            "preserve_intermediate_results": original_processing.get("preserve_intermediate_results", True),
            "batch_size": 1,  # New feature
            "enable_checkpointing": True  # New feature
        }
    
    logger.info("Migrated processing configuration")
    return processing


def create_yaml_config(json_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the new YAML configuration structure from JSON.
    
    Args:
        json_config: Original JSON configuration
        
    Returns:
        New YAML configuration structure
    """
    yaml_config = {
        "environment": "development",
        "debug": False,
        "version": "1.0.0",
        
        # Core settings from JSON
        "quality_threshold": json_config.get("quality_threshold", 0.75),
        "improvement_threshold": json_config.get("improvement_threshold", 0.3),
        "max_improvement_cycles": json_config.get("max_improvement_cycles", 3),
        "communication_interval": json_config.get("communication_interval", 30),
        
        # Migrated components
        "agents": migrate_agent_prompts(json_config),
        "feedback_loops": migrate_feedback_loops(json_config),
        "inter_agent_communication": migrate_inter_agent_communication(json_config),
        "improvement_criteria": migrate_improvement_criteria(json_config),
        "processing": migrate_processing_config(json_config),
        
        # New configuration sections
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
    
    logger.info("Created new YAML configuration structure")
    return yaml_config


def save_yaml_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save the migrated configuration to a YAML file.
    
    Args:
        config: Configuration to save
        output_path: Path to save the YAML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    logger.info(f"Saved migrated configuration to: {output_path}")


def create_environment_configs(base_config: Dict[str, Any], config_dir: Path) -> None:
    """
    Create environment-specific configuration files.
    
    Args:
        base_config: Base configuration
        config_dir: Configuration directory
    """
    # Development configuration
    dev_config = {
        "debug": True,
        "logging": {
            "level": "DEBUG"
        },
        "processing": {
            "max_concurrent_agents": 1
        }
    }
    save_yaml_config(dev_config, config_dir / "development.yaml")
    
    # Production configuration
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
    save_yaml_config(prod_config, config_dir / "production.yaml")
    
    # Test configuration
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
    save_yaml_config(test_config, config_dir / "test.yaml")
    
    logger.info("Created environment-specific configuration files")


def backup_original_config(json_config_path: Path) -> None:
    """
    Create a backup of the original JSON configuration.
    
    Args:
        json_config_path: Path to the original JSON configuration
    """
    backup_path = json_config_path.with_suffix('.json.backup')
    
    if backup_path.exists():
        logger.warning(f"Backup already exists: {backup_path}")
        return
    
    import shutil
    shutil.copy2(json_config_path, backup_path)
    logger.info(f"Created backup of original configuration: {backup_path}")


def main():
    """Main migration function."""
    # Define paths
    project_root = Path(__file__).parent.parent
    json_config_path = project_root / "config" / "academic_agent_config.json"
    config_dir = project_root / "config"
    
    try:
        # Load existing JSON configuration
        json_config = load_json_config(json_config_path)
        
        # Create backup
        backup_original_config(json_config_path)
        
        # Migrate to new YAML structure
        yaml_config = create_yaml_config(json_config)
        
        # Save base configuration
        save_yaml_config(yaml_config, config_dir / "base.yaml")
        
        # Create environment-specific configurations
        create_environment_configs(yaml_config, config_dir)
        
        logger.info("Migration completed successfully!")
        logger.info("Please review the new configuration files in the config directory.")
        logger.info("The original JSON configuration has been backed up.")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()