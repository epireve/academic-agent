#!/usr/bin/env python3
"""
Configuration utility functions and CLI for academic-agent.

This module provides utility functions for configuration management,
including creation, validation, and migration of configuration files.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from config_manager import ConfigurationManager, AcademicAgentConfig, AgentConfig


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_agent_config(agent_id: str, **kwargs) -> AgentConfig:
    """
    Create an agent configuration with default values.
    
    Args:
        agent_id: Agent identifier
        **kwargs: Additional configuration parameters
        
    Returns:
        Agent configuration
    """
    config_data = {
        "agent_id": agent_id,
        "enabled": kwargs.get("enabled", True),
        "max_retries": kwargs.get("max_retries", 3),
        "timeout": kwargs.get("timeout", 300),
        "quality_threshold": kwargs.get("quality_threshold", 0.7),
        "specialized_prompt": kwargs.get("specialized_prompt")
    }
    
    return AgentConfig(**config_data)


def export_config_to_json(config: AcademicAgentConfig, output_path: Path) -> None:
    """
    Export configuration to JSON format.
    
    Args:
        config: Configuration to export
        output_path: Path to save JSON file
    """
    config_dict = config.model_dump(exclude_defaults=False, exclude_none=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Configuration exported to JSON: {output_path}")


def import_config_from_json(json_path: Path) -> AcademicAgentConfig:
    """
    Import configuration from JSON format.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Imported configuration
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return AcademicAgentConfig(**config_dict)


def merge_configurations(base_config: AcademicAgentConfig, override_config: Dict[str, Any]) -> AcademicAgentConfig:
    """
    Merge a base configuration with override values.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration values
        
    Returns:
        Merged configuration
    """
    base_dict = base_config.model_dump(exclude_defaults=False, exclude_none=True)
    
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    merged_dict = deep_merge(base_dict, override_config)
    return AcademicAgentConfig(**merged_dict)


def get_config_diff(config1: AcademicAgentConfig, config2: AcademicAgentConfig) -> Dict[str, Any]:
    """
    Get differences between two configurations.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary of differences
    """
    dict1 = config1.model_dump(exclude_defaults=False, exclude_none=True)
    dict2 = config2.model_dump(exclude_defaults=False, exclude_none=True)
    
    def find_differences(d1: Dict[str, Any], d2: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Find differences recursively."""
        diff = {}
        
        all_keys = set(d1.keys()) | set(d2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in d1:
                diff[current_path] = {"added": d2[key]}
            elif key not in d2:
                diff[current_path] = {"removed": d1[key]}
            elif d1[key] != d2[key]:
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    nested_diff = find_differences(d1[key], d2[key], current_path)
                    diff.update(nested_diff)
                else:
                    diff[current_path] = {"changed": {"from": d1[key], "to": d2[key]}}
        
        return diff
    
    return find_differences(dict1, dict2)


def validate_agent_ids(config: AcademicAgentConfig) -> List[str]:
    """
    Validate agent IDs and references.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors
    """
    errors = []
    agent_ids = set(config.agents.keys())
    
    # Check for duplicate agent IDs
    if len(agent_ids) != len(config.agents):
        errors.append("Duplicate agent IDs found")
    
    # Check feedback loop references
    for i, loop in enumerate(config.feedback_loops):
        if loop.source not in agent_ids:
            errors.append(f"Feedback loop {i} references unknown source agent: {loop.source}")
        if loop.target not in agent_ids:
            errors.append(f"Feedback loop {i} references unknown target agent: {loop.target}")
    
    return errors


def optimize_configuration(config: AcademicAgentConfig) -> AcademicAgentConfig:
    """
    Optimize configuration for better performance.
    
    Args:
        config: Configuration to optimize
        
    Returns:
        Optimized configuration
    """
    optimized_dict = config.model_dump(exclude_defaults=False, exclude_none=True)
    
    # Optimize processing settings based on environment
    if config.environment == "production":
        optimized_dict["processing"]["max_concurrent_agents"] = min(
            optimized_dict["processing"]["max_concurrent_agents"] * 2, 8
        )
        optimized_dict["processing"]["processing_timeout"] = max(
            optimized_dict["processing"]["processing_timeout"], 7200
        )
    
    # Optimize logging for production
    if config.environment == "production":
        optimized_dict["logging"]["level"] = "INFO"
        optimized_dict["logging"]["console_enabled"] = False
    
    # Optimize quality thresholds
    if config.environment == "test":
        optimized_dict["quality_threshold"] = 0.6
        for criterion in optimized_dict["improvement_criteria"].values():
            if isinstance(criterion, dict):
                criterion["threshold"] = max(criterion.get("threshold", 0.7) - 0.1, 0.5)
    
    return AcademicAgentConfig(**optimized_dict)


def generate_config_documentation(config: AcademicAgentConfig) -> str:
    """
    Generate documentation for the configuration.
    
    Args:
        config: Configuration to document
        
    Returns:
        Documentation string
    """
    doc = []
    doc.append("# Academic Agent Configuration Documentation\n")
    
    doc.append(f"**Environment:** {config.environment}")
    doc.append(f"**Version:** {config.version}")
    doc.append(f"**Debug Mode:** {config.debug}\n")
    
    doc.append("## Core Settings")
    doc.append(f"- Quality Threshold: {config.quality_threshold}")
    doc.append(f"- Improvement Threshold: {config.improvement_threshold}")
    doc.append(f"- Max Improvement Cycles: {config.max_improvement_cycles}")
    doc.append(f"- Communication Interval: {config.communication_interval}s\n")
    
    doc.append("## Agents")
    if config.agents:
        for agent_id, agent_config in config.agents.items():
            doc.append(f"### {agent_id}")
            doc.append(f"- Enabled: {agent_config.enabled}")
            doc.append(f"- Max Retries: {agent_config.max_retries}")
            doc.append(f"- Timeout: {agent_config.timeout}s")
            doc.append(f"- Quality Threshold: {agent_config.quality_threshold}")
            if agent_config.specialized_prompt:
                doc.append(f"- Has Specialized Prompt: Yes")
            doc.append("")
    else:
        doc.append("No agents configured.\n")
    
    doc.append("## Feedback Loops")
    if config.feedback_loops:
        for i, loop in enumerate(config.feedback_loops):
            doc.append(f"### Loop {i + 1}")
            doc.append(f"- Source: {loop.source}")
            doc.append(f"- Target: {loop.target}")
            doc.append(f"- Type: {loop.type}")
            doc.append(f"- Interval: {loop.interval}s")
            doc.append(f"- Enabled: {loop.enabled}")
            doc.append("")
    else:
        doc.append("No feedback loops configured.\n")
    
    doc.append("## Processing Configuration")
    doc.append(f"- Max Concurrent Agents: {config.processing.max_concurrent_agents}")
    doc.append(f"- Processing Timeout: {config.processing.processing_timeout}s")
    doc.append(f"- Retry on Failure: {config.processing.retry_on_failure}")
    doc.append(f"- Preserve Intermediate Results: {config.processing.preserve_intermediate_results}")
    doc.append(f"- Batch Size: {config.processing.batch_size}")
    doc.append(f"- Enable Checkpointing: {config.processing.enable_checkpointing}\n")
    
    doc.append("## Improvement Criteria")
    doc.append(f"- Content Quality: {config.improvement_criteria.content_quality.weight} (threshold: {config.improvement_criteria.content_quality.threshold})")
    doc.append(f"- Clarity: {config.improvement_criteria.clarity.weight} (threshold: {config.improvement_criteria.clarity.threshold})")
    doc.append(f"- Structure: {config.improvement_criteria.structure.weight} (threshold: {config.improvement_criteria.structure.threshold})")
    doc.append(f"- Citations: {config.improvement_criteria.citations.weight} (threshold: {config.improvement_criteria.citations.threshold})")
    
    return "\n".join(doc)


def create_sample_agent_configs() -> Dict[str, AgentConfig]:
    """
    Create sample agent configurations.
    
    Returns:
        Dictionary of sample agent configurations
    """
    agents = {}
    
    # Ingestion Agent
    agents["ingestion_agent"] = create_agent_config(
        "ingestion_agent",
        timeout=600,
        quality_threshold=0.8,
        specialized_prompt="You are an academic content processor focused on accurately converting PDFs to markdown."
    )
    
    # Outline Agent
    agents["outline_agent"] = create_agent_config(
        "outline_agent",
        timeout=300,
        quality_threshold=0.7,
        specialized_prompt="You are an academic knowledge organizer tasked with creating comprehensive outlines."
    )
    
    # Notes Agent
    agents["notes_agent"] = create_agent_config(
        "notes_agent",
        timeout=900,
        quality_threshold=0.75,
        specialized_prompt="You are an academic notes enhancer responsible for expanding outlines into comprehensive notes."
    )
    
    # Quality Manager
    agents["quality_manager"] = create_agent_config(
        "quality_manager",
        timeout=300,
        quality_threshold=0.9,
        specialized_prompt="You are an academic quality evaluator with expertise in academic content assessment."
    )
    
    # Update Agent
    agents["update_agent"] = create_agent_config(
        "update_agent",
        timeout=600,
        quality_threshold=0.8,
        specialized_prompt="You are an academic content enhancer focused on improving notes while preserving original meaning."
    )
    
    return agents


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Academic Agent Configuration Utilities")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create sample configuration files")
    create_parser.add_argument("--config-dir", type=Path, default="config", help="Configuration directory")
    create_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration files")
    validate_parser.add_argument("--config-dir", type=Path, default="config", help="Configuration directory")
    validate_parser.add_argument("--environment", type=str, help="Environment to validate")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export configuration to different formats")
    export_parser.add_argument("--config-dir", type=Path, default="config", help="Configuration directory")
    export_parser.add_argument("--environment", type=str, default="development", help="Environment to export")
    export_parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    export_parser.add_argument("--output", type=Path, help="Output file path")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize configuration for environment")
    optimize_parser.add_argument("--config-dir", type=Path, default="config", help="Configuration directory")
    optimize_parser.add_argument("--environment", type=str, default="development", help="Environment to optimize")
    optimize_parser.add_argument("--output", type=Path, help="Output file path")
    
    # Document command
    doc_parser = subparsers.add_parser("document", help="Generate configuration documentation")
    doc_parser.add_argument("--config-dir", type=Path, default="config", help="Configuration directory")
    doc_parser.add_argument("--environment", type=str, default="development", help="Environment to document")
    doc_parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        config_manager = ConfigurationManager(args.config_dir)
        
        if args.command == "create":
            config_manager.create_sample_configs()
            logger.info("Sample configuration files created successfully")
        
        elif args.command == "validate":
            if args.environment:
                config = config_manager.load_config(args.environment)
                errors = validate_agent_ids(config)
                if errors:
                    logger.error(f"Validation errors: {errors}")
                    sys.exit(1)
                else:
                    logger.info("Configuration is valid")
            else:
                logger.info("Use --environment to specify which environment to validate")
        
        elif args.command == "export":
            config = config_manager.load_config(args.environment)
            
            if args.format == "json":
                output_path = args.output or Path(f"{args.environment}_config.json")
                export_config_to_json(config, output_path)
            else:
                output_path = args.output or Path(f"{args.environment}_config.yaml")
                config_manager.save_config(config, output_path.name)
        
        elif args.command == "optimize":
            config = config_manager.load_config(args.environment)
            optimized_config = optimize_configuration(config)
            
            if args.output:
                config_manager.save_config(optimized_config, args.output.name)
            else:
                logger.info("Optimized configuration (use --output to save):")
                print(yaml.dump(optimized_config.model_dump(), default_flow_style=False))
        
        elif args.command == "document":
            config = config_manager.load_config(args.environment)
            documentation = generate_config_documentation(config)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(documentation)
                logger.info(f"Documentation saved to: {args.output}")
            else:
                print(documentation)
        
        logger.info("Command completed successfully")
        
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()