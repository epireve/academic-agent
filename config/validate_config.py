#!/usr/bin/env python3
"""
Configuration validation script for academic-agent.

This script validates YAML configuration files against the Pydantic models
and provides detailed error reporting.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import yaml
from pydantic import ValidationError

from config_manager import ConfigurationManager, AcademicAgentConfig


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_single_config(config_path: Path) -> List[str]:
    """
    Validate a single configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        # Load YAML file
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        # Validate against Pydantic model
        AcademicAgentConfig(**config_data)
        
        logger.info(f"✓ Configuration file is valid: {config_path}")
        
    except FileNotFoundError:
        errors.append(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        errors.append(f"YAML parsing error in {config_path}: {str(e)}")
    except ValidationError as e:
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"Validation error in {config_path} at {field}: {error['msg']}")
    except Exception as e:
        errors.append(f"Unexpected error validating {config_path}: {str(e)}")
    
    return errors


def validate_environment_config(config_manager: ConfigurationManager, environment: str) -> List[str]:
    """
    Validate a complete environment configuration (base + environment-specific).
    
    Args:
        config_manager: Configuration manager instance
        environment: Environment name
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    try:
        # Try to load the complete configuration
        config = config_manager.load_config(environment)
        logger.info(f"✓ Environment configuration is valid: {environment}")
        
        # Additional validation checks
        validation_errors = validate_configuration_consistency(config)
        if validation_errors:
            errors.extend(validation_errors)
        
    except Exception as e:
        errors.append(f"Failed to load environment configuration '{environment}': {str(e)}")
    
    return errors


def validate_configuration_consistency(config: AcademicAgentConfig) -> List[str]:
    """
    Validate configuration consistency and business rules.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check improvement criteria weights
    if not config.improvement_criteria.validate_weights():
        total_weight = config.improvement_criteria.get_total_weight()
        errors.append(f"Improvement criteria weights sum to {total_weight:.3f}, should be 1.0")
    
    # Check feedback loop references
    agent_ids = set(config.agents.keys())
    for i, loop in enumerate(config.feedback_loops):
        if loop.source not in agent_ids:
            errors.append(f"Feedback loop {i} references unknown source agent: {loop.source}")
        if loop.target not in agent_ids:
            errors.append(f"Feedback loop {i} references unknown target agent: {loop.target}")
    
    # Check for circular dependencies in feedback loops
    circular_deps = find_circular_dependencies(config.feedback_loops)
    if circular_deps:
        errors.append(f"Circular dependencies detected in feedback loops: {circular_deps}")
    
    # Check timeout values are reasonable
    if config.processing.processing_timeout < config.inter_agent_communication.message_timeout:
        errors.append("Processing timeout should be greater than message timeout")
    
    # Check quality thresholds are reasonable
    if config.quality_threshold > 0.95:
        errors.append("Quality threshold above 0.95 may be too strict")
    
    # Check concurrent agents vs available resources
    if config.processing.max_concurrent_agents > 8:
        errors.append("Max concurrent agents above 8 may cause resource issues")
    
    return errors


def find_circular_dependencies(feedback_loops: List[Any]) -> List[str]:
    """
    Find circular dependencies in feedback loops.
    
    Args:
        feedback_loops: List of feedback loop configurations
        
    Returns:
        List of circular dependency descriptions
    """
    # Build dependency graph
    graph = {}
    for loop in feedback_loops:
        if loop.source not in graph:
            graph[loop.source] = []
        graph[loop.source].append(loop.target)
    
    # Find cycles using DFS
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node, path):
        if node in rec_stack:
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(" -> ".join(cycle))
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            dfs(neighbor, path + [node])
        
        rec_stack.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node, [])
    
    return cycles


def validate_yaml_syntax(config_path: Path) -> List[str]:
    """
    Validate YAML syntax of a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        List of syntax errors (empty if valid)
    """
    errors = []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        logger.info(f"✓ YAML syntax is valid: {config_path}")
    except FileNotFoundError:
        errors.append(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        errors.append(f"YAML syntax error in {config_path}: {str(e)}")
    except Exception as e:
        errors.append(f"Unexpected error reading {config_path}: {str(e)}")
    
    return errors


def generate_validation_report(errors: Dict[str, List[str]]) -> str:
    """
    Generate a formatted validation report.
    
    Args:
        errors: Dictionary of validation errors by category
        
    Returns:
        Formatted validation report
    """
    report = []
    report.append("=== CONFIGURATION VALIDATION REPORT ===\n")
    
    total_errors = sum(len(error_list) for error_list in errors.values())
    
    if total_errors == 0:
        report.append("✅ All configurations are valid!\n")
    else:
        report.append(f"❌ Found {total_errors} validation errors:\n")
        
        for category, error_list in errors.items():
            if error_list:
                report.append(f"📁 {category}:")
                for error in error_list:
                    report.append(f"  • {error}")
                report.append("")
    
    return "\n".join(report)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate academic-agent configuration files")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default="config",
        help="Configuration directory (default: config)"
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "test"],
        help="Specific environment to validate"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Specific configuration file to validate"
    )
    parser.add_argument(
        "--syntax-only",
        action="store_true",
        help="Only validate YAML syntax"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    errors = {}
    
    try:
        # Validate specific file
        if args.file:
            if args.syntax_only:
                errors["YAML Syntax"] = validate_yaml_syntax(args.file)
            else:
                errors["Single File"] = validate_single_config(args.file)
        
        # Validate specific environment
        elif args.environment:
            config_manager = ConfigurationManager(args.config_dir)
            errors[f"Environment: {args.environment}"] = validate_environment_config(config_manager, args.environment)
        
        # Validate all configurations
        else:
            config_manager = ConfigurationManager(args.config_dir)
            
            # Validate base configuration
            base_config_path = args.config_dir / "base.yaml"
            if args.syntax_only:
                errors["Base Config (Syntax)"] = validate_yaml_syntax(base_config_path)
            else:
                errors["Base Config"] = validate_single_config(base_config_path)
            
            # Validate environment configurations
            environments = ["development", "production", "test"]
            for env in environments:
                env_config_path = args.config_dir / f"{env}.yaml"
                if env_config_path.exists():
                    if args.syntax_only:
                        errors[f"{env.title()} Config (Syntax)"] = validate_yaml_syntax(env_config_path)
                    else:
                        errors[f"{env.title()} Environment"] = validate_environment_config(config_manager, env)
        
        # Generate and display report
        report = generate_validation_report(errors)
        print(report)
        
        # Exit with error code if validation failed
        total_errors = sum(len(error_list) for error_list in errors.values())
        if total_errors > 0:
            sys.exit(1)
        else:
            logger.info("All validations passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()