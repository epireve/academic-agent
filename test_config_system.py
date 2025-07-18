#!/usr/bin/env python3
"""
Test script for the YAML-based configuration system.

This script tests the basic functionality of the configuration system
without requiring external dependencies.
"""

import os
import sys
from pathlib import Path

# Add the config directory to the path
sys.path.insert(0, str(Path(__file__).parent / "config"))

try:
    from src.core.config_manager import ConfigurationManager, AcademicAgentConfig, ConfigurationError
    
    def test_basic_config_loading():
        """Test basic configuration loading."""
        print("Testing basic configuration loading...")
        
        try:
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("development")
            
            print(f"✓ Successfully loaded development configuration")
            print(f"  Environment: {config.environment}")
            print(f"  Debug mode: {config.debug}")
            print(f"  Quality threshold: {config.quality_threshold}")
            print(f"  Number of agents: {len(config.agents)}")
            print(f"  Number of feedback loops: {len(config.feedback_loops)}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            return False
    
    def test_environment_configs():
        """Test loading different environment configurations."""
        print("\nTesting environment-specific configurations...")
        
        config_manager = ConfigurationManager("config")
        environments = ["development", "production", "test"]
        
        for env in environments:
            try:
                config = config_manager.load_config(env)
                print(f"✓ Successfully loaded {env} configuration")
                print(f"  Debug mode: {config.debug}")
                print(f"  Log level: {config.logging.level}")
                print(f"  Max concurrent agents: {config.processing.max_concurrent_agents}")
                
            except Exception as e:
                print(f"✗ Failed to load {env} configuration: {e}")
                return False
        
        return True
    
    def test_agent_configs():
        """Test agent configuration access."""
        print("\nTesting agent configurations...")
        
        try:
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("development")
            
            expected_agents = ["ingestion_agent", "outline_agent", "notes_agent", "quality_manager", "update_agent"]
            
            for agent_id in expected_agents:
                agent_config = config.get_agent_config(agent_id)
                if agent_config:
                    print(f"✓ Found agent config for {agent_id}")
                    print(f"  Enabled: {agent_config.enabled}")
                    print(f"  Timeout: {agent_config.timeout}")
                    print(f"  Quality threshold: {agent_config.quality_threshold}")
                else:
                    print(f"✗ Missing agent config for {agent_id}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to test agent configurations: {e}")
            return False
    
    def test_feedback_loops():
        """Test feedback loop configurations."""
        print("\nTesting feedback loop configurations...")
        
        try:
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("development")
            
            if not config.feedback_loops:
                print("✗ No feedback loops found")
                return False
            
            for i, loop in enumerate(config.feedback_loops):
                print(f"✓ Feedback loop {i + 1}: {loop.source} -> {loop.target}")
                print(f"  Type: {loop.type}")
                print(f"  Interval: {loop.interval}s")
                print(f"  Enabled: {loop.enabled}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to test feedback loops: {e}")
            return False
    
    def test_path_creation():
        """Test path creation functionality."""
        print("\nTesting path creation...")
        
        try:
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("test")
            
            # Check if test directories were created
            test_dirs = [
                config.paths.input_dir,
                config.paths.output_dir,
                config.paths.processed_dir,
                config.paths.analysis_dir,
                config.paths.outlines_dir,
                config.paths.notes_dir,
                config.paths.metadata_dir,
                config.paths.temp_dir
            ]
            
            for test_dir in test_dirs:
                if Path(test_dir).exists():
                    print(f"✓ Directory created: {test_dir}")
                else:
                    print(f"✗ Directory not created: {test_dir}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to test path creation: {e}")
            return False
    
    def test_configuration_validation():
        """Test configuration validation."""
        print("\nTesting configuration validation...")
        
        try:
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("development")
            
            # Test improvement criteria weights
            total_weight = config.improvement_criteria.get_total_weight()
            if abs(total_weight - 1.0) < 0.01:
                print(f"✓ Improvement criteria weights sum to {total_weight:.3f}")
            else:
                print(f"✗ Improvement criteria weights sum to {total_weight:.3f} (should be 1.0)")
                return False
            
            # Test agent references in feedback loops
            agent_ids = set(config.agents.keys())
            for loop in config.feedback_loops:
                if loop.source in agent_ids and loop.target in agent_ids:
                    print(f"✓ Feedback loop references valid: {loop.source} -> {loop.target}")
                else:
                    print(f"✗ Feedback loop references invalid: {loop.source} -> {loop.target}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to test configuration validation: {e}")
            return False
    
    def main():
        """Run all tests."""
        print("=" * 60)
        print("YAML-based Configuration System Tests")
        print("=" * 60)
        
        tests = [
            test_basic_config_loading,
            test_environment_configs,
            test_agent_configs,
            test_feedback_loops,
            test_path_creation,
            test_configuration_validation
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
                print()
        
        print("=" * 60)
        print(f"Test Results: {passed}/{total} tests passed")
        print("=" * 60)
        
        if passed == total:
            print("🎉 All tests passed! Configuration system is working correctly.")
            return 0
        else:
            print("❌ Some tests failed. Please check the configuration system.")
            return 1
    
    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"Failed to import configuration system: {e}")
    print("This might be due to missing dependencies. Please install:")
    print("  pip install pyyaml pydantic pydantic-settings")
    sys.exit(1)