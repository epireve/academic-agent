"""
Academic Agent v2 - Simplified Academic Agent System

This package provides a streamlined academic agent implementation that replaces
the complex smolagents architecture with a more maintainable and extensible system.
"""

from .academic_agent import (
    AcademicAgent,
    BasePlugin,
    ContentAnalysisPlugin,
    PDFProcessorPlugin,
    TaskPlanner,
    TaskResult,
    analyze_content,
    create_academic_agent,
    process_pdfs,
)
from .agent_config import (
    AgentConfig,
    AgentConfigManager,
    MonitoringConfig,
    PluginConfig,
    QualityConfig,
    TaskPlanningConfig,
    create_content_analysis_config,
    create_pdf_processing_config,
)
from .plugin_system import (
    AdvancedBasePlugin,
    PluginInterface,
    PluginLoader,
    PluginManager,
    PluginRegistry,
    create_plugin_manager,
)
from .state_manager import (
    AgentMetrics,
    AgentState,
    StateManager,
    TaskState,
    create_state_manager,
    create_task_state,
)

__version__ = "2.0.0"
__author__ = "Academic Agent Team"
__description__ = "Simplified Academic Agent System"

# Default exports for easy import
__all__ = [
    # Core academic agent
    "AcademicAgent",
    "create_academic_agent",
    "analyze_content",
    "process_pdfs",
    
    # Task and result classes
    "TaskResult",
    "TaskPlanner",
    
    # Plugin system
    "BasePlugin",
    "AdvancedBasePlugin",
    "PluginInterface",
    "PDFProcessorPlugin",
    "ContentAnalysisPlugin",
    "PluginManager",
    "PluginRegistry",
    "PluginLoader",
    "create_plugin_manager",
    
    # Configuration system
    "AgentConfig",
    "AgentConfigManager",
    "PluginConfig",
    "TaskPlanningConfig",
    "QualityConfig",
    "MonitoringConfig",
    "create_pdf_processing_config",
    "create_content_analysis_config",
    
    # State management
    "StateManager",
    "AgentState",
    "TaskState",
    "AgentMetrics",
    "create_state_manager",
    "create_task_state",
]

# System information
SYSTEM_INFO = {
    "name": "Academic Agent v2",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "components": {
        "academic_agent": "Core agent system with task execution and workflow management",
        "plugin_system": "Extensible plugin architecture for adding new capabilities",
        "state_manager": "Comprehensive state persistence and recovery system",
        "agent_config": "YAML-based configuration management with validation",
        "cli": "Command-line interface for agent operations",
    },
    "features": [
        "Simplified architecture replacing smolagents",
        "Plugin-based extensibility",
        "State persistence and recovery",
        "Quality management and assessment",
        "Configuration-driven behavior",
        "Comprehensive monitoring and metrics",
        "Error handling with recovery mechanisms",
        "Async operation support",
        "Integration with v2 core systems",
    ],
}


def get_system_info():
    """Get system information dictionary."""
    return SYSTEM_INFO.copy()


def print_system_info():
    """Print formatted system information."""
    info = get_system_info()
    
    print(f"\n{info['name']} - {info['description']}")
    print("=" * 60)
    print(f"Version: {info['version']}")
    print(f"Author: {info['author']}")
    
    print(f"\nComponents:")
    for component, description in info['components'].items():
        print(f"  • {component}: {description}")
    
    print(f"\nKey Features:")
    for feature in info['features']:
        print(f"  • {feature}")
    
    print("=" * 60)


# Quick start function
def quick_start(agent_id="academic_agent", config_path=None):
    """Quick start function to create a basic academic agent.
    
    Args:
        agent_id: Agent identifier
        config_path: Optional path to configuration file
        
    Returns:
        Configured AcademicAgent instance
    """
    try:
        # Create agent
        agent = create_academic_agent(
            config_path=config_path,
            agent_id=agent_id
        )
        
        print(f"Academic Agent '{agent_id}' created successfully!")
        print(f"Available plugins: {', '.join(agent.get_available_plugins())}")
        
        return agent
        
    except Exception as e:
        print(f"Failed to create academic agent: {e}")
        raise


# Version check function
def check_version():
    """Check and display version information."""
    import sys
    from pathlib import Path
    
    print(f"Academic Agent v2: {__version__}")
    print(f"Python version: {sys.version}")
    print(f"Installation path: {Path(__file__).parent}")
    
    # Check dependencies
    try:
        import yaml
        print(f"PyYAML: {yaml.__version__}")
    except ImportError:
        print("PyYAML: Not installed")
    
    try:
        import pydantic
        print(f"Pydantic: {pydantic.VERSION}")
    except ImportError:
        print("Pydantic: Not installed")
    
    try:
        import click
        print(f"Click: {click.__version__}")
    except ImportError:
        print("Click: Not installed")


if __name__ == "__main__":
    print_system_info()
    check_version()