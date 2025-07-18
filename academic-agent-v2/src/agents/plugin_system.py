"""
Plugin System for Academic Agent v2.

This module provides a flexible plugin system that allows for easy extension
of academic agent capabilities with dynamic loading, validation, and management.
"""

import asyncio
import importlib
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..core.exceptions import PluginError, ValidationError
from ..core.logging import get_logger
from .agent_config import PluginConfig
from .academic_agent import BasePlugin, TaskResult

logger = get_logger(__name__)


class PluginInterface(ABC):
    """Abstract interface that all plugins must implement."""
    
    @abstractmethod
    def get_plugin_name(self) -> str:
        """Get the plugin name."""
        pass
    
    @abstractmethod
    def get_plugin_version(self) -> str:
        """Get the plugin version."""
        pass
    
    @abstractmethod
    def get_supported_task_types(self) -> List[str]:
        """Get list of supported task types."""
        pass
    
    @abstractmethod
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute plugin task."""
        pass
    
    @abstractmethod
    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        pass
    
    def get_plugin_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.get_plugin_name(),
            "version": self.get_plugin_version(),
            "supported_tasks": self.get_supported_task_types(),
            "description": getattr(self, "__doc__", "No description available"),
        }


class AdvancedBasePlugin(BasePlugin, PluginInterface):
    """Enhanced base plugin with additional functionality."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize advanced base plugin."""
        super().__init__(config)
        self.plugin_name = self.__class__.__name__
        self.plugin_version = "1.0.0"
        self.supported_task_types = []
        
        # Plugin state management
        self.state = {}
        self.metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_processing_time": 0.0,
        }
    
    def get_plugin_name(self) -> str:
        """Get the plugin name."""
        return self.plugin_name
    
    def get_plugin_version(self) -> str:
        """Get the plugin version."""
        return self.plugin_version
    
    def get_supported_task_types(self) -> List[str]:
        """Get list of supported task types."""
        return self.supported_task_types
    
    async def execute(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute plugin task with metrics tracking."""
        start_time = asyncio.get_event_loop().time()
        task_id = task_data.get("task_id", "unknown")
        
        try:
            self.metrics["executions"] += 1
            
            # Validate input
            if not self.validate_input(task_data):
                raise ValidationError("Invalid input data for plugin")
            
            # Execute plugin-specific logic
            result = await self._execute_plugin_logic(task_data)
            
            # Update metrics
            self.metrics["successes"] += 1
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics["total_processing_time"] += processing_time
            
            result.processing_time = processing_time
            return result
            
        except Exception as e:
            self.metrics["failures"] += 1
            processing_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.error(f"Plugin {self.plugin_name} execution failed: {str(e)}")
            
            return TaskResult(
                task_id=task_id,
                task_type=task_data.get("task_type", "unknown"),
                success=False,
                error_message=str(e),
                processing_time=processing_time,
            )
    
    @abstractmethod
    async def _execute_plugin_logic(self, task_data: Dict[str, Any]) -> TaskResult:
        """Execute plugin-specific logic - to be implemented by subclasses."""
        pass
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get comprehensive plugin information."""
        base_info = super().get_plugin_info()
        base_info.update({
            "metadata": self.get_plugin_metadata(),
            "metrics": self.metrics,
            "state": self.state,
        })
        return base_info
    
    def reset_metrics(self):
        """Reset plugin metrics."""
        self.metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "total_processing_time": 0.0,
        }
        self.logger.info(f"Reset metrics for plugin {self.plugin_name}")


class PluginRegistry:
    """Registry for managing available plugins."""
    
    def __init__(self):
        """Initialize plugin registry."""
        self.logger = get_logger(__name__)
        self._plugins: Dict[str, Type[PluginInterface]] = {}
        self._plugin_metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded_modules: Dict[str, Any] = {}
    
    def register_plugin(self, plugin_class: Type[PluginInterface], force: bool = False):
        """Register a plugin class.
        
        Args:
            plugin_class: Plugin class to register
            force: Whether to force registration if plugin already exists
        """
        if not issubclass(plugin_class, PluginInterface):
            raise PluginError(f"Plugin {plugin_class.__name__} must implement PluginInterface")
        
        # Get plugin name from class
        plugin_name = getattr(plugin_class, "plugin_name", plugin_class.__name__)
        
        if plugin_name in self._plugins and not force:
            raise PluginError(f"Plugin {plugin_name} is already registered")
        
        self._plugins[plugin_name] = plugin_class
        
        # Store metadata
        try:
            # Create temporary instance to get metadata
            temp_instance = plugin_class()
            self._plugin_metadata[plugin_name] = temp_instance.get_plugin_metadata()
        except Exception as e:
            self.logger.warning(f"Could not get metadata for plugin {plugin_name}: {e}")
            self._plugin_metadata[plugin_name] = {
                "name": plugin_name,
                "version": "unknown",
                "supported_tasks": [],
                "description": "Metadata unavailable",
            }
        
        self.logger.info(f"Registered plugin: {plugin_name}")
    
    def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            del self._plugin_metadata[plugin_name]
            self.logger.info(f"Unregistered plugin: {plugin_name}")
        else:
            self.logger.warning(f"Plugin {plugin_name} not found in registry")
    
    def get_plugin_class(self, plugin_name: str) -> Optional[Type[PluginInterface]]:
        """Get plugin class by name.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin class or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def list_plugins(self) -> List[str]:
        """Get list of registered plugin names."""
        return list(self._plugins.keys())
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get plugin metadata.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin metadata or None if not found
        """
        return self._plugin_metadata.get(plugin_name)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all registered plugins."""
        return self._plugin_metadata.copy()
    
    def find_plugins_by_task_type(self, task_type: str) -> List[str]:
        """Find plugins that support a specific task type.
        
        Args:
            task_type: Task type to search for
            
        Returns:
            List of plugin names that support the task type
        """
        matching_plugins = []
        
        for plugin_name, metadata in self._plugin_metadata.items():
            supported_tasks = metadata.get("supported_tasks", [])
            if task_type in supported_tasks:
                matching_plugins.append(plugin_name)
        
        return matching_plugins
    
    def validate_plugin(self, plugin_class: Type[PluginInterface]) -> List[str]:
        """Validate a plugin class.
        
        Args:
            plugin_class: Plugin class to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check if class implements PluginInterface
        if not issubclass(plugin_class, PluginInterface):
            issues.append("Plugin must implement PluginInterface")
            return issues
        
        # Check required methods
        required_methods = [
            "get_plugin_name",
            "get_plugin_version",
            "get_supported_task_types",
            "execute",
            "validate_input",
        ]
        
        for method_name in required_methods:
            if not hasattr(plugin_class, method_name):
                issues.append(f"Plugin missing required method: {method_name}")
            else:
                method = getattr(plugin_class, method_name)
                if not callable(method):
                    issues.append(f"Plugin method {method_name} is not callable")
        
        # Check if execute method is async
        if hasattr(plugin_class, "execute"):
            execute_method = getattr(plugin_class, "execute")
            if not asyncio.iscoroutinefunction(execute_method):
                issues.append("Plugin execute method must be async")
        
        return issues


class PluginLoader:
    """Loads plugins from various sources."""
    
    def __init__(self, registry: PluginRegistry):
        """Initialize plugin loader.
        
        Args:
            registry: Plugin registry to load plugins into
        """
        self.registry = registry
        self.logger = get_logger(__name__)
    
    def load_plugin_from_file(self, plugin_file: Path) -> List[str]:
        """Load plugins from a Python file.
        
        Args:
            plugin_file: Path to Python file containing plugins
            
        Returns:
            List of loaded plugin names
        """
        if not plugin_file.exists():
            raise PluginError(f"Plugin file not found: {plugin_file}")
        
        if plugin_file.suffix != ".py":
            raise PluginError(f"Plugin file must be a Python file: {plugin_file}")
        
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in module
            loaded_plugins = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj != PluginInterface and 
                    issubclass(obj, PluginInterface) and 
                    obj.__module__ == module.__name__):
                    
                    # Validate plugin
                    issues = self.registry.validate_plugin(obj)
                    if issues:
                        self.logger.error(f"Plugin {name} validation failed: {issues}")
                        continue
                    
                    # Register plugin
                    self.registry.register_plugin(obj)
                    loaded_plugins.append(name)
            
            self.logger.info(f"Loaded {len(loaded_plugins)} plugins from {plugin_file}")
            return loaded_plugins
            
        except Exception as e:
            raise PluginError(f"Failed to load plugin from {plugin_file}: {e}")
    
    def load_plugins_from_directory(self, plugin_dir: Path, recursive: bool = True) -> List[str]:
        """Load plugins from a directory.
        
        Args:
            plugin_dir: Directory containing plugin files
            recursive: Whether to search subdirectories
            
        Returns:
            List of loaded plugin names
        """
        if not plugin_dir.exists():
            raise PluginError(f"Plugin directory not found: {plugin_dir}")
        
        loaded_plugins = []
        pattern = "**/*.py" if recursive else "*.py"
        
        for plugin_file in plugin_dir.glob(pattern):
            if plugin_file.name.startswith("__"):
                continue  # Skip __init__.py and similar files
            
            try:
                plugins = self.load_plugin_from_file(plugin_file)
                loaded_plugins.extend(plugins)
            except Exception as e:
                self.logger.error(f"Failed to load plugin from {plugin_file}: {e}")
        
        self.logger.info(f"Loaded {len(loaded_plugins)} plugins from {plugin_dir}")
        return loaded_plugins
    
    def load_builtin_plugins(self) -> List[str]:
        """Load built-in plugins."""
        from .academic_agent import PDFProcessorPlugin, ContentAnalysisPlugin
        
        builtin_plugins = [PDFProcessorPlugin, ContentAnalysisPlugin]
        loaded_plugins = []
        
        for plugin_class in builtin_plugins:
            try:
                # Validate plugin
                issues = self.registry.validate_plugin(plugin_class)
                if issues:
                    self.logger.error(f"Built-in plugin {plugin_class.__name__} validation failed: {issues}")
                    continue
                
                # Register plugin
                self.registry.register_plugin(plugin_class)
                loaded_plugins.append(plugin_class.__name__)
                
            except Exception as e:
                self.logger.error(f"Failed to load built-in plugin {plugin_class.__name__}: {e}")
        
        self.logger.info(f"Loaded {len(loaded_plugins)} built-in plugins")
        return loaded_plugins


class PluginManager:
    """Manages plugin lifecycle and instances."""
    
    def __init__(self, registry: Optional[PluginRegistry] = None):
        """Initialize plugin manager.
        
        Args:
            registry: Plugin registry (creates new one if None)
        """
        self.registry = registry or PluginRegistry()
        self.loader = PluginLoader(self.registry)
        self.logger = get_logger(__name__)
        
        # Plugin instances
        self._plugin_instances: Dict[str, PluginInterface] = {}
        self._plugin_configs: Dict[str, PluginConfig] = {}
    
    def initialize_plugin(self, plugin_name: str, config: PluginConfig) -> bool:
        """Initialize a plugin instance.
        
        Args:
            plugin_name: Name of plugin to initialize
            config: Plugin configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            plugin_class = self.registry.get_plugin_class(plugin_name)
            if not plugin_class:
                raise PluginError(f"Plugin {plugin_name} not found in registry")
            
            # Create plugin instance
            plugin_instance = plugin_class(config.config)
            
            # Store instance and config
            self._plugin_instances[plugin_name] = plugin_instance
            self._plugin_configs[plugin_name] = config
            
            self.logger.info(f"Initialized plugin: {plugin_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            return False
    
    def shutdown_plugin(self, plugin_name: str):
        """Shutdown a plugin instance.
        
        Args:
            plugin_name: Name of plugin to shutdown
        """
        if plugin_name in self._plugin_instances:
            # Call shutdown method if available
            plugin = self._plugin_instances[plugin_name]
            if hasattr(plugin, "shutdown"):
                try:
                    plugin.shutdown()
                except Exception as e:
                    self.logger.error(f"Error during plugin {plugin_name} shutdown: {e}")
            
            # Remove from instances
            del self._plugin_instances[plugin_name]
            if plugin_name in self._plugin_configs:
                del self._plugin_configs[plugin_name]
            
            self.logger.info(f"Shutdown plugin: {plugin_name}")
    
    def get_plugin_instance(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get plugin instance.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugin_instances.get(plugin_name)
    
    def list_active_plugins(self) -> List[str]:
        """Get list of active plugin names."""
        return list(self._plugin_instances.keys())
    
    def get_plugin_status(self, plugin_name: str) -> Dict[str, Any]:
        """Get plugin status information.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin status information
        """
        if plugin_name not in self._plugin_instances:
            return {"status": "not_initialized"}
        
        plugin = self._plugin_instances[plugin_name]
        config = self._plugin_configs.get(plugin_name)
        
        status = {
            "status": "active",
            "enabled": config.enabled if config else True,
            "plugin_info": plugin.get_plugin_info(),
        }
        
        return status
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin.
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if successful, False otherwise
        """
        if plugin_name not in self._plugin_instances:
            self.logger.warning(f"Plugin {plugin_name} not active, cannot reload")
            return False
        
        # Get current config
        config = self._plugin_configs.get(plugin_name)
        if not config:
            self.logger.error(f"No config found for plugin {plugin_name}")
            return False
        
        # Shutdown current instance
        self.shutdown_plugin(plugin_name)
        
        # Reinitialize
        return self.initialize_plugin(plugin_name, config)
    
    def shutdown_all_plugins(self):
        """Shutdown all active plugins."""
        plugin_names = list(self._plugin_instances.keys())
        for plugin_name in plugin_names:
            self.shutdown_plugin(plugin_name)
        
        self.logger.info("All plugins shut down")


# Factory function for creating plugin managers

def create_plugin_manager(
    plugin_dirs: Optional[List[Path]] = None,
    load_builtin: bool = True
) -> PluginManager:
    """Create and initialize a plugin manager.
    
    Args:
        plugin_dirs: Optional list of directories to load plugins from
        load_builtin: Whether to load built-in plugins
        
    Returns:
        Initialized PluginManager
    """
    registry = PluginRegistry()
    manager = PluginManager(registry)
    
    # Load built-in plugins
    if load_builtin:
        manager.loader.load_builtin_plugins()
    
    # Load plugins from directories
    if plugin_dirs:
        for plugin_dir in plugin_dirs:
            if plugin_dir.exists():
                manager.loader.load_plugins_from_directory(plugin_dir)
    
    return manager