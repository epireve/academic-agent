"""
Feature Flags System for Academic Agent Unified Architecture

This module provides a comprehensive feature flags system for controlling
behavior during the architecture transition from legacy to unified systems.
Supports thread-safe flag management, runtime toggling, hierarchical dependencies,
audit logging, and safe fallback mechanisms.
"""

import json
import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field

from .logging import get_logger
from .exceptions import AcademicAgentError


logger = get_logger("feature_flags")


class FlagState(Enum):
    """Feature flag states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLOUT = "rollout"  # Gradual rollout
    DEPRECATED = "deprecated"  # Scheduled for removal


class FlagScope(Enum):
    """Feature flag scopes."""
    GLOBAL = "global"
    AGENT = "agent"
    PROCESSING = "processing"
    SYSTEM = "system"


@dataclass
class FlagDependency:
    """Feature flag dependency definition."""
    flag_name: str
    required_state: FlagState
    relationship: str = "requires"  # requires, conflicts_with, depends_on


@dataclass
class FlagAuditEntry:
    """Audit entry for flag changes."""
    timestamp: datetime
    flag_name: str
    old_state: FlagState
    new_state: FlagState
    changed_by: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureFlag:
    """Feature flag definition."""
    name: str
    description: str
    state: FlagState
    scope: FlagScope
    default_value: bool = False
    rollout_percentage: float = 0.0
    dependencies: List[FlagDependency] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert flag to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "state": self.state.value,
            "scope": self.scope.value,
            "default_value": self.default_value,
            "rollout_percentage": self.rollout_percentage,
            "dependencies": [
                {
                    "flag_name": dep.flag_name,
                    "required_state": dep.required_state.value,
                    "relationship": dep.relationship
                }
                for dep in self.dependencies
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureFlag":
        """Create flag from dictionary."""
        dependencies = [
            FlagDependency(
                flag_name=dep["flag_name"],
                required_state=FlagState(dep["required_state"]),
                relationship=dep.get("relationship", "requires")
            )
            for dep in data.get("dependencies", [])
        ]
        
        return cls(
            name=data["name"],
            description=data["description"],
            state=FlagState(data["state"]),
            scope=FlagScope(data["scope"]),
            default_value=data.get("default_value", False),
            rollout_percentage=data.get("rollout_percentage", 0.0),
            dependencies=dependencies,
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            metadata=data.get("metadata", {})
        )


class FlagEvaluator(ABC):
    """Abstract base class for flag evaluation strategies."""
    
    @abstractmethod
    def evaluate(self, flag: FeatureFlag, context: Dict[str, Any]) -> bool:
        """Evaluate if flag should be enabled for given context."""
        pass


class SimpleEvaluator(FlagEvaluator):
    """Simple flag evaluator based on state."""
    
    def evaluate(self, flag: FeatureFlag, context: Dict[str, Any]) -> bool:
        """Evaluate flag based on state."""
        if flag.state == FlagState.ENABLED:
            return True
        elif flag.state == FlagState.DISABLED:
            return False
        elif flag.state == FlagState.ROLLOUT:
            # Use hash of context for consistent rollout
            import hashlib
            context_str = json.dumps(context, sort_keys=True)
            hash_value = int(hashlib.md5(context_str.encode()).hexdigest()[:8], 16)
            percentage = (hash_value % 100) / 100.0
            return percentage < flag.rollout_percentage
        elif flag.state == FlagState.DEPRECATED:
            # Deprecated flags default to disabled
            return flag.default_value
        else:
            return flag.default_value


class PercentageRolloutEvaluator(FlagEvaluator):
    """Evaluator for percentage-based rollouts."""
    
    def evaluate(self, flag: FeatureFlag, context: Dict[str, Any]) -> bool:
        """Evaluate flag using percentage rollout."""
        if flag.state == FlagState.ENABLED:
            return True
        elif flag.state == FlagState.DISABLED:
            return False
        
        # Use consistent hash for rollout
        import hashlib
        user_id = context.get("user_id", "default")
        agent_id = context.get("agent_id", "default")
        key = f"{flag.name}:{user_id}:{agent_id}"
        hash_value = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        percentage = (hash_value % 100) / 100.0
        
        return percentage < flag.rollout_percentage


class FeatureFlagManager:
    """Thread-safe feature flag manager with runtime toggling and dependencies."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize feature flag manager.
        
        Args:
            config_path: Path to feature flags configuration file
        """
        self.config_path = config_path or Path("config/feature_flags.json")
        self._flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()
        self._evaluators: Dict[str, FlagEvaluator] = {
            "simple": SimpleEvaluator(),
            "percentage": PercentageRolloutEvaluator(),
        }
        self._default_evaluator = "simple"
        self._audit_log: List[FlagAuditEntry] = []
        self._watchers: List[Callable[[str, bool, bool], None]] = []
        
        # Initialize default flags
        self._initialize_default_flags()
        
        # Load configuration
        self.load_configuration()
        
        # Load environment overrides
        self._load_environment_overrides()
    
    def _initialize_default_flags(self):
        """Initialize default feature flags for unified architecture."""
        default_flags = [
            FeatureFlag(
                name="USE_UNIFIED_AGENTS",
                description="Enable unified agent architecture",
                state=FlagState.DISABLED,
                scope=FlagScope.AGENT,
                default_value=False,
                metadata={"category": "architecture", "impact": "high"}
            ),
            FeatureFlag(
                name="USE_LEGACY_ADAPTER",
                description="Enable legacy compatibility mode",
                state=FlagState.ENABLED,
                scope=FlagScope.AGENT,
                default_value=True,
                dependencies=[
                    FlagDependency("USE_UNIFIED_AGENTS", FlagState.DISABLED, "conflicts_with")
                ],
                metadata={"category": "compatibility", "impact": "medium"}
            ),
            FeatureFlag(
                name="ENABLE_ASYNC_PROCESSING",
                description="Control async vs sync behavior",
                state=FlagState.DISABLED,
                scope=FlagScope.PROCESSING,
                default_value=False,
                dependencies=[
                    FlagDependency("USE_UNIFIED_AGENTS", FlagState.ENABLED, "requires")
                ],
                metadata={"category": "performance", "impact": "medium"}
            ),
            FeatureFlag(
                name="USE_UNIFIED_CONFIG",
                description="New vs old configuration system",
                state=FlagState.DISABLED,
                scope=FlagScope.SYSTEM,
                default_value=False,
                metadata={"category": "configuration", "impact": "medium"}
            ),
            FeatureFlag(
                name="ENABLE_PERFORMANCE_MONITORING",
                description="Enhanced monitoring features",
                state=FlagState.ENABLED,
                scope=FlagScope.SYSTEM,
                default_value=True,
                metadata={"category": "monitoring", "impact": "low"}
            ),
            FeatureFlag(
                name="USE_BATCH_PROCESSING",
                description="New batch processing capabilities",
                state=FlagState.DISABLED,
                scope=FlagScope.PROCESSING,
                default_value=False,
                dependencies=[
                    FlagDependency("USE_UNIFIED_AGENTS", FlagState.ENABLED, "requires")
                ],
                metadata={"category": "processing", "impact": "medium"}
            ),
            FeatureFlag(
                name="ENABLE_MEMORY_MANAGEMENT",
                description="Enhanced memory management features",
                state=FlagState.ENABLED,
                scope=FlagScope.SYSTEM,
                default_value=True,
                metadata={"category": "memory", "impact": "low"}
            ),
            FeatureFlag(
                name="USE_ENHANCED_LOGGING",
                description="Enhanced logging and audit features",
                state=FlagState.ENABLED,
                scope=FlagScope.SYSTEM,
                default_value=True,
                metadata={"category": "logging", "impact": "low"}
            ),
        ]
        
        with self._lock:
            for flag in default_flags:
                self._flags[flag.name] = flag
    
    def load_configuration(self) -> bool:
        """Load feature flags from configuration file.
        
        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        if not self.config_path.exists():
            logger.info(f"Feature flags config not found at {self.config_path}, using defaults")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            with self._lock:
                # Load flags from config
                flags_data = config_data.get("flags", {})
                for flag_name, flag_data in flags_data.items():
                    try:
                        flag = FeatureFlag.from_dict(flag_data)
                        self._flags[flag_name] = flag
                    except Exception as e:
                        logger.error(f"Failed to load flag {flag_name}: {e}")
                
                # Load evaluator settings
                evaluator_config = config_data.get("evaluator", {})
                self._default_evaluator = evaluator_config.get("default", "simple")
            
            logger.info(f"Loaded {len(flags_data)} feature flags from configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load feature flags configuration: {e}")
            return False
    
    def save_configuration(self) -> bool:
        """Save current feature flags to configuration file.
        
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self._lock:
                config_data = {
                    "flags": {
                        name: flag.to_dict()
                        for name, flag in self._flags.items()
                    },
                    "evaluator": {
                        "default": self._default_evaluator
                    },
                    "saved_at": datetime.now().isoformat()
                }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved feature flags configuration to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save feature flags configuration: {e}")
            return False
    
    def _load_environment_overrides(self):
        """Load feature flag overrides from environment variables."""
        with self._lock:
            for flag_name in self._flags.keys():
                env_var = f"ACADEMIC_AGENT_FLAG_{flag_name}"
                env_value = os.getenv(env_var)
                
                if env_value is not None:
                    try:
                        # Parse environment value
                        if env_value.lower() in ("true", "1", "enabled"):
                            self.set_flag_state(flag_name, FlagState.ENABLED, changed_by="environment")
                        elif env_value.lower() in ("false", "0", "disabled"):
                            self.set_flag_state(flag_name, FlagState.DISABLED, changed_by="environment")
                        elif env_value.startswith("rollout:"):
                            percentage = float(env_value.split(":")[1])
                            self.set_flag_rollout(flag_name, percentage, changed_by="environment")
                        else:
                            logger.warning(f"Invalid environment value for {env_var}: {env_value}")
                    except Exception as e:
                        logger.error(f"Failed to parse environment override for {flag_name}: {e}")
    
    def is_enabled(self, flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled.
        
        Args:
            flag_name: Name of the feature flag
            context: Optional context for evaluation
            
        Returns:
            True if flag is enabled, False otherwise
        """
        context = context or {}
        
        with self._lock:
            if flag_name not in self._flags:
                logger.warning(f"Unknown feature flag: {flag_name}")
                return False
            
            flag = self._flags[flag_name]
            
            # Check dependencies first
            if not self._check_dependencies(flag):
                logger.debug(f"Flag {flag_name} disabled due to dependency constraints")
                return False
            
            # Evaluate flag
            evaluator = self._evaluators.get(self._default_evaluator, self._evaluators["simple"])
            result = evaluator.evaluate(flag, context)
            
            logger.debug(f"Flag {flag_name} evaluated to {result}")
            return result
    
    def _check_dependencies(self, flag: FeatureFlag) -> bool:
        """Check if flag dependencies are satisfied.
        
        Args:
            flag: Feature flag to check
            
        Returns:
            True if all dependencies are satisfied, False otherwise
        """
        for dependency in flag.dependencies:
            if dependency.flag_name not in self._flags:
                logger.warning(f"Dependency flag not found: {dependency.flag_name}")
                continue
            
            dep_flag = self._flags[dependency.flag_name]
            
            if dependency.relationship == "requires":
                if dep_flag.state != dependency.required_state:
                    return False
            elif dependency.relationship == "conflicts_with":
                if dep_flag.state == dependency.required_state:
                    return False
        
        return True
    
    def set_flag_state(
        self,
        flag_name: str,
        state: FlagState,
        changed_by: str = "system",
        reason: str = ""
    ) -> bool:
        """Set feature flag state.
        
        Args:
            flag_name: Name of the feature flag
            state: New state for the flag
            changed_by: Who changed the flag
            reason: Reason for the change
            
        Returns:
            True if state was changed successfully, False otherwise
        """
        with self._lock:
            if flag_name not in self._flags:
                logger.error(f"Cannot set state for unknown flag: {flag_name}")
                return False
            
            flag = self._flags[flag_name]
            old_state = flag.state
            
            # Validate state change
            if not self._validate_state_change(flag, state):
                logger.error(f"Invalid state change for flag {flag_name}: {old_state} -> {state}")
                return False
            
            # Update flag
            flag.state = state
            flag.updated_at = datetime.now()
            
            # Log audit entry
            audit_entry = FlagAuditEntry(
                timestamp=datetime.now(),
                flag_name=flag_name,
                old_state=old_state,
                new_state=state,
                changed_by=changed_by,
                reason=reason
            )
            self._audit_log.append(audit_entry)
            
            # Notify watchers
            old_enabled = old_state == FlagState.ENABLED
            new_enabled = state == FlagState.ENABLED
            if old_enabled != new_enabled:
                self._notify_watchers(flag_name, old_enabled, new_enabled)
            
            logger.info(f"Flag {flag_name} state changed: {old_state.value} -> {state.value}")
            return True
    
    def set_flag_rollout(
        self,
        flag_name: str,
        percentage: float,
        changed_by: str = "system",
        reason: str = ""
    ) -> bool:
        """Set feature flag rollout percentage.
        
        Args:
            flag_name: Name of the feature flag
            percentage: Rollout percentage (0.0 to 1.0)
            changed_by: Who changed the flag
            reason: Reason for the change
            
        Returns:
            True if rollout was set successfully, False otherwise
        """
        if not 0.0 <= percentage <= 1.0:
            logger.error(f"Invalid rollout percentage: {percentage}")
            return False
        
        with self._lock:
            if flag_name not in self._flags:
                logger.error(f"Cannot set rollout for unknown flag: {flag_name}")
                return False
            
            flag = self._flags[flag_name]
            old_percentage = flag.rollout_percentage
            
            # Update flag
            flag.rollout_percentage = percentage
            flag.state = FlagState.ROLLOUT
            flag.updated_at = datetime.now()
            
            # Log audit entry
            audit_entry = FlagAuditEntry(
                timestamp=datetime.now(),
                flag_name=flag_name,
                old_state=flag.state,
                new_state=FlagState.ROLLOUT,
                changed_by=changed_by,
                reason=reason,
                metadata={"old_percentage": old_percentage, "new_percentage": percentage}
            )
            self._audit_log.append(audit_entry)
            
            logger.info(f"Flag {flag_name} rollout set to {percentage:.1%}")
            return True
    
    def _validate_state_change(self, flag: FeatureFlag, new_state: FlagState) -> bool:
        """Validate that a state change is allowed.
        
        Args:
            flag: Feature flag
            new_state: Proposed new state
            
        Returns:
            True if state change is valid, False otherwise
        """
        # Basic validation - could be extended with more complex rules
        if flag.state == FlagState.DEPRECATED and new_state != FlagState.DISABLED:
            return False
        
        return True
    
    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get feature flag definition.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            FeatureFlag instance or None if not found
        """
        with self._lock:
            return self._flags.get(flag_name)
    
    def list_flags(self, scope: Optional[FlagScope] = None) -> Dict[str, FeatureFlag]:
        """List all feature flags, optionally filtered by scope.
        
        Args:
            scope: Optional scope filter
            
        Returns:
            Dictionary of flag name to FeatureFlag
        """
        with self._lock:
            if scope is None:
                return self._flags.copy()
            else:
                return {
                    name: flag for name, flag in self._flags.items()
                    if flag.scope == scope
                }
    
    def add_watcher(self, callback: Callable[[str, bool, bool], None]):
        """Add a callback to be notified when flags change.
        
        Args:
            callback: Function(flag_name, old_enabled, new_enabled)
        """
        with self._lock:
            self._watchers.append(callback)
    
    def remove_watcher(self, callback: Callable[[str, bool, bool], None]):
        """Remove a flag change watcher.
        
        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._watchers:
                self._watchers.remove(callback)
    
    def _notify_watchers(self, flag_name: str, old_enabled: bool, new_enabled: bool):
        """Notify watchers of flag changes.
        
        Args:
            flag_name: Name of the changed flag
            old_enabled: Previous enabled state
            new_enabled: New enabled state
        """
        for watcher in self._watchers:
            try:
                watcher(flag_name, old_enabled, new_enabled)
            except Exception as e:
                logger.error(f"Error in flag watcher: {e}")
    
    def get_audit_log(self, flag_name: Optional[str] = None) -> List[FlagAuditEntry]:
        """Get audit log entries.
        
        Args:
            flag_name: Optional flag name filter
            
        Returns:
            List of audit entries
        """
        with self._lock:
            if flag_name is None:
                return self._audit_log.copy()
            else:
                return [entry for entry in self._audit_log if entry.flag_name == flag_name]
    
    def get_status(self) -> Dict[str, Any]:
        """Get feature flags system status.
        
        Returns:
            Status information dictionary
        """
        with self._lock:
            enabled_flags = sum(1 for flag in self._flags.values() if flag.state == FlagState.ENABLED)
            rollout_flags = sum(1 for flag in self._flags.values() if flag.state == FlagState.ROLLOUT)
            deprecated_flags = sum(1 for flag in self._flags.values() if flag.state == FlagState.DEPRECATED)
            
            return {
                "total_flags": len(self._flags),
                "enabled_flags": enabled_flags,
                "rollout_flags": rollout_flags,
                "deprecated_flags": deprecated_flags,
                "audit_entries": len(self._audit_log),
                "watchers": len(self._watchers),
                "evaluator": self._default_evaluator,
                "config_path": str(self.config_path),
                "flags_by_scope": {
                    scope.value: sum(1 for flag in self._flags.values() if flag.scope == scope)
                    for scope in FlagScope
                }
            }


# Global feature flag manager instance
_flag_manager: Optional[FeatureFlagManager] = None


def get_flag_manager(config_path: Optional[Path] = None) -> FeatureFlagManager:
    """Get the global feature flag manager instance.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        FeatureFlagManager instance
    """
    global _flag_manager
    
    if _flag_manager is None:
        _flag_manager = FeatureFlagManager(config_path)
    
    return _flag_manager


def is_flag_enabled(flag_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if a feature flag is enabled.
    
    Args:
        flag_name: Name of the feature flag
        context: Optional context for evaluation
        
    Returns:
        True if flag is enabled, False otherwise
    """
    return get_flag_manager().is_enabled(flag_name, context)


def with_feature_flag(flag_name: str, context: Optional[Dict[str, Any]] = None):
    """Decorator to conditionally execute function based on feature flag.
    
    Args:
        flag_name: Name of the feature flag
        context: Optional context for evaluation
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if is_flag_enabled(flag_name, context):
                return func(*args, **kwargs)
            else:
                logger.debug(f"Function {func.__name__} skipped due to disabled flag: {flag_name}")
                return None
        return wrapper
    return decorator


class FeatureFlagContext:
    """Context manager for feature flag evaluation."""
    
    def __init__(self, **context):
        """Initialize context.
        
        Args:
            **context: Context key-value pairs
        """
        self.context = context
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if flag is enabled in this context.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            True if flag is enabled, False otherwise
        """
        return is_flag_enabled(flag_name, self.context)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass