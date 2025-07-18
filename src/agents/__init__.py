"""Agents for Academic Agent."""

from .base_agent import BaseAgent, AgentCapability, TaskResult, AgentMessage
from .legacy_adapter import LegacyAgentAdapter, create_legacy_adapter

__all__ = [
    "BaseAgent",
    "AgentCapability", 
    "TaskResult",
    "AgentMessage",
    "LegacyAgentAdapter",
    "create_legacy_adapter",
]