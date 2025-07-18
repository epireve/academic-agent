"""
Simple monitoring module for unified architecture.
Provides basic monitoring functionality without complex dependencies.
"""

from typing import Any, Dict, Optional


class SimpleMonitor:
    """Simple monitoring implementation."""
    
    def __init__(self):
        self.metrics_collector = self
    
    def collect_metric(self, metric: Dict[str, Any]):
        """Collect a metric (no-op for now)."""
        pass
    
    def record_metric(self, name: str, data: Dict[str, Any]):
        """Record a metric (no-op for now)."""
        pass


# Global monitor instance
_monitor: Optional[SimpleMonitor] = None


def get_system_monitor() -> SimpleMonitor:
    """Get the global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = SimpleMonitor()
    return _monitor