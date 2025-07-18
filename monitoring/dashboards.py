"""Dashboard generation and visualization for Academic Agent monitoring."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from academic_agent_v2.src.core.logging import get_logger


@dataclass
class DashboardPanel:
    """Configuration for a dashboard panel."""
    
    title: str
    panel_type: str  # graph, stat, table, heatmap, etc.
    targets: List[Dict[str, str]]  # Prometheus queries
    grid_pos: Dict[str, int]  # x, y, w, h
    options: Dict[str, Any] = None
    field_config: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to Grafana JSON format."""
        panel = {
            "id": hash(self.title) % 10000,  # Simple ID generation
            "title": self.title,
            "type": self.panel_type,
            "targets": self.targets,
            "gridPos": self.grid_pos,
            "datasource": {
                "type": "prometheus",
                "uid": "${DS_PROMETHEUS}"
            }
        }
        
        if self.options:
            panel["options"] = self.options
        
        if self.field_config:
            panel["fieldConfig"] = self.field_config
        
        return panel


class GrafanaDashboardGenerator:
    """Generates Grafana dashboards for Academic Agent monitoring."""
    
    def __init__(self):
        self.logger = get_logger("dashboard_generator")
        
    def create_overview_dashboard(self) -> Dict[str, Any]:
        """Create the main overview dashboard."""
        
        # System overview panels
        system_panels = [
            DashboardPanel(
                title="Service Status",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:service_availability",
                    "legendFormat": "Service Up"
                }],
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                options={
                    "colorMode": "background",
                    "graphMode": "none",
                    "justifyMode": "center",
                    "orientation": "horizontal",
                    "reduceOptions": {
                        "values": False,
                        "calcs": ["lastNotNull"],
                        "fields": ""
                    }
                },
                field_config={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "green", "value": 1}
                            ]
                        },
                        "unit": "bool"
                    }
                }
            ),
            
            DashboardPanel(
                title="Memory Usage",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:memory_usage_rss_mb",
                    "legendFormat": "RSS Memory"
                }],
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                options={
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "center",
                    "orientation": "horizontal"
                },
                field_config={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 500},
                                {"color": "red", "value": 1000}
                            ]
                        },
                        "unit": "MB"
                    }
                }
            ),
            
            DashboardPanel(
                title="CPU Usage",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent_cpu_usage_percent",
                    "legendFormat": "CPU %"
                }],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                options={
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "center"
                },
                field_config={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 60},
                                {"color": "red", "value": 80}
                            ]
                        },
                        "unit": "percent"
                    }
                }
            ),
            
            DashboardPanel(
                title="Disk Usage",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:disk_usage_percent",
                    "legendFormat": "Disk %"
                }],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                options={
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "center"
                },
                field_config={
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "green", "value": 0},
                                {"color": "yellow", "value": 70},
                                {"color": "red", "value": 85}
                            ]
                        },
                        "unit": "percent"
                    }
                }
            )
        ]
        
        # Performance panels
        performance_panels = [
            DashboardPanel(
                title="Operation Rate",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:operation_rate_by_type",
                    "legendFormat": "{{operation_type}}"
                }],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8},
                options={
                    "tooltip": {"mode": "multi", "sort": "desc"},
                    "legend": {"displayMode": "table", "placement": "right"}
                },
                field_config={
                    "defaults": {
                        "unit": "ops",
                        "color": {"mode": "palette-classic"}
                    }
                }
            ),
            
            DashboardPanel(
                title="Success Rate",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:operation_success_rate * 100",
                    "legendFormat": "Success Rate"
                }],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8},
                options={
                    "tooltip": {"mode": "single"},
                    "legend": {"displayMode": "list", "placement": "bottom"}
                },
                field_config={
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "color": {"mode": "continuous-GrYlRd"}
                    }
                }
            )
        ]
        
        # Quality panels
        quality_panels = [
            DashboardPanel(
                title="Quality Scores",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:quality_score_avg_by_type",
                    "legendFormat": "{{content_type}}"
                }],
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 8},
                options={
                    "tooltip": {"mode": "multi", "sort": "desc"},
                    "legend": {"displayMode": "table", "placement": "right"}
                },
                field_config={
                    "defaults": {
                        "unit": "short",
                        "min": 0,
                        "max": 1,
                        "color": {"mode": "palette-classic"}
                    }
                }
            ),
            
            DashboardPanel(
                title="Improvement Cycles",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:improvement_cycle_rate",
                    "legendFormat": "Improvement Rate"
                }],
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8},
                options={
                    "tooltip": {"mode": "single"},
                    "legend": {"displayMode": "list", "placement": "bottom"}
                },
                field_config={
                    "defaults": {
                        "unit": "cps",
                        "color": {"mode": "palette-classic"}
                    }
                }
            )
        ]
        
        all_panels = system_panels + performance_panels + quality_panels
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Academic Agent - Overview",
                "tags": ["academic-agent", "overview"],
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in all_panels],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "timepicker": {},
                "templating": {"list": []},
                "annotations": {"list": []},
                "refresh": "30s",
                "schemaVersion": 30,
                "version": 1,
                "links": []
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_performance_dashboard(self) -> Dict[str, Any]:
        """Create detailed performance monitoring dashboard."""
        
        panels = [
            # Operation duration histogram
            DashboardPanel(
                title="Operation Duration Distribution",
                panel_type="heatmap",
                targets=[{
                    "expr": "rate(academic_agent_operation_duration_seconds_bucket[5m])",
                    "legendFormat": "{{le}}"
                }],
                grid_pos={"x": 0, "y": 0, "w": 24, "h": 8}
            ),
            
            # Response time percentiles
            DashboardPanel(
                title="Response Time Percentiles",
                panel_type="graph",
                targets=[
                    {
                        "expr": "academic_agent:operation_duration_avg",
                        "legendFormat": "Average"
                    },
                    {
                        "expr": "academic_agent:operation_duration_p95",
                        "legendFormat": "95th Percentile"
                    },
                    {
                        "expr": "academic_agent:operation_duration_p99",
                        "legendFormat": "99th Percentile"
                    }
                ],
                grid_pos={"x": 0, "y": 8, "w": 12, "h": 8}
            ),
            
            # Throughput metrics
            DashboardPanel(
                title="Processing Throughput",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:throughput_by_operation",
                    "legendFormat": "{{operation_type}}"
                }],
                grid_pos={"x": 12, "y": 8, "w": 12, "h": 8}
            ),
            
            # Concurrent operations
            DashboardPanel(
                title="Concurrent Operations",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent_concurrent_operations",
                    "legendFormat": "{{operation_type}}"
                }],
                grid_pos={"x": 0, "y": 16, "w": 12, "h": 8}
            ),
            
            # Error rates
            DashboardPanel(
                title="Error Rates by Component",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:error_rate_by_component",
                    "legendFormat": "{{component}}"
                }],
                grid_pos={"x": 12, "y": 16, "w": 12, "h": 8}
            )
        ]
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Academic Agent - Performance",
                "tags": ["academic-agent", "performance"],
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in panels],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "15s",
                "schemaVersion": 30,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_pdf_processing_dashboard(self) -> Dict[str, Any]:
        """Create PDF processing specific dashboard."""
        
        panels = [
            # PDF processing rate
            DashboardPanel(
                title="PDF Processing Rate",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:pdf_processing_rate",
                    "legendFormat": "PDFs/sec"
                }],
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4}
            ),
            
            # Success rate
            DashboardPanel(
                title="PDF Success Rate",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:pdf_processing_success_rate * 100",
                    "legendFormat": "Success %"
                }],
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),
            
            # Average processing time
            DashboardPanel(
                title="Avg Processing Time",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:pdf_processing_duration_avg",
                    "legendFormat": "Seconds"
                }],
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),
            
            # Pages per second
            DashboardPanel(
                title="Pages/Second",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:pdf_pages_per_second",
                    "legendFormat": "Pages/sec"
                }],
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),
            
            # Processing time distribution
            DashboardPanel(
                title="PDF Processing Time Distribution",
                panel_type="graph",
                targets=[{
                    "expr": "histogram_quantile(0.5, rate(academic_agent_pdf_processing_duration_seconds_bucket[5m]))",
                    "legendFormat": "50th percentile"
                }, {
                    "expr": "histogram_quantile(0.95, rate(academic_agent_pdf_processing_duration_seconds_bucket[5m]))",
                    "legendFormat": "95th percentile"
                }, {
                    "expr": "histogram_quantile(0.99, rate(academic_agent_pdf_processing_duration_seconds_bucket[5m]))",
                    "legendFormat": "99th percentile"
                }],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8}
            ),
            
            # File size distribution
            DashboardPanel(
                title="PDF File Size Distribution",
                panel_type="graph",
                targets=[{
                    "expr": "histogram_quantile(0.5, rate(academic_agent_pdf_file_size_bytes_bucket[5m]))",
                    "legendFormat": "Median size"
                }, {
                    "expr": "histogram_quantile(0.95, rate(academic_agent_pdf_file_size_bytes_bucket[5m]))",
                    "legendFormat": "95th percentile"
                }],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8}
            ),
            
            # Images extracted
            DashboardPanel(
                title="Images Extracted Rate",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:images_extracted_rate",
                    "legendFormat": "Images/sec"
                }],
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 8}
            ),
            
            # Processing errors
            DashboardPanel(
                title="PDF Processing Errors",
                panel_type="graph",
                targets=[{
                    "expr": "rate(academic_agent_pdf_processing_duration_seconds_count{status=\"error\"}[5m])",
                    "legendFormat": "Errors/sec"
                }],
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8}
            )
        ]
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Academic Agent - PDF Processing",
                "tags": ["academic-agent", "pdf"],
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in panels],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s",
                "schemaVersion": 30,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def create_quality_dashboard(self) -> Dict[str, Any]:
        """Create quality metrics dashboard."""
        
        panels = [
            # Quality score overview
            DashboardPanel(
                title="Overall Quality Score",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:quality_score_avg",
                    "legendFormat": "Quality Score"
                }],
                grid_pos={"x": 0, "y": 0, "w": 8, "h": 4},
                field_config={
                    "defaults": {
                        "min": 0,
                        "max": 1,
                        "unit": "short",
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 0.5},
                                {"color": "green", "value": 0.7}
                            ]
                        }
                    }
                }
            ),
            
            # Improvement success rate
            DashboardPanel(
                title="Improvement Success Rate",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:improvement_cycle_success_rate * 100",
                    "legendFormat": "Success Rate %"
                }],
                grid_pos={"x": 8, "y": 0, "w": 8, "h": 4}
            ),
            
            # Quality violations
            DashboardPanel(
                title="Quality Violations/Hour",
                panel_type="stat",
                targets=[{
                    "expr": "academic_agent:quality_violation_rate * 3600",
                    "legendFormat": "Violations/hr"
                }],
                grid_pos={"x": 16, "y": 0, "w": 8, "h": 4}
            ),
            
            # Quality scores by content type
            DashboardPanel(
                title="Quality Scores by Content Type",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:quality_score_avg_by_type",
                    "legendFormat": "{{content_type}}"
                }],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8}
            ),
            
            # Quality scores by agent
            DashboardPanel(
                title="Quality Scores by Agent",
                panel_type="graph",
                targets=[{
                    "expr": "academic_agent:quality_score_avg_by_agent",
                    "legendFormat": "{{agent}}"
                }],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8}
            ),
            
            # Quality distribution
            DashboardPanel(
                title="Quality Score Distribution",
                panel_type="heatmap",
                targets=[{
                    "expr": "rate(academic_agent_quality_score_bucket[5m])",
                    "legendFormat": "{{le}}"
                }],
                grid_pos={"x": 0, "y": 12, "w": 24, "h": 8}
            )
        ]
        
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Academic Agent - Quality Metrics",
                "tags": ["academic-agent", "quality"],
                "timezone": "browser",
                "panels": [panel.to_dict() for panel in panels],
                "time": {
                    "from": "now-6h",
                    "to": "now"
                },
                "refresh": "1m",
                "schemaVersion": 30,
                "version": 1
            },
            "overwrite": True
        }
        
        return dashboard
    
    def export_dashboards(self, output_dir: Path):
        """Export all dashboards to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dashboards = {
            "overview": self.create_overview_dashboard(),
            "performance": self.create_performance_dashboard(),
            "pdf_processing": self.create_pdf_processing_dashboard(),
            "quality": self.create_quality_dashboard()
        }
        
        for name, dashboard in dashboards.items():
            file_path = output_dir / f"{name}_dashboard.json"
            with open(file_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            self.logger.info(f"Dashboard exported: {file_path}")
        
        # Create dashboard index
        index_file = output_dir / "dashboard_index.json"
        index = {
            "dashboards": [
                {
                    "name": name,
                    "title": dashboard["dashboard"]["title"],
                    "file": f"{name}_dashboard.json",
                    "tags": dashboard["dashboard"]["tags"]
                }
                for name, dashboard in dashboards.items()
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        self.logger.info(f"Dashboard index created: {index_file}")


def generate_all_dashboards(output_dir: Path = None):
    """Generate all monitoring dashboards."""
    if output_dir is None:
        output_dir = Path("monitoring/dashboards")
    
    generator = GrafanaDashboardGenerator()
    generator.export_dashboards(output_dir)
    
    return output_dir


if __name__ == "__main__":
    # Generate dashboards when run directly
    output_path = generate_all_dashboards()
    print(f"Dashboards generated in: {output_path}")