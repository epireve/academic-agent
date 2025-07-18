# Academic Agent Monitoring System

This directory contains a comprehensive monitoring solution for the Academic Agent project using Prometheus, Grafana, and related tools.

## Overview

The monitoring system provides:

- **Metrics Collection**: Prometheus for collecting and storing metrics
- **Visualization**: Grafana dashboards for monitoring system health and performance
- **Alerting**: Alertmanager for handling alerts and notifications
- **Log Aggregation**: Loki and Promtail for centralized logging
- **System Monitoring**: Node exporter and process exporter for system-level metrics
- **Long-term Storage**: VictoriaMetrics for extended metric retention

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Academic Agent │────│   Prometheus    │────│     Grafana     │
│   Application   │    │   (Metrics)     │    │  (Dashboards)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐             │
         │              │  Alertmanager   │             │
         │              │   (Alerts)      │             │
         │              └─────────────────┘             │
         │                                              │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Loki       │────│    Promtail     │    │ VictoriaMetrics │
│   (Logs)        │    │ (Log Collection)│    │ (Long-term)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install prometheus-client psutil requests
   ```

2. **Docker and Docker Compose**:
   ```bash
   # Verify Docker installation
   docker --version
   docker-compose --version
   ```

### Setup and Start

1. **Run the setup script**:
   ```bash
   python monitoring/setup_monitoring.py --setup
   ```

2. **Start the monitoring stack**:
   ```bash
   python monitoring/setup_monitoring.py --start
   ```

3. **Access the monitoring interfaces**:
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000 (admin/admin)
   - **Alertmanager**: http://localhost:9093

### Quick Commands

```bash
# Check dependencies
python monitoring/setup_monitoring.py --check-deps

# Start specific services
python monitoring/setup_monitoring.py --start --services prometheus grafana

# Stop the monitoring stack
python monitoring/setup_monitoring.py --stop
```

## Configuration Files

### Core Configuration

- `prometheus.yml` - Prometheus server configuration
- `alert_rules.yml` - Alert rule definitions
- `recording_rules.yml` - Recording rule definitions for aggregations
- `alertmanager.yml` - Alertmanager notification configuration
- `docker-compose.yml` - Complete monitoring stack definition

### Integration

- `prometheus_config.py` - Prometheus integration and metrics collection
- `integration.py` - Integration layer with existing Academic Agent systems
- `dashboards.py` - Grafana dashboard generation

## Metrics Overview

### System Metrics

- **Memory Usage**: RSS and VMS memory consumption
- **CPU Usage**: Current CPU utilization percentage
- **Disk Usage**: Disk space utilization and free space
- **Uptime**: System and service uptime tracking

### Performance Metrics

- **Operation Duration**: Time spent on various operations
- **Operation Count**: Total number of operations by type and status
- **Concurrent Operations**: Number of simultaneous operations
- **Processing Throughput**: Items processed per second

### Quality Metrics

- **Quality Scores**: Content quality assessment scores
- **Improvement Cycles**: Number and success rate of improvement iterations
- **Quality Violations**: Threshold violations and quality issues

### PDF Processing Metrics

- **Processing Duration**: Time spent processing PDF files
- **Pages Processed**: Number of pages processed per operation
- **File Size Distribution**: Size distribution of processed files
- **Image Extraction**: Number of images extracted from PDFs

### Agent Communication Metrics

- **Message Count**: Number of messages between agents
- **Message Latency**: Communication latency between agents
- **Agent Status**: Active/inactive status of each agent
- **Failed Messages**: Communication failures and retry attempts

### Error Metrics

- **Error Count**: Total errors by type, component, and severity
- **Error Rate**: Current error rate per component
- **Recovery Count**: Number of successful error recoveries

## Dashboards

The system includes pre-configured Grafana dashboards:

### 1. Overview Dashboard
- System status and health indicators
- Key performance metrics
- Alert summaries
- Resource utilization

### 2. Performance Dashboard
- Operation duration distributions
- Response time percentiles
- Throughput metrics
- Concurrent operation tracking
- Error rate analysis

### 3. PDF Processing Dashboard
- PDF processing rates and success metrics
- Processing time distributions
- File size analysis
- Image extraction statistics
- Processing error tracking

### 4. Quality Dashboard
- Quality score trends and distributions
- Improvement cycle success rates
- Quality violation tracking
- Agent-specific quality metrics

## Alerting

### Alert Categories

1. **Critical Alerts** (Immediate attention)
   - Service down
   - Critical memory/disk usage
   - High error rates (>25%)

2. **Warning Alerts** (Attention within hours)
   - High resource usage
   - Performance degradation
   - Quality issues

3. **Info Alerts** (Informational)
   - Configuration changes
   - Maintenance notifications

### Alert Routing

Alerts are routed based on:
- **Severity**: critical, warning, info
- **Component**: system, performance, quality, pdf_processing, communication, errors
- **Source**: specific agent or system component

### Notification Channels

- **Email**: Configured per alert type
- **Slack**: Critical alerts to #alerts-critical channel
- **Webhook**: Custom notification endpoints

## Integration with Academic Agent

### Automatic Metric Collection

The monitoring system automatically collects metrics from:

1. **Core System Monitor**: Existing monitoring infrastructure
2. **Operation Tracking**: Decorated functions and context managers
3. **Agent Communication**: Inter-agent message tracking
4. **Quality Assessment**: Quality score and improvement cycle tracking

### Usage in Code

```python
from monitoring.integration import track_operation, track_pdf_processing, track_quality_score

# Track operations
@track_operation("pdf_processing")
def process_pdf(file_path):
    # PDF processing logic
    pass

# Track PDF processing with context manager
with track_operation("content_analysis", "analysis_123"):
    # Analysis logic
    pass

# Track quality scores
@track_quality_score("notes", "notes_agent")
def generate_notes(content):
    # Note generation logic
    return {"quality_score": 0.85}
```

### Starting Monitoring in Application

```python
from monitoring.integration import start_integrated_monitoring

# Start monitoring when application starts
monitoring = start_integrated_monitoring()

# Monitor will automatically collect metrics and sync with Prometheus
```

## Advanced Configuration

### Custom Metrics

Add custom metrics to `prometheus_config.py`:

```python
# In PrometheusMetrics.__init__
self.custom_metric = Counter(
    'academic_agent_custom_operations_total',
    'Custom operation counter',
    ['operation_type', 'result'],
    registry=self.registry
)

# Usage
self.custom_metric.labels(operation_type="custom", result="success").inc()
```

### Custom Alerts

Add custom alert rules to `alert_rules.yml`:

```yaml
- alert: CustomAlert
  expr: custom_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Custom alert triggered"
    description: "Custom condition met: {{ $value }}"
```

### Custom Dashboards

Generate custom dashboards using `dashboards.py`:

```python
from monitoring.dashboards import DashboardPanel, GrafanaDashboardGenerator

generator = GrafanaDashboardGenerator()

# Create custom panel
panel = DashboardPanel(
    title="Custom Metric",
    panel_type="graph",
    targets=[{"expr": "custom_metric", "legendFormat": "Custom"}],
    grid_pos={"x": 0, "y": 0, "w": 12, "h": 8}
)

# Add to dashboard
```

## Troubleshooting

### Common Issues

1. **Prometheus not collecting metrics**:
   ```bash
   # Check if metrics endpoint is accessible
   curl http://localhost:9090/metrics
   
   # Check Prometheus configuration
   docker-compose logs prometheus
   ```

2. **Grafana dashboards not loading**:
   ```bash
   # Check Grafana logs
   docker-compose logs grafana
   
   # Verify datasource configuration
   # Access Grafana -> Configuration -> Data Sources
   ```

3. **Alerts not firing**:
   ```bash
   # Check Alertmanager configuration
   docker-compose logs alertmanager
   
   # Verify alert rules in Prometheus
   # Access Prometheus -> Alerts
   ```

4. **High memory usage**:
   ```bash
   # Check retention settings in prometheus.yml
   # Reduce retention period if needed
   --storage.tsdb.retention.time=7d
   ```

### Debug Commands

```bash
# Check service status
docker-compose ps

# View logs for specific service
docker-compose logs -f prometheus
docker-compose logs -f grafana
docker-compose logs -f alertmanager

# Restart specific service
docker-compose restart prometheus

# Check metrics endpoint
curl -s http://localhost:9090/metrics | grep academic_agent

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets
```

## Maintenance

### Regular Tasks

1. **Monitor disk usage**: Prometheus and Grafana data growth
2. **Review alerts**: Tune thresholds based on operational experience
3. **Update dashboards**: Add new metrics and improve visualizations
4. **Backup configurations**: Save custom dashboards and alert rules

### Backup

```bash
# Backup Prometheus data
docker run --rm -v monitoring_prometheus_data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus_backup.tar.gz /data

# Backup Grafana data
docker run --rm -v monitoring_grafana_data:/data -v $(pwd):/backup alpine tar czf /backup/grafana_backup.tar.gz /data
```

### Updates

```bash
# Update monitoring stack
docker-compose pull
docker-compose up -d

# Regenerate dashboards
python monitoring/dashboards.py
```

## Performance Considerations

### Resource Usage

- **Prometheus**: ~100-500MB RAM, depends on metric cardinality
- **Grafana**: ~50-100MB RAM
- **Alertmanager**: ~20-50MB RAM
- **Total**: ~200-700MB RAM for full stack

### Scaling

- **High metric volume**: Consider VictoriaMetrics for better performance
- **Many alerts**: Tune alert grouping and inhibition rules
- **Long retention**: Use remote storage (VictoriaMetrics, Thanos)

### Optimization

- Reduce metric cardinality by limiting label values
- Use recording rules for frequently queried complex expressions
- Configure appropriate retention periods
- Monitor monitoring system itself for resource usage

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review Docker Compose logs
3. Consult Prometheus and Grafana documentation
4. Check the Academic Agent project documentation

## License

This monitoring configuration is part of the Academic Agent project and follows the same license terms.