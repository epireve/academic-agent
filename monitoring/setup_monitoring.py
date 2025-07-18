#!/usr/bin/env python3
"""Setup script for Academic Agent monitoring system."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from academic_agent_v2.src.core.logging import get_logger
from .dashboards import generate_all_dashboards
from .integration import get_monitoring_integration


class MonitoringSetup:
    """Setup and configuration manager for the monitoring system."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.monitoring_dir = self.project_root / "monitoring"
        self.logger = get_logger("monitoring_setup")
        
        # Ensure monitoring directory exists
        self.monitoring_dir.mkdir(exist_ok=True)
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        required_packages = [
            "prometheus-client",
            "psutil",
            "requests"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            self.logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        return True
    
    def check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.logger.info(f"Docker available: {result.stdout.strip()}")
                return True
            else:
                self.logger.warning("Docker not available")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.logger.warning("Docker not found or not responding")
            return False
    
    def setup_prometheus_config(self):
        """Setup Prometheus configuration files."""
        self.logger.info("Setting up Prometheus configuration...")
        
        # Check if configuration files exist
        config_files = [
            "prometheus.yml",
            "alert_rules.yml",
            "recording_rules.yml",
            "alertmanager.yml"
        ]
        
        missing_files = []
        for config_file in config_files:
            file_path = self.monitoring_dir / config_file
            if not file_path.exists():
                missing_files.append(config_file)
        
        if missing_files:
            self.logger.error(f"Missing configuration files: {', '.join(missing_files)}")
            self.logger.info("Please ensure all monitoring configuration files are present")
            return False
        
        self.logger.info("Prometheus configuration files are ready")
        return True
    
    def setup_grafana_config(self):
        """Setup Grafana configuration and dashboards."""
        self.logger.info("Setting up Grafana configuration...")
        
        # Create Grafana directories
        grafana_dir = self.monitoring_dir / "grafana"
        provisioning_dir = grafana_dir / "provisioning"
        datasources_dir = provisioning_dir / "datasources"
        dashboards_dir = provisioning_dir / "dashboards"
        
        for directory in [grafana_dir, provisioning_dir, datasources_dir, dashboards_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create datasource configuration
        datasource_config = """
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"""
        
        with open(datasources_dir / "prometheus.yml", 'w') as f:
            f.write(datasource_config)
        
        # Create dashboard configuration
        dashboard_config = """
apiVersion: 1

providers:
  - name: 'Academic Agent Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
"""
        
        with open(dashboards_dir / "dashboards.yml", 'w') as f:
            f.write(dashboard_config)
        
        # Generate dashboard files
        dashboard_output_dir = self.monitoring_dir / "dashboards"
        generate_all_dashboards(dashboard_output_dir)
        
        self.logger.info("Grafana configuration and dashboards are ready")
        return True
    
    def setup_process_exporter_config(self):
        """Setup process exporter configuration."""
        self.logger.info("Setting up process exporter configuration...")
        
        process_config = """
process_names:
  - name: "{{.Comm}}"
    cmdline:
    - '.+'

  - name: "academic-agent"
    cmdline:
    - 'python.*academic.*agent'
    - 'academic-agent'
  
  - name: "python"
    cmdline:
    - 'python'
"""
        
        config_path = self.monitoring_dir / "process_exporter.yml"
        with open(config_path, 'w') as f:
            f.write(process_config)
        
        self.logger.info("Process exporter configuration created")
        return True
    
    def setup_loki_config(self):
        """Setup Loki configuration for log aggregation."""
        self.logger.info("Setting up Loki configuration...")
        
        loki_config = """
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

compactor:
  working_directory: /loki/boltdb-shipper-compactor
  shared_store: filesystem

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules-temp
  alertmanager_url: http://alertmanager:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
"""
        
        config_path = self.monitoring_dir / "loki.yml"
        with open(config_path, 'w') as f:
            f.write(loki_config)
        
        # Setup Promtail configuration
        promtail_config = """
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: academic-agent-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: academic-agent
          __path__: /var/log/academic-agent/*.log
    
    pipeline_stages:
      - json:
          expressions:
            timestamp: timestamp
            level: level
            message: message
            component: logger_name
      
      - labels:
          level:
          component:
      
      - timestamp:
          source: timestamp
          format: RFC3339Nano
"""
        
        promtail_path = self.monitoring_dir / "promtail.yml"
        with open(promtail_path, 'w') as f:
            f.write(promtail_config)
        
        self.logger.info("Loki and Promtail configurations created")
        return True
    
    def create_docker_network(self):
        """Create Docker networks for monitoring."""
        try:
            # Check if networks already exist
            result = subprocess.run(
                ["docker", "network", "ls", "--format", "{{.Name}}"],
                capture_output=True,
                text=True
            )
            
            existing_networks = result.stdout.strip().split('\n')
            
            networks_to_create = []
            if "academic-agent-monitoring" not in existing_networks:
                networks_to_create.append("academic-agent-monitoring")
            
            if "academic-agent-app" not in existing_networks:
                networks_to_create.append("academic-agent-app")
            
            for network in networks_to_create:
                subprocess.run(
                    ["docker", "network", "create", network],
                    check=True,
                    capture_output=True
                )
                self.logger.info(f"Created Docker network: {network}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create Docker networks: {e}")
            return False
    
    def start_monitoring_stack(self, services: Optional[List[str]] = None):
        """Start the monitoring stack using Docker Compose."""
        if not self.check_docker():
            self.logger.error("Docker is required to start the monitoring stack")
            return False
        
        try:
            # Change to monitoring directory
            original_dir = os.getcwd()
            os.chdir(self.monitoring_dir)
            
            # Build the docker-compose command
            cmd = ["docker-compose", "up", "-d"]
            
            if services:
                cmd.extend(services)
            
            # Start the services
            self.logger.info("Starting monitoring stack...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            self.logger.info("Monitoring stack started successfully")
            self.logger.info("Access points:")
            self.logger.info("  - Prometheus: http://localhost:9090")
            self.logger.info("  - Grafana: http://localhost:3000 (admin/admin)")
            self.logger.info("  - Alertmanager: http://localhost:9093")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to start monitoring stack: {e}")
            if e.stdout:
                self.logger.error(f"stdout: {e.stdout}")
            if e.stderr:
                self.logger.error(f"stderr: {e.stderr}")
            return False
            
        finally:
            os.chdir(original_dir)
    
    def stop_monitoring_stack(self):
        """Stop the monitoring stack."""
        try:
            original_dir = os.getcwd()
            os.chdir(self.monitoring_dir)
            
            subprocess.run(
                ["docker-compose", "down"],
                check=True,
                capture_output=True,
                text=True
            )
            
            self.logger.info("Monitoring stack stopped")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to stop monitoring stack: {e}")
            return False
            
        finally:
            os.chdir(original_dir)
    
    def setup_monitoring_integration(self):
        """Setup the monitoring integration with the Academic Agent."""
        self.logger.info("Setting up monitoring integration...")
        
        try:
            # Initialize monitoring integration
            integration = get_monitoring_integration()
            
            # This would typically be called from the main application
            # integration.start_monitoring()
            
            self.logger.info("Monitoring integration configured")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring integration: {e}")
            return False
    
    def run_full_setup(self):
        """Run the complete monitoring setup."""
        self.logger.info("Starting full monitoring setup...")
        
        steps = [
            ("Checking dependencies", self.check_dependencies),
            ("Setting up Prometheus config", self.setup_prometheus_config),
            ("Setting up Grafana config", self.setup_grafana_config),
            ("Setting up process exporter config", self.setup_process_exporter_config),
            ("Setting up Loki config", self.setup_loki_config),
            ("Setting up monitoring integration", self.setup_monitoring_integration)
        ]
        
        if self.check_docker():
            steps.append(("Creating Docker networks", self.create_docker_network))
        
        failed_steps = []
        
        for step_name, step_func in steps:
            self.logger.info(f"Running: {step_name}")
            try:
                if not step_func():
                    failed_steps.append(step_name)
                    self.logger.error(f"Failed: {step_name}")
                else:
                    self.logger.info(f"Completed: {step_name}")
            except Exception as e:
                failed_steps.append(step_name)
                self.logger.error(f"Error in {step_name}: {e}")
        
        if failed_steps:
            self.logger.error(f"Setup completed with errors in: {', '.join(failed_steps)}")
            return False
        else:
            self.logger.info("Full monitoring setup completed successfully!")
            
            # Print next steps
            self.logger.info("\nNext steps:")
            self.logger.info("1. Start the monitoring stack: python monitoring/setup_monitoring.py --start")
            self.logger.info("2. Access Grafana at http://localhost:3000 (admin/admin)")
            self.logger.info("3. Import dashboards and configure data sources")
            self.logger.info("4. Configure alert notification channels")
            
            return True


def main():
    """Main function for the setup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Academic Agent Monitoring Setup")
    parser.add_argument("--check-deps", action="store_true", help="Check dependencies only")
    parser.add_argument("--setup", action="store_true", help="Run full setup")
    parser.add_argument("--start", action="store_true", help="Start monitoring stack")
    parser.add_argument("--stop", action="store_true", help="Stop monitoring stack")
    parser.add_argument("--services", nargs="*", help="Specific services to start")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    
    args = parser.parse_args()
    
    setup = MonitoringSetup(args.project_root)
    
    if args.check_deps:
        if setup.check_dependencies():
            print("All dependencies are available")
            sys.exit(0)
        else:
            print("Missing dependencies")
            sys.exit(1)
    
    elif args.setup:
        if setup.run_full_setup():
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.start:
        if setup.start_monitoring_stack(args.services):
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.stop:
        if setup.stop_monitoring_stack():
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        # Run full setup by default
        if setup.run_full_setup():
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()