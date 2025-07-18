#!/usr/bin/env python3
"""
Health check script for Academic Agent container.
Returns exit code 0 for healthy, 1 for unhealthy.
"""

import sys
import json
import time
import socket
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health checker for Academic Agent."""
    
    def __init__(self):
        self.checks = []
        self.results = {}
        
    def add_check(self, name: str, check_func, critical: bool = True):
        """Add a health check."""
        self.checks.append({
            'name': name,
            'func': check_func,
            'critical': critical
        })
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        overall_healthy = True
        
        for check in self.checks:
            try:
                result = check['func']()
                self.results[check['name']] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'critical': check['critical'],
                    'details': result if isinstance(result, dict) else {'passed': result}
                }
                
                if not result and check['critical']:
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"Health check '{check['name']}' failed: {e}")
                self.results[check['name']] = {
                    'status': 'error',
                    'critical': check['critical'],
                    'error': str(e)
                }
                
                if check['critical']:
                    overall_healthy = False
        
        self.results['overall'] = {
            'status': 'healthy' if overall_healthy else 'unhealthy',
            'timestamp': time.time()
        }
        
        return self.results
    
    def check_port_listening(self, port: int, host: str = 'localhost') -> bool:
        """Check if a port is listening."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"Port check failed for {host}:{port}: {e}")
            return False
    
    def check_http_endpoint(self, url: str, timeout: int = 10) -> Dict[str, Any]:
        """Check HTTP endpoint health."""
        try:
            response = requests.get(url, timeout=timeout)
            return {
                'reachable': True,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'healthy': 200 <= response.status_code < 400
            }
        except Exception as e:
            logger.error(f"HTTP check failed for {url}: {e}")
            return {
                'reachable': False,
                'error': str(e),
                'healthy': False
            }
    
    def check_file_writable(self, path: str) -> bool:
        """Check if a directory is writable."""
        try:
            test_path = Path(path) / '.health_check_test'
            test_path.write_text('test')
            test_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Write check failed for {path}: {e}")
            return False
    
    def check_disk_space(self, path: str = '/', min_free_mb: int = 100) -> Dict[str, Any]:
        """Check disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            free_mb = free // (1024 * 1024)
            
            return {
                'free_mb': free_mb,
                'free_percent': (free / total) * 100,
                'sufficient': free_mb >= min_free_mb
            }
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return {'sufficient': False, 'error': str(e)}
    
    def check_memory_usage(self, max_usage_percent: int = 90) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                'used_percent': memory.percent,
                'available_mb': memory.available // (1024 * 1024),
                'healthy': memory.percent < max_usage_percent
            }
        except ImportError:
            logger.warning("psutil not available for memory check")
            return {'healthy': True, 'note': 'psutil not available'}
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def check_log_directory(self) -> bool:
        """Check log directory accessibility."""
        log_paths = ['/app/logs', './logs']
        
        for path in log_paths:
            if Path(path).exists():
                return self.check_file_writable(path)
        
        # Try to create logs directory
        try:
            Path('./logs').mkdir(exist_ok=True)
            return self.check_file_writable('./logs')
        except Exception as e:
            logger.error(f"Cannot create/access log directory: {e}")
            return False
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files."""
        config_paths = [
            '/app/config/production.yaml',
            './config/production.yaml',
            '/app/config/base.yaml',
            './config/base.yaml'
        ]
        
        for config_path in config_paths:
            if Path(config_path).exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    return {
                        'found': True,
                        'path': config_path,
                        'valid': True,
                        'keys': list(config.keys()) if isinstance(config, dict) else []
                    }
                except Exception as e:
                    return {
                        'found': True,
                        'path': config_path,
                        'valid': False,
                        'error': str(e)
                    }
        
        return {
            'found': False,
            'searched_paths': config_paths
        }
    
    def check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment."""
        try:
            import sys
            
            # Check critical imports
            critical_modules = [
                'fastapi', 'uvicorn', 'openai', 'pydantic', 
                'aiofiles', 'requests', 'yaml'
            ]
            
            missing_modules = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            return {
                'python_version': sys.version,
                'missing_modules': missing_modules,
                'healthy': len(missing_modules) == 0
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}


def main():
    """Main health check function."""
    checker = HealthChecker()
    
    # Add health checks
    checker.add_check('app_port', lambda: checker.check_port_listening(8080), critical=True)
    checker.add_check('metrics_port', lambda: checker.check_port_listening(9090), critical=False)
    checker.add_check('logs', checker.check_log_directory, critical=True)
    checker.add_check('disk_space', lambda: checker.check_disk_space()['sufficient'], critical=True)
    checker.add_check('memory', lambda: checker.check_memory_usage()['healthy'], critical=False)
    checker.add_check('config', lambda: checker.check_configuration()['found'], critical=True)
    checker.add_check('python_env', lambda: checker.check_python_environment()['healthy'], critical=True)
    
    # Try to check application endpoint if available
    try:
        app_check = checker.check_http_endpoint('http://localhost:8080/health', timeout=5)
        checker.add_check('app_endpoint', lambda: app_check['healthy'], critical=True)
    except Exception:
        logger.info("Application endpoint not available for health check")
    
    # Run all checks
    results = checker.run_checks()
    
    # Output results
    if len(sys.argv) > 1 and sys.argv[1] == '--json':
        print(json.dumps(results, indent=2))
    else:
        overall_status = results['overall']['status']
        print(f"Health Status: {overall_status.upper()}")
        
        for check_name, result in results.items():
            if check_name == 'overall':
                continue
            
            status = result['status']
            critical = result.get('critical', False)
            priority = 'CRITICAL' if critical else 'WARNING'
            
            if status == 'healthy':
                print(f"✓ {check_name}: OK")
            else:
                print(f"✗ {check_name}: {status.upper()} ({priority})")
                if 'error' in result:
                    print(f"  Error: {result['error']}")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall']['status'] == 'healthy' else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()