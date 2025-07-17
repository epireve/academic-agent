#!/usr/bin/env python3
"""
Setup and Installation Script for High-Performance PDF Processor
Academic Agent v2 - Task 11 Implementation

This script handles installation, configuration, and verification of the PDF processor.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessorSetup:
    """Setup and installation manager for PDF processor."""
    
    def __init__(self):
        self.system_info = {
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': sys.version_info,
            'python_executable': sys.executable
        }
        
    def check_python_version(self):
        """Check if Python version is compatible."""
        logger.info("Checking Python version...")
        
        min_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            logger.error(f"Python {min_version[0]}.{min_version[1]}+ required, found {current_version[0]}.{current_version[1]}")
            return False
            
        logger.info(f"✅ Python {current_version[0]}.{current_version[1]} is compatible")
        return True
        
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        
        # Core dependencies
        core_deps = [
            'torch',
            'torchvision', 
            'pillow',
            'psutil',
            'pyyaml',
            'pytest',
            'asyncio'
        ]
        
        # Optional dependencies for better functionality
        optional_deps = [
            'marker-pdf',  # Main PDF processing library
            'transformers',  # For enhanced text recognition
            'GPUtil',  # For GPU monitoring
        ]
        
        # Install core dependencies
        logger.info("Installing core dependencies...")
        for dep in core_deps:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
                logger.info(f"✅ Installed {dep}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {dep}: {e}")
                
        # Install optional dependencies
        logger.info("Installing optional dependencies...")
        for dep in optional_deps:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
                logger.info(f"✅ Installed {dep}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Failed to install {dep}: {e}")
                logger.warning(f"   This dependency is optional and may not be critical")
                
    def check_gpu_support(self):
        """Check for GPU support and availability."""
        logger.info("Checking GPU support...")
        
        gpu_info = {
            'cuda_available': False,
            'mps_available': False,
            'gpu_count': 0,
            'recommended_device': 'cpu'
        }
        
        try:
            import torch
            
            # Check CUDA
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['gpu_count'] = torch.cuda.device_count()
                gpu_info['recommended_device'] = 'cuda'
                logger.info(f"✅ CUDA available with {gpu_info['gpu_count']} GPU(s)")
                
            # Check MPS (Apple Silicon)
            if torch.backends.mps.is_available():
                gpu_info['mps_available'] = True
                if not gpu_info['cuda_available']:  # Prefer MPS on Apple Silicon
                    gpu_info['recommended_device'] = 'mps'
                logger.info("✅ MPS (Apple Silicon) available")
                
            if not gpu_info['cuda_available'] and not gpu_info['mps_available']:
                logger.info("ℹ️  No GPU acceleration available, using CPU")
                
        except ImportError:
            logger.warning("⚠️  PyTorch not available, cannot check GPU support")
            
        return gpu_info
        
    def create_sample_config(self):
        """Create a sample configuration file."""
        logger.info("Creating sample configuration...")
        
        config_path = Path("pdf_processor_config_sample.yaml")
        
        gpu_info = self.check_gpu_support()
        
        sample_config = f"""# High-Performance PDF Processor Configuration
# Generated automatically by setup script

# Device Configuration
device:
  type: {gpu_info['recommended_device']}  # Recommended for your system
  use_gpu: {'true' if gpu_info['cuda_available'] or gpu_info['mps_available'] else 'false'}

# Processing Configuration  
processing:
  batch_size: {'3 if gpu_info["cuda_available"] else 2'}
  max_workers: 4
  max_pages: null
  split_chapters: true
  extract_images: true

# Marker Library Configuration
marker:
  models:
    enable_editor_model: true
    enable_ocr: true
    default_language: en
    
  quality:
    extract_tables: true
    extract_equations: true
    preserve_formatting: true

# Monitoring Configuration
monitoring:
  enabled: true
  log_level: INFO
  
  alerts:
    memory_threshold: 80.0
    time_threshold: 300.0
    error_threshold: 10.0

# Output Configuration
output:
  images:
    extract: true
    save_images: true
    format: png
    
  chapters:
    enable: true
    min_length: 100

# System Information (detected automatically)
# Platform: {self.system_info['platform']}
# Architecture: {self.system_info['architecture']}
# Python: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}
# GPU Support: {'Yes' if gpu_info['cuda_available'] or gpu_info['mps_available'] else 'No'}
"""
        
        config_path.write_text(sample_config)
        logger.info(f"✅ Sample configuration saved to {config_path}")
        
        return config_path
        
    def run_verification_tests(self):
        """Run verification tests to ensure everything is working."""
        logger.info("Running verification tests...")
        
        try:
            # Test imports
            logger.info("Testing imports...")
            
            # Test core modules
            from marker_pdf_processor import create_pdf_processor
            from monitoring import MonitoringSystem
            logger.info("✅ Core modules imported successfully")
            
            # Test basic functionality
            logger.info("Testing basic functionality...")
            
            # Create a simple processor
            processor = create_pdf_processor({'device': 'cpu', 'batch_size': 1})
            logger.info("✅ PDF processor created successfully")
            
            # Test monitoring
            monitoring = MonitoringSystem()
            monitoring.stop_monitoring()  # Clean shutdown
            logger.info("✅ Monitoring system initialized successfully")
            
            logger.info("✅ All verification tests passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification test failed: {e}")
            return False
            
    def create_directories(self):
        """Create necessary directories."""
        logger.info("Creating directories...")
        
        directories = [
            'input',
            'output', 
            'logs',
            'cache',
            'tests/fixtures'
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
            
    def show_usage_examples(self):
        """Show usage examples."""
        logger.info("Usage Examples:")
        print("""
📚 High-Performance PDF Processor - Usage Examples

1. Basic Processing:
   python -c "
   import asyncio
   from marker_pdf_processor import create_pdf_processor
   from pathlib import Path
   
   async def main():
       processor = create_pdf_processor()
       result = await processor.process_single_pdf(
           Path('input/document.pdf'), 
           Path('output/')
       )
       print(f'Success: {result.success}')
   
   asyncio.run(main())
   "

2. Batch Processing:
   python pdf_processor_cli.py batch input/ output/ --batch-size 3

3. With Custom Configuration:
   python pdf_processor_cli.py process document.pdf output/ --device cuda --batch-size 2

4. Running Tests:
   python -m pytest test_pdf_processor.py -v

5. Complete Demo:
   python example_usage.py

For more information, see README.md
""")
        
    def run_setup(self, install_deps=True, create_config=True, run_tests=True):
        """Run the complete setup process."""
        logger.info("🚀 Starting PDF Processor Setup")
        logger.info("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            logger.error("❌ Setup failed: Incompatible Python version")
            return False
            
        # Install dependencies
        if install_deps:
            self.install_dependencies()
            
        # Check GPU support
        gpu_info = self.check_gpu_support()
        
        # Create directories
        self.create_directories()
        
        # Create sample configuration
        if create_config:
            config_path = self.create_sample_config()
            
        # Run verification tests
        if run_tests:
            if not self.run_verification_tests():
                logger.warning("⚠️  Some verification tests failed")
                logger.info("💡 The system may still work, but check the error messages above")
            
        # Show system information
        logger.info("📊 System Information:")
        logger.info(f"   Platform: {self.system_info['platform']}")
        logger.info(f"   Architecture: {self.system_info['architecture']}")
        logger.info(f"   Python: {self.system_info['python_version'].major}.{self.system_info['python_version'].minor}")
        logger.info(f"   Recommended device: {gpu_info['recommended_device']}")
        
        # Show usage examples
        self.show_usage_examples()
        
        logger.info("✅ Setup completed successfully!")
        logger.info("🎯 You can now use the PDF processor")
        
        return True


def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup High-Performance PDF Processor")
    parser.add_argument('--no-install', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--no-config', action='store_true',
                       help='Skip configuration file creation') 
    parser.add_argument('--no-test', action='store_true',
                       help='Skip verification tests')
    
    args = parser.parse_args()
    
    setup = PDFProcessorSetup()
    
    success = setup.run_setup(
        install_deps=not args.no_install,
        create_config=not args.no_config,
        run_tests=not args.no_test
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()