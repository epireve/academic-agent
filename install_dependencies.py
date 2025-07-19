#!/usr/bin/env python3
"""
Academic Agent Dependency Installation Script
Installs all required dependencies for the Academic Agent project.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package, description=""):
    """Install a package using pip."""
    print(f"Installing {package}..." + (f" ({description})" if description else ""))
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def check_package(package):
    """Check if a package is installed."""
    try:
        __import__(package.replace('-', '_'))
        return True
    except ImportError:
        return False

def main():
    """Main installation function."""
    print("Academic Agent Dependency Installation")
    print("=" * 50)
    
    # Core dependencies with descriptions
    core_dependencies = [
        ("pytest>=7.0.0", "Testing framework"),
        ("python-frontmatter>=1.0.0", "YAML frontmatter processing"),
        ("docling>=1.2.0", "PDF processing library"),
        ("pydantic>=2.5.0", "Data validation"),
        ("pydantic-settings>=2.0.0", "Settings management"),
        ("pyyaml>=6.0", "YAML configuration parsing"),
        ("toml>=0.10.2", "TOML configuration parsing"),
        ("smolagents>=0.3.0", "Agent framework"),
        ("litellm>=1.0.0", "LLM interface"),
        ("python-dotenv>=1.0.0", "Environment variables"),
        ("requests>=2.31.0", "HTTP client"),
        ("aiofiles>=23.2.1", "Async file operations"),
        ("fastapi>=0.104.1", "Web framework"),
        ("uvicorn>=0.24.0", "ASGI server"),
        ("openai>=1.3.0", "OpenAI API client"),
    ]
    
    # Optional dependencies
    optional_dependencies = [
        ("marker-pdf>=0.2.14", "Advanced PDF processing"),
        ("weasyprint>=60.0", "PDF generation"),
        ("markdown>=3.5.0", "Markdown processing"),
        ("psutil>=5.9.0", "System monitoring"),
        ("prometheus-client>=0.17.0", "Metrics collection"),
    ]
    
    # Development dependencies
    dev_dependencies = [
        ("black>=22.0.0", "Code formatting"),
        ("flake8>=6.0.0", "Code linting"),
        ("mypy>=1.5.0", "Type checking"),
        ("pytest-cov>=4.1.0", "Coverage reporting"),
        ("pytest-mock>=3.11.1", "Mocking utilities"),
        ("pytest-asyncio>=0.21.0", "Async testing support"),
        ("pre-commit>=3.4.0", "Git hooks"),
        ("isort>=5.12.0", "Import sorting"),
        ("bandit>=1.7.5", "Security linting"),
    ]
    
    print("Installing core dependencies...")
    core_success = 0
    for package, description in core_dependencies:
        if install_package(package, description):
            core_success += 1
    
    print(f"\nCore dependencies: {core_success}/{len(core_dependencies)} installed successfully")
    
    print("\nInstalling optional dependencies...")
    optional_success = 0
    for package, description in optional_dependencies:
        if install_package(package, description):
            optional_success += 1
    
    print(f"Optional dependencies: {optional_success}/{len(optional_dependencies)} installed successfully")
    
    print("\nInstalling development dependencies...")
    dev_success = 0
    for package, description in dev_dependencies:
        if install_package(package, description):
            dev_success += 1
    
    print(f"Development dependencies: {dev_success}/{len(dev_dependencies)} installed successfully")
    
    # Verify critical imports
    print("\nVerifying critical package imports...")
    critical_packages = [
        'pytest', 'frontmatter', 'docling', 'pydantic', 'yaml', 'smolagents', 'litellm'
    ]
    
    working_packages = []
    for package in critical_packages:
        if check_package(package):
            print(f"✓ {package} - available")
            working_packages.append(package)
        else:
            print(f"✗ {package} - still missing")
    
    print(f"\nInstallation Summary:")
    print(f"Core: {core_success}/{len(core_dependencies)}")
    print(f"Optional: {optional_success}/{len(optional_dependencies)}")
    print(f"Development: {dev_success}/{len(dev_dependencies)}")
    print(f"Critical imports working: {len(working_packages)}/{len(critical_packages)}")
    
    if len(working_packages) >= len(critical_packages) * 0.8:
        print("\n✓ Installation appears successful! You can now run tests.")
    else:
        print("\n⚠ Some critical packages are still missing. Manual intervention may be required.")

if __name__ == "__main__":
    main()