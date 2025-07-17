#!/usr/bin/env python
"""
Test verification script for the academic-agent testing framework.

This script verifies that the testing framework is properly set up
without requiring pytest installation.
"""

import sys
import os
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False


def check_directory_structure() -> bool:
    """Check if the test directory structure is correct."""
    print("Checking test directory structure...")
    
    required_dirs = [
        "tests",
        "tests/unit", 
        "tests/integration",
        "tests/data"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist


def check_test_files() -> bool:
    """Check if test files are present."""
    print("\nChecking test files...")
    
    test_files = [
        ("tests/__init__.py", "Test package init"),
        ("tests/conftest.py", "Pytest configuration"),
        ("tests/utils.py", "Test utilities"),
        ("tests/unit/__init__.py", "Unit tests package init"),
        ("tests/unit/test_base_agent.py", "Base agent unit tests"),
        ("tests/unit/test_academic_agent.py", "Academic agent unit tests"),
        ("tests/unit/test_pdf_processing.py", "PDF processing unit tests"),
        ("tests/integration/__init__.py", "Integration tests package init"),
        ("tests/integration/test_agent_communication.py", "Agent communication integration tests"),
        ("tests/integration/test_end_to_end_workflow.py", "End-to-end workflow integration tests")
    ]
    
    all_exist = True
    for filepath, description in test_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def check_configuration_files() -> bool:
    """Check if configuration files are present."""
    print("\nChecking configuration files...")
    
    config_files = [
        ("pytest.ini", "Pytest configuration"),
        ("tox.ini", "Tox configuration"),
        (".github/workflows/tests.yml", "GitHub Actions workflow"),
        ("Makefile", "Make configuration"),
        ("run_tests.py", "Test runner script"),
        ("TESTING.md", "Testing documentation")
    ]
    
    all_exist = True
    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def validate_test_file_content() -> bool:
    """Validate that test files have proper content."""
    print("\nValidating test file content...")
    
    # Check that test files contain expected imports and classes
    test_validations = [
        ("tests/conftest.py", ["pytest", "fixture", "def"]),
        ("tests/utils.py", ["class", "def", "import"]),
        ("tests/unit/test_base_agent.py", ["class Test", "def test_", "import pytest"]),
        ("tests/integration/test_agent_communication.py", ["class Test", "def test_", "import pytest"])
    ]
    
    all_valid = True
    for filepath, expected_content in test_validations:
        if Path(filepath).exists():
            content = Path(filepath).read_text()
            missing_content = []
            
            for expected in expected_content:
                if expected not in content:
                    missing_content.append(expected)
            
            if missing_content:
                print(f"❌ {filepath}: Missing expected content: {missing_content}")
                all_valid = False
            else:
                print(f"✓ {filepath}: Contains expected content")
        else:
            print(f"❌ {filepath}: File not found")
            all_valid = False
    
    return all_valid


def check_requirements() -> bool:
    """Check if requirements.txt includes testing dependencies."""
    print("\nChecking requirements.txt for testing dependencies...")
    
    required_packages = [
        "pytest",
        "pytest-cov",
        "pytest-mock",
        "black",
        "flake8",
        "mypy",
        "bandit",
        "safety"
    ]
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    content = Path("requirements.txt").read_text().lower()
    missing_packages = []
    
    for package in required_packages:
        if package not in content:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing testing packages in requirements.txt: {missing_packages}")
        return False
    else:
        print("✓ requirements.txt contains all required testing packages")
        return True


def run_syntax_check() -> bool:
    """Run basic syntax check on test files."""
    print("\nRunning syntax check on test files...")
    
    test_files = [
        "tests/conftest.py",
        "tests/utils.py", 
        "tests/unit/test_base_agent.py",
        "tests/unit/test_academic_agent.py",
        "tests/unit/test_pdf_processing.py",
        "tests/integration/test_agent_communication.py",
        "tests/integration/test_end_to_end_workflow.py"
    ]
    
    all_valid = True
    for filepath in test_files:
        if Path(filepath).exists():
            try:
                with open(filepath, 'r') as f:
                    compile(f.read(), filepath, 'exec')
                print(f"✓ Syntax valid: {filepath}")
            except SyntaxError as e:
                print(f"❌ Syntax error in {filepath}: {e}")
                all_valid = False
        else:
            print(f"❌ File not found: {filepath}")
            all_valid = False
    
    return all_valid


def main():
    """Main verification function."""
    print("Academic Agent Testing Framework Verification")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Test Files", check_test_files),
        ("Configuration Files", check_configuration_files),
        ("Test File Content", validate_test_file_content),
        ("Requirements", check_requirements),
        ("Syntax Check", run_syntax_check)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{check_name}")
        print("-" * len(check_name))
        results[check_name] = check_func()
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\nOverall: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 All verification checks passed!")
        print("The testing framework is properly set up.")
        return True
    else:
        print(f"\n⚠️  {total_checks - passed_checks} verification check(s) failed.")
        print("Please address the issues above before running tests.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)