#!/usr/bin/env python
"""
Test runner script for the academic-agent project.

This script provides a convenient way to run different types of tests
with various configuration options.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any


def setup_environment():
    """Set up the test environment."""
    # Create necessary directories
    dirs_to_create = [
        "logs",
        "tests/data",
        "processed/raw",
        "processed/markdown", 
        "processed/metadata",
        "processed/analysis",
        "processed/outlines",
        "processed/notes"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    print("✓ Test environment set up successfully")


def run_command(command: List[str], description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        command: Command to run as list of strings
        description: Description of what the command does
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        print(f"✓ {description} completed successfully")
        
        if result.stdout:
            print("Output:")
            print(result.stdout)
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Exit code: {e.returncode}")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
            
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
            
        return False


def run_unit_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run unit tests."""
    command = ["python", "-m", "pytest", "tests/unit/"]
    
    if verbose:
        command.append("-v")
        
    if coverage:
        command.extend([
            "--cov=agents",
            "--cov=tools", 
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    return run_command(command, "Running unit tests")


def run_integration_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run integration tests."""
    command = ["python", "-m", "pytest", "tests/integration/"]
    
    if verbose:
        command.append("-v")
        
    if coverage:
        command.extend([
            "--cov=agents",
            "--cov=tools",
            "--cov-append",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Skip slow tests by default in integration tests
    command.extend(["-m", "not slow"])
    
    return run_command(command, "Running integration tests")


def run_slow_tests(verbose: bool = False, coverage: bool = False) -> bool:
    """Run slow tests."""
    command = ["python", "-m", "pytest", "tests/"]
    
    if verbose:
        command.append("-v")
        
    if coverage:
        command.extend([
            "--cov=agents",
            "--cov=tools",
            "--cov-append",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Only run slow tests
    command.extend(["-m", "slow"])
    
    return run_command(command, "Running slow tests")


def run_specific_tests(test_pattern: str, verbose: bool = False) -> bool:
    """Run specific tests matching a pattern."""
    command = ["python", "-m", "pytest", "-k", test_pattern]
    
    if verbose:
        command.append("-v")
    
    return run_command(command, f"Running tests matching pattern: {test_pattern}")


def run_linting() -> bool:
    """Run code linting checks."""
    success = True
    
    # Black formatting check
    if not run_command(
        ["black", "--check", "--diff", "agents/", "tools/", "tests/"],
        "Checking code formatting with Black"
    ):
        success = False
    
    # Flake8 linting
    if not run_command(
        ["flake8", "agents/", "tools/", "tests/", "--count", "--statistics"],
        "Running Flake8 linting"
    ):
        success = False
    
    return success


def run_type_checking() -> bool:
    """Run type checking with MyPy."""
    return run_command(
        ["mypy", "agents/", "tools/", "--ignore-missing-imports"],
        "Running MyPy type checking"
    )


def run_security_checks() -> bool:
    """Run security checks."""
    success = True
    
    # Bandit security linting
    if not run_command(
        ["bandit", "-r", "agents/", "tools/", "-f", "txt"],
        "Running Bandit security checks"
    ):
        success = False
    
    # Safety dependency check
    if not run_command(
        ["safety", "check"],
        "Running Safety dependency checks"
    ):
        success = False
    
    return success


def generate_test_report(results: Dict[str, bool]) -> str:
    """Generate a test report."""
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    report = f"""
Test Report
===========

Total test suites: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Success rate: {(passed_tests/total_tests)*100:.1f}%

Detailed Results:
"""
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        report += f"  {status} {test_name}\n"
    
    if failed_tests > 0:
        report += f"\n{failed_tests} test suite(s) failed. Please check the output above for details."
    else:
        report += "\nAll test suites passed! 🎉"
    
    return report


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for the academic-agent project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all                    # Run all tests
  python run_tests.py --unit --coverage        # Run unit tests with coverage
  python run_tests.py --integration -v         # Run integration tests verbosely
  python run_tests.py --pattern "test_pdf"     # Run tests matching pattern
  python run_tests.py --lint --type-check      # Run linting and type checking
  python run_tests.py --security               # Run security checks
        """
    )
    
    # Test selection options
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--slow", action="store_true", help="Run slow tests")
    parser.add_argument("--pattern", help="Run tests matching pattern")
    
    # Code quality options
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--security", action="store_true", help="Run security checks")
    
    # Test options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--no-setup", action="store_true", help="Skip environment setup")
    
    args = parser.parse_args()
    
    # Set up environment unless skipped
    if not args.no_setup:
        setup_environment()
    
    # Track results
    results = {}
    
    # If no specific tests selected, default to unit tests
    if not any([args.all, args.unit, args.integration, args.slow, args.pattern, 
                args.lint, args.type_check, args.security]):
        args.unit = True
    
    # Run selected tests
    if args.all:
        results["Unit Tests"] = run_unit_tests(args.verbose, args.coverage)
        results["Integration Tests"] = run_integration_tests(args.verbose, args.coverage)
        results["Linting"] = run_linting()
        results["Type Checking"] = run_type_checking()
    
    if args.unit:
        results["Unit Tests"] = run_unit_tests(args.verbose, args.coverage)
    
    if args.integration:
        results["Integration Tests"] = run_integration_tests(args.verbose, args.coverage)
    
    if args.slow:
        results["Slow Tests"] = run_slow_tests(args.verbose, args.coverage)
    
    if args.pattern:
        results[f"Pattern Tests ({args.pattern})"] = run_specific_tests(args.pattern, args.verbose)
    
    if args.lint:
        results["Linting"] = run_linting()
    
    if args.type_check:
        results["Type Checking"] = run_type_checking()
    
    if args.security:
        results["Security Checks"] = run_security_checks()
    
    # Generate and display report
    if results:
        print("\n" + "="*50)
        print(generate_test_report(results))
        
        # Exit with error code if any tests failed
        if any(not result for result in results.values()):
            sys.exit(1)
        else:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
    else:
        print("No tests were run. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()