# Makefile for academic-agent project

.PHONY: help test test-unit test-integration test-slow test-all lint type-check security coverage clean setup install

# Default target
help:
	@echo "Academic Agent Testing Commands"
	@echo "==============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup        - Set up the development environment"
	@echo "  install      - Install project dependencies"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test         - Run unit tests (default)"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-slow    - Run slow/performance tests only"
	@echo "  test-all     - Run all tests with coverage"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint         - Run linting checks (Black, Flake8)"
	@echo "  type-check   - Run type checking (MyPy)"
	@echo "  security     - Run security checks (Bandit, Safety)"
	@echo "  format       - Format code with Black"
	@echo ""
	@echo "Coverage Commands:"
	@echo "  coverage     - Generate coverage report"
	@echo "  coverage-html - Generate HTML coverage report"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean        - Clean up generated files"
	@echo "  docs         - Generate documentation"

# Setup and installation
setup:
	@echo "Setting up development environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	mkdir -p logs tests/data
	mkdir -p processed/{raw,markdown,metadata,analysis,outlines,notes}
	@echo "✓ Development environment setup complete"

install:
	pip install -r requirements.txt

# Testing commands
test:
	python run_tests.py --unit -v

test-unit:
	python run_tests.py --unit -v --coverage

test-integration:
	python run_tests.py --integration -v

test-slow:
	python run_tests.py --slow -v

test-all:
	python run_tests.py --all --coverage

# Code quality commands
lint:
	python run_tests.py --lint

type-check:
	python run_tests.py --type-check

security:
	python run_tests.py --security

format:
	black agents/ tools/ tests/
	@echo "✓ Code formatted with Black"

# Quick quality check (lint + type-check)
quality: lint type-check
	@echo "✓ Code quality checks completed"

# Coverage commands
coverage:
	pytest tests/unit/ --cov=agents --cov=tools --cov-report=term-missing --cov-report=xml

coverage-html:
	pytest tests/unit/ tests/integration/ --cov=agents --cov=tools --cov-report=html --cov-report=term-missing
	@echo "✓ HTML coverage report generated in htmlcov/"

# Utility commands
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf bandit-report.json
	rm -rf safety-report.json
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✓ Cleaned up generated files"

docs:
	@echo "Generating documentation..."
	sphinx-apidoc -o docs/api agents/ tools/
	sphinx-build -b html docs/ docs/_build/html
	@echo "✓ Documentation generated in docs/_build/html/"

# Development workflow commands
dev-check: format lint type-check test-unit
	@echo "✓ Development checks completed"

ci-check: lint type-check security test-all
	@echo "✓ CI checks completed"

# Quick commands for common patterns
test-pdf:
	python run_tests.py --pattern "pdf" -v

test-agent:
	python run_tests.py --pattern "agent" -v

test-quality:
	python run_tests.py --pattern "quality" -v

# Performance testing
perf:
	pytest tests/ -m "performance" --benchmark-only

# Dependency management
deps-check:
	pip check
	safety check

deps-update:
	pip list --outdated
	@echo "Run 'pip install --upgrade <package>' to update packages"

# Docker commands (if applicable)
docker-build:
	docker build -t academic-agent .

docker-test:
	docker run --rm academic-agent python run_tests.py --all

# Git hooks setup
hooks:
	@echo "Setting up git hooks..."
	echo "#!/bin/sh\nmake dev-check" > .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "✓ Git pre-commit hook installed"

# Environment info
env-info:
	@echo "Environment Information:"
	@echo "========================"
	python --version
	pip --version
	@echo ""
	@echo "Installed packages:"
	pip list | grep -E "(pytest|black|flake8|mypy|bandit|safety)"