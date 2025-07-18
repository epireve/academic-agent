#!/usr/bin/env python3
"""
Migration script to help transition from dual architecture to unified system.

This script:
1. Updates import statements
2. Migrates configuration files
3. Creates compatibility wrappers
4. Validates the migration
"""

import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml


class ImportMigrator(ast.NodeTransformer):
    """AST transformer to update import statements."""
    
    def __init__(self):
        self.import_mapping = {
            # V2 imports to unified
            "academic-agent-v2.src.core": "src.core",
            "academic-agent-v2.src.agents": "src.agents",
            "academic-agent-v2.src.processors": "src.processors",
            "academic-agent-v2.src.utils": "src.utils",
            
            # Legacy imports to unified
            "agents.academic.base_agent": "src.agents.legacy_adapter",
            "core.logging": "src.core.logging",
            "core.exceptions": "src.core.exceptions",
            "core.config_manager": "src.core.config_manager",
        }
    
    def visit_Import(self, node):
        """Transform import statements."""
        for alias in node.names:
            for old_path, new_path in self.import_mapping.items():
                if alias.name.startswith(old_path):
                    alias.name = alias.name.replace(old_path, new_path, 1)
        return node
    
    def visit_ImportFrom(self, node):
        """Transform from imports."""
        if node.module:
            for old_path, new_path in self.import_mapping.items():
                if node.module.startswith(old_path):
                    node.module = node.module.replace(old_path, new_path, 1)
        return node


def migrate_python_file(file_path: Path) -> bool:
    """Migrate a single Python file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        # Transform imports
        migrator = ImportMigrator()
        new_tree = migrator.visit(tree)
        
        # Generate new code
        new_content = ast.unparse(new_tree)
        
        # Backup original
        backup_path = file_path.with_suffix('.py.bak')
        shutil.copy2(file_path, backup_path)
        
        # Write updated file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Migrated: {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to migrate {file_path}: {e}")
        return False


def migrate_config_file(config_path: Path, output_path: Path) -> bool:
    """Migrate configuration file to unified format."""
    try:
        # Import unified config module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.config.unified_config import ConfigMigrator
        
        # Migrate configuration
        unified_config = ConfigMigrator.migrate_file(config_path, output_path)
        
        print(f"✓ Migrated config: {config_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to migrate config {config_path}: {e}")
        return False


def create_compatibility_wrapper(module_path: Path) -> bool:
    """Create a compatibility wrapper for a module."""
    try:
        wrapper_content = f'''"""
Compatibility wrapper for {module_path.name}

This module provides backward compatibility during migration.
"""

import warnings
from src.agents.legacy_adapter import create_legacy_adapter

warnings.warn(
    f"{{__name__}} is deprecated. Please use the unified architecture.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location
from src.agents import *
from src.core import *

# Create compatibility aliases
BaseAgent = create_legacy_adapter
'''
        
        wrapper_path = module_path.parent / f"{module_path.stem}_compat.py"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        print(f"✓ Created wrapper: {wrapper_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to create wrapper for {module_path}: {e}")
        return False


def validate_migration(project_root: Path) -> List[str]:
    """Validate the migration by checking for issues."""
    issues = []
    
    # Check for remaining old imports
    python_files = list(project_root.rglob("*.py"))
    old_import_patterns = [
        "academic-agent-v2.src",
        "agents.academic",
        "from agents import",
        "from academic-agent-v2 import",
    ]
    
    for py_file in python_files:
        if py_file.suffix == '.bak':
            continue
            
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            
            for pattern in old_import_patterns:
                if pattern in content:
                    issues.append(f"Old import found in {py_file}: {pattern}")
        except:
            pass
    
    # Check for missing unified modules
    required_modules = [
        "src/agents/base_agent.py",
        "src/core/logging.py",
        "src/core/exceptions.py",
        "src/processors/pdf_processor.py",
    ]
    
    for module in required_modules:
        module_path = project_root / module
        if not module_path.exists():
            issues.append(f"Missing unified module: {module}")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Academic Agent to unified architecture"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    parser.add_argument(
        "--migrate-imports",
        action="store_true",
        help="Migrate import statements in Python files"
    )
    parser.add_argument(
        "--migrate-config",
        type=Path,
        help="Migrate configuration file"
    )
    parser.add_argument(
        "--create-wrappers",
        action="store_true",
        help="Create compatibility wrappers"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate migration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    print(f"Academic Agent Migration Tool")
    print(f"Project root: {args.root.absolute()}")
    print()
    
    if args.migrate_imports:
        print("=== Migrating Imports ===")
        python_files = list(args.root.rglob("*.py"))
        
        if args.dry_run:
            print(f"Would migrate {len(python_files)} Python files")
        else:
            success = 0
            for py_file in python_files:
                if py_file.suffix == '.bak':
                    continue
                if migrate_python_file(py_file):
                    success += 1
            
            print(f"\nMigrated {success}/{len(python_files)} files")
    
    if args.migrate_config:
        print("\n=== Migrating Configuration ===")
        output_path = args.migrate_config.with_name("unified_config.yaml")
        
        if args.dry_run:
            print(f"Would migrate {args.migrate_config} -> {output_path}")
        else:
            migrate_config_file(args.migrate_config, output_path)
    
    if args.create_wrappers:
        print("\n=== Creating Compatibility Wrappers ===")
        modules_to_wrap = [
            args.root / "agents" / "academic" / "base_agent.py",
            args.root / "academic-agent-v2" / "src" / "agents" / "academic_agent.py",
        ]
        
        if args.dry_run:
            print(f"Would create {len(modules_to_wrap)} wrappers")
        else:
            for module in modules_to_wrap:
                if module.exists():
                    create_compatibility_wrapper(module)
    
    if args.validate:
        print("\n=== Validating Migration ===")
        issues = validate_migration(args.root)
        
        if issues:
            print(f"Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✓ No issues found!")
    
    print("\n=== Migration Summary ===")
    print("1. Unified architecture created in src/")
    print("2. Legacy adapters available for compatibility")
    print("3. Configuration migration supported")
    print("4. Run tests to verify functionality")
    print("\nNext steps:")
    print("- Update remaining imports manually")
    print("- Test agent functionality")
    print("- Remove .bak files after verification")
    print("- Archive old implementations")


if __name__ == "__main__":
    main()