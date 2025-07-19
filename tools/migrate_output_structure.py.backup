#!/usr/bin/env python3
"""
Output Structure Migration Utility

This tool helps migrate from the current hardcoded output structure
to the new centralized output management system.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.output_manager import (
    OutputManager, 
    OutputConfig, 
    OutputCategory, 
    ContentType,
    get_output_manager,
    initialize_output_manager
)
from src.core.logging import get_logger


def analyze_current_structure(project_root: Path) -> Dict[str, Any]:
    """Analyze the current output structure and identify migration needs."""
    analysis = {
        "existing_directories": [],
        "file_counts": {},
        "size_analysis": {},
        "hardcoded_paths_found": [],
        "recommendations": []
    }
    
    logger = get_logger("migration_analysis")
    
    # Check for existing output directories
    potential_output_dirs = [
        "output", "outputs", "processed", "markdown", "raw", 
        "metadata", "logs", "results", "generated"
    ]
    
    for dir_name in potential_output_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            analysis["existing_directories"].append(str(dir_path))
            
            # Count files and calculate size
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            
            analysis["file_counts"][dir_name] = file_count
            analysis["size_analysis"][dir_name] = {
                "file_count": file_count,
                "total_size_mb": total_size / (1024 * 1024),
                "subdirectories": [d.name for d in dir_path.iterdir() if d.is_dir()]
            }
    
    # Scan for hardcoded paths in Python files
    logger.info("Scanning for hardcoded paths in source code...")
    hardcoded_patterns = [
        "/output/", "/outputs/", "/processed/", "/markdown/",
        "output/sra/", "outputs/", project_root.name + "/output"
    ]
    
    for py_file in project_root.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in hardcoded_patterns:
                    if pattern in content:
                        analysis["hardcoded_paths_found"].append({
                            "file": str(py_file.relative_to(project_root)),
                            "pattern": pattern,
                            "line_count": content.count(pattern)
                        })
        except Exception as e:
            logger.warning(f"Could not scan {py_file}: {e}")
    
    # Generate recommendations
    if analysis["existing_directories"]:
        analysis["recommendations"].append(
            "Multiple output directories found. Consider consolidating with migration."
        )
    
    if analysis["hardcoded_paths_found"]:
        unique_files = len(set(item["file"] for item in analysis["hardcoded_paths_found"]))
        analysis["recommendations"].append(
            f"Found hardcoded paths in {unique_files} files. Use path replacement tool."
        )
    
    total_files = sum(analysis["file_counts"].values())
    if total_files > 1000:
        analysis["recommendations"].append(
            f"Large number of files ({total_files}) found. Consider batch migration."
        )
    
    return analysis


def create_migration_plan(analysis: Dict[str, Any], output_manager: OutputManager) -> Dict[str, Any]:
    """Create a detailed migration plan based on analysis."""
    plan = {
        "migration_steps": [],
        "directory_mappings": {},
        "file_operations": [],
        "estimated_time_minutes": 0,
        "backup_required": True
    }
    
    # Map existing directories to new structure
    directory_mappings = {
        "output/sra/ai_enhanced_study_notes": (OutputCategory.FINAL, ContentType.STUDY_NOTES),
        "output/sra/alignment_analysis": (OutputCategory.ANALYSIS, ContentType.ALIGNMENT),
        "output/sra/enhanced_integrated_notes": (OutputCategory.PROCESSED, ContentType.ENHANCED),
        "output/sra/integrated_notes": (OutputCategory.PROCESSED, ContentType.INTEGRATED),
        "output/sra/lectures": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
        "output/sra/mermaid_diagrams": (OutputCategory.ASSETS, ContentType.DIAGRAMS),
        "output/sra/notes": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
        "output/sra/resolved_content": (OutputCategory.PROCESSED, ContentType.RESOLVED),
        "output/sra/textbook": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
        "output/sra/transcripts": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
        "processed": (OutputCategory.PROCESSED, None),
        "markdown": (OutputCategory.PROCESSED, ContentType.MARKDOWN),
        "metadata": (OutputCategory.METADATA, None),
        "logs": (OutputCategory.LOGS, None),
    }
    
    plan["directory_mappings"] = directory_mappings
    
    # Create migration steps
    plan["migration_steps"] = [
        "1. Create backup of existing output directories",
        "2. Initialize new output directory structure", 
        "3. Migrate files to new locations based on content type",
        "4. Update configuration files to use new paths",
        "5. Replace hardcoded paths in source code",
        "6. Validate migration and test functionality",
        "7. Clean up old directory structure (after validation)"
    ]
    
    # Estimate time based on file count
    total_files = sum(analysis["file_counts"].values())
    plan["estimated_time_minutes"] = max(5, total_files // 100)  # Rough estimate
    
    return plan


def execute_migration(
    output_manager: OutputManager, 
    migration_plan: Dict[str, Any], 
    dry_run: bool = True
) -> Dict[str, Any]:
    """Execute the migration plan."""
    logger = get_logger("migration_execution")
    
    logger.info(f"Executing migration (dry_run={dry_run})")
    
    # Use the output manager's migration functionality
    migration_result = output_manager.migrate_legacy_outputs(dry_run=dry_run)
    
    return migration_result


def generate_path_replacement_script(
    analysis: Dict[str, Any], 
    output_path: Path
) -> Path:
    """Generate a script to replace hardcoded paths in source files."""
    script_content = """#!/usr/bin/env python3
\"\"\"
Generated script to replace hardcoded paths with OutputManager calls.
\"\"\"

import re
import sys
from pathlib import Path

def replace_hardcoded_paths(file_path: Path) -> bool:
    \"\"\"Replace hardcoded paths in a Python file.\"\"\"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replacement patterns
        replacements = {
"""
    
    # Add replacement patterns based on found hardcoded paths
    for item in analysis.get("hardcoded_paths_found", []):
        pattern = item["pattern"]
        if "/output/sra/ai_enhanced_study_notes" in pattern:
            script_content += f"""            r'{pattern}': 'get_final_output_path(ContentType.STUDY_NOTES)',\n"""
        elif "/output/sra/alignment_analysis" in pattern:
            script_content += f"""            r'{pattern}': 'get_analysis_output_path(ContentType.ALIGNMENT)',\n"""
        # Add more patterns as needed
    
    script_content += """        }
        
        # Apply replacements
        for old_pattern, new_pattern in replacements.items():
            content = content.replace(old_pattern, new_pattern)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    \"\"\"Main function to process all files.\"\"\"
    project_root = Path(__file__).parent.parent
    
    # Add necessary imports to files that need them
    import_statement = '''
from src.core.output_manager import (
    get_final_output_path, 
    get_processed_output_path,
    get_analysis_output_path,
    ContentType
)
'''
    
    files_to_process = [
"""
    
    # Add specific files that need updating
    for item in analysis.get("hardcoded_paths_found", []):
        script_content += f"""        "{item['file']}",\n"""
    
    script_content += """    ]
    
    updated_count = 0
    for file_path in files_to_process:
        full_path = project_root / file_path
        if full_path.exists() and replace_hardcoded_paths(full_path):
            updated_count += 1
    
    print(f"Updated {updated_count} files")

if __name__ == "__main__":
    main()
"""
    
    script_path = output_path / "replace_hardcoded_paths.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    return script_path


def main():
    """Main function for the migration utility."""
    parser = argparse.ArgumentParser(description="Migrate academic agent output structure")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze current structure, don't migrate")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show what would be migrated without actually moving files")
    parser.add_argument("--execute", action="store_true",
                       help="Actually execute the migration (overrides dry-run)")
    parser.add_argument("--output-report", type=Path,
                       help="Save analysis and migration report to file")
    
    args = parser.parse_args()
    
    # Initialize logging
    logger = get_logger("migration_tool")
    logger.info(f"Starting migration utility for {args.project_root}")
    
    # Initialize output manager
    initialize_output_manager(args.project_root)
    output_manager = get_output_manager()
    
    # Analyze current structure
    print("🔍 Analyzing current output structure...")
    analysis = analyze_current_structure(args.project_root)
    
    print(f"\n📊 Analysis Results:")
    print(f"   Existing directories: {len(analysis['existing_directories'])}")
    print(f"   Total files: {sum(analysis['file_counts'].values())}")
    print(f"   Hardcoded paths found: {len(analysis['hardcoded_paths_found'])}")
    
    if analysis["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"   • {rec}")
    
    if args.analyze_only:
        print("\n✅ Analysis complete (analyze-only mode)")
        return
    
    # Create migration plan
    print("\n📋 Creating migration plan...")
    migration_plan = create_migration_plan(analysis, output_manager)
    
    print(f"\n📋 Migration Plan:")
    print(f"   Estimated time: {migration_plan['estimated_time_minutes']} minutes")
    print(f"   Backup required: {migration_plan['backup_required']}")
    print(f"   Steps: {len(migration_plan['migration_steps'])}")
    
    # Execute migration
    dry_run = not args.execute
    if dry_run:
        print("\n🧪 Executing migration (DRY RUN)...")
    else:
        print("\n🚀 Executing migration (LIVE)...")
        response = input("Are you sure? This will move files. Type 'yes' to continue: ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            return
    
    migration_result = execute_migration(output_manager, migration_plan, dry_run)
    
    print(f"\n📊 Migration Results:")
    print(f"   Files to migrate: {migration_result['summary']['total_files']}")
    print(f"   Total size: {migration_result['summary']['total_size_mb']:.2f} MB")
    print(f"   Errors: {migration_result['summary']['errors_count']}")
    
    # Generate path replacement script
    if analysis["hardcoded_paths_found"]:
        print("\n🔧 Generating path replacement script...")
        tools_dir = args.project_root / "tools"
        tools_dir.mkdir(exist_ok=True)
        script_path = generate_path_replacement_script(analysis, tools_dir)
        print(f"   Generated: {script_path}")
        print(f"   Run this script to update hardcoded paths in source code.")
    
    # Save report if requested
    if args.output_report:
        report_data = {
            "analysis": analysis,
            "migration_plan": migration_plan,
            "migration_result": migration_result,
            "timestamp": output_manager.get_output_summary()["last_updated"]
        }
        
        with open(args.output_report, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Report saved to: {args.output_report}")
    
    print("\n✅ Migration utility complete!")
    
    if dry_run:
        print("\n💡 To execute the migration for real, run with --execute flag")


if __name__ == "__main__":
    main()