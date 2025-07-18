#!/usr/bin/env python
"""
CMS Setup Script

Quick setup script for the Academic Agent Content Management System.
This script initializes the CMS, creates necessary directories, and
performs basic configuration.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add the agents directory to the path
sys.path.append(str(Path(__file__).parent / "agents"))

from academic.content_management_system import (
    ContentManagementSystem, CourseInfo, ContentType
)
from academic.cms_integration import CMSIntegration


def create_directory_structure(base_path: Path) -> bool:
    """Create the standard CMS directory structure"""
    try:
        directories = [
            "cms",
            "cms/storage", 
            "cms/versions",
            "cms/search_index",
            "cms/reports",
            "cms/backups",
            "logs",
            "config"
        ]
        
        for directory in directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating directory structure: {str(e)}")
        return False


def initialize_cms(base_path: str) -> ContentManagementSystem:
    """Initialize the CMS with default configuration"""
    try:
        cms = ContentManagementSystem(base_path)
        print(f"✅ CMS initialized at: {base_path}")
        return cms
        
    except Exception as e:
        print(f"❌ Error initializing CMS: {str(e)}")
        return None


def create_sample_course(cms: ContentManagementSystem) -> bool:
    """Create a sample course for testing"""
    try:
        sample_course = CourseInfo(
            course_id="sample_course_001",
            course_name="Sample Academic Course",
            course_code="SAMPLE101",
            academic_year=str(datetime.now().year),
            semester="Fall",
            instructor="Dr. Sample",
            department="Computer Science",
            description="A sample course for testing the CMS functionality",
            credits=3,
            learning_outcomes=[
                "Understand content management concepts",
                "Learn to use the academic agent CMS",
                "Practice with content organization"
            ],
            assessment_methods=[
                "Assignments",
                "Projects", 
                "Final Exam"
            ]
        )
        
        success = cms.create_course(sample_course)
        if success:
            print(f"✅ Sample course created: {sample_course.course_id}")
            return True
        else:
            print(f"❌ Failed to create sample course")
            return False
            
    except Exception as e:
        print(f"❌ Error creating sample course: {str(e)}")
        return False


def setup_content_types(cms: ContentManagementSystem) -> bool:
    """Set up content type directories"""
    try:
        content_types = [
            ContentType.LECTURE,
            ContentType.TRANSCRIPT, 
            ContentType.NOTES,
            ContentType.TEXTBOOK,
            ContentType.ASSIGNMENT,
            ContentType.TUTORIAL,
            ContentType.EXAM,
            ContentType.IMAGE,
            ContentType.DIAGRAM
        ]
        
        base_storage = cms.storage_path / "sample_course_001"
        
        for content_type in content_types:
            type_dir = base_storage / content_type.value
            type_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created content type directory: {type_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up content types: {str(e)}")
        return False


def create_readme_files(base_path: Path) -> bool:
    """Create helpful README files"""
    try:
        # Main CMS README
        cms_readme = base_path / "cms" / "README.md"
        with open(cms_readme, 'w', encoding='utf-8') as f:
            f.write("""# Academic Agent CMS

This directory contains the Content Management System for the Academic Agent.

## Directory Structure

- `storage/` - Content file storage organized by course and type
- `versions/` - Version control storage for content revisions
- `search_index/` - Search index files and metadata
- `reports/` - Generated analytics and reports
- `backups/` - Database and content backups
- `content_database.db` - Main SQLite database

## Usage

Use the CLI tool to manage content:

```bash
# List courses
python tools/cms_cli.py --base-path . list-courses

# Import content
python tools/cms_cli.py --base-path . import-content --course-id COURSE_ID --paths /path/to/content

# Search content
python tools/cms_cli.py --base-path . search --query "search terms" --course-id COURSE_ID

# Generate analytics
python tools/cms_cli.py --base-path . analytics --course-id COURSE_ID --output report.json
```

For more information, see CMS_IMPLEMENTATION.md in the project root.
""")
        
        # Storage README
        storage_readme = base_path / "cms" / "storage" / "README.md"
        with open(storage_readme, 'w', encoding='utf-8') as f:
            f.write("""# Content Storage

This directory contains the actual content files organized by course and content type.

## Organization

```
storage/
├── course_id_1/
│   ├── lectures/
│   ├── transcripts/
│   ├── notes/
│   ├── textbook/
│   ├── assignments/
│   └── ...
└── course_id_2/
    └── ...
```

Each course has its own directory with subdirectories for different content types.
Files are stored with their original names plus content ID prefixes for uniqueness.
""")
        
        print("✅ Created README files")
        return True
        
    except Exception as e:
        print(f"❌ Error creating README files: {str(e)}")
        return False


def run_basic_tests(cms: ContentManagementSystem) -> bool:
    """Run basic functionality tests"""
    try:
        print("\n🧪 Running basic functionality tests...")
        
        # Test database connectivity
        courses = cms.list_courses()
        print(f"✅ Database connectivity test passed ({len(courses)} courses found)")
        
        # Test search index
        success = cms.build_search_index()
        if success:
            print("✅ Search index test passed")
        else:
            print("⚠️ Search index test had issues")
        
        # Test analytics
        report = cms.generate_analytics_report()
        if report:
            print("✅ Analytics generation test passed")
        else:
            print("⚠️ Analytics generation test had issues")
        
        # Test CMS status
        try:
            quality_score = cms.check_quality(None)
            print(f"✅ CMS quality check passed (score: {quality_score:.2f})")
        except:
            print("⚠️ CMS quality check had issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during basic tests: {str(e)}")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup script for Academic Agent CMS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_cms.py --base-path ./academic-agent --full-setup
  python setup_cms.py --base-path /path/to/cms --init-only
  python setup_cms.py --base-path ./cms --with-sample-data --run-tests
        """
    )
    
    parser.add_argument("--base-path", required=True, 
                       help="Base path for CMS installation")
    parser.add_argument("--init-only", action="store_true",
                       help="Only initialize CMS, skip sample data")
    parser.add_argument("--full-setup", action="store_true",
                       help="Full setup including sample course and content types")
    parser.add_argument("--with-sample-data", action="store_true",
                       help="Include sample course and data")
    parser.add_argument("--run-tests", action="store_true",
                       help="Run basic functionality tests after setup")
    parser.add_argument("--force", action="store_true",
                       help="Force setup even if CMS already exists")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path).resolve()
    
    print("🚀 Academic Agent CMS Setup")
    print("=" * 50)
    print(f"Base path: {base_path}")
    print(f"Setup type: {'Full' if args.full_setup else 'Basic'}")
    print()
    
    # Check if CMS already exists
    cms_path = base_path / "cms"
    db_path = cms_path / "content_database.db"
    
    if db_path.exists() and not args.force:
        print("⚠️ CMS database already exists!")
        print("Use --force to overwrite existing installation")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled")
            return 1
    
    success_steps = []
    
    try:
        # Step 1: Create directory structure
        print("📁 Creating directory structure...")
        if create_directory_structure(base_path):
            success_steps.append("Directory structure")
        
        # Step 2: Initialize CMS
        print("🔧 Initializing CMS...")
        cms = initialize_cms(str(base_path))
        if cms:
            success_steps.append("CMS initialization")
        else:
            print("❌ Failed to initialize CMS")
            return 1
        
        # Step 3: Create sample course (if requested)
        if args.full_setup or args.with_sample_data:
            print("📚 Creating sample course...")
            if create_sample_course(cms):
                success_steps.append("Sample course creation")
            
            print("📂 Setting up content type directories...")
            if setup_content_types(cms):
                success_steps.append("Content type setup")
        
        # Step 4: Create README files
        print("📄 Creating documentation...")
        if create_readme_files(base_path):
            success_steps.append("Documentation creation")
        
        # Step 5: Run tests (if requested)
        if args.run_tests:
            if run_basic_tests(cms):
                success_steps.append("Basic functionality tests")
        
        # Step 6: Generate initial analytics (if we have data)
        courses = cms.list_courses()
        if courses:
            print("📊 Generating initial analytics report...")
            try:
                report = cms.generate_analytics_report()
                if report:
                    report_path = base_path / "cms" / "reports" / "initial_setup_report.json"
                    with open(report_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, default=str)
                    print(f"✅ Initial analytics report saved: {report_path}")
                    success_steps.append("Initial analytics")
            except Exception as e:
                print(f"⚠️ Analytics generation had issues: {str(e)}")
        
        # Cleanup
        cms.shutdown()
        
        # Success summary
        print("\n" + "=" * 50)
        print("🎉 CMS Setup Complete!")
        print("=" * 50)
        
        print("✅ Completed steps:")
        for step in success_steps:
            print(f"   - {step}")
        
        print(f"\n📍 CMS Location: {cms_path}")
        print(f"🗄️ Database: {db_path}")
        
        print("\n🚀 Next Steps:")
        print("1. Review the configuration in config/cms_config.yaml")
        print("2. Import your content using the CLI tool:")
        print(f"   python tools/cms_cli.py --base-path {base_path} import-content --course-id COURSE_ID --paths /path/to/content")
        print("3. Explore the CMS using the CLI:")
        print(f"   python tools/cms_cli.py --base-path {base_path} list-courses")
        print("4. Read CMS_IMPLEMENTATION.md for detailed documentation")
        
        if courses:
            print(f"\n📚 Sample course available: {courses[0].course_id}")
            print("   Use this course ID for testing import and other operations")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())