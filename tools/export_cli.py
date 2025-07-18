#!/usr/bin/env python3
"""
Command Line Interface for Academic Agent Export System

This CLI provides easy access to the comprehensive export system with support for
multiple formats, batch processing, and quality validation.

Usage:
    python export_cli.py export --input file.md --output ./exports --format pdf
    python export_cli.py batch --input-dir ./content --output-dir ./exports --format all
    python export_cli.py validate --file output.pdf
    python export_cli.py consolidate-images --content-dir ./content --output-dir ./images
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.academic.export_system import (
    ExportSystemTool, ExportConfig, ImageConsolidator, 
    ExportQualityValidator, AcademicTemplate
)

class ExportCLI:
    """Command line interface for the export system"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.export_tool = ExportSystemTool(self.base_dir)
        self.config_path = self.base_dir / "config" / "export_system_config.yaml"
        self.default_config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    return config_data.get('export_system', {}).get('defaults', {})
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        # Fallback defaults
        return {
            "output_format": "pdf",
            "template_name": "academic",
            "image_sizing": "medium",
            "include_diagrams": True,
            "consolidate_images": True,
            "resolve_references": True
        }
    
    def run(self):
        """Main CLI entry point"""
        parser = self._create_parser()
        args = parser.parse_args()
        
        if hasattr(args, 'func'):
            args.func(args)
        else:
            parser.print_help()
    
    def _create_parser(self):
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Academic Agent Export System CLI",
            prog="export_cli"
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export single file')
        self._add_export_arguments(export_parser)
        export_parser.set_defaults(func=self.cmd_export)
        
        # Batch export command
        batch_parser = subparsers.add_parser('batch', help='Batch export multiple files')
        self._add_batch_arguments(batch_parser)
        batch_parser.set_defaults(func=self.cmd_batch)
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate export quality')
        self._add_validate_arguments(validate_parser)
        validate_parser.set_defaults(func=self.cmd_validate)
        
        # Image consolidation command
        consolidate_parser = subparsers.add_parser('consolidate-images', 
                                                  help='Consolidate and optimize images')
        self._add_consolidate_arguments(consolidate_parser)
        consolidate_parser.set_defaults(func=self.cmd_consolidate_images)
        
        # List templates command
        templates_parser = subparsers.add_parser('templates', help='List available templates')
        templates_parser.set_defaults(func=self.cmd_list_templates)
        
        # Configuration command
        config_parser = subparsers.add_parser('config', help='Show configuration')
        config_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                                 help='Output format')
        config_parser.set_defaults(func=self.cmd_show_config)
        
        return parser
    
    def _add_export_arguments(self, parser):
        """Add export command arguments"""
        parser.add_argument('--input', '-i', required=True,
                          help='Input file path')
        parser.add_argument('--output', '-o', required=True,
                          help='Output directory')
        parser.add_argument('--format', '-f', 
                          choices=['pdf', 'html', 'docx', 'all'],
                          default=self.default_config.get('output_format', 'pdf'),
                          help='Export format')
        parser.add_argument('--template', '-t',
                          default=self.default_config.get('template_name', 'academic'),
                          help='Template name')
        parser.add_argument('--image-sizing',
                          choices=['small', 'medium', 'large', 'original'],
                          default=self.default_config.get('image_sizing', 'medium'),
                          help='Image sizing preset')
        parser.add_argument('--no-diagrams', action='store_true',
                          help='Exclude diagrams from export')
        parser.add_argument('--no-consolidation', action='store_true',
                          help='Skip image consolidation')
        parser.add_argument('--no-references', action='store_true',
                          help='Skip reference resolution')
        parser.add_argument('--metadata', '-m',
                          help='Metadata JSON file or string')
        parser.add_argument('--validate', action='store_true',
                          help='Validate export quality')
    
    def _add_batch_arguments(self, parser):
        """Add batch command arguments"""
        parser.add_argument('--input-dir', required=True,
                          help='Input directory containing files')
        parser.add_argument('--output-dir', required=True,
                          help='Output directory')
        parser.add_argument('--format', '-f',
                          choices=['pdf', 'html', 'docx', 'all'],
                          default=self.default_config.get('output_format', 'pdf'),
                          help='Export format')
        parser.add_argument('--template', '-t',
                          default=self.default_config.get('template_name', 'academic'),
                          help='Template name')
        parser.add_argument('--image-sizing',
                          choices=['small', 'medium', 'large', 'original'],
                          default=self.default_config.get('image_sizing', 'medium'),
                          help='Image sizing preset')
        parser.add_argument('--pattern', '-p', default='*.md',
                          help='File pattern to match')
        parser.add_argument('--recursive', '-r', action='store_true',
                          help='Process directories recursively')
        parser.add_argument('--no-diagrams', action='store_true',
                          help='Exclude diagrams from export')
        parser.add_argument('--no-consolidation', action='store_true',
                          help='Skip image consolidation')
        parser.add_argument('--validate', action='store_true',
                          help='Validate export quality')
        parser.add_argument('--max-concurrent', type=int, default=4,
                          help='Maximum concurrent exports')
    
    def _add_validate_arguments(self, parser):
        """Add validate command arguments"""
        parser.add_argument('--file', required=True,
                          help='File to validate')
        parser.add_argument('--original', 
                          help='Original source file for comparison')
        parser.add_argument('--format', choices=['json', 'yaml', 'text'],
                          default='text',
                          help='Output format')
        parser.add_argument('--detailed', action='store_true',
                          help='Show detailed validation results')
    
    def _add_consolidate_arguments(self, parser):
        """Add consolidate command arguments"""
        parser.add_argument('--content-dir', required=True,
                          help='Directory containing content with images')
        parser.add_argument('--output-dir', required=True,
                          help='Output directory for consolidated images')
        parser.add_argument('--sizing',
                          choices=['small', 'medium', 'large', 'original'],
                          default='medium',
                          help='Image sizing preset')
        parser.add_argument('--format', choices=['json', 'yaml', 'text'],
                          default='text',
                          help='Report format')
    
    def cmd_export(self, args):
        """Handle export command"""
        print(f"📄 Exporting {args.input} to {args.format.upper()} format...")
        
        try:
            # Parse metadata
            metadata = self._parse_metadata(args.metadata) if args.metadata else None
            
            # Create export configuration
            config = {
                "output_format": args.format,
                "template_name": args.template,
                "image_sizing": args.image_sizing,
                "include_diagrams": not args.no_diagrams,
                "consolidate_images": not args.no_consolidation,
                "resolve_references": not args.no_references,
                "metadata": metadata
            }
            
            # Perform export
            start_time = time.time()
            result = self.export_tool.forward(
                content_paths=[args.input],
                output_directory=args.output,
                export_config=config,
                batch_mode=False
            )
            
            # Display results
            self._display_export_results(result, time.time() - start_time)
            
            # Validate if requested
            if args.validate and result["summary"]["success"]:
                self._validate_results(result, args.input)
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
            sys.exit(1)
    
    def cmd_batch(self, args):
        """Handle batch export command"""
        print(f"📚 Batch exporting from {args.input_dir} to {args.format.upper()} format...")
        
        try:
            # Find input files
            input_dir = Path(args.input_dir)
            if args.recursive:
                input_files = list(input_dir.rglob(args.pattern))
            else:
                input_files = list(input_dir.glob(args.pattern))
            
            if not input_files:
                print(f"❌ No files found matching pattern '{args.pattern}'")
                sys.exit(1)
            
            print(f"📋 Found {len(input_files)} files to process")
            
            # Create export configuration
            config = {
                "output_format": args.format,
                "template_name": args.template,
                "image_sizing": args.image_sizing,
                "include_diagrams": not args.no_diagrams,
                "consolidate_images": not args.no_consolidation,
                "resolve_references": True
            }
            
            # Perform batch export
            start_time = time.time()
            result = self.export_tool.forward(
                content_paths=[str(f) for f in input_files],
                output_directory=args.output_dir,
                export_config=config,
                batch_mode=True
            )
            
            # Display results
            self._display_batch_results(result, time.time() - start_time)
            
            # Validate if requested
            if args.validate:
                self._validate_batch_results(result, input_files)
                
        except Exception as e:
            print(f"❌ Batch export failed: {e}")
            sys.exit(1)
    
    def cmd_validate(self, args):
        """Handle validate command"""
        print(f"🔍 Validating {args.file}...")
        
        try:
            file_path = Path(args.file)
            if not file_path.exists():
                print(f"❌ File not found: {args.file}")
                sys.exit(1)
            
            # Create mock export result for validation
            from agents.academic.export_system import ExportResult
            
            file_size = file_path.stat().st_size / (1024 * 1024)
            format_type = file_path.suffix[1:].lower()
            
            mock_result = ExportResult(
                success=True,
                output_files=[str(file_path)],
                format_type=format_type,
                file_size_mb=file_size,
                processing_time=0,
                validation_score=0,
                errors=[],
                warnings=[],
                metadata={}
            )
            
            # Load original content if provided
            original_content = ""
            if args.original and Path(args.original).exists():
                with open(args.original, 'r', encoding='utf-8') as f:
                    original_content = f.read()
            
            # Validate
            validator = ExportQualityValidator()
            validation_result = validator.validate_export(mock_result, original_content)
            
            # Display results
            self._display_validation_result(validation_result, args.format, args.detailed)
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            sys.exit(1)
    
    def cmd_consolidate_images(self, args):
        """Handle image consolidation command"""
        print(f"🖼️  Consolidating images from {args.content_dir}...")
        
        try:
            consolidator = ImageConsolidator(Path.cwd())
            
            # Find content files
            content_dir = Path(args.content_dir)
            content_files = list(content_dir.rglob("*.md"))
            
            if not content_files:
                print("❌ No markdown files found in content directory")
                sys.exit(1)
            
            # Consolidate images
            start_time = time.time()
            result = consolidator.consolidate_images(content_files, args.sizing)
            processing_time = time.time() - start_time
            
            # Display results
            self._display_consolidation_result(result, processing_time, args.format)
            
        except Exception as e:
            print(f"❌ Image consolidation failed: {e}")
            sys.exit(1)
    
    def cmd_list_templates(self, args):
        """Handle list templates command"""
        print("📋 Available Export Templates:")
        print()
        
        templates_info = {
            'academic': {
                'name': 'Academic Paper',
                'description': 'Professional academic template with proper formatting',
                'formats': ['PDF', 'HTML', 'DOCX'],
                'features': ['Citations', 'Table of Contents', 'Figure Captions']
            }
        }
        
        for template_id, info in templates_info.items():
            print(f"🎨 {info['name']} ({template_id})")
            print(f"   Description: {info['description']}")
            print(f"   Formats: {', '.join(info['formats'])}")
            print(f"   Features: {', '.join(info['features'])}")
            print()
    
    def cmd_show_config(self, args):
        """Handle show config command"""
        print(f"⚙️  Export System Configuration ({args.format.upper()}):")
        print()
        
        if args.format == 'json':
            print(json.dumps(self.default_config, indent=2))
        else:
            print(yaml.dump({'export_config': self.default_config}, default_flow_style=False))
    
    def _parse_metadata(self, metadata_input: str) -> Dict[str, Any]:
        """Parse metadata from file or JSON string"""
        try:
            # Try to parse as JSON string first
            return json.loads(metadata_input)
        except json.JSONDecodeError:
            # Try to load as file
            try:
                metadata_path = Path(metadata_input)
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        if metadata_path.suffix == '.yaml' or metadata_path.suffix == '.yml':
                            return yaml.safe_load(f)
                        else:
                            return json.load(f)
            except Exception:
                pass
        
        return {}
    
    def _display_export_results(self, result: Dict[str, Any], processing_time: float):
        """Display export results"""
        summary = result["summary"]
        
        if summary["success"]:
            print("✅ Export completed successfully!")
            print(f"   📄 Files generated: {summary['successful_exports']}")
            print(f"   📁 Output size: {summary['total_output_size_mb']:.2f} MB")
            print(f"   ⏱️  Processing time: {processing_time:.2f}s")
            
            # Show output files
            if summary.get("all_output_files"):
                print("   📋 Output files:")
                for file_path in summary["all_output_files"]:
                    file_size = Path(file_path).stat().st_size / (1024 * 1024)
                    print(f"      • {file_path} ({file_size:.2f} MB)")
        else:
            print("❌ Export failed!")
            if "error" in summary:
                print(f"   Error: {summary['error']}")
        
        # Show image consolidation results
        if result.get("consolidation_result"):
            consolidation = result["consolidation_result"]
            if consolidation["consolidated_images"]:
                print(f"   🖼️  Images processed: {len(consolidation['consolidated_images'])}")
                print(f"   💾 Image optimization: {consolidation['optimization_ratio']*100:.1f}%")
    
    def _display_batch_results(self, result: Dict[str, Any], processing_time: float):
        """Display batch export results"""
        summary = result["summary"]
        
        print(f"📊 Batch Export Results:")
        print(f"   ✅ Successful: {summary['successful_exports']}")
        print(f"   ❌ Failed: {summary['failed_exports']}")
        print(f"   📄 Total files: {summary['total_files_processed']}")
        print(f"   📁 Total output: {summary['total_output_size_mb']:.2f} MB")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        
        # Show formats
        if summary.get("output_formats"):
            print(f"   📋 Formats: {', '.join(summary['output_formats']).upper()}")
        
        # Show image consolidation
        if result.get("consolidation_result"):
            consolidation = result["consolidation_result"]
            if consolidation["consolidated_images"]:
                print(f"   🖼️  Images: {len(consolidation['consolidated_images'])} processed")
                print(f"   💾 Optimization: {consolidation['optimization_ratio']*100:.1f}%")
    
    def _display_validation_result(self, validation: Dict[str, Any], 
                                  format_type: str, detailed: bool):
        """Display validation results"""
        score = validation["overall_score"]
        passed = validation["passed"]
        
        print(f"📊 Validation Results:")
        print(f"   Overall Score: {score:.2f}/1.00")
        print(f"   Status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        if detailed:
            print(f"   📋 Detailed Scores:")
            for criterion, score in validation.get("criteria_scores", {}).items():
                print(f"      • {criterion.replace('_', ' ').title()}: {score:.2f}")
        
        if validation.get("suggestions"):
            print(f"   💡 Suggestions:")
            for suggestion in validation["suggestions"]:
                print(f"      • {suggestion}")
        
        if validation.get("issues"):
            print(f"   ⚠️  Issues:")
            for issue in validation["issues"]:
                print(f"      • {issue}")
        
        # Output in requested format
        if format_type == 'json':
            print("\n📄 JSON Output:")
            print(json.dumps(validation, indent=2))
        elif format_type == 'yaml':
            print("\n📄 YAML Output:")
            print(yaml.dump(validation, default_flow_style=False))
    
    def _display_consolidation_result(self, result, processing_time: float, 
                                    format_type: str):
        """Display image consolidation results"""
        print(f"📊 Image Consolidation Results:")
        print(f"   🖼️  Images processed: {len(result.consolidated_images)}")
        print(f"   📁 Total size: {result.total_size_mb:.2f} MB")
        print(f"   💾 Optimization: {result.optimization_ratio*100:.1f}%")
        print(f"   ⏱️  Processing time: {processing_time:.2f}s")
        
        if result.errors:
            print(f"   ⚠️  Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"      • {error}")
        
        # Output detailed results in requested format
        if format_type == 'json':
            print("\n📄 Detailed Results (JSON):")
            from dataclasses import asdict
            print(json.dumps(asdict(result), indent=2))
        elif format_type == 'yaml':
            print("\n📄 Detailed Results (YAML):")
            from dataclasses import asdict
            print(yaml.dump(asdict(result), default_flow_style=False))
    
    def _validate_results(self, result: Dict[str, Any], original_file: str):
        """Validate export results"""
        print("\n🔍 Validating export quality...")
        
        # Load original content
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception:
            original_content = ""
        
        # Validate each successful export
        validator = ExportQualityValidator()
        
        for export_result_data in result["export_results"]:
            if export_result_data["success"]:
                # Recreate ExportResult object
                from agents.academic.export_system import ExportResult
                export_result = ExportResult(**export_result_data)
                
                validation = validator.validate_export(export_result, original_content)
                
                print(f"   📄 {export_result.format_type.upper()}: {validation['overall_score']:.2f} "
                      f"({'✅ PASSED' if validation['passed'] else '❌ FAILED'})")
    
    def _validate_batch_results(self, result: Dict[str, Any], input_files: List[Path]):
        """Validate batch export results"""
        print("\n🔍 Validating batch export quality...")
        
        validator = ExportQualityValidator()
        passed_count = 0
        total_count = 0
        
        for i, export_result_data in enumerate(result["export_results"]):
            if export_result_data["success"] and i < len(input_files):
                try:
                    # Load original content
                    with open(input_files[i], 'r', encoding='utf-8') as f:
                        original_content = f.read()
                except Exception:
                    original_content = ""
                
                # Recreate ExportResult object
                from agents.academic.export_system import ExportResult
                export_result = ExportResult(**export_result_data)
                
                validation = validator.validate_export(export_result, original_content)
                
                total_count += 1
                if validation['passed']:
                    passed_count += 1
        
        print(f"   📊 Validation Summary: {passed_count}/{total_count} passed "
              f"({passed_count/total_count*100:.1f}%)" if total_count > 0 else "   📊 No files to validate")

def main():
    """Main CLI entry point"""
    cli = ExportCLI()
    cli.run()

if __name__ == "__main__":
    main()