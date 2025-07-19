#!/usr/bin/env python3
"""
Export System Integration for Academic Agent

This module integrates the export system with the existing academic agent components,
providing seamless export capabilities for study notes, processed PDFs, and other
academic content.

Key Features:
- Integration with study notes generator
- Automatic export of processed content
- CMS integration for export tracking
- Quality assurance for exported documents
- Batch processing for large document sets
- Template management and customization
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

from .export_system import (
    ExportSystemTool, ExportConfig, ExportResult, 
    ImageConsolidator, ReferenceResolver, ExportQualityValidator
)
from .study_notes_generator import StudyNotesGeneratorTool
from ...src.agents.base_agent import BaseAgent


@dataclass
class IntegratedExportRequest:
    """Request for integrated export operation"""
    source_type: str  # 'study_notes', 'pdf_processed', 'markdown', 'directory'
    source_paths: List[str]
    output_directory: str
    export_formats: List[str] = None  # ['pdf', 'html', 'docx']
    template_name: str = 'academic'
    include_quality_validation: bool = True
    consolidate_images: bool = True
    generate_index: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class IntegratedExportResult:
    """Result of integrated export operation"""
    request_id: str
    success: bool
    export_results: List[ExportResult]
    quality_validation: Dict[str, Any]
    consolidation_result: Dict[str, Any]
    index_generated: bool
    processing_time: float
    errors: List[str]
    warnings: List[str]
    output_summary: Dict[str, Any]


class StudyNotesExportIntegration:
    """Integration between study notes generator and export system"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.study_notes_tool = StudyNotesGeneratorTool(base_dir)
        self.export_tool = ExportSystemTool(base_dir)
        
    def generate_and_export_study_notes(
        self, 
        content_path: str, 
        title: str, 
        subject: str,
        export_formats: List[str] = None,
        template_name: str = 'academic'
    ) -> Dict[str, Any]:
        """Generate study notes and export them to multiple formats"""
        
        export_formats = export_formats or ['pdf', 'html']
        
        try:
            # Step 1: Generate study notes
            study_notes_result = self.study_notes_tool.forward(
                content_path=content_path,
                title=title,
                subject=subject,
                output_formats=['markdown'],  # Generate markdown for export
                include_diagrams=True
            )
            
            if not study_notes_result['processing_stats']['success']:
                return {
                    'success': False,
                    'error': 'Study notes generation failed',
                    'study_notes_result': study_notes_result
                }
            
            # Step 2: Find generated markdown file
            markdown_files = [
                f for f in study_notes_result['output_files'] 
                if f.endswith('.md')
            ]
            
            if not markdown_files:
                return {
                    'success': False,
                    'error': 'No markdown file generated for export'
                }
            
            # Step 3: Export to requested formats
            export_results = []
            
            for format_type in export_formats:
                export_config = {
                    "output_format": format_type,
                    "template_name": template_name,
                    "include_diagrams": True,
                    "consolidate_images": True,
                    "resolve_references": True,
                    "metadata": {
                        "title": title,
                        "subject": subject,
                        "generated_from": "study_notes_generator",
                        "source_content": content_path
                    }
                }
                
                export_result = self.export_tool.forward(
                    content_paths=markdown_files,
                    output_directory=str(self.base_dir / "export" / "study_notes"),
                    export_config=export_config,
                    batch_mode=False
                )
                
                export_results.append({
                    'format': format_type,
                    'result': export_result
                })
            
            return {
                'success': True,
                'study_notes_result': study_notes_result,
                'export_results': export_results,
                'summary': self._generate_integration_summary(
                    study_notes_result, export_results
                )
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Integration failed: {str(e)}'
            }
    
    def _generate_integration_summary(
        self, 
        study_notes_result: Dict[str, Any],
        export_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of integrated operation"""
        
        successful_exports = sum(
            1 for result in export_results 
            if result['result']['summary']['success']
        )
        
        total_output_files = []
        for result in export_results:
            if result['result']['summary']['success']:
                total_output_files.extend(
                    result['result']['summary']['all_output_files']
                )
        
        return {
            'study_notes_generated': study_notes_result['processing_stats']['success'],
            'export_formats_successful': successful_exports,
            'total_export_formats': len(export_results),
            'total_output_files': len(total_output_files),
            'output_files': total_output_files,
            'study_notes_stats': study_notes_result['processing_stats'],
            'formats_processed': [r['format'] for r in export_results]
        }


class PDFProcessorExportIntegration:
    """Integration between PDF processor and export system"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.export_tool = ExportSystemTool(base_dir)
    
    def export_processed_pdfs(
        self,
        processed_content_dir: str,
        export_formats: List[str] = None,
        template_name: str = 'academic'
    ) -> Dict[str, Any]:
        """Export processed PDF content to multiple formats"""
        
        export_formats = export_formats or ['pdf', 'html']
        processed_dir = Path(processed_content_dir)
        
        if not processed_dir.exists():
            return {
                'success': False,
                'error': f'Processed content directory not found: {processed_content_dir}'
            }
        
        try:
            # Find processed markdown files
            markdown_files = list(processed_dir.rglob("*.md"))
            
            if not markdown_files:
                return {
                    'success': False,
                    'error': 'No processed markdown files found'
                }
            
            # Export each format
            export_results = []
            
            for format_type in export_formats:
                export_config = {
                    "output_format": format_type,
                    "template_name": template_name,
                    "include_diagrams": True,
                    "consolidate_images": True,
                    "resolve_references": True,
                    "metadata": {
                        "source_type": "pdf_processor",
                        "processing_date": datetime.now().isoformat()
                    }
                }
                
                export_result = self.export_tool.forward(
                    content_paths=[str(f) for f in markdown_files],
                    output_directory=str(self.base_dir / "export" / "processed_pdfs"),
                    export_config=export_config,
                    batch_mode=True
                )
                
                export_results.append({
                    'format': format_type,
                    'result': export_result
                })
            
            return {
                'success': True,
                'processed_files': len(markdown_files),
                'export_results': export_results,
                'summary': self._generate_pdf_export_summary(export_results)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'PDF export integration failed: {str(e)}'
            }
    
    def _generate_pdf_export_summary(self, export_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary for PDF export results"""
        
        total_successful = 0
        total_files = 0
        
        for result in export_results:
            if result['result']['summary']['success']:
                total_successful += result['result']['summary']['successful_exports']
            total_files += result['result']['summary']['total_files_processed']
        
        return {
            'total_files_processed': total_files,
            'successful_exports': total_successful,
            'formats_generated': [r['format'] for r in export_results],
            'success_rate': total_successful / total_files if total_files > 0 else 0
        }


class CMSExportIntegration:
    """Integration with Content Management System for export tracking"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.export_metadata_dir = base_dir / "cms" / "export_metadata"
        self.export_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def register_export_operation(
        self, 
        request: IntegratedExportRequest,
        result: IntegratedExportResult
    ) -> str:
        """Register export operation in CMS"""
        
        export_record = {
            'id': result.request_id,
            'timestamp': datetime.now().isoformat(),
            'request': asdict(request),
            'result_summary': {
                'success': result.success,
                'processing_time': result.processing_time,
                'export_count': len(result.export_results),
                'formats': list(set(r.format_type for r in result.export_results)),
                'total_size_mb': sum(r.file_size_mb for r in result.export_results),
                'quality_score': result.quality_validation.get('overall_score', 0)
            },
            'output_files': [
                file_path for export_result in result.export_results 
                for file_path in export_result.output_files
            ],
            'errors': result.errors,
            'warnings': result.warnings
        }
        
        # Save to CMS
        record_file = self.export_metadata_dir / f"{result.request_id}.json"
        with open(record_file, 'w') as f:
            json.dump(export_record, f, indent=2)
        
        return str(record_file)
    
    def get_export_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get export operation history"""
        
        records = []
        
        for record_file in sorted(
            self.export_metadata_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )[:limit]:
            try:
                with open(record_file, 'r') as f:
                    record = json.load(f)
                    records.append(record)
            except Exception:
                continue
        
        return records
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export system statistics"""
        
        history = self.get_export_history(limit=1000)
        
        if not history:
            return {'total_operations': 0}
        
        total_operations = len(history)
        successful_operations = sum(1 for r in history if r['result_summary']['success'])
        
        # Format statistics
        formats_used = {}
        total_size_mb = 0
        total_processing_time = 0
        
        for record in history:
            for fmt in record['result_summary']['formats']:
                formats_used[fmt] = formats_used.get(fmt, 0) + 1
            total_size_mb += record['result_summary']['total_size_mb']
            total_processing_time += record['result_summary']['processing_time']
        
        return {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / total_operations,
            'formats_used': formats_used,
            'total_output_size_mb': total_size_mb,
            'average_processing_time': total_processing_time / total_operations,
            'most_recent_operation': history[0]['timestamp'] if history else None
        }


class QualityAssuranceIntegration:
    """Integration with quality assurance for exported documents"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.quality_validator = ExportQualityValidator()
        self.qa_reports_dir = base_dir / "quality" / "export_reports"
        self.qa_reports_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_export_batch(
        self, 
        export_results: List[ExportResult],
        original_content_paths: List[str]
    ) -> Dict[str, Any]:
        """Validate a batch of export results"""
        
        validation_results = []
        
        for i, export_result in enumerate(export_results):
            if export_result.success:
                # Load original content
                original_content = ""
                if i < len(original_content_paths):
                    try:
                        with open(original_content_paths[i], 'r', encoding='utf-8') as f:
                            original_content = f.read()
                    except Exception:
                        pass
                
                # Validate
                validation = self.quality_validator.validate_export(
                    export_result, original_content
                )
                
                validation_results.append({
                    'file': export_result.output_files[0] if export_result.output_files else '',
                    'format': export_result.format_type,
                    'validation': validation
                })
        
        # Generate overall QA report
        overall_score = (
            sum(v['validation']['overall_score'] for v in validation_results) 
            / len(validation_results) if validation_results else 0
        )
        
        passed_count = sum(
            1 for v in validation_results 
            if v['validation']['passed']
        )
        
        qa_report = {
            'timestamp': datetime.now().isoformat(),
            'total_validated': len(validation_results),
            'passed_validation': passed_count,
            'overall_score': overall_score,
            'pass_rate': passed_count / len(validation_results) if validation_results else 0,
            'detailed_results': validation_results,
            'recommendations': self._generate_qa_recommendations(validation_results)
        }
        
        # Save QA report
        report_file = self.qa_reports_dir / f"qa_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(qa_report, f, indent=2)
        
        return qa_report
    
    def _generate_qa_recommendations(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """Generate QA recommendations based on validation results"""
        
        recommendations = []
        
        # Analyze common issues
        low_scores = [
            v for v in validation_results 
            if v['validation']['overall_score'] < 0.7
        ]
        
        if low_scores:
            recommendations.append(
                f"Review {len(low_scores)} files with low quality scores"
            )
        
        # Check for format-specific issues
        format_issues = {}
        for v in validation_results:
            if not v['validation']['passed']:
                fmt = v['format']
                format_issues[fmt] = format_issues.get(fmt, 0) + 1
        
        for fmt, count in format_issues.items():
            if count > 1:
                recommendations.append(
                    f"Multiple {fmt.upper()} exports failing validation - "
                    f"check {fmt} export configuration"
                )
        
        return recommendations


class ExportSystemIntegrator(BaseAgent):
    """Main integrator for the export system with academic agent components"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__("export_system_integrator")
        self.base_dir = base_dir or Path.cwd()
        
        # Initialize integration components
        self.study_notes_integration = StudyNotesExportIntegration(self.base_dir)
        self.pdf_integration = PDFProcessorExportIntegration(self.base_dir)
        self.cms_integration = CMSExportIntegration(self.base_dir)
        self.qa_integration = QualityAssuranceIntegration(self.base_dir)
        
        # Main export tool
        self.export_tool = ExportSystemTool(self.base_dir)
    
    def process_integrated_export_request(
        self, 
        request: IntegratedExportRequest
    ) -> IntegratedExportResult:
        """Process an integrated export request"""
        
        request_id = f"export_{int(time.time())}_{hash(str(request)) % 10000}"
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Route request based on source type
            if request.source_type == 'study_notes':
                result = self._process_study_notes_export(request)
            elif request.source_type == 'pdf_processed':
                result = self._process_pdf_export(request)
            elif request.source_type in ['markdown', 'directory']:
                result = self._process_direct_export(request)
            else:
                raise ValueError(f"Unknown source type: {request.source_type}")
            
            # Quality validation if requested
            quality_validation = {}
            if request.include_quality_validation and result.get('export_results'):
                quality_validation = self.qa_integration.validate_export_batch(
                    result['export_results'], request.source_paths
                )
            
            # Generate index if requested
            index_generated = False
            if request.generate_index:
                index_generated = self._generate_export_index(result, request)
            
            # Create integrated result
            integrated_result = IntegratedExportResult(
                request_id=request_id,
                success=result.get('success', False),
                export_results=result.get('export_results', []),
                quality_validation=quality_validation,
                consolidation_result=result.get('consolidation_result', {}),
                index_generated=index_generated,
                processing_time=time.time() - start_time,
                errors=errors + result.get('errors', []),
                warnings=warnings + result.get('warnings', []),
                output_summary=result.get('summary', {})
            )
            
            # Register with CMS
            self.cms_integration.register_export_operation(request, integrated_result)
            
            return integrated_result
            
        except Exception as e:
            errors.append(f"Integrated export failed: {str(e)}")
            
            return IntegratedExportResult(
                request_id=request_id,
                success=False,
                export_results=[],
                quality_validation={},
                consolidation_result={},
                index_generated=False,
                processing_time=time.time() - start_time,
                errors=errors,
                warnings=warnings,
                output_summary={}
            )
    
    def _process_study_notes_export(self, request: IntegratedExportRequest) -> Dict[str, Any]:
        """Process study notes export request"""
        
        if not request.source_paths:
            raise ValueError("No source paths provided for study notes export")
        
        # Use first source path as content, extract title and subject from metadata
        content_path = request.source_paths[0]
        metadata = request.metadata or {}
        
        title = metadata.get('title', Path(content_path).stem.replace('_', ' ').title())
        subject = metadata.get('subject', 'Academic Study')
        
        result = self.study_notes_integration.generate_and_export_study_notes(
            content_path=content_path,
            title=title,
            subject=subject,
            export_formats=request.export_formats or ['pdf'],
            template_name=request.template_name
        )
        
        # Transform result format
        export_results = []
        if result['success']:
            for export_result in result['export_results']:
                if export_result['result']['summary']['success']:
                    # Convert to ExportResult objects
                    from .export_system import ExportResult
                    for export_data in export_result['result']['export_results']:
                        export_results.append(ExportResult(**export_data))
        
        return {
            'success': result['success'],
            'export_results': export_results,
            'summary': result.get('summary', {}),
            'errors': [result.get('error')] if not result['success'] else [],
            'consolidation_result': result.get('export_results', [{}])[0].get('result', {}).get('consolidation_result', {})
        }
    
    def _process_pdf_export(self, request: IntegratedExportRequest) -> Dict[str, Any]:
        """Process PDF processor export request"""
        
        if not request.source_paths:
            raise ValueError("No source paths provided for PDF export")
        
        # Assume first path is the processed content directory
        processed_content_dir = request.source_paths[0]
        
        result = self.pdf_integration.export_processed_pdfs(
            processed_content_dir=processed_content_dir,
            export_formats=request.export_formats or ['pdf'],
            template_name=request.template_name
        )
        
        # Transform result format similar to study notes
        export_results = []
        if result['success']:
            for export_result in result['export_results']:
                if export_result['result']['summary']['success']:
                    from .export_system import ExportResult
                    for export_data in export_result['result']['export_results']:
                        export_results.append(ExportResult(**export_data))
        
        return {
            'success': result['success'],
            'export_results': export_results,
            'summary': result.get('summary', {}),
            'errors': [result.get('error')] if not result['success'] else []
        }
    
    def _process_direct_export(self, request: IntegratedExportRequest) -> Dict[str, Any]:
        """Process direct export request"""
        
        export_config = {
            "output_format": request.export_formats[0] if request.export_formats else 'pdf',
            "template_name": request.template_name,
            "include_diagrams": True,
            "consolidate_images": request.consolidate_images,
            "resolve_references": True,
            "metadata": request.metadata
        }
        
        # Handle multiple formats
        all_results = []
        for format_type in (request.export_formats or ['pdf']):
            export_config["output_format"] = format_type
            
            result = self.export_tool.forward(
                content_paths=request.source_paths,
                output_directory=request.output_directory,
                export_config=export_config,
                batch_mode=len(request.source_paths) > 1
            )
            
            all_results.append(result)
        
        # Combine results
        combined_export_results = []
        combined_success = True
        combined_errors = []
        
        for result in all_results:
            if result['summary']['success']:
                from .export_system import ExportResult
                for export_data in result['export_results']:
                    combined_export_results.append(ExportResult(**export_data))
            else:
                combined_success = False
                combined_errors.append(result.get('summary', {}).get('error', 'Unknown error'))
        
        return {
            'success': combined_success,
            'export_results': combined_export_results,
            'summary': all_results[0]['summary'] if all_results else {},
            'errors': combined_errors,
            'consolidation_result': all_results[0].get('consolidation_result', {}) if all_results else {}
        }
    
    def _generate_export_index(self, result: Dict[str, Any], request: IntegratedExportRequest) -> bool:
        """Generate an index file for exported documents"""
        
        try:
            index_content = []
            index_content.append(f"# Export Index")
            index_content.append(f"")
            index_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            index_content.append(f"**Source Type:** {request.source_type}")
            index_content.append(f"**Export Formats:** {', '.join(request.export_formats or ['pdf'])}")
            index_content.append(f"")
            
            # List exported files
            if result.get('export_results'):
                index_content.append("## Exported Files")
                index_content.append("")
                
                for i, export_result in enumerate(result['export_results'], 1):
                    for output_file in export_result.output_files:
                        file_path = Path(output_file)
                        file_size = file_path.stat().st_size / (1024 * 1024)
                        
                        index_content.append(f"{i}. **{file_path.name}**")
                        index_content.append(f"   - Format: {export_result.format_type.upper()}")
                        index_content.append(f"   - Size: {file_size:.2f} MB")
                        index_content.append(f"   - Path: `{output_file}`")
                        index_content.append("")
            
            # Write index file
            index_file = Path(request.output_directory) / "export_index.md"
            with open(index_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(index_content))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate export index: {e}")
            return False
    
    def get_export_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the export system"""
        
        return {
            'cms_statistics': self.cms_integration.get_export_statistics(),
            'recent_exports': self.cms_integration.get_export_history(limit=10),
            'system_health': {
                'export_tool_available': self.export_tool is not None,
                'integrations_loaded': {
                    'study_notes': self.study_notes_integration is not None,
                    'pdf_processor': self.pdf_integration is not None,
                    'cms': self.cms_integration is not None,
                    'quality_assurance': self.qa_integration is not None
                }
            },
            'directories': {
                'base_dir': str(self.base_dir),
                'export_dir': str(self.base_dir / "export"),
                'cms_metadata_dir': str(self.cms_integration.export_metadata_dir),
                'qa_reports_dir': str(self.qa_integration.qa_reports_dir)
            }
        }

# Convenience functions for easy integration

def export_study_notes(
    content_path: str,
    title: str,
    subject: str,
    output_dir: str = None,
    formats: List[str] = None,
    base_dir: Path = None
) -> Dict[str, Any]:
    """Convenience function to generate and export study notes"""
    
    integrator = ExportSystemIntegrator(base_dir)
    
    request = IntegratedExportRequest(
        source_type='study_notes',
        source_paths=[content_path],
        output_directory=output_dir or str(integrator.base_dir / "export" / "study_notes"),
        export_formats=formats or ['pdf'],
        metadata={'title': title, 'subject': subject}
    )
    
    result = integrator.process_integrated_export_request(request)
    return asdict(result)

def export_processed_pdfs(
    processed_content_dir: str,
    output_dir: str = None,
    formats: List[str] = None,
    base_dir: Path = None
) -> Dict[str, Any]:
    """Convenience function to export processed PDF content"""
    
    integrator = ExportSystemIntegrator(base_dir)
    
    request = IntegratedExportRequest(
        source_type='pdf_processed',
        source_paths=[processed_content_dir],
        output_directory=output_dir or str(integrator.base_dir / "export" / "processed_pdfs"),
        export_formats=formats or ['pdf']
    )
    
    result = integrator.process_integrated_export_request(request)
    return asdict(result)

def batch_export_directory(
    content_directory: str,
    output_dir: str = None,
    formats: List[str] = None,
    template: str = 'academic',
    base_dir: Path = None
) -> Dict[str, Any]:
    """Convenience function for batch export of directory content"""
    
    integrator = ExportSystemIntegrator(base_dir)
    
    # Find all markdown files in directory
    content_dir = Path(content_directory)
    markdown_files = list(content_dir.rglob("*.md"))
    
    request = IntegratedExportRequest(
        source_type='directory',
        source_paths=[str(f) for f in markdown_files],
        output_directory=output_dir or str(integrator.base_dir / "export" / "batch"),
        export_formats=formats or ['pdf'],
        template_name=template,
        include_quality_validation=True,
        generate_index=True
    )
    
    result = integrator.process_integrated_export_request(request)
    return asdict(result)

if __name__ == "__main__":
    # Example usage
    integrator = ExportSystemIntegrator()
    
    # Example: Export study notes
    study_notes_result = export_study_notes(
        content_path="sample_content.md",
        title="Sample Study Notes",
        subject="Computer Science",
        formats=['pdf', 'html']
    )
    
    print("Study Notes Export Result:")
    print(f"Success: {study_notes_result['success']}")
    print(f"Processing time: {study_notes_result['processing_time']:.2f}s")
    
    # Example: Get system status
    status = integrator.get_export_system_status()
    print("\nExport System Status:")
    print(f"Total operations: {status['cms_statistics']['total_operations']}")
    print(f"Success rate: {status['cms_statistics'].get('success_rate', 0)*100:.1f}%")