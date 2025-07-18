#!/usr/bin/env python
"""
Ingestion Agent - Specialized agent for ingesting and processing PDF files

Migrated to use unified base agent architecture.
Part of the Academic Agent system
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio

from .base_agent import BaseAgent
from ..processors.pdf_processor import PDFProcessor
from ..core.exceptions import ProcessingError, ValidationError


@dataclass
class IngestionResult:
    """Results from PDF ingestion"""
    source_file: str
    output_paths: Dict[str, str]  # type -> path mapping
    metadata: Dict[str, Any]
    processing_time: float
    quality_score: float
    success: bool
    errors: List[str] = None


@dataclass
class IngestionJob:
    """Represents an ingestion job"""
    job_id: str
    source_path: str
    output_dir: str
    options: Dict[str, Any]
    created_at: datetime
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[IngestionResult] = None


class IngestionAgent(BaseAgent):
    """Specialized agent for ingesting and processing PDF files using unified architecture"""

    def __init__(self, config_path: Optional[str] = None):
        super().__init__("ingestion_agent", config_path)
        
        # Initialize PDF processor with unified architecture
        self.pdf_processor = PDFProcessor()
        
        # Ingestion settings
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
        self.output_formats = ['markdown', 'json', 'metadata']
        
        # Job tracking
        self.active_jobs: Dict[str, IngestionJob] = {}
        self.completed_jobs: List[IngestionJob] = []
        
        # Output directories
        self.base_output_dir = self.base_dir / "processed" / "ingestion"
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("IngestionAgent initialized with unified architecture")

    async def ingest_file(self, file_path: str, output_dir: Optional[str] = None, 
                         options: Optional[Dict[str, Any]] = None) -> IngestionResult:
        """Ingest a single file"""
        try:
            self.logger.info(f"Starting ingestion of: {file_path}")
            
            if not await self.validate_input(file_path):
                raise ValidationError(f"Invalid input file: {file_path}")
            
            start_time = datetime.now()
            
            # Create job
            job = IngestionJob(
                job_id=f"ingest_{int(start_time.timestamp())}",
                source_path=file_path,
                output_dir=output_dir or str(self.base_output_dir),
                options=options or {},
                created_at=start_time,
                status="processing"
            )
            
            self.active_jobs[job.job_id] = job
            
            # Process file based on type
            source_path = Path(file_path)
            if source_path.suffix.lower() == '.pdf':
                result = await self._process_pdf(job)
            elif source_path.suffix.lower() in ['.txt', '.md']:
                result = await self._process_text(job)
            else:
                raise ProcessingError(f"Unsupported file format: {source_path.suffix}")
            
            # Update job
            job.status = "completed" if result.success else "failed"
            job.result = result
            
            # Move to completed jobs
            self.completed_jobs.append(job)
            del self.active_jobs[job.job_id]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            if await self.validate_output(result):
                self.logger.info(f"Ingestion completed: {file_path} in {processing_time:.2f}s")
                return result
            else:
                raise ValidationError("Output validation failed")
                
        except Exception as e:
            self.logger.error(f"Error ingesting file {file_path}: {e}")
            return IngestionResult(
                source_file=file_path,
                output_paths={},
                metadata={"error": str(e)},
                processing_time=0.0,
                quality_score=0.0,
                success=False,
                errors=[str(e)]
            )

    async def _process_pdf(self, job: IngestionJob) -> IngestionResult:
        """Process PDF file using unified PDF processor"""
        try:
            # Use unified PDF processor
            result = await self.pdf_processor.process_file(
                job.source_path, 
                job.output_dir,
                **job.options
            )
            
            if result.get("success"):
                output_paths = {
                    "markdown": result.get("markdown_path", ""),
                    "metadata": result.get("metadata_path", ""),
                    "images": result.get("images_dir", "")
                }
                
                return IngestionResult(
                    source_file=job.source_path,
                    output_paths=output_paths,
                    metadata=result.get("metadata", {}),
                    processing_time=0.0,  # Will be set by caller
                    quality_score=result.get("quality_score", 0.8),
                    success=True
                )
            else:
                return IngestionResult(
                    source_file=job.source_path,
                    output_paths={},
                    metadata={},
                    processing_time=0.0,
                    quality_score=0.0,
                    success=False,
                    errors=[result.get("error", "Unknown error")]
                )
                
        except Exception as e:
            return IngestionResult(
                source_file=job.source_path,
                output_paths={},
                metadata={"error": str(e)},
                processing_time=0.0,
                quality_score=0.0,
                success=False,
                errors=[str(e)]
            )

    async def _process_text(self, job: IngestionJob) -> IngestionResult:
        """Process text file (markdown, txt)"""
        try:
            source_path = Path(job.source_path)
            output_dir = Path(job.output_dir)
            
            # Read content
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create output paths
            output_base = output_dir / source_path.stem
            output_base.mkdir(parents=True, exist_ok=True)
            
            # Copy/process content
            if source_path.suffix.lower() == '.md':
                # Already markdown
                markdown_path = output_base / f"{source_path.stem}.md"
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                # Convert to markdown
                markdown_path = output_base / f"{source_path.stem}.md"
                markdown_content = f"# {source_path.stem}\n\n{content}"
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
            
            # Create metadata
            metadata = {
                "title": source_path.stem,
                "source_file": str(source_path),
                "processed_date": datetime.now().isoformat(),
                "file_size": source_path.stat().st_size,
                "content_length": len(content)
            }
            
            metadata_path = output_base / f"{source_path.stem}_meta.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            output_paths = {
                "markdown": str(markdown_path),
                "metadata": str(metadata_path)
            }
            
            return IngestionResult(
                source_file=job.source_path,
                output_paths=output_paths,
                metadata=metadata,
                processing_time=0.0,
                quality_score=0.9,  # High score for text files
                success=True
            )
            
        except Exception as e:
            return IngestionResult(
                source_file=job.source_path,
                output_paths={},
                metadata={"error": str(e)},
                processing_time=0.0,
                quality_score=0.0,
                success=False,
                errors=[str(e)]
            )

    async def batch_ingest_directory(self, directory_path: str, 
                                   output_dir: Optional[str] = None,
                                   file_pattern: str = "*",
                                   max_concurrent: int = 3) -> Dict[str, Any]:
        """Batch ingest all supported files in a directory"""
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise ValidationError(f"Directory not found: {directory_path}")
            
            # Find supported files
            supported_files = []
            for ext in self.supported_formats:
                supported_files.extend(directory.glob(f"**/*{ext}"))
            
            if not supported_files:
                return {
                    "success": True,
                    "message": "No supported files found",
                    "results": []
                }
            
            self.logger.info(f"Found {len(supported_files)} files to process")
            
            # Process files with concurrency limit
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(file_path):
                async with semaphore:
                    return await self.ingest_file(str(file_path), output_dir)
            
            # Execute batch processing
            results = await asyncio.gather(*[
                process_with_semaphore(file_path) 
                for file_path in supported_files
            ], return_exceptions=True)
            
            # Analyze results
            successful = sum(1 for r in results if isinstance(r, IngestionResult) and r.success)
            failed = len(results) - successful
            
            return {
                "success": True,
                "total_files": len(supported_files),
                "successful": successful,
                "failed": failed,
                "results": [r for r in results if isinstance(r, IngestionResult)],
                "errors": [str(r) for r in results if isinstance(r, Exception)]
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch ingestion: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }

    async def validate_input(self, input_data: Any) -> bool:
        """Validate input file"""
        if isinstance(input_data, str):
            path = Path(input_data)
            return (path.exists() and 
                    path.is_file() and 
                    path.suffix.lower() in self.supported_formats)
        return False

    async def validate_output(self, output_data: Any) -> bool:
        """Validate output result"""
        if isinstance(output_data, IngestionResult):
            return output_data.success and bool(output_data.output_paths)
        return False

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status,
                "source_path": job.source_path,
                "created_at": job.created_at.isoformat(),
                "progress": "processing"
            }
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return {
                    "job_id": job_id,
                    "status": job.status,
                    "source_path": job.source_path,
                    "created_at": job.created_at.isoformat(),
                    "result": asdict(job.result) if job.result else None
                }
        
        return None

    def get_all_jobs(self) -> Dict[str, Any]:
        """Get status of all jobs"""
        return {
            "active_jobs": {
                job_id: {
                    "status": job.status,
                    "source_path": job.source_path,
                    "created_at": job.created_at.isoformat()
                }
                for job_id, job in self.active_jobs.items()
            },
            "completed_jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "source_path": job.source_path,
                    "created_at": job.created_at.isoformat(),
                    "success": job.result.success if job.result else False
                }
                for job in self.completed_jobs[-10:]  # Last 10 jobs
            ]
        }