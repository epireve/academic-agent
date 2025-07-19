#!/usr/bin/env python
"""
Ingestion Agent - Specialized agent for ingesting and processing PDF files
Part of the Academic Agent system
"""

import os
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from PIL import Image
import magic
from docling.document_converter import DocumentConverter
# Use unified BaseAgent for standardized interface
from ...src.agents.base_agent import BaseAgent

# Import from unified architecture
from ...src.core.output_manager import get_output_manager, OutputCategory, ContentType
from ...src.processors.pdf_processor import PDFProcessor

# Load environment variables
load_dotenv()

# Optional AI agents - only import if available
try:
    from smolagents import CodeAgent, HfApiModel
    AI_AGENTS_AVAILABLE = True
except ImportError:
    AI_AGENTS_AVAILABLE = False
    CodeAgent = None
    HfApiModel = None


class IngestionAgent(BaseAgent):
    """Agent responsible for processing PDFs into markdown with image preservation"""

    def __init__(self):
        super().__init__("ingestion_agent")
        self.converter = DocumentConverter()
        self.supported_formats = ["application/pdf", "text/plain"]
        self.image_formats = ["jpg", "jpeg", "png", "gif"]
        self.min_processing_speed = 10  # pages per second
        self.output_manager = None
        self.pdf_processor = None

    async def initialize(self):
        """Initialize agent-specific resources."""
        try:
            # Initialize output manager
            self.output_manager = get_output_manager()
            
            # Setup output directories
            ingestion_dir = self.output_manager.get_output_path(
                OutputCategory.PROCESSED, 
                ContentType.MARKDOWN, 
                subdirectory="ingestion"
            )
            ingestion_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize PDF processor if available
            if hasattr(self, 'pdf_processor'):
                self.pdf_processor = PDFProcessor()
                await self.pdf_processor.initialize()
            
            # Setup AI agents if available
            if AI_AGENTS_AVAILABLE:
                try:
                    self.ai_model = HfApiModel()
                    self.code_agent = CodeAgent(tools=[], model=self.ai_model)
                except Exception as e:
                    self.logger.warning(f"Could not initialize AI agents: {e}")
                    self.ai_model = None
                    self.code_agent = None
            
            self.logger.info(f"{self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.agent_id}: {e}")
            raise

    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            # Cleanup PDF processor
            if hasattr(self, 'pdf_processor') and self.pdf_processor:
                await self.pdf_processor.cleanup()
            
            # Close any open file handles
            if hasattr(self, 'converter'):
                # DocumentConverter doesn't need explicit cleanup
                pass
            
            # Clear AI agents
            if hasattr(self, 'code_agent'):
                self.code_agent = None
            if hasattr(self, 'ai_model'):
                self.ai_model = None
            
            self.logger.info(f"{self.agent_id} cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during {self.agent_id} cleanup: {e}")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        try:
            # Validate input file
            if not self._validate_file(pdf_path):
                raise ValueError(f"Invalid or unsupported file: {pdf_path}")

            start_time = datetime.now()

            # Convert PDF to markdown
            result = self.converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()

            # Extract and process images
            images = self._process_images(result.document.images)

            # Generate metadata
            metadata = self._generate_metadata(pdf_path, result)

            # Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            pages_per_second = len(result.document.pages) / processing_time

            if pages_per_second < self.min_processing_speed:
                self.logger.warning(
                    f"Processing speed below threshold: {pages_per_second} pages/sec"
                )

            # Save processed content
            output = self._save_processed_content(
                pdf_path, markdown_content, images, metadata
            )

            # Check quality
            quality_score = self.check_quality(
                {
                    "markdown_content": markdown_content,
                    "images": images,
                    "metadata": metadata,
                }
            )

            return {
                "success": True,
                "markdown_path": output["markdown_path"],
                "images": output["images"],
                "metadata": metadata,
                "quality_score": quality_score,
                "processing_metrics": {
                    "processing_time": processing_time,
                    "pages_per_second": pages_per_second,
                    "total_pages": len(result.document.pages),
                    "total_images": len(images),
                },
            }

        except Exception as e:
            self.handle_error(e, {"operation": "pdf_processing", "file": pdf_path})
            return {"success": False, "error": str(e)}

    def _validate_file(self, file_path: str) -> bool:
        """Validate input file format and accessibility"""
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return False

        file_type = magic.from_file(file_path, mime=True)
        if file_type not in self.supported_formats:
            self.logger.error(f"Unsupported file type: {file_type}")
            return False

        return True

    def _process_images(self, images: List[Dict]) -> List[Dict[str, Any]]:
        """Process and optimize images from the PDF"""
        processed_images = []

        for idx, img_data in enumerate(images):
            try:
                # Create image object
                img = Image.open(img_data["data"])

                # Generate filename
                filename = f"img_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{img.format.lower()}"

                # Save original
                original_path = os.path.join("processed/img", f"original_{filename}")
                img.save(original_path)

                # Create web-optimized version
                web_path = os.path.join("processed/img", f"web_{filename}")
                self._optimize_image(img, web_path)

                processed_images.append(
                    {
                        "original_path": original_path,
                        "web_path": web_path,
                        "metadata": {
                            "format": img.format,
                            "size": img.size,
                            "mode": img.mode,
                        },
                    }
                )

            except Exception as e:
                self.logger.error(f"Error processing image {idx}: {str(e)}")
                continue

        return processed_images

    def _optimize_image(self, img: Image, output_path: str) -> None:
        """Create web-optimized version of image"""
        # Max dimensions for web version
        MAX_SIZE = (1200, 1200)

        # Create copy for web version
        web_img = img.copy()

        # Resize if needed
        if web_img.size[0] > MAX_SIZE[0] or web_img.size[1] > MAX_SIZE[1]:
            web_img.thumbnail(MAX_SIZE)

        # Save with optimization
        web_img.save(output_path, optimize=True, quality=85)

    def _generate_metadata(self, pdf_path: str, result: Any) -> Dict[str, Any]:
        """Generate metadata for processed content"""
        return {
            "source_file": os.path.basename(pdf_path),
            "processing_timestamp": datetime.now().isoformat(),
            "page_count": len(result.document.pages),
            "image_count": len(result.document.images),
            "document_info": result.document.metadata,
            "processing_version": {
                "docling": self.converter.__version__,
                "agent": "1.0.0",
            },
        }

    def _save_processed_content(
        self, pdf_path: str, markdown_content: str, images: List[Dict], metadata: Dict
    ) -> Dict[str, Any]:
        """Save processed content to appropriate locations"""
        # Generate filenames
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_path = os.path.join("processed/markdown", f"{base_name}.md")

        # Save markdown content
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return {"markdown_path": markdown_path, "images": images}

    def check_quality(self, content: Dict[str, Any]) -> float:
        """Check quality of processed content"""
        quality_score = 1.0
        deductions = []

        # Check markdown content
        if not content["markdown_content"]:
            deductions.append(0.5)
        elif len(content["markdown_content"]) < 100:
            deductions.append(0.3)

        # Check images
        if not content["images"]:
            if "images" in content["metadata"]["document_info"]:
                deductions.append(0.4)

        # Check metadata
        if not all(
            key in content["metadata"]
            for key in ["source_file", "processing_timestamp", "page_count"]
        ):
            deductions.append(0.2)

        # Apply deductions
        for deduction in deductions:
            quality_score -= deduction

        return max(0.0, min(1.0, quality_score))

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if not isinstance(input_data, str):
            return False
        return self._validate_file(input_data)

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if not isinstance(output_data, dict):
            return False

        required_fields = ["success", "markdown_path", "images", "metadata"]
        return all(field in output_data for field in required_fields)


def main():
    """Main entry point for the ingestion agent"""
    parser = argparse.ArgumentParser(
        description="Ingestion Agent for processing PDF files"
    )

    # Define command line arguments
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument(
        "--output", help="Output directory (defaults to processed/ingestion)"
    )
    parser.add_argument("--api-key", help="Hugging Face API key")
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device to use for PDF processing",
    )
    parser.add_argument(
        "--no-smart-rename", action="store_true", help="Disable smart file renaming"
    )

    args = parser.parse_args()

    # Get API key from environment or command line
    api_key = args.api_key or os.getenv("HF_API_KEY")
    if not api_key:
        print(
            "Error: Hugging Face API key is required. Set HF_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    # Create the ingestion agent with specified device
    agent = IngestionAgent()

    # Process PDF or directory
    if args.pdf:
        output_dir = args.output or str(OUTPUT_DIR)
        result = agent.process_pdf(args.pdf)
        if result["success"]:
            print(f"Processed PDF: {result['markdown_path']}")
            if result["metadata"].get("file_info", {}).get("sequence_number"):
                seq_type = (
                    result["metadata"].get("file_info", {}).get("sequence_type", "")
                )
                seq_num = (
                    result["metadata"].get("file_info", {}).get("sequence_number", "")
                )
                print(f"Detected as {seq_type} {seq_num}")
        else:
            print("Processing failed")
    elif args.dir:
        output_dir = args.output or str(OUTPUT_DIR)
        results = agent.process_directory(args.dir, output_dir)
        print(f"Processed {len(results)} PDF files")
        for path, metadata in results:
            seq_info = ""
            if metadata.get("file_info", {}).get("sequence_number"):
                seq_type = metadata.get("file_info", {}).get("sequence_type", "")
                seq_num = metadata.get("file_info", {}).get("sequence_number", "")
                seq_info = f" ({seq_type} {seq_num})"
            print(f"- {path}{seq_info}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
