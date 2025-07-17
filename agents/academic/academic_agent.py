#!/usr/bin/env python
"""
Academic Agent - An intelligent system for processing and synthesizing academic materials
using smolagents framework and Groq LLM.

This agent:
1. Ingests PDF data from various sources (lectures, transcripts, notes)
2. Analyzes and organizes content
3. Creates comprehensive outlines and notes
4. Updates materials as new information becomes available
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import frontmatter
from docling.document_converter import DocumentConverter
from smolagents import CodeAgent, Tool
import re
import litellm

# Load environment variables from .env file
load_dotenv()

try:
    from smolagents import LiteLLMModel
except ImportError:
    print("Installing smolagents...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import LiteLLMModel

# Add parent directory to path for importing the pdf_processor
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from tools.pdf_processor import DoclingProcessor
from tools.markdown_processor.processor import MarkdownProcessor

# Define base paths
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROCESSED_DIR = BASE_DIR / "processed"
INGESTION_DIR = PROCESSED_DIR / "ingestion"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"
OUTLINES_DIR = PROCESSED_DIR / "outlines"
NOTES_DIR = PROCESSED_DIR / "notes"


class PDFIngestionTool(Tool):
    """Tool for ingesting PDFs using Docling"""

    name = "pdf_ingestion_tool"
    description = (
        "Process PDF files using Docling and convert to markdown with metadata"
    )
    inputs = {
        "pdf_path": {
            "type": "string",
            "description": "Path to the PDF file or directory to process",
        },
        "recursive": {
            "type": "boolean",
            "description": "Whether to recursively process directories",
            "nullable": True,
        },
    }
    outputs = {
        "processed_files": {
            "type": "array",
            "description": "List of processed files and their metadata",
        },
        "errors": {
            "type": "array",
            "description": "List of errors encountered during processing",
        },
        "stats": {
            "type": "object",
            "description": "Processing statistics",
        },
    }
    output_type = "object"

    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__()
        self.base_dir = base_dir or Path.cwd()
        try:
            self.converter = DocumentConverter()
            print("Debug: DocumentConverter initialized successfully")
        except Exception as e:
            print(f"Error initializing DocumentConverter: {str(e)}")
            raise
        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Ensure required directories exist"""
        dirs = ["raw", "markdown", "metadata"]
        for dir_name in dirs:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Debug: Created directory {dir_path}")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        print(f"\nProcessing PDF: {pdf_path}")
        try:
            # Convert PDF using Docling
            print("Debug: Starting PDF conversion...")
            result = self.converter.convert(pdf_path)
            print("Debug: PDF converted successfully")

            print("Debug: Exporting to markdown...")
            markdown_content = result.document.export_to_markdown()
            # Convert bytes to string if necessary
            if isinstance(markdown_content, bytes):
                markdown_content = markdown_content.decode("utf-8")
            print("Debug: Markdown export successful")

            # Extract metadata safely
            metadata = {
                "source_file": pdf_path,
                "processed_date": datetime.now().isoformat(),
                "title": getattr(result.document, "title", Path(pdf_path).stem),
                "language": getattr(result.document, "language", "en"),
            }
            print("Debug: Metadata extracted successfully")

            # Save markdown and metadata separately
            filename = Path(pdf_path).stem
            markdown_path = self.base_dir / "markdown" / f"{filename}.md"
            metadata_path = self.base_dir / "metadata" / f"{filename}.json"

            print(f"Debug: Saving markdown to {markdown_path}")
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print("Debug: Markdown saved successfully")

            print(f"Debug: Saving metadata to {metadata_path}")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print("Debug: Metadata saved successfully")

            return {
                "status": "success",
                "markdown_path": str(markdown_path),
                "metadata_path": str(metadata_path),
                "metadata": metadata,
            }

        except Exception as e:
            print(f"\nError processing PDF {pdf_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback

            print(f"Traceback:\n{traceback.format_exc()}")
            return {"status": "error", "error": str(e), "file": pdf_path}

    def forward(self, pdf_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Process PDF files"""
        results = {
            "processed_files": [],
            "errors": [],
            "stats": {"total": 0, "success": 0, "failed": 0},
        }

        if os.path.isfile(pdf_path):
            # Process single file
            results["stats"]["total"] = 1
            result = self.process_pdf(pdf_path)

            if result["status"] == "success":
                results["processed_files"].append(result)
                results["stats"]["success"] = 1
            else:
                results["errors"].append(result)
                results["stats"]["failed"] = 1

        elif os.path.isdir(pdf_path):
            # Process directory
            for root, _, files in os.walk(pdf_path):
                if not recursive and root != pdf_path:
                    continue

                for file in files:
                    if file.lower().endswith(".pdf"):
                        file_path = os.path.join(root, file)
                        results["stats"]["total"] += 1

                        result = self.process_pdf(file_path)
                        if result["status"] == "success":
                            results["processed_files"].append(result)
                            results["stats"]["success"] += 1
                        else:
                            results["errors"].append(result)
                            results["stats"]["failed"] += 1

        else:
            raise FileNotFoundError(f"Path not found: {pdf_path}")

        return results


class ContentAnalysisTool(Tool):
    """Tool for analyzing academic content and identifying key concepts"""

    name = "content_analysis_tool"
    description = (
        "Analyze academic content to identify topics, structure, and key concepts"
    )
    inputs = {
        "markdown_path": {
            "type": "string",
            "description": "Path to the markdown file or directory to analyze",
        },
        "recursive": {
            "type": "boolean",
            "description": "Whether to recursively process directories",
            "nullable": True,
        },
    }
    outputs = {
        "files_analyzed": {
            "type": "array",
            "description": "List of analyzed files and their results",
        },
        "errors": {
            "type": "array",
            "description": "List of errors encountered during analysis",
        },
        "stats": {
            "type": "object",
            "description": "Analysis statistics",
        },
    }
    output_type = "object"

    def __init__(self, base_dir: Optional[Path] = None):
        super().__init__()
        # Initialize markdown processor with base directory
        self.base_dir = base_dir or PROCESSED_DIR
        self.markdown_processor = MarkdownProcessor(base_dir=self.base_dir)
        # Ensure output directories exist
        for dir_name in ["analysis", "raw", "markdown", "metadata"]:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def analyze_single_file(self, markdown_path: str) -> Dict[str, Any]:
        """
        Analyze a single markdown file

        Args:
            markdown_path: Path to the markdown file

        Returns:
            Dictionary of analysis results
        """
        try:
            # Read the markdown file
            with open(markdown_path, "r") as f:
                content = f.read()

            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(markdown_path))[0]

            # Clean and process the markdown content
            content = self.markdown_processor.clean_markdown(content)
            sections = self.markdown_processor.extract_sections(content)

            # Create analysis structure
            analysis = {
                "main_topics": [],
                "key_concepts": [],
                "structure": [],
                "summary": "",
                "source_file": markdown_path,
                "analysis_date": datetime.now().isoformat(),
                "sections": sections,
            }

            # Process each section and extract key information
            for section in sections:
                section_title = section.get("title", "")
                section_content = section.get("content", "")

                if section_title:
                    analysis["structure"].append(
                        {
                            "title": section_title,
                            "summary": (
                                section_content[:200] + "..."
                                if len(section_content) > 200
                                else section_content
                            ),
                        }
                    )

                # Extract topics and concepts from section content
                if section_content:
                    # Add section title as a main topic if it exists
                    if section_title:
                        analysis["main_topics"].append(section_title)

                    # Extract key phrases as concepts
                    sentences = section_content.split(". ")
                    for sentence in sentences[:5]:  # Limit to first 5 sentences
                        if (
                            len(sentence.strip()) > 20
                        ):  # Only consider substantial sentences
                            analysis["key_concepts"].append(sentence.strip())

            # Generate a summary from the first few sections
            summary_content = "\n".join(
                [s.get("content", "")[:300] for s in sections[:3]]
            )
            analysis["summary"] = summary_content

            # Save analysis results
            saved_paths = self.markdown_processor.save_analysis(analysis, filename)
            analysis.update(saved_paths)

            print(f"\nAnalysis saved to:")
            print(f"- Markdown: {saved_paths['markdown_path']}")
            print(f"- JSON: {saved_paths['json_path']}\n")

            return analysis

        except Exception as e:
            print(f"\nError processing analysis: {str(e)}")
            print(f"File: {markdown_path}")
            import traceback

            print(f"Traceback: {traceback.format_exc()}\n")
            return {"error": str(e), "source_file": markdown_path}

    def forward(self, markdown_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Analyze markdown content from a file or directory

        Args:
            markdown_path: Path to the markdown file or directory
            recursive: Whether to recursively process directories

        Returns:
            Dictionary with analysis results
        """
        results = {
            "files_analyzed": [],
            "errors": [],
            "stats": {"total": 0, "success": 0, "failed": 0},
        }

        if os.path.isfile(markdown_path):
            # Process single file
            results["stats"]["total"] = 1
            analysis = self.analyze_single_file(markdown_path)

            if "error" not in analysis:
                results["files_analyzed"].append(analysis)
                results["stats"]["success"] = 1
            else:
                results["errors"].append(analysis)
                results["stats"]["failed"] = 1

        elif os.path.isdir(markdown_path):
            # Process directory
            for root, _, files in os.walk(markdown_path):
                if not recursive and root != markdown_path:
                    continue

                for file in files:
                    if file.lower().endswith(".md"):
                        file_path = os.path.join(root, file)
                        results["stats"]["total"] += 1

                        analysis = self.analyze_single_file(file_path)
                        if "error" not in analysis:
                            results["files_analyzed"].append(analysis)
                            results["stats"]["success"] += 1
                        else:
                            results["errors"].append(analysis)
                            results["stats"]["failed"] += 1

            print(f"\nProcessing Summary:")
            print(f"Total files: {results['stats']['total']}")
            print(f"Successfully analyzed: {results['stats']['success']}")
            print(f"Failed: {results['stats']['failed']}")

            if results["errors"]:
                print("\nErrors encountered:")
                for error in results["errors"]:
                    print(f"- {error['source_file']}: {error['error']}")

        else:
            raise FileNotFoundError(f"Path not found: {markdown_path}")

        return results


class OutlineGenerationTool(Tool):
    """Tool for generating structured outlines from analyzed content"""

    name = "outline_generation_tool"
    description = "Generate structured outlines from analyzed content"
    inputs = {
        "markdown_path": {
            "type": "string",
            "description": "Path to the markdown file to outline",
        },
        "analysis_path": {
            "type": "string",
            "description": "Path to the analysis file",
            "nullable": True,
        },
    }
    outputs = {
        "outline": {
            "type": "object",
            "description": "Generated outline structure",
            "properties": {
                "source_file": {"type": "string"},
                "generated_date": {"type": "string"},
                "sections": {"type": "array"},
            },
        }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        # Ensure output directory exists
        os.makedirs(OUTLINES_DIR, exist_ok=True)
        self.markdown_processor = MarkdownProcessor(base_dir=PROCESSED_DIR)

    def forward(
        self, markdown_path: str, analysis_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an outline from markdown content

        Args:
            markdown_path: Path to the markdown file
            analysis_path: Path to the analysis file (optional)

        Returns:
            Dictionary with generated outline
        """
        try:
            # Read the markdown file
            with open(markdown_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(markdown_path))[0]

            # Clean and process the markdown content
            content = self.markdown_processor.clean_markdown(content)
            sections = self.markdown_processor.extract_sections(content)

            # Create outline structure
            outline = {
                "source_file": markdown_path,
                "generated_date": datetime.now().isoformat(),
                "sections": [],
            }

            # Process sections into outline
            current_section = None
            current_subsection = None

            for section in sections:
                title = section.get("title", "").strip()
                content = section.get("content", "").strip()

                if not title and not content:
                    continue

                # Create section entry
                section_entry = {
                    "title": title or "Untitled Section",
                    "key_points": [],
                    "subsections": [],
                }

                # Extract key points from content
                sentences = content.split(". ")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Only substantial sentences
                        section_entry["key_points"].append(sentence)

                outline["sections"].append(section_entry)

            # Save outline
            outline_path = OUTLINES_DIR / f"{filename}_outline.md"
            outline_json_path = OUTLINES_DIR / f"{filename}_outline.json"

            # Save JSON version
            with open(outline_json_path, "w", encoding="utf-8") as f:
                json.dump(outline, f, indent=2, ensure_ascii=False)

            # Generate markdown version
            markdown_content = f"# Outline: {filename}\n\n"
            markdown_content += f"Generated: {outline['generated_date']}\n\n"
            markdown_content += f"Source: {outline['source_file']}\n\n"

            for i, section in enumerate(outline["sections"], 1):
                markdown_content += f"## {i}. {section['title']}\n\n"

                if section["key_points"]:
                    markdown_content += "Key Points:\n"
                    for point in section["key_points"]:
                        markdown_content += f"- {point}\n"
                    markdown_content += "\n"

                if section["subsections"]:
                    for j, subsection in enumerate(section["subsections"], 1):
                        markdown_content += f"### {i}.{j}. {subsection['title']}\n"
                        if subsection.get("key_points"):
                            for point in subsection["key_points"]:
                                markdown_content += f"- {point}\n"
                        markdown_content += "\n"

            # Save markdown version
            with open(outline_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            return {
                "outline": outline,
                "outline_path": str(outline_path),
                "outline_json_path": str(outline_json_path),
            }

        except Exception as e:
            print(f"\nError generating outline for {markdown_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback

            print(f"Traceback:\n{traceback.format_exc()}")
            return {"error": str(e), "file": markdown_path}


class NotesGenerationTool(Tool):
    """Tool for generating comprehensive notes from content"""

    name = "notes_generation_tool"
    description = "Generate comprehensive notes from content"
    inputs = {
        "markdown_path": {
            "type": "string",
            "description": "Path to the markdown file",
        },
        "outline_path": {
            "type": "string",
            "description": "Path to the outline file",
            "nullable": True,
        },
    }
    outputs = {
        "notes": {
            "type": "object",
            "description": "Generated notes",
            "properties": {
                "source_file": {"type": "string"},
                "generated_date": {"type": "string"},
                "content": {"type": "array"},
            },
        }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()
        # Ensure output directory exists
        os.makedirs(NOTES_DIR, exist_ok=True)
        self.markdown_processor = MarkdownProcessor(base_dir=PROCESSED_DIR)

    def forward(
        self, markdown_path: str, outline_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive notes from content

        Args:
            markdown_path: Path to the markdown file
            outline_path: Path to the outline file (optional)

        Returns:
            Dictionary with generated notes
        """
        try:
            # Read the markdown file
            with open(markdown_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(markdown_path))[0]

            # Clean and process the markdown content
            content = self.markdown_processor.clean_markdown(content)
            sections = self.markdown_processor.extract_sections(content)

            # Create notes structure
            notes = {
                "source_file": markdown_path,
                "generated_date": datetime.now().isoformat(),
                "sections": [],
            }

            # Process sections into detailed notes
            for section in sections:
                title = section.get("title", "").strip()
                content = section.get("content", "").strip()

                if not title and not content:
                    continue

                # Create section entry
                section_entry = {
                    "title": title or "Untitled Section",
                    "summary": "",
                    "key_concepts": [],
                    "detailed_notes": [],
                }

                # Process content
                sentences = content.split(". ")

                # Generate summary (first 2-3 sentences)
                summary_sentences = sentences[:3]
                section_entry["summary"] = ". ".join(summary_sentences) + "."

                # Extract key concepts (important phrases/terms)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Only substantial sentences
                        section_entry["key_concepts"].append(sentence)

                # Create detailed notes
                section_entry["detailed_notes"] = content.split("\n")

                notes["sections"].append(section_entry)

            # Save notes
            notes_path = NOTES_DIR / f"{filename}_notes.md"
            notes_json_path = NOTES_DIR / f"{filename}_notes.json"

            # Save JSON version
            with open(notes_json_path, "w", encoding="utf-8") as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)

            # Generate markdown version
            markdown_content = f"# Notes: {filename}\n\n"
            markdown_content += f"Generated: {notes['generated_date']}\n\n"
            markdown_content += f"Source: {notes['source_file']}\n\n"

            for i, section in enumerate(notes["sections"], 1):
                markdown_content += f"## {i}. {section['title']}\n\n"

                if section["summary"]:
                    markdown_content += "### Summary\n"
                    markdown_content += f"{section['summary']}\n\n"

                if section["key_concepts"]:
                    markdown_content += "### Key Concepts\n"
                    for concept in section["key_concepts"]:
                        markdown_content += f"- {concept}\n"
                    markdown_content += "\n"

                if section["detailed_notes"]:
                    markdown_content += "### Detailed Notes\n"
                    for note in section["detailed_notes"]:
                        if note.strip():
                            markdown_content += f"{note}\n"
                    markdown_content += "\n"

            # Save markdown version
            with open(notes_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            return {
                "notes": notes,
                "notes_path": str(notes_path),
                "notes_json_path": str(notes_json_path),
            }

        except Exception as e:
            print(f"\nError generating notes for {markdown_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback

            print(f"Traceback:\n{traceback.format_exc()}")
            return {"error": str(e), "file": markdown_path}


class ContentUpdateTool(Tool):
    """Tool for updating existing notes with new information"""

    name = "content_update_tool"
    description = "Update existing notes with new information"
    inputs = {
        "notes_path": {
            "type": "string",
            "description": "Path to the existing notes file",
        },
        "new_content_path": {
            "type": "string",
            "description": "Path to the new content file",
        },
    }
    outputs = {
        "updated_notes": {
            "type": "object",
            "description": "Updated notes",
            "properties": {
                "source_file": {"type": "string"},
                "update_date": {"type": "string"},
                "content": {"type": "array"},
            },
        }
    }
    output_type = "object"

    def __init__(self):
        super().__init__()

    def forward(self, notes_path: str, new_content_path: str) -> Dict[str, Any]:
        """
        Update existing notes with new information

        Args:
            notes_path: Path to the existing notes file
            new_content_path: Path to the new content file

        Returns:
            Dictionary with updated notes
        """
        # Read the existing notes
        with open(notes_path, "r") as f:
            existing_notes = json.load(f)

        # Read the new content
        with open(new_content_path, "r") as f:
            new_content = f.read()

        # Basic update structure
        updated_notes = {
            "source_file": notes_path,
            "update_date": datetime.now().isoformat(),
            "content": existing_notes.get("content", []),
        }

        return {"updated_notes": updated_notes}


class QualityManagerTool(Tool):
    """Tool for managing content quality using DeepSeek model"""

    name = "quality_manager_tool"
    description = (
        "Evaluate and manage quality of processed content using DeepSeek Reasoning"
    )
    inputs = {
        "content_path": {
            "type": "string",
            "description": "Path to the content to evaluate",
        },
        "content_type": {
            "type": "string",
            "description": "Type of content to evaluate (markdown, analysis, outline, notes)",
        },
        "quality_threshold": {
            "type": "number",
            "description": "Minimum quality score required (0-1)",
            "nullable": True,
        },
    }
    outputs = {
        "evaluation": {
            "type": "object",
            "description": "Quality evaluation results with reasoning",
            "properties": {
                "quality_score": {"type": "number"},
                "feedback": {"type": "array"},
                "reasoning": {"type": "string"},
                "assessment": {"type": "string"},
                "approved": {"type": "boolean"},
                "improvement_suggestions": {"type": "array"},
            },
        }
    }
    output_type = "object"

    def __init__(self):
        """Initialize the quality manager tool"""
        super().__init__()
        self.model_name = "groq/deepseek-r1-distill-llama-70b"  # Specify provider
        self.model_config = {
            "temperature": 0.2,
            "max_tokens": 2048,
            "model": self.model_name,
        }

        # Set up environment variables for litellm
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        os.environ["LITELLM_DISABLE_LOGGING"] = "true"  # Disable litellm logging

    def get_reasoning_evaluation(
        self, content_type: str, content: Union[str, Dict]
    ) -> Dict[str, Any]:
        """
        Get reasoning-based evaluation using the DeepSeek model

        Args:
            content_type: Type of content being evaluated
            content: Content to evaluate (string for markdown, dict for others)

        Returns:
            Dictionary with reasoning evaluation results
        """
        # Prepare content for evaluation
        if isinstance(content, dict):
            content_str = json.dumps(content, indent=2)
        else:
            content_str = content

        # Create evaluation prompt based on content type
        prompt = self.get_evaluation_prompt(content_type, content_str)

        # Call DeepSeek model for evaluation
        try:
            from litellm import completion

            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.model_config["temperature"],
                max_tokens=self.model_config["max_tokens"],
            )

            # Extract assessment, reasoning, and suggestions using regex
            response_text = response.choices[0].message.content
            assessment_match = re.search(
                r"Assessment:(.*?)(?=Reasoning:|$)", response_text, re.DOTALL
            )
            reasoning_match = re.search(
                r"Reasoning:(.*?)(?=Suggestions:|$)", response_text, re.DOTALL
            )
            suggestions_match = re.search(
                r"Suggestions:(.*?)$", response_text, re.DOTALL
            )

            # Process the matches
            assessment = (
                assessment_match.group(1).strip()
                if assessment_match
                else "No clear assessment provided"
            )
            reasoning = (
                reasoning_match.group(1).strip()
                if reasoning_match
                else "No clear reasoning provided"
            )
            suggestions = []
            if suggestions_match:
                suggestions = [
                    s.strip()
                    for s in suggestions_match.group(1).strip().split("\n")
                    if s.strip() and not s.strip().startswith("-")
                ]

            # Analyze the assessment text for quality indicators
            quality_indicators = {
                "excellent": [
                    "excellent",
                    "outstanding",
                    "exceptional",
                    "thorough",
                    "comprehensive",
                ],
                "good": ["good", "strong", "well", "solid", "effective"],
                "adequate": ["adequate", "satisfactory", "acceptable", "sufficient"],
                "needs_improvement": [
                    "needs improvement",
                    "could be better",
                    "lacking",
                    "limited",
                    "insufficient",
                ],
                "poor": ["poor", "inadequate", "weak", "missing", "incomplete"],
            }

            assessment_lower = assessment.lower()
            quality_level = None

            # Find the highest matching quality level
            for level, indicators in quality_indicators.items():
                if any(indicator in assessment_lower for indicator in indicators):
                    quality_level = level
                    break

            # Map quality level to score
            quality_scores = {
                "excellent": 0.9,
                "good": 0.8,
                "adequate": 0.7,
                "needs_improvement": 0.5,
                "poor": 0.3,
            }

            # If no clear quality indicators found in assessment, analyze content
            if not quality_level:
                # Count positive and negative terms
                positive_terms = [
                    "clear",
                    "detailed",
                    "organized",
                    "structured",
                    "complete",
                ]
                negative_terms = [
                    "unclear",
                    "missing",
                    "disorganized",
                    "incomplete",
                    "confusing",
                ]

                positive_count = sum(
                    term in assessment_lower for term in positive_terms
                )
                negative_count = sum(
                    term in assessment_lower for term in negative_terms
                )

                # Calculate relative quality
                total_terms = positive_count + negative_count
                if total_terms > 0:
                    relative_quality = positive_count / total_terms
                    quality_score = 0.5 + (
                        relative_quality * 0.4
                    )  # Scale to 0.5-0.9 range
                else:
                    quality_score = 0.6  # Default if no clear indicators
            else:
                quality_score = quality_scores[quality_level]

            # Add quality assessment to suggestions if score is low
            if quality_score < 0.7:
                suggestions.append(
                    f"Current quality score ({quality_score:.2f}) indicates room for improvement"
                )
                suggestions.append(
                    "Consider addressing the points mentioned in the assessment"
                )

            return {
                "assessment": assessment,
                "reasoning": reasoning,
                "suggestions": suggestions,
                "quality_score": quality_score,
            }

        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            return {
                "assessment": f"Error during evaluation: {str(e)}",
                "reasoning": "Model evaluation failed",
                "suggestions": ["Check model configuration and try again"],
                "quality_score": 0.5,
            }

    def get_evaluation_prompt(self, content_type: str, content: str) -> str:
        """
        Generate evaluation prompt based on content type

        Args:
            content_type: Type of content being evaluated
            content: Content to evaluate

        Returns:
            Evaluation prompt for the model
        """
        base_prompt = f"""You are an expert academic content evaluator using the DeepSeek Reasoning model.
Please evaluate the following {content_type} content and provide a detailed analysis in the following format:

Assessment:
[Provide a brief overall assessment of the content quality]

Reasoning:
[Provide detailed reasoning for your assessment, considering the evaluation criteria below]

Suggestions:
[List specific improvement suggestions, one per line]

Content type: {content_type}

Evaluation criteria:
"""

        criteria_map = {
            "markdown": """
- Content structure and organization
- Clarity and readability
- Academic rigor and depth
- Citation and reference quality
- Overall presentation""",
            "analysis": """
- Depth of analysis
- Logical flow and coherence
- Key concept identification
- Supporting evidence
- Critical thinking demonstration""",
            "outline": """
- Hierarchical structure
- Topic coverage
- Logical progression
- Section balance
- Clear headings and subheadings""",
            "notes": """
- Comprehensiveness
- Key point capture
- Organization and flow
- Clarity and conciseness
- Practical usability""",
        }

        prompt = base_prompt
        prompt += criteria_map.get(content_type, "")
        prompt += f"\n\nContent to evaluate:\n{content}\n\nProvide your evaluation:"

        return prompt

    def calculate_quality_score(self, assessment: str) -> float:
        """Calculate quality score based on the model's assessment"""
        # Look for explicit quality indicators in the assessment
        assessment = assessment.lower()

        if any(
            term in assessment for term in ["excellent", "outstanding", "exceptional"]
        ):
            return 0.9
        elif any(term in assessment for term in ["good", "strong", "well"]):
            return 0.8
        elif any(
            term in assessment for term in ["adequate", "satisfactory", "acceptable"]
        ):
            return 0.7
        elif any(
            term in assessment
            for term in ["needs improvement", "could be better", "lacking"]
        ):
            return 0.5
        elif any(term in assessment for term in ["poor", "inadequate", "insufficient"]):
            return 0.3
        else:
            # Default to middle score if no clear indicators
            return 0.6

    def forward(
        self, content_path: str, content_type: str, quality_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Evaluate content quality and provide feedback with reasoning

        Args:
            content_path: Path to the content file
            content_type: Type of content to evaluate
            quality_threshold: Minimum quality score required (default: 0.7)

        Returns:
            Dictionary with quality evaluation results and reasoning
        """
        try:
            # Read the content
            with open(content_path, "r", encoding="utf-8") as f:
                if content_type == "markdown":
                    content = f.read()
                else:
                    content = json.load(f)

            # Get reasoning-based evaluation
            reasoning_eval = self.get_reasoning_evaluation(content_type, content)

            # Calculate quality score based on reasoning
            quality_score = self.calculate_quality_score(reasoning_eval["assessment"])

            # Determine if content meets quality threshold
            approved = quality_score >= quality_threshold

            # Prepare improvement suggestions
            improvement_suggestions = []
            if not approved:
                improvement_suggestions = [
                    f"Current quality score ({quality_score:.2f}) is below threshold ({quality_threshold})",
                    "Consider the following improvements:",
                ] + reasoning_eval["suggestions"]

            return {
                "evaluation": {
                    "quality_score": quality_score,
                    "feedback": reasoning_eval["suggestions"],
                    "reasoning": reasoning_eval["reasoning"],
                    "assessment": reasoning_eval["assessment"],
                    "approved": approved,
                    "improvement_suggestions": improvement_suggestions,
                }
            }

        except Exception as e:
            print(f"\nError evaluating {content_type} quality for {content_path}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback

            print(f"Traceback:\n{traceback.format_exc()}")
            return {"error": str(e), "file": content_path, "content_type": content_type}


def process_with_quality_control(
    agent: "AcademicAgent",
    content_path: str,
    content_type: str,
    max_attempts: int = 3,
    quality_threshold: float = 0.7,
) -> bool:
    """
    Process content with quality control feedback loop

    Args:
        agent: Academic agent instance
        content_path: Path to content file
        content_type: Type of content being processed
        max_attempts: Maximum number of processing attempts
        quality_threshold: Minimum quality score required

    Returns:
        bool: True if processing succeeded within quality threshold, False otherwise
    """
    attempt = 1
    while attempt <= max_attempts:
        # Get quality evaluation
        quality_result = agent.quality_manager.forward(
            content_path, content_type, quality_threshold
        )

        if "error" in quality_result:
            print(f"\nError in quality evaluation: {quality_result['error']}")
            print(f"Quality check failed for {content_path}")
            return False

        quality_score = quality_result["evaluation"]["quality_score"]
        approved = quality_result["evaluation"]["approved"]

        if approved:
            print(f"\n✓ Quality check passed (Score: {quality_score:.2f})")
            return True

        print(f"\n⚠️ Analysis needs improvement (Attempt {attempt}/{max_attempts})")
        print(f"Quality Score: {quality_score:.2f}\n")

        if "improvement_suggestions" in quality_result["evaluation"]:
            print("Improvement Suggestions:")
            for suggestion in quality_result["evaluation"]["improvement_suggestions"]:
                print(f"- {suggestion}")

        if attempt < max_attempts:
            # Reprocess with feedback
            if content_type == "analysis":
                agent.analysis_tool.forward(content_path)
            elif content_type == "outline":
                agent.outline_tool.forward(content_path)
            elif content_type == "notes":
                agent.notes_tool.forward(content_path)

        attempt += 1

    print(f"\n❌ Failed to meet quality threshold after {max_attempts} attempts")
    return False


def setup_agent(api_key: str, base_dir: Optional[Path] = None):
    """Set up the academic agent"""
    print("\nDebug: Setting up academic agent...")
    print(f"Debug: API Key (first 10 chars): {api_key[:10]}...")

    try:
        # Set up base directories
        base_dir = base_dir or PROCESSED_DIR
        for dir_name in [
            "ingestion",
            "analysis",
            "outlines",
            "notes",
            "raw",
            "markdown",
            "metadata",
        ]:
            (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print("Debug: Directories created successfully")

        # Configure model
        model = LiteLLMModel(
            model_id="groq/llama-3.3-70b-versatile",
            api_key=api_key,
            api_base="https://api.groq.com/openai/v1",
            temperature=0.2,
            max_tokens=4096,
            stream=False,
        )
        print("Debug: Model configured successfully")

        # Create tools
        pdf_tool = PDFIngestionTool(base_dir)
        analysis_tool = ContentAnalysisTool(base_dir)
        outline_tool = OutlineGenerationTool()
        notes_tool = NotesGenerationTool()
        update_tool = ContentUpdateTool()
        quality_manager = QualityManagerTool()

        tools = [
            pdf_tool,
            analysis_tool,
            outline_tool,
            notes_tool,
            update_tool,
            quality_manager,
        ]
        print("Debug: Tools created successfully")

        # Create agent
        agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=4,
        )
        print("Debug: Agent created successfully")

        # Initialize tools with agent
        for tool in tools:
            tool.agent = agent
        print("Debug: Tools initialized with agent")

        # Store tools in agent for easy access
        agent.pdf_tool = pdf_tool
        agent.analysis_tool = analysis_tool
        agent.outline_tool = outline_tool
        agent.notes_tool = notes_tool
        agent.update_tool = update_tool
        agent.quality_manager = quality_manager

        return agent

    except Exception as e:
        print(f"\nError configuring agent: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        raise


def handle_help():
    """Display help information about available commands"""
    help_text = """
Available Commands:
-----------------
1. Process PDF:
   process pdf <path_to_pdf_or_directory> [--recursive] [--rename-smartly]
   Examples: 
   - Single file: process pdf documents/paper.pdf
   - Directory: process pdf documents/papers --recursive
   - No smart renaming: process pdf documents/paper.pdf --rename-smartly=false

2. Analyze Content:
   analyze content <path_to_markdown>
   Example: analyze content processed/paper.md

3. Generate Outline:
   generate outline <path_to_markdown>
   Example: generate outline processed/paper.md

4. Generate Notes:
   generate notes <path_to_markdown>
   Example: generate notes processed/paper.md

5. Update Notes:
   update notes <path_to_existing_notes> <path_to_new_content>
   Example: update notes notes/old.md notes/new.md

6. Help:
   help - Display this help message

7. Exit:
   exit or quit - Exit the program
"""
    print(help_text)


def main():
    """Main entry point for the academic agent"""
    parser = argparse.ArgumentParser(
        description="Academic Agent for processing and analyzing academic materials"
    )

    # Only keep essential arguments
    parser.add_argument("--api-key", help="Groq API key")
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device to use for PDF processing",
    )

    args = parser.parse_args()

    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    # Set up the agent
    agent = setup_agent(api_key)
    print("\nStarting automated processing pipeline with quality control...")

    try:
        # Step 1: Process all PDFs in the input directory
        input_dir = BASE_DIR / "input"
        if input_dir.exists():
            print("\n1. Processing PDFs from input directory...")
            pdf_result = agent.pdf_tool.forward(str(input_dir), recursive=True)
            print(f"PDF Processing Stats: {json.dumps(pdf_result['stats'], indent=2)}")
        else:
            print("\nNo input directory found. Skipping PDF processing.")

        # Step 2: Analyze all markdown files with quality control
        markdown_dir = PROCESSED_DIR / "markdown"
        if markdown_dir.exists() and any(markdown_dir.iterdir()):
            print("\n2. Analyzing markdown content with quality control...")
            for md_file in markdown_dir.glob("**/*.md"):
                print(f"\nProcessing: {md_file.name}")

                # Initial analysis
                analysis_result = agent.analysis_tool.forward(str(md_file))
                analysis_path = ANALYSIS_DIR / f"{md_file.stem}_analysis.json"

                # Quality control loop for analysis
                if process_with_quality_control(agent, str(analysis_path), "analysis"):
                    print(f"Analysis completed successfully for {md_file.name}")
                else:
                    print(f"Analysis quality check failed for {md_file.name}")
                    continue

                # Generate outline with quality control
                outline_result = agent.outline_tool.forward(str(md_file))
                outline_path = OUTLINES_DIR / f"{md_file.stem}_outline.json"

                if process_with_quality_control(agent, str(outline_path), "outline"):
                    print(f"Outline generated successfully for {md_file.name}")
                else:
                    print(f"Outline quality check failed for {md_file.name}")
                    continue

                # Generate notes with quality control
                notes_result = agent.notes_tool.forward(str(md_file))
                notes_path = NOTES_DIR / f"{md_file.stem}_notes.json"

                if process_with_quality_control(agent, str(notes_path), "notes"):
                    print(f"Notes generated successfully for {md_file.name}")
                else:
                    print(f"Notes quality check failed for {md_file.name}")

        print("\nProcessing pipeline completed!")
        print("\nOutput Locations:")
        print(f"- Processed Markdown: {PROCESSED_DIR / 'markdown'}")
        print(f"- Analysis Results: {PROCESSED_DIR / 'analysis'}")
        print(f"- Generated Outlines: {PROCESSED_DIR / 'outlines'}")
        print(f"- Comprehensive Notes: {PROCESSED_DIR / 'notes'}")

    except Exception as e:
        print(f"\nError in processing pipeline: {str(e)}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
