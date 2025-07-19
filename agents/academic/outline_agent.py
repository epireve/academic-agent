#!/usr/bin/env python
"""
Outline Agent - Specialized agent for creating structured outlines from academic content
Part of the Academic Agent system
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import networkx as nx
from groq import Groq
from .base_agent import BaseAgent

try:
    from smolagents import CodeAgent
    from smolagents import HfApiModel
except ImportError:
    print("Installing smolagents...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents import HfApiModel

# Define base paths
BASE_DIR = Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
OUTPUT_DIR = BASE_DIR / str(get_output_manager().outputs_dir)
PROCESSED_DIR = BASE_DIR / "processed"
OUTLINES_DIR = PROCESSED_DIR / "outlines"
ANALYSIS_DIR = PROCESSED_DIR / "analysis"


class OutlineAgent(BaseAgent):
    """Agent responsible for creating unified outline from markdown files"""

    def __init__(self, groq_api_key: str):
        super().__init__("outline_agent")
        self.groq = Groq(api_key=groq_api_key)
        self.concept_graph = nx.DiGraph()
        self.min_concepts_per_section = 5

    def create_unified_outline(self, markdown_files: List[str]) -> Dict[str, Any]:
        """Create a unified outline from multiple markdown files"""
        try:
            start_time = datetime.now()

            # Process all files
            processed_files = []
            for file_path in markdown_files:
                file_concepts = self._process_markdown_file(file_path)
                if file_concepts:
                    processed_files.append(
                        {"path": file_path, "concepts": file_concepts}
                    )

            # Generate concept map
            concept_map = self._generate_concept_map(processed_files)

            # Create hierarchical outline
            outline = self._create_hierarchical_outline(concept_map)

            # Generate section summaries
            outline_with_summaries = self._add_section_summaries(outline)

            # Save outline
            output = self._save_outline(outline_with_summaries)

            # Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()

            # Check quality
            quality_score = self.check_quality(
                {
                    "outline": outline_with_summaries,
                    "concept_map": concept_map,
                    "processed_files": processed_files,
                }
            )

            return {
                "success": True,
                "outline_path": output["outline_path"],
                "concept_map": concept_map,
                "quality_score": quality_score,
                "processing_metrics": {
                    "processing_time": processing_time,
                    "files_processed": len(processed_files),
                    "total_concepts": len(concept_map["nodes"]),
                },
            }

        except Exception as e:
            self.handle_error(e, {"operation": "outline_creation"})
            return {"success": False, "error": str(e)}

    def _process_markdown_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single markdown file to extract concepts"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract concepts using LLM
            concepts = self._extract_concepts(content)

            # Validate minimum concepts requirement
            if len(concepts) < self.min_concepts_per_section:
                self.logger.warning(
                    f"File {file_path} has fewer concepts than required: {len(concepts)}"
                )

            return {
                "main_concepts": concepts["main_concepts"],
                "relationships": concepts["relationships"],
                "importance_scores": concepts["importance_scores"],
            }

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def _extract_concepts(self, content: str) -> Dict[str, Any]:
        """Extract concepts from content using LLM"""
        prompt = """
        Analyze the following academic content and extract:
        1. Main concepts (minimum 5)
        2. Relationships between concepts
        3. Importance score for each concept (0-1)
        
        Content:
        {content}
        
        Return a JSON with three keys:
        - main_concepts: list of concept strings
        - relationships: list of [concept1, concept2, relationship_type]
        - importance_scores: dict mapping concepts to scores
        """

        response = self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "user",
                    "content": prompt.format(
                        content=content[:4000]
                    ),  # Limit content length
                }
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        # Parse LLM response
        try:
            result = response.choices[0].message.content
            import json

            concepts_data = json.loads(result)
            return concepts_data
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return {"main_concepts": [], "relationships": [], "importance_scores": {}}

    def _generate_concept_map(self, processed_files: List[Dict]) -> Dict[str, Any]:
        """Generate concept map from processed files"""
        # Clear existing graph
        self.concept_graph.clear()

        # Add nodes (concepts)
        for file_data in processed_files:
            for concept in file_data["concepts"]["main_concepts"]:
                if concept not in self.concept_graph:
                    self.concept_graph.add_node(
                        concept,
                        importance=file_data["concepts"]["importance_scores"].get(
                            concept, 0.5
                        ),
                    )

        # Add edges (relationships)
        for file_data in processed_files:
            for c1, c2, rel_type in file_data["concepts"]["relationships"]:
                self.concept_graph.add_edge(c1, c2, relationship=rel_type)

        return {
            "nodes": list(self.concept_graph.nodes(data=True)),
            "edges": list(self.concept_graph.edges(data=True)),
        }

    def _create_hierarchical_outline(
        self, concept_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create hierarchical outline from concept map"""
        # Sort concepts by importance
        sorted_concepts = sorted(
            concept_map["nodes"], key=lambda x: x[1].get("importance", 0), reverse=True
        )

        # Group related concepts
        outline = {"title": "Unified Knowledge Outline", "sections": []}

        processed_concepts = set()

        for concept, data in sorted_concepts:
            if concept in processed_concepts:
                continue

            # Find related concepts
            related = [
                (n2, rel["relationship"])
                for n1, n2, rel in concept_map["edges"]
                if n1 == concept and n2 not in processed_concepts
            ]

            section = {
                "title": concept,
                "importance": data.get("importance", 0.5),
                "subsections": [
                    {"title": rel_concept, "relationship": rel_type}
                    for rel_concept, rel_type in related
                ],
            }

            outline["sections"].append(section)
            processed_concepts.add(concept)
            processed_concepts.update(rc for rc, _ in related)

        return outline

    def _add_section_summaries(self, outline: Dict[str, Any]) -> Dict[str, Any]:
        """Add summaries to outline sections using LLM"""
        for section in outline["sections"]:
            prompt = f"""
            Create a brief summary for an academic outline section with:
            Title: {section['title']}
            Related concepts: {[sub['title'] for sub in section['subsections']]}
            
            Return a 2-3 sentence summary that explains the main concept and its relationship
            to the related concepts.
            """

            response = self.groq.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            section["summary"] = response.choices[0].message.content

        return outline

    def _save_outline(self, outline: Dict[str, Any]) -> Dict[str, str]:
        """Save outline to file"""
        outline_path = os.path.join("processed", "outline.md")

        # Convert outline to markdown
        markdown_content = self._convert_outline_to_markdown(outline)

        # Save file
        with open(outline_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return {"outline_path": outline_path}

    def _convert_outline_to_markdown(self, outline: Dict[str, Any]) -> str:
        """Convert outline dictionary to markdown format"""
        markdown = [f"# {outline['title']}\n", "## Table of Contents\n"]

        # Add TOC
        for i, section in enumerate(outline["sections"], 1):
            markdown.append(f"{i}. [{section['title']}](#section-{i})\n")

        markdown.append("\n---\n")

        # Add sections
        for i, section in enumerate(outline["sections"], 1):
            markdown.extend(
                [
                    f"## Section {i}: {section['title']}\n",
                    f"*Importance Score: {section['importance']:.2f}*\n\n",
                    f"{section['summary']}\n\n",
                    "### Related Concepts:\n",
                ]
            )

            for subsection in section["subsections"]:
                markdown.append(
                    f"- **{subsection['title']}** ({subsection['relationship']})\n"
                )

            markdown.append("\n")

        return "".join(markdown)

    def check_quality(self, content: Dict[str, Any]) -> float:
        """Check quality of generated outline"""
        quality_score = 1.0
        deductions = []

        # Check number of sections
        if len(content["outline"]["sections"]) < 3:
            deductions.append(0.3)

        # Check concept coverage
        total_concepts = len(content["concept_map"]["nodes"])
        if (
            total_concepts
            < len(content["processed_files"]) * self.min_concepts_per_section
        ):
            deductions.append(0.2)

        # Check relationships
        if not content["concept_map"]["edges"]:
            deductions.append(0.3)

        # Check summaries
        for section in content["outline"]["sections"]:
            if "summary" not in section or not section["summary"]:
                deductions.append(0.1)

        # Apply deductions
        for deduction in deductions:
            quality_score -= deduction

        return max(0.0, min(1.0, quality_score))

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if not isinstance(input_data, list):
            return False
        return all(isinstance(f, str) and os.path.exists(f) for f in input_data)

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if not isinstance(output_data, dict):
            return False

        required_fields = ["success", "outline_path", "concept_map", "quality_score"]
        return all(field in output_data for field in required_fields)


def main():
    """Main entry point for the outline agent"""
    parser = argparse.ArgumentParser(
        description="Outline Agent for creating academic outlines"
    )

    # Define command line arguments
    parser.add_argument("--files", nargs="+", help="Paths to markdown files to outline")
    parser.add_argument("--analysis", help="Path to existing analysis file (optional)")
    parser.add_argument("--output", help="Custom output path for the outline")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also generate a markdown version of the outline",
    )
    parser.add_argument("--api-key", help="Groq API key")

    args = parser.parse_args()

    # Get API key from environment or command line
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    # Check for required arguments
    if not args.files:
        print("Error: At least one markdown file must be specified with --files")
        parser.print_help()
        sys.exit(1)

    # Create the outline agent
    agent = OutlineAgent(api_key)

    # Create the outline
    outline = agent.create_unified_outline(args.files)

    # Determine output path for the JSON outline
    if args.output:
        output_path = args.output
    else:
        # Generate a default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(args.files) == 1:
            base_name = os.path.basename(args.files[0]).replace(".md", "")
            outline_filename = f"{base_name}_outline_{timestamp}.json"
        else:
            outline_filename = f"combined_outline_{timestamp}.json"
        output_path = str(OUTLINES_DIR / outline_filename)

    # Save the outline in JSON format
    with open(output_path, "w") as f:
        json.dump(outline, f, indent=2)

    print(f"Outline created and saved to: {output_path}")

    # Generate markdown version if requested
    if args.markdown:
        markdown_path = output_path.replace(".json", ".md")
        agent.generate_markdown_outline(outline, markdown_path)
        print(f"Markdown outline saved to: {markdown_path}")


if __name__ == "__main__":
    main()
