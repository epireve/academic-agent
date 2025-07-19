#!/usr/bin/env python
"""
Analysis Agent - Specialized agent for analyzing academic content and extracting
key concepts and relationships.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from smolagents import CodeAgent
    from smolagents import HfApiModel
except ImportError:
    print("Installing smolagents...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents import HfApiModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Define base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = BASE_DIR / str(get_output_manager().outputs_dir)
ANALYSIS_DIR = BASE_DIR / "processed" / "analysis"


class AnalysisAgent:
    """
    Agent specialized in analyzing academic content, identifying structure,
    themes, topics, and key concepts
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Get API key from environment if not provided
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it to the constructor.")
            
        # Configure Groq model
        self.model = HfApiModel(
            model_id="llama3-70b-8192",
            provider="groq",
            api_key=self.api_key
        )
        
        # Create the agent
        self.agent = CodeAgent(
            tools=[],  # No special tools needed for this agent
            model=self.model
        )
        
        # Ensure analysis directory exists
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        
    def analyze_document(self, markdown_path: str) -> Dict[str, Any]:
        """
        Analyze a single document and extract key elements
        
        Args:
            markdown_path: Path to the markdown file
            
        Returns:
            Dictionary of analysis results
        """
        if not os.path.exists(markdown_path):
            return {"error": f"File not found: {markdown_path}"}
            
        # Read the markdown file
        with open(markdown_path, 'r') as f:
            content = f.read()
            
        # Keep the content within token limits
        content_to_analyze = content[:8000]  # First 8000 chars as a sample
        
        # Ask the LLM to analyze the content
        prompt = f"""
        Perform a comprehensive analysis of the following academic content:
        
        ```
        {content_to_analyze}
        ```
        
        Provide a detailed analysis including:
        
        1. Main subject area and sub-disciplines
        2. Key concepts and theories discussed
        3. Hierarchical structure of topics
        4. Important definitions, formulas, or frameworks
        5. Potential connections to other academic areas
        
        Return your analysis as a structured JSON object.
        """
        
        response = self.agent.run(prompt)
        
        # In a full implementation, we would parse the JSON response
        # For now, we'll create a basic analysis file
        
        # Generate output path
        filename = os.path.basename(markdown_path).replace('.md', '_analysis.json')
        output_path = ANALYSIS_DIR / filename
        
        # Create a placeholder analysis result
        analysis = {
            "source_file": markdown_path,
            "analysis_timestamp": None,  # Would be set in a full implementation
            "subject_area": None,
            "key_concepts": [],
            "structure": {},
            "definitions": [],
            "connections": []
        }
        
        # Save the analysis
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return analysis
    
    def analyze_multiple_documents(self, markdown_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze multiple documents and synthesize their content
        
        Args:
            markdown_paths: List of paths to markdown files
            
        Returns:
            Dictionary of combined analysis results
        """
        individual_analyses = []
        
        # First analyze each document individually
        for path in markdown_paths:
            analysis = self.analyze_document(path)
            individual_analyses.append(analysis)
            
        # Then synthesize the results
        return self.synthesize_analyses(individual_analyses)
    
    def synthesize_analyses(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize multiple individual analyses into a comprehensive overview
        
        Args:
            analyses: List of individual document analyses
            
        Returns:
            Dictionary of synthesized analysis
        """
        # This is a placeholder for the combined analysis
        # In a full implementation, this would involve more sophisticated logic
        # and likely additional LLM queries
        
        combined = {
            "source_count": len(analyses),
            "sources": [a.get("source_file") for a in analyses if "source_file" in a],
            "subject_areas": [],
            "key_concepts": [],
            "integrated_structure": {},
        }
        
        # Generate output path for the combined analysis
        output_path = ANALYSIS_DIR / "combined_analysis.json"
        
        # Save the combined analysis
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
            
        return combined


def main():
    """Main entry point for the analysis agent"""
    parser = argparse.ArgumentParser(description="Analysis Agent for academic content")
    
    # Define command line arguments
    parser.add_argument("--file", help="Path to a markdown file to analyze")
    parser.add_argument("--dir", help="Directory containing markdown files to analyze")
    parser.add_argument("--multiple", nargs='+', help="Multiple specific files to analyze together")
    parser.add_argument("--api-key", help="Groq API key")
    
    args = parser.parse_args()
    
    # Get API key from environment or command line
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create the analysis agent
    agent = AnalysisAgent(api_key)
    
    # Process according to the arguments
    if args.file:
        result = agent.analyze_document(args.file)
        print(f"Analyzed file: {args.file}")
        print(f"Output saved to: {ANALYSIS_DIR}")
    
    elif args.dir:
        # Get all markdown files in the directory
        markdown_files = []
        for root, _, files in os.walk(args.dir):
            for file in files:
                if file.lower().endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
        
        if markdown_files:
            result = agent.analyze_multiple_documents(markdown_files)
            print(f"Analyzed {len(markdown_files)} files from directory: {args.dir}")
            print(f"Output saved to: {ANALYSIS_DIR}")
        else:
            print(f"No markdown files found in directory: {args.dir}")
    
    elif args.multiple:
        result = agent.analyze_multiple_documents(args.multiple)
        print(f"Analyzed {len(args.multiple)} specified files")
        print(f"Output saved to: {ANALYSIS_DIR}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
