#!/usr/bin/env python
"""
Advanced PDF Processing Agent using smolagents framework
This agent uses CodeAgent to intelligently process PDFs using docling
"""

import os
import sys
import subprocess
from typing import List, Dict, Any, Optional
import argparse

try:
    from smolagents import CodeAgent
    from smolagents.models import OpenAIModel, HuggingfaceModel
except ImportError:
    print("smolagents not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "smolagents"])
    from smolagents import CodeAgent
    from smolagents.models import OpenAIModel, HuggingfaceModel

# Define PDF processing tools
class DoclingTools:
    """Collection of tools for processing PDFs with docling"""
    
    def convert_pdf_to_markdown(self, pdf_path: str, output_dir: str, 
                              image_mode: str = "referenced",
                              enrich_pictures: bool = True,
                              enrich_formulas: bool = True) -> str:
        """
        Convert a PDF to Markdown using docling with enhanced image handling
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory where to save the markdown file
            image_mode: How to handle images (referenced, embedded, placeholder)
            enrich_pictures: Whether to enable picture classification
            enrich_formulas: Whether to enable formula enrichment
            
        Returns:
            Path to the generated markdown file
        """
        if not os.path.exists(pdf_path):
            return f"Error: PDF file {pdf_path} does not exist"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Build the docling command
        cmd = ["docling", "--to", "md", "--output", output_dir, "--device", "mps", 
              f"--image-export-mode={image_mode}"]
        
        if enrich_pictures:
            cmd.append("--enrich-picture-classes")
        
        if enrich_formulas:
            cmd.append("--enrich-formula")
        
        cmd.append(pdf_path)
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            filename = os.path.basename(pdf_path)
            filename_no_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{filename_no_ext}.md")
            return f"Successfully converted {pdf_path} to {output_path}"
        except subprocess.CalledProcessError as e:
            return f"Error converting PDF: {e.stderr}"
    
    def analyze_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF file to extract metadata and content structure
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata and content structure
        """
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file {pdf_path} does not exist"}
        
        try:
            # This is a placeholder for actual PDF analysis
            # In a real implementation, we would use a library like PyPDF2 or pdfminer
            return {
                "filename": os.path.basename(pdf_path),
                "path": pdf_path,
                "size_bytes": os.path.getsize(pdf_path),
                "pages": "Unknown", # Would be determined by analysis
                "has_images": "Unknown", # Would be determined by analysis
                "has_tables": "Unknown", # Would be determined by analysis
                "has_formulas": "Unknown", # Would be determined by analysis
            }
        except Exception as e:
            return {"error": f"Error analyzing PDF: {str(e)}"}
    
    def list_pdfs_in_directory(self, directory: str) -> List[Dict[str, Any]]:
        """
        List all PDF files in a directory with basic metadata
        
        Args:
            directory: Path to the directory to scan
            
        Returns:
            List of dictionaries containing PDF filenames and paths
        """
        if not os.path.exists(directory):
            return [{"error": f"Directory {directory} does not exist"}]
        
        pdfs = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    pdfs.append({
                        "filename": file,
                        "path": pdf_path,
                        "size_bytes": os.path.getsize(pdf_path)
                    })
        
        return pdfs

def setup_agent(api_key: Optional[str] = None, use_openai: bool = False):
    """
    Set up and return the PDF processing agent
    
    Args:
        api_key: API key for the model provider (OpenAI or HuggingFace)
        use_openai: Whether to use OpenAI (True) or HuggingFace (False)
        
    Returns:
        Configured CodeAgent
    """
    # Set up the tools
    tools = [DoclingTools()]
    
    # Set up the model
    if use_openai and api_key:
        model = OpenAIModel(model="gpt-4o", api_key=api_key)
    else:
        # Default to a HuggingFace model if no OpenAI key provided
        model = HuggingfaceModel(model="HuggingFaceH4/zephyr-7b-beta")
    
    # Create and return the agent
    return CodeAgent(tools=tools, model=model)

def main():
    """Main function to set up and run the agent"""
    parser = argparse.ArgumentParser(description="Advanced PDF Processing Agent")
    parser.add_argument("--pdf", help="Path to a PDF file to process")
    parser.add_argument("--dir", help="Directory containing PDFs to process")
    parser.add_argument("--output", required=True, help="Output directory for markdown files")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI model (requires API key)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # If not using an agent, just use the tools directly
    if not args.interactive and not args.use_openai:
        tools = DoclingTools()
        
        if args.pdf:
            result = tools.convert_pdf_to_markdown(args.pdf, args.output)
            print(result)
        elif args.dir:
            pdfs = tools.list_pdfs_in_directory(args.dir)
            for pdf in pdfs:
                if "error" not in pdf:
                    result = tools.convert_pdf_to_markdown(pdf["path"], args.output)
                    print(result)
        else:
            print("Error: Please provide either --pdf or --dir argument")
            parser.print_help()
            sys.exit(1)
    else:
        # Set up the agent
        agent = setup_agent(api_key=args.openai_key, use_openai=args.use_openai)
        
        if args.interactive:
            # Interactive mode
            print("PDF Processing Agent (interactive mode)")
            print("Type 'exit' to quit")
            
            while True:
                user_input = input("\nWhat would you like to do? ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                response = agent.run(user_input)
                print(f"\nAgent: {response}")
        else:
            # Non-interactive mode with an agent
            if args.pdf:
                prompt = f"Convert the PDF file at {args.pdf} to markdown and save it to {args.output} directory. Use referenced image mode and enable picture classification and formula enrichment."
                response = agent.run(prompt)
                print(response)
            elif args.dir:
                prompt = f"Process all PDF files in the directory {args.dir} and convert them to markdown files in the {args.output} directory. Use referenced image mode and enable picture classification and formula enrichment."
                response = agent.run(prompt)
                print(response)

if __name__ == "__main__":
    main()
