# Academic Agent

A simple CLI tool for converting PDF files to Markdown while preserving images, diagrams, and formulas using docling.

## Project Structure

- **docling-processor/**: Contains scripts for PDF to Markdown conversion
  - `pdf_processor_agent.py`: CLI tool for PDF processing
  - `requirements.txt`: Project dependencies

- **.venv/**: Virtual environment with installed dependencies

- **output/**: Contains processed data (gitignored)
  - Markdown files
  - Extracted images

## Getting Started

### Prerequisites

- Python 3.8+
- docling (will be installed via requirements.txt)
- Virtual environment setup

### Installation

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -r docling-processor/requirements.txt
   ```

### Usage

The tool provides two main functionalities:

1. Convert a single PDF file:
   ```bash
   python docling-processor/pdf_processor_agent.py --pdf /path/to/file.pdf --output /path/to/output
   ```

2. Process all PDFs in a directory:
   ```bash
   python docling-processor/pdf_processor_agent.py --dir /path/to/pdfs --output /path/to/output
   ```

### Features

- PDF to Markdown conversion
- Automatic image extraction and referencing
- Picture classification
- Formula enrichment and LaTeX support
- Support for batch processing

### Arguments

- `--pdf`: Path to a single PDF file to convert
- `--dir`: Path to a directory containing PDFs to convert
- `--output`: Output directory for markdown files and images (required)
