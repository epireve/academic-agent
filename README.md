# Academic Agent

An intelligent agent for processing academic material, particularly converting PDF files to Markdown while maintaining the richness of the content including images, diagrams, and formulas.

## Project Structure

- **docling-processor/**: Contains scripts and tools for PDF to Markdown conversion
  - `.venv/`: Virtual environment with installed dependencies
  - `convert_pdfs.sh`: Bash script for batch conversion of PDFs
  - `pdf_processor_agent.py`: Simple agent for PDF processing
  - `advanced_pdf_agent.py`: Advanced agent using smolagents framework
  - `requirements.txt`: Project dependencies

- **.rules/**: Contains project rules and guidelines
  - `academic_agent_rules.md`: Rules based on smolagents framework

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for npm/npx tools)
- Virtual environment setup

### Installation

1. Activate the virtual environment:
   ```bash
   cd docling-processor
   source .venv/bin/activate
   ```

2. Install additional dependencies if needed:
   ```bash
   uv pip install -r requirements.txt
   ```

### Usage

#### Basic PDF Conversion

To convert a PDF to Markdown while preserving images and formulas:

```bash
./convert_pdfs.sh
```

#### Using the Python Agent

For more advanced processing:

```bash
python pdf_processor_agent.py --pdf /path/to/your.pdf --output /output/directory
```

Or process an entire directory:

```bash
python pdf_processor_agent.py --dir /path/to/pdf/directory --output /output/directory
```

#### Using the Advanced Agent

For interactive processing with the smolagents framework:

```bash
python advanced_pdf_agent.py --interactive --output /output/directory
```

## Features

- PDF to Markdown conversion with docling
- Preservation of images, diagrams, and charts
- Formula understanding and conversion
- Picture classification and captioning
- Batch processing of multiple PDF files
- Intelligent agent capabilities with smolagents

## Framework

This project uses:
- [docling](https://github.com/docling-project/docling) for PDF processing
- [smolagents](https://github.com/huggingface/smolagents) for intelligent agent capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.
