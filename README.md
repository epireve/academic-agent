# Academic Agent

An intelligent agent for processing academic material, particularly converting PDF files to Markdown while maintaining the richness of the content including images, diagrams, and formulas.

## Project Structure

- **docling-processor/**: Contains scripts and tools for PDF to Markdown conversion
  - `convert_pdfs.sh`: Bash script for batch conversion of PDFs
  - `pdf_processor_agent.py`: Simple agent for PDF processing
  - `advanced_pdf_agent.py`: Advanced agent using smolagents framework
  - `requirements.txt`: Project dependencies
  - `schema/`: Contains JSON schemas for document metadata

- **.venv/**: Virtual environment with installed dependencies

- **output/**: Contains processed data
  - `{course}/`: Course category (e.g., sra for Security Risk Analysis)
    - `lectures/`: Course slides and presentation materials
      - `markdown/`: Converted markdown files
      - `images/`: Extracted images from PDFs
    - `notes/`: Personal notes and supplementary materials
      - `markdown/`: Converted markdown files
      - `images/`: Extracted images from PDFs
    - `transcripts/`: Class transcriptions and recordings
      - `markdown/`: Converted markdown files
      - `images/`: Extracted images from PDFs

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js (for npm/npx tools)
- Virtual environment setup

### Installation

1. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Install additional dependencies if needed:
   ```bash
   uv pip install -r docling-processor/requirements.txt
   ```

### Usage

#### Basic PDF Conversion

To convert a PDF to Markdown while preserving images and formulas:

```bash
cd docling-processor
./convert_pdfs.sh [course] [category]
```

Example:
```bash
./convert_pdfs.sh sra lectures
```

#### Using the Python Agent

For more advanced processing:

```bash
python docling-processor/pdf_processor_agent.py --pdf /path/to/your.pdf --course sra --category lectures
```

Or process an entire directory:

```bash
python docling-processor/pdf_processor_agent.py --dir /path/to/pdf/directory --course sra --category notes
```

#### Using the Advanced Agent

For automatic metadata extraction from file paths:

```bash
python docling-processor/advanced_pdf_agent.py --pdf /path/to/your.pdf --auto-metadata
```

For interactive processing with the smolagents framework:

```bash
python docling-processor/advanced_pdf_agent.py --interactive
```

## Output Structure

The project uses a simplified folder structure organized by courses and categories:

```
output/
└── sra/                      # Course (e.g., Security Risk Analysis)
    ├── lectures/             # Course slides and presentations
    │   ├── markdown/         # Converted markdown files
    │   └── images/          # Extracted images
    ├── notes/               # Personal notes
    │   ├── markdown/
    │   └── images/
    └── transcripts/         # Class transcriptions
        ├── markdown/
        └── images/
```

## Features

- Converts PDF files to Markdown format
- Preserves images and formulas
- Organizes content by course and category
- Supports batch processing of multiple PDFs
- Automatic metadata extraction
- Enhanced image handling with classification
- Formula enrichment and LaTeX support

## Framework

This project uses:
- [docling](https://github.com/docling-project/docling) for PDF processing
- [smolagents](https://github.com/huggingface/smolagents) for intelligent agent capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details
