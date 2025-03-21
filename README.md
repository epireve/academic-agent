# Academic Agent

An intelligent academic content processing system built with smolagents framework and Groq LLM. This agent specializes in converting academic PDFs into structured, searchable knowledge by leveraging:

- **smolagents**: A powerful agent framework for building composable AI systems
- **Groq LLM**: State-of-the-art language model for content analysis
- **docling**: Advanced PDF processing with image and formula preservation

The system intelligently processes academic materials, extracts key concepts, generates structured outlines, and creates comprehensive notes while maintaining semantic relationships between different pieces of content.

## Project Structure

- **agents/academic/**: Contains the main agent implementations
  - `academic_agent.py`: Main agent orchestrating all operations
  - `ingestion_agent.py`: Handles PDF ingestion and initial processing
  - `analysis_agent.py`: Analyzes content and extracts key information
  - `outline_agent.py`: Generates structured outlines
  - `notes_agent.py`: Creates comprehensive notes
  - `update_agent.py`: Updates existing notes with new information
  - `requirements.txt`: Project dependencies

- **tools/**: Reusable tools and utilities
  - **pdf_processor/**: PDF processing tools
    - `processor.py`: Core PDF processing functionality
    - `cli.py`: Command-line interface
    - `__init__.py`: Package initialization

- **processed/**: Contains processed data (gitignored)
  - `ingestion/`: Initial processed PDFs and metadata
  - `analysis/`: Content analysis results
  - `outlines/`: Generated outlines
  - `notes/`: Comprehensive notes

## Getting Started

### Prerequisites

- Python 3.8+
- Groq API key for LLM functionality
- Virtual environment setup

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd academic-agent
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r agents/academic/requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Groq API key and other settings
   ```

### Usage

The agent can be used in two modes:

1. Command-line mode with specific actions:
   ```bash
   python -m agents.academic.academic_agent --pdf /path/to/file.pdf --analyze --generate-outline --generate-notes
   ```

2. Interactive mode for ongoing operations:
   ```bash
   python -m agents.academic.academic_agent --interactive
   ```

### Features

- Smart PDF processing with content-based file naming
- Automatic metadata extraction and organization
- Content analysis and key concept identification
- Structured outline generation with semantic relationships
- Comprehensive notes creation with cross-references
- Support for updating existing notes with new information
- Interactive mode for complex workflows
- Groq LLM integration for intelligent processing
- Built on smolagents framework for robust agent interactions

### Configuration

The following environment variables can be configured in `.env`:

- `GROQ_API_KEY`: Your Groq API key (required)
- `PDF_PROCESSOR_DEVICE`: Device to use for PDF processing (cpu/mps/cuda)
- `OUTPUT_BASE_DIR`: Base directory for processed files
- `LOG_LEVEL`: Logging verbosity (default: INFO)

## Architecture

The system uses the smolagents framework to create a network of specialized agents that work together:

1. **Ingestion Agent**: Handles initial PDF processing and metadata extraction
2. **Analysis Agent**: Performs deep content analysis and concept mapping
3. **Outline Agent**: Creates structured outlines based on content relationships
4. **Notes Agent**: Generates comprehensive notes with cross-references
5. **Update Agent**: Manages content updates and version control

Each agent is designed to be modular and can be extended or modified independently.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
