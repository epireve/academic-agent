# Academic Agent System

An intelligent system for processing academic materials using smolagents framework and Groq LLM.

## Overview

This system consists of specialized agents that work together to process PDF files, analyze content, create outlines, generate comprehensive notes, and update materials as new information becomes available.

## Agent Components

1. **Main Academic Agent** - Coordinates the overall process
2. **Ingestion Agent** - Processes PDF files using docling
3. **Analysis Agent** - Analyzes content for structure and key concepts
4. **Outline Agent** - Creates structured outlines
5. **Notes Agent** - Generates comprehensive notes
6. **Update Agent** - Updates existing materials with new information

## Directory Structure

```
processed/
├── analysis/ - Content analysis results
├── outlines/ - Generated outlines
└── notes/ - Comprehensive notes
```

## Requirements

- Python 3.8+
- smolagents
- Groq API key

## Installation

1. Install dependencies:
   ```bash
   pip install -r agents/academic/requirements.txt
   ```

2. Set up your Groq API key:
   ```bash
   export GROQ_API_KEY="your-api-key"
   ```

## Usage

### Main Academic Agent

```bash
python agents/academic/academic_agent.py --pdf /path/to/file.pdf --output /path/to/output --analyze --generate-outline --generate-notes
```

Or use interactive mode:
```bash
python agents/academic/academic_agent.py --interactive
```

### Individual Agents

#### Ingestion Agent
```bash
python agents/academic/ingestion_agent.py --pdf /path/to/file.pdf --output /path/to/output
```

#### Analysis Agent
```bash
python agents/academic/analysis_agent.py --file /path/to/markdown.md
```

#### Outline Agent
```bash
python agents/academic/outline_agent.py --files /path/to/markdown1.md /path/to/markdown2.md --markdown
```

#### Notes Agent
```bash
python agents/academic/notes_agent.py --sources /path/to/markdown1.md /path/to/markdown2.md --outline /path/to/outline.json
```

#### Update Agent
```bash
python agents/academic/update_agent.py --notes /path/to/notes.md --sources /path/to/new1.md /path/to/new2.md --style inline
```

## Agent Workflow

1. **PDF Ingestion**: Convert PDFs to markdown files
2. **Content Analysis**: Identify structure, topics, and key concepts
3. **Outline Creation**: Generate a comprehensive outline
4. **Notes Generation**: Create detailed notes following the outline
5. **Content Updates**: Incorporate new information as it becomes available

## Customization

Each agent can be used independently or as part of the overall system. The main academic agent provides a unified interface to coordinate the entire process.
