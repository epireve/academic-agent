#!/bin/bash

# Activate the virtual environment
source ../.venv/bin/activate

# Define source directory
SOURCE_DIR="/Users/invoture/Documents/UM Masters/Security Risk Analysis and Evaluation (WOA7017)/Old materials"

# Base directories
BASE_OUTPUT_DIR="/Users/invoture/dev.local/academic-agent/output"

# Default values - specifically setting 'sra' for Security Risk Analysis
DEFAULT_COURSE="sra"     # sra = Security Risk Analysis
DEFAULT_CATEGORY="lectures"  # Default category (lectures, notes, transcripts)

# Get course and category from command line arguments
COURSE=${1:-$DEFAULT_COURSE}
CATEGORY=${2:-$DEFAULT_CATEGORY}

# Create output directories
TARGET_DIR="${BASE_OUTPUT_DIR}/${COURSE}/${CATEGORY}/markdown"
IMAGE_DIR="${BASE_OUTPUT_DIR}/${COURSE}/${CATEGORY}/images"

mkdir -p "$TARGET_DIR"
mkdir -p "$IMAGE_DIR"

echo "Using output directories:"
echo "Markdown: $TARGET_DIR"
echo "Images: $IMAGE_DIR"

# Process all PDF files in the source directory
find "$SOURCE_DIR" -name "*.pdf" -type f | while read -r pdf_file; do
    echo "Processing: $pdf_file"
    python pdf_processor_agent.py --pdf "$pdf_file" --output "$BASE_OUTPUT_DIR" --course "$COURSE" --category "$CATEGORY"
done
