#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Define source and target directories
SOURCE_DIR="/Users/invoture/Documents/UM Masters/Security Risk Analysis and Evaluation (WOA7017)/Old materials"
TARGET_DIR="/Users/invoture/dev.local/mse-st/sra/markdown"
IMAGE_DIR="/Users/invoture/dev.local/mse-st/sra/images"

# Ensure the image directory exists
mkdir -p "$IMAGE_DIR"

# Convert all PDF files with enriched features for image handling
find "$SOURCE_DIR" -name "*.pdf" | while read -r pdf_path; do
    # Get the filename without the directory path
    filename=$(basename "$pdf_path")
    # Remove the .pdf extension
    filename_no_ext="${filename%.pdf}"
    # Target markdown file
    target_file="$TARGET_DIR/$filename_no_ext.md"
    
    echo "Converting $pdf_path to $target_file with enhanced image handling"
    
    # Convert PDF to markdown with enhanced image handling
    # Using image-export-mode as referenced to maintain image references
    # Enable picture classification and formula enrichment
    docling --to md \
        --output "$TARGET_DIR" \
        --device mps \
        --image-export-mode referenced \
        --enrich-picture-classes \
        --enrich-formula \
        "$pdf_path"
done

echo "Conversion completed with enhanced image handling!"
