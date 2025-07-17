# High-Performance PDF Processor

A comprehensive PDF processing system using the Marker library for superior PDF-to-markdown conversion, specifically designed for academic documents.

## Features

- **High-Performance Processing**: Utilizes the Marker library for fast and accurate PDF-to-markdown conversion
- **GPU Acceleration**: Supports CUDA and MPS (Apple Silicon) for enhanced performance
- **Batch Processing**: Efficient processing of multiple PDFs with configurable batch sizes
- **Chapter Splitting**: Intelligent chapter detection and splitting for textbooks
- **Image Extraction**: Automatic extraction and handling of images from PDFs
- **Academic Optimization**: Specialized handling for academic documents (papers, textbooks, slides)
- **Comprehensive Monitoring**: Real-time performance tracking and alerting
- **Error Handling**: Robust error handling with retry mechanisms
- **Caching**: Intelligent caching to avoid reprocessing unchanged files
- **CLI Interface**: Command-line interface for easy integration

## Installation

1. Install the Marker library:
```bash
pip install marker-pdf
```

2. Install optional dependencies for better performance:
```bash
pip install torch torchvision  # For GPU acceleration
pip install pillow  # For image processing
pip install transformers  # For enhanced text recognition
```

3. Install monitoring dependencies:
```bash
pip install psutil  # For system monitoring
pip install pyyaml  # For configuration files
```

## Quick Start

### Basic Usage

```python
import asyncio
from pathlib import Path
from marker_pdf_processor import create_pdf_processor

async def main():
    # Create processor with default configuration
    processor = create_pdf_processor()
    
    # Process a single PDF
    pdf_path = Path("document.pdf")
    output_dir = Path("output")
    
    result = await processor.process_single_pdf(pdf_path, output_dir)
    
    if result.success:
        print(f"✅ Processing successful!")
        print(f"Output: {result.output_path}")
        print(f"Processing time: {result.metrics.processing_time:.2f}s")
        print(f"Pages processed: {result.metrics.pages_processed}")
    else:
        print(f"❌ Processing failed: {result.error_message}")

asyncio.run(main())
```

### Batch Processing

```python
import asyncio
from pathlib import Path
from marker_pdf_processor import create_pdf_processor

async def batch_process():
    # Create processor with custom configuration
    config = {
        'batch_size': 3,
        'max_workers': 4,
        'extract_images': True,
        'split_chapters': True
    }
    processor = create_pdf_processor(config)
    
    # Process multiple PDFs
    pdf_paths = [
        Path("document1.pdf"),
        Path("document2.pdf"),
        Path("document3.pdf")
    ]
    
    output_dir = Path("batch_output")
    
    results = await processor.process_batch(pdf_paths, output_dir)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"Batch processing completed:")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Success rate: {len(successful)/len(results)*100:.1f}%")

asyncio.run(batch_process())
```

### Command Line Interface

```bash
# Process a single PDF
python pdf_processor_cli.py process input.pdf output/

# Process multiple PDFs with custom settings
python pdf_processor_cli.py batch input_directory/ output/ --batch-size 3 --max-workers 4

# Process with GPU acceleration
python pdf_processor_cli.py process textbook.pdf output/ --device cuda

# Disable chapter splitting
python pdf_processor_cli.py process document.pdf output/ --no-split-chapters

# Show processing statistics
python pdf_processor_cli.py stats

# Get help
python pdf_processor_cli.py --help
```

## Configuration

### YAML Configuration

Create a `pdf_processor_config.yaml` file:

```yaml
# Device Configuration
device:
  type: auto  # auto, cpu, cuda, mps
  use_gpu: true

# Processing Configuration
processing:
  batch_size: 2
  max_workers: 4
  max_pages: null  # null for unlimited
  split_chapters: true
  extract_images: true

# Marker Library Configuration
marker:
  models:
    enable_editor_model: true
    enable_ocr: true
    default_language: en
  
  quality:
    extract_tables: true
    extract_equations: true
    preserve_formatting: true

# Monitoring Configuration
monitoring:
  enabled: true
  log_level: INFO
  
  alerts:
    memory_threshold: 80.0
    time_threshold: 300.0
    error_threshold: 10.0
```

### Python Configuration

```python
config = {
    'device': 'auto',
    'batch_size': 2,
    'max_workers': 4,
    'extract_images': True,
    'split_chapters': True,
    'enable_editor_model': True,
    'enable_ocr': True,
    'max_pages': None,
    'output_format': 'markdown'
}

processor = create_pdf_processor(config)
```

## Advanced Features

### Chapter Splitting

The processor automatically detects and splits chapters in textbooks:

```python
# Process with chapter splitting enabled
result = await processor.process_single_pdf(
    textbook_path, 
    output_dir, 
    chapter_splitting=True
)

# Access individual chapters
for chapter_name, chapter_content in result.chapters.items():
    print(f"Chapter: {chapter_name}")
    print(f"Content length: {len(chapter_content)} characters")
```

### Image Extraction

Images are automatically extracted and saved:

```python
# Process with image extraction
result = await processor.process_single_pdf(pdf_path, output_dir)

# Access extracted images
for image_path in result.images:
    print(f"Extracted image: {image_path}")
```

### Performance Monitoring

Enable comprehensive monitoring:

```python
from monitoring import MonitoringSystem

# Create monitoring system
monitoring = MonitoringSystem()

# Process with monitoring
async with processor:
    result = await processor.process_single_pdf(pdf_path, output_dir)
    
    # Get performance metrics
    stats = await processor.get_processing_stats()
    dashboard = monitoring.get_dashboard_data()
```

### Error Handling

Robust error handling with detailed feedback:

```python
try:
    result = await processor.process_single_pdf(pdf_path, output_dir)
    
    if result.success:
        print("Processing successful!")
    else:
        print(f"Processing failed: {result.error_message}")
        print(f"Processing time: {result.metrics.processing_time:.2f}s")
        
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### GPU Acceleration

For optimal performance with GPU acceleration:

```python
# CUDA configuration
config = {
    'device': 'cuda',
    'batch_size': 4,  # Larger batches for GPU
    'max_workers': 2,  # Fewer workers for GPU
    'enable_editor_model': True,
    'enable_ocr': True
}

# MPS (Apple Silicon) configuration
config = {
    'device': 'mps',
    'batch_size': 2,
    'max_workers': 4,
    'enable_editor_model': True,
    'enable_ocr': True
}
```

### Memory Management

Configure memory usage for large documents:

```python
config = {
    'batch_size': 1,  # Smaller batches for large files
    'max_pages': 100,  # Limit pages for very large documents
    'clear_cache_after_batch': True
}
```

### Batch Processing Optimization

For optimal batch processing:

```python
# Small files (< 10MB each)
config = {'batch_size': 5, 'max_workers': 6}

# Medium files (10-50MB each)
config = {'batch_size': 3, 'max_workers': 4}

# Large files (> 50MB each)
config = {'batch_size': 1, 'max_workers': 2}
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest test_pdf_processor.py -v

# Run specific test categories
python -m pytest test_pdf_processor.py::TestMarkerPDFProcessor -v

# Run with coverage
python -m pytest test_pdf_processor.py --cov=marker_pdf_processor --cov-report=html

# Run performance benchmarks
python -m pytest test_pdf_processor.py::TestPerformanceBenchmarks -v
```

## Architecture

### Core Components

1. **MarkerPDFProcessor**: Main processing engine
2. **ChapterSplitter**: Intelligent chapter detection and splitting
3. **PerformanceMonitor**: Performance tracking and metrics
4. **MonitoringSystem**: Comprehensive monitoring and alerting
5. **CLI Interface**: Command-line interface for easy usage

### Processing Pipeline

```
PDF Input → Validation → Marker Processing → Image Extraction → Chapter Splitting → Output Generation → Monitoring
```

### Key Classes

- `MarkerPDFProcessor`: Main processor class
- `ProcessingResult`: Result container with metrics
- `ProcessingMetrics`: Performance metrics tracking
- `ChapterSplit`: Chapter information container
- `MonitoringSystem`: System monitoring and alerting

## Error Handling

The processor includes comprehensive error handling:

- **File Validation**: Checks for file existence and format
- **Processing Errors**: Handles Marker library errors gracefully
- **Memory Errors**: Monitors and prevents memory overflow
- **Timeout Handling**: Configurable processing timeouts
- **Retry Logic**: Automatic retry on transient failures

## Monitoring and Alerting

### Available Metrics

- Processing time per document
- Memory usage during processing
- Success/failure rates
- Pages processed per minute
- GPU utilization (if available)
- System resource usage

### Alert Conditions

- High memory usage (> 80%)
- Long processing times (> 5 minutes)
- High error rates (> 10%)
- GPU temperature warnings
- System resource exhaustion

## Best Practices

1. **GPU Usage**: Use GPU acceleration for better performance
2. **Batch Sizing**: Adjust batch size based on document size and available memory
3. **Monitoring**: Enable monitoring for production usage
4. **Error Handling**: Always check processing results
5. **Resource Management**: Use async context managers for proper cleanup
6. **Configuration**: Use YAML configuration files for complex setups

## Troubleshooting

### Common Issues

1. **Marker Library Not Found**
   ```bash
   pip install marker-pdf
   ```

2. **GPU Not Available**
   - Check CUDA installation
   - Verify PyTorch GPU support
   - Use CPU fallback if needed

3. **Memory Issues**
   - Reduce batch size
   - Enable cache clearing
   - Process smaller document chunks

4. **Performance Issues**
   - Enable GPU acceleration
   - Adjust worker count
   - Use appropriate batch sizes

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or through CLI
python pdf_processor_cli.py process input.pdf output/ --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is part of the Academic Agent system and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run tests to verify installation
3. Enable debug logging for detailed error information
4. Check system requirements and dependencies