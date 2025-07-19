#!/usr/bin/env python3
"""
PDF to Markdown Processor for WOC7017 Security Risk Analysis and Evaluation

This script processes split PDF chapters into markdown format using the marker library.
Uses LLM enhancement and disables image extraction for faster processing.
"""

import os
import subprocess
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import shutil

class PDFToMarkdownProcessor:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        
        # Define input and output paths
        self.input_dir = self.project_root / "Split_Chapters"
        self.output_dir = self.project_root / str(get_output_manager().outputs_dir) / "sra" / "textbook" / get_processed_output_path(ContentType.MARKDOWN)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Marker configuration
        self.google_api_key = 'AIzaSyDDAjJf4GbT83nomvIMMuOwQ511qCR0J4A'
        self.marker_path = self.project_root / ".venv" / "bin" / "marker_single"
        
        # Processing parameters
        self.marker_params = [
            "--output_format", get_processed_output_path(ContentType.MARKDOWN),
            "--use_llm"
        ]
        
        # Define textbook chapters structure
        self.textbook_chapters = [
            {"number": 1, "title": "Introduction", "filename": "Chapter_1_Introduction.pdf"},
            {"number": 2, "title": "Information Security Risk Assessment Basics", "filename": "Chapter_2_Information_Security_Risk_Assessment_Basics.pdf"},
            {"number": 3, "title": "Project Definition", "filename": "Chapter_3_Project_Definition.pdf"},
            {"number": 4, "title": "Security Risk Assessment Preparation", "filename": "Chapter_4_Security_Risk_Assessment_Preparation.pdf"},
            {"number": 5, "title": "Data Gathering", "filename": "Chapter_5_Data_Gathering.pdf"},
            {"number": 6, "title": "Administrative Data Gathering", "filename": "Chapter_6_Administrative_Data_Gathering.pdf"},
            {"number": 7, "title": "Technical Data Gathering", "filename": "Chapter_7_Technical_Data_Gathering.pdf"},
            {"number": 8, "title": "Physical Data Gathering", "filename": "Chapter_8_Physical_Data_Gathering.pdf"},
            {"number": 9, "title": "Security Risk Analysis", "filename": "Chapter_9_Security_Risk_Analysis.pdf"},
            {"number": 10, "title": "Security Risk Mitigation", "filename": "Chapter_10_Security_Risk_Mitigation.pdf"},
            {"number": 11, "title": "Security Risk Assessment Reporting", "filename": "Chapter_11_Security_Risk_Assessment_Reporting.pdf"},
            {"number": 12, "title": "Security Risk Assessment Project Management", "filename": "Chapter_12_Security_Risk_Assessment_Project_Management.pdf"},
            {"number": 13, "title": "Security Risk Assessment Approaches", "filename": "Chapter_13_Security_Risk_Assessment_Approaches.pdf"},
        ]
        
        # Processing status
        self.processing_results = []
        self.start_time = None
        self.end_time = None

    def verify_prerequisites(self) -> bool:
        """Verify all prerequisites are met"""
        print("Verifying prerequisites...")
        
        # Check if marker_single exists
        if not self.marker_path.exists():
            print(f"❌ marker_single not found at {self.marker_path}")
            return False
        print(f"✅ marker_single found at {self.marker_path}")
        
        # Check if input directory exists
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return False
        print(f"✅ Input directory found: {self.input_dir}")
        
        # Check if API key is set
        if not self.google_api_key:
            print("❌ Google API key not set")
            return False
        print("✅ Google API key configured")
        
        # Check if PDF files exist
        missing_files = []
        for chapter in self.textbook_chapters:
            pdf_path = self.input_dir / chapter["filename"]
            if not pdf_path.exists():
                missing_files.append(chapter["filename"])
        
        if missing_files:
            print(f"❌ Missing PDF files: {missing_files}")
            return False
        print(f"✅ All {len(self.textbook_chapters)} PDF files found")
        
        return True

    def process_single_chapter(self, chapter_info: dict) -> dict:
        """Process a single PDF chapter to markdown"""
        chapter_num = chapter_info["number"]
        chapter_title = chapter_info["title"]
        pdf_filename = chapter_info["filename"]
        
        print(f"\n🔄 Processing Chapter {chapter_num}: {chapter_title}")
        
        # Define paths
        input_pdf = self.input_dir / pdf_filename
        output_chapter_dir = self.output_dir / f"chapter_{chapter_num:02d}"
        
        # Create chapter-specific output directory
        output_chapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare command
        cmd = [
            str(self.marker_path),
            str(input_pdf),
            "--output_dir", str(output_chapter_dir)
        ] + self.marker_params
        
        # Set environment variables
        env = os.environ.copy()
        env['GOOGLE_API_KEY'] = self.google_api_key
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Run marker_single command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                # Find the generated markdown file
                md_files = list(output_chapter_dir.glob("*.md"))
                if md_files:
                    # Rename to standard format
                    generated_md = md_files[0]
                    standard_md = output_chapter_dir / f"chapter_{chapter_num:02d}_{chapter_title.replace(' ', '_').replace('/', '_')}.md"
                    
                    # Move and rename
                    shutil.move(str(generated_md), str(standard_md))
                    
                    # Get file size
                    file_size = standard_md.stat().st_size
                    
                    result_info = {
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_title,
                        "status": "success",
                        "processing_time": processing_time,
                        "output_file": str(standard_md),
                        "file_size": file_size,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                    
                    print(f"✅ Chapter {chapter_num} processed successfully")
                    print(f"   Output: {standard_md}")
                    print(f"   Size: {file_size:,} bytes")
                    print(f"   Time: {processing_time:.1f} seconds")
                    
                else:
                    result_info = {
                        "chapter_number": chapter_num,
                        "chapter_title": chapter_title,
                        "status": "error",
                        "processing_time": processing_time,
                        "error": "No markdown file generated",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                    print(f"❌ Chapter {chapter_num} failed - No markdown file generated")
                    
            else:
                result_info = {
                    "chapter_number": chapter_num,
                    "chapter_title": chapter_title,
                    "status": "error",
                    "processing_time": processing_time,
                    "error": f"Command failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
                print(f"❌ Chapter {chapter_num} failed with return code {result.returncode}")
                print(f"   Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            result_info = {
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "status": "timeout",
                "processing_time": 600,
                "error": "Processing timeout (10 minutes)",
                "stdout": "",
                "stderr": ""
            }
            print(f"⏱️ Chapter {chapter_num} timed out after 10 minutes")
            
        except Exception as e:
            result_info = {
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "status": "error",
                "processing_time": time.time() - start_time,
                "error": str(e),
                "stdout": "",
                "stderr": ""
            }
            print(f"❌ Chapter {chapter_num} failed with exception: {e}")
        
        return result_info

    def process_all_chapters(self) -> list:
        """Process all chapters sequentially"""
        print(f"\n🚀 Starting batch processing of {len(self.textbook_chapters)} chapters")
        print(f"📁 Output directory: {self.output_dir}")
        
        self.start_time = time.time()
        
        for chapter_info in self.textbook_chapters:
            result = self.process_single_chapter(chapter_info)
            self.processing_results.append(result)
            
            # Small delay between chapters to avoid overwhelming the API
            time.sleep(2)
        
        self.end_time = time.time()
        
        return self.processing_results

    def generate_processing_report(self) -> str:
        """Generate a comprehensive processing report"""
        if not self.processing_results:
            return "No processing results available"
        
        total_time = self.end_time - self.start_time if self.end_time and self.start_time else 0
        successful = [r for r in self.processing_results if r["status"] == "success"]
        failed = [r for r in self.processing_results if r["status"] == "error"]
        timeout = [r for r in self.processing_results if r["status"] == "timeout"]
        
        report = []
        report.append("# PDF to Markdown Processing Report")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Course**: WOC7017 Security Risk Analysis and Evaluation")
        report.append("")
        
        # Summary
        report.append("## Processing Summary")
        report.append(f"- **Total Chapters**: {len(self.processing_results)}")
        report.append(f"- **Successful**: {len(successful)}")
        report.append(f"- **Failed**: {len(failed)}")
        report.append(f"- **Timeout**: {len(timeout)}")
        report.append(f"- **Total Processing Time**: {total_time:.1f} seconds")
        report.append(f"- **Average Time per Chapter**: {total_time/len(self.processing_results):.1f} seconds")
        report.append("")
        
        # Successful chapters
        if successful:
            report.append("## Successfully Processed Chapters")
            total_size = sum(r["file_size"] for r in successful)
            report.append(f"**Total Size**: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
            report.append("")
            
            for result in successful:
                report.append(f"### Chapter {result['chapter_number']}: {result['chapter_title']}")
                report.append(f"- **File**: `{Path(result['output_file']).name}`")
                report.append(f"- **Size**: {result['file_size']:,} bytes")
                report.append(f"- **Processing Time**: {result['processing_time']:.1f} seconds")
                report.append("")
        
        # Failed chapters
        if failed or timeout:
            report.append("## Failed Chapters")
            for result in failed + timeout:
                report.append(f"### Chapter {result['chapter_number']}: {result['chapter_title']}")
                report.append(f"- **Status**: {result['status']}")
                report.append(f"- **Error**: {result['error']}")
                report.append(f"- **Processing Time**: {result['processing_time']:.1f} seconds")
                if result['stderr']:
                    report.append(f"- **Error Details**: {result['stderr'][:500]}...")
                report.append("")
        
        # Usage instructions
        report.append("## Usage Instructions")
        report.append("1. **Individual Chapter Access**: Navigate to `output/sra/textbook/markdown/chapter_XX/`")
        report.append("2. **Integration**: Use these markdown files with the three-source processor")
        report.append("3. **Quality Check**: Review generated markdown for accuracy and completeness")
        report.append("")
        
        return "\n".join(report)

    def save_results(self):
        """Save processing results and report"""
        # Save JSON results
        results_data = {
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_chapters": len(self.processing_results),
                "successful_chapters": len([r for r in self.processing_results if r["status"] == "success"]),
                "failed_chapters": len([r for r in self.processing_results if r["status"] == "error"]),
                "timeout_chapters": len([r for r in self.processing_results if r["status"] == "timeout"]),
                "total_processing_time": self.end_time - self.start_time if self.end_time and self.start_time else 0
            },
            "chapter_results": self.processing_results
        }
        
        with open(self.output_dir / "processing_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Save processing report
        report = self.generate_processing_report()
        with open(self.output_dir / "processing_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📊 Results saved to {self.output_dir}")
        print(f"   - processing_results.json")
        print(f"   - processing_report.md")

    def run_complete_processing(self):
        """Run the complete PDF to markdown processing pipeline"""
        print("🔄 Starting PDF to Markdown Processing")
        print("=" * 50)
        
        # Verify prerequisites
        if not self.verify_prerequisites():
            print("❌ Prerequisites not met. Exiting.")
            return False
        
        # Process all chapters
        results = self.process_all_chapters()
        
        # Save results
        self.save_results()
        
        # Print summary
        successful = len([r for r in results if r["status"] == "success"])
        total = len(results)
        
        print("\n" + "=" * 50)
        print("🎉 Processing Complete!")
        print(f"   Successfully processed: {successful}/{total} chapters")
        print(f"   Output directory: {self.output_dir}")
        
        return successful == total

def main():
    """Main function"""
    project_root = Path.cwd()
    
    processor = PDFToMarkdownProcessor(str(project_root))
    success = processor.run_complete_processing()
    
    if success:
        print("\n✅ All chapters processed successfully!")
        sys.exit(0)
    else:
        print("\n⚠️  Some chapters failed to process. Check the report for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()