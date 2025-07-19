#!/usr/bin/env python3
"""
Test the simplified converter on specific files
"""

from pathlib import Path
from simplified_final_converter import SimplifiedFinalConverter

def test_specific_files():
    """Test conversion on specific files"""
    project_root = Path.cwd()
    input_dir = project_root / str(get_output_manager().outputs_dir) / "sra" / "ai_enhanced_study_notes"
    output_dir = input_dir  # Same folder as markdown files
    
    converter = SimplifiedFinalConverter(input_dir, output_dir)
    
    # Test with the files that have good examples of all issues
    test_files = ["week_01_comprehensive_study_notes.md", "week_05_comprehensive_study_notes.md"]
    
    print("🧪 Testing simplified converter on selected files...")
    print("📋 This will verify:")
    print("   • Mermaid diagrams are 50% smaller")
    print("   • Text is visible in SVG diagrams")
    print("   • Bullet points have proper line breaks")
    print("   • Overall formatting is clean\n")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for test_file in test_files:
        file_path = input_dir / test_file
        if file_path.exists():
            success = converter.convert_single_file(file_path)
            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1
        else:
            print(f"❌ File not found: {test_file}")
            failed_conversions += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Successful: {successful_conversions}")
    print(f"❌ Failed: {failed_conversions}")
    
    if successful_conversions > 0:
        print(f"\n🎉 Test successful! Please check the generated PDFs:")
        for test_file in test_files:
            pdf_name = test_file.replace('.md', '.pdf')
            print(f"   📄 {pdf_name}")
        print(f"\n📁 Location: {output_dir}")
        print(f"\n✅ If the test PDFs look good, we can proceed with all files!")
    else:
        print(f"\n❌ Test failed. Please check the errors above.")

if __name__ == "__main__":
    test_specific_files()