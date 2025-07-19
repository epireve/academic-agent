#!/usr/bin/env python3
"""
Generated script to replace hardcoded paths with OutputManager calls.
"""

import re
import sys
from pathlib import Path

def replace_hardcoded_paths(file_path: Path) -> bool:
    """Replace hardcoded paths in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replacement patterns
        replacements = {
        }
        
        # Apply replacements
        for old_pattern, new_pattern in replacements.items():
            content = content.replace(old_pattern, new_pattern)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all files."""
    project_root = Path(__file__).parent.parent
    
    # Add necessary imports to files that need them
    import_statement = '''
from src.core.output_manager import (
    get_final_output_path, 
    get_processed_output_path,
    get_analysis_output_path,
    ContentType
)
'''
    
    files_to_process = [
        "tools/demo_week_resolver.py",
        "tools/migrate_output_structure.py",
        "tools/migrate_output_structure.py",
        "tools/migrate_output_structure.py",
        "tools/migrate_output_structure.py",
        "tools/migrate_output_structure.py",
        "tools/migrate_output_structure.py",
        "tools/tutorial_pdf_converter_full.py",
        "tools/tutorial_pdf_converter_full.py",
        "tools/tutorial_pdf_converter_full.py",
        "tools/demo_consolidation.py",
        "tools/markdown_to_pdf_with_images.py",
        "tools/markdown_to_pdf_with_images.py",
        "tools/markdown_to_pdf_with_images.py",
        "tools/markdown_to_pdf_custom_sizing.py",
        "tools/markdown_to_pdf_custom_sizing.py",
        "tools/markdown_to_pdf_custom_sizing.py",
        "tools/consolidate_all_images.py",
        "tools/consolidate_all_images.py",
        "tools/consolidate_all_images.py",
        "tools/consolidate_all_images.py",
        "tools/batch_consolidation.py",
        "tools/batch_consolidation.py",
        "tools/batch_consolidation.py",
        "tools/sra_pandoc_converter.py",
        "tools/sra_pandoc_converter.py",
        "tools/sra_pandoc_converter.py",
        "tools/tutorial_pdf_converter.py",
        "tools/tutorial_pdf_converter.py",
        "tools/tutorial_pdf_converter.py",
        "tools/replace_hardcoded_paths.py",
        "tools/exams_pdf_converter.py",
        "tools/exams_pdf_converter.py",
        "tools/exams_pdf_converter.py",
        "tools/pdf_to_markdown_processor.py",
        "tools/pdf_to_markdown_processor.py",
        "tools/sra_pdf_converter.py",
        "tools/sra_pdf_converter.py",
        "tools/sra_pdf_converter.py",
        "tools/markdown_to_pdf_fixed_bullets.py",
        "tools/markdown_to_pdf_fixed_bullets.py",
        "tools/markdown_to_pdf_fixed_bullets.py",
        "tools/transcript_integration_helper.py",
        "tests/test_output_manager.py",
        "tests/test_output_manager.py",
        "tests/test_output_manager.py",
        "agents/academic/main_agent.py",
        "agents/academic/content_templates.py",
        ".venv/lib/python3.12/site-packages/markdown2.py",
        ".venv/lib/python3.12/site-packages/markdown/blockparser.py",
        ".venv/lib/python3.12/site-packages/markdown/blockprocessors.py",
        ".venv/lib/python3.12/site-packages/markdown/treeprocessors.py",
        ".venv/lib/python3.12/site-packages/markdown/htmlparser.py",
        ".venv/lib/python3.12/site-packages/markdown/__meta__.py",
        ".venv/lib/python3.12/site-packages/markdown/util.py",
        ".venv/lib/python3.12/site-packages/markdown/__init__.py",
        ".venv/lib/python3.12/site-packages/markdown/core.py",
        ".venv/lib/python3.12/site-packages/markdown/preprocessors.py",
        ".venv/lib/python3.12/site-packages/markdown/postprocessors.py",
        ".venv/lib/python3.12/site-packages/markdown/inlinepatterns.py",
        ".venv/lib/python3.12/site-packages/markdown/test_tools.py",
        ".venv/lib/python3.12/site-packages/markdown/__main__.py",
        ".venv/lib/python3.12/site-packages/PIL/ImageCms.py",
        ".venv/lib/python3.12/site-packages/sympy/physics/control/lti.py",
        ".venv/lib/python3.12/site-packages/markdown/extensions/__init__.py",
        ".venv/lib/python3.12/site-packages/markdown/extensions/legacy_attrs.py",
        ".venv/lib/python3.12/site-packages/google/genai/types.py",
        ".venv/lib/python3.12/site-packages/torch/_functorch/config.py",
        ".venv/lib/python3.12/site-packages/torch/_functorch/config.py",
        ".venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py",
        ".venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/schemas.py",
        ".venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/schemas.py",
        ".venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/ddp_fusion.py",
        ".venv/lib/python3.12/site-packages/torch/utils/hipify/hipify_python.py",
        ".venv/lib/python3.12/site-packages/torch/utils/tensorboard/_pytorch_graph.py",
        ".venv/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py",
        ".venv/lib/python3.12/site-packages/transformers/models/big_bird/modeling_big_bird.py",
        ".venv/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py",
        ".venv/lib/python3.12/site-packages/transformers/models/fsmt/modeling_fsmt.py",
        ".venv/lib/python3.12/site-packages/transformers/models/deprecated/realm/modeling_realm.py",
        ".venv/lib/python3.13/site-packages/markdown2.py",
        ".venv/lib/python3.13/site-packages/markdown/blockparser.py",
        ".venv/lib/python3.13/site-packages/markdown/blockprocessors.py",
        ".venv/lib/python3.13/site-packages/markdown/treeprocessors.py",
        ".venv/lib/python3.13/site-packages/markdown/htmlparser.py",
        ".venv/lib/python3.13/site-packages/markdown/__meta__.py",
        ".venv/lib/python3.13/site-packages/markdown/util.py",
        ".venv/lib/python3.13/site-packages/markdown/__init__.py",
        ".venv/lib/python3.13/site-packages/markdown/core.py",
        ".venv/lib/python3.13/site-packages/markdown/preprocessors.py",
        ".venv/lib/python3.13/site-packages/markdown/postprocessors.py",
        ".venv/lib/python3.13/site-packages/markdown/inlinepatterns.py",
        ".venv/lib/python3.13/site-packages/markdown/test_tools.py",
        ".venv/lib/python3.13/site-packages/markdown/__main__.py",
        ".venv/lib/python3.13/site-packages/PIL/ImageCms.py",
        ".venv/lib/python3.13/site-packages/sympy/physics/control/lti.py",
        ".venv/lib/python3.13/site-packages/pygments/lexers/markup.py",
        ".venv/lib/python3.13/site-packages/markdown/extensions/__init__.py",
        ".venv/lib/python3.13/site-packages/markdown/extensions/legacy_attrs.py",
        ".venv/lib/python3.13/site-packages/google/genai/types.py",
        ".venv/lib/python3.13/site-packages/torch/_functorch/config.py",
        ".venv/lib/python3.13/site-packages/torch/_functorch/config.py",
        ".venv/lib/python3.13/site-packages/torch/_functorch/aot_autograd.py",
        ".venv/lib/python3.13/site-packages/torch/_functorch/_aot_autograd/schemas.py",
        ".venv/lib/python3.13/site-packages/torch/_functorch/_aot_autograd/schemas.py",
        ".venv/lib/python3.13/site-packages/torch/_inductor/fx_passes/ddp_fusion.py",
        ".venv/lib/python3.13/site-packages/torch/utils/hipify/hipify_python.py",
        ".venv/lib/python3.13/site-packages/torch/utils/tensorboard/_pytorch_graph.py",
        ".venv/lib/python3.13/site-packages/transformers/models/albert/modeling_albert.py",
        ".venv/lib/python3.13/site-packages/transformers/models/big_bird/modeling_big_bird.py",
        ".venv/lib/python3.13/site-packages/transformers/models/convbert/modeling_convbert.py",
        ".venv/lib/python3.13/site-packages/transformers/models/fsmt/modeling_fsmt.py",
        ".venv/lib/python3.13/site-packages/transformers/models/deprecated/realm/modeling_realm.py",
        ".venv/lib/python3.13/site-packages/litellm/proxy/guardrails/guardrail_hooks/aim.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/humanfriendly/testing.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/PIL/ImageCms.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/sympy/physics/control/lti.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/pygments/lexers/markup.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/_functorch/config.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/_functorch/config.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/_functorch/aot_autograd.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/schemas.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/schemas.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/_inductor/fx_passes/ddp_fusion.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/utils/hipify/hipify_python.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/torch/utils/tensorboard/_pytorch_graph.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/transformers/models/albert/modeling_albert.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/transformers/models/big_bird/modeling_big_bird.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/transformers/models/convbert/modeling_convbert.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/transformers/models/fsmt/modeling_fsmt.py",
        "academic-agent-v2/venv/lib/python3.12/site-packages/transformers/models/deprecated/realm/modeling_realm.py",
        "src/core/output_manager.py",
    ]
    
    updated_count = 0
    for file_path in files_to_process:
        full_path = project_root / file_path
        if full_path.exists() and replace_hardcoded_paths(full_path):
            updated_count += 1
    
    print(f"Updated {updated_count} files")

if __name__ == "__main__":
    main()
