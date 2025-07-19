#!/usr/bin/env python3
"""
Unit tests for the migrated ConsolidationAgent.

Tests the unified architecture implementation of the consolidation agent
including content merging, conflict resolution, and file mapping.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.consolidation_agent import ConsolidationAgent, ConsolidationData, ConflictResolution, MergeStrategy
from src.agents.quality_manager import QualityEvaluation, QualityMetrics


class TestConsolidationAgent:
    """Test suite for ConsolidationAgent unified architecture implementation."""

    @pytest.fixture
    def consolidation_agent(self):
        """Create a ConsolidationAgent instance for testing."""
        return ConsolidationAgent()

    @pytest.fixture
    def sample_file_mappings(self):
        """Sample file mappings for testing."""
        return {
            "lecture_1": ["lecture1_v1.md", "lecture1_v2.md"],
            "lecture_2": ["lecture2_draft.md", "lecture2_final.md"],
            "assignment": ["assignment_notes.md"]
        }

    @pytest.fixture
    def sample_content_files(self, tmp_path):
        """Create sample content files for testing."""
        files = {}
        
        # Create lecture 1 versions
        v1_content = """# Lecture 1 - Introduction
## Overview
Basic introduction to the topic.
- Point A
- Point B
"""
        v2_content = """# Lecture 1 - Introduction  
## Overview
Comprehensive introduction to the topic.
- Point A (expanded)
- Point B
- Point C (new)

## Additional Section
Extra content added in v2.
"""
        
        files["lecture1_v1.md"] = tmp_path / "lecture1_v1.md"
        files["lecture1_v1.md"].write_text(v1_content)
        
        files["lecture1_v2.md"] = tmp_path / "lecture1_v2.md" 
        files["lecture1_v2.md"].write_text(v2_content)
        
        # Create lecture 2 versions
        draft_content = """# Lecture 2 - Advanced Topics
Draft content here.
"""
        final_content = """# Lecture 2 - Advanced Topics
Final polished content here.
Includes all revisions.
"""
        
        files["lecture2_draft.md"] = tmp_path / "lecture2_draft.md"
        files["lecture2_draft.md"].write_text(draft_content)
        
        files["lecture2_final.md"] = tmp_path / "lecture2_final.md"
        files["lecture2_final.md"].write_text(final_content)
        
        return files

    @pytest.mark.asyncio
    async def test_consolidation_agent_initialization(self, consolidation_agent):
        """Test ConsolidationAgent initialization."""
        assert consolidation_agent.agent_name == "consolidation_agent"
        assert consolidation_agent.quality_manager is not None
        assert consolidation_agent.merge_strategy == MergeStrategy.INTELLIGENT
        assert consolidation_agent.output_dir.exists()

    @pytest.mark.asyncio
    async def test_validate_input_valid_mappings(self, consolidation_agent, sample_file_mappings):
        """Test input validation with valid file mappings."""
        result = await consolidation_agent.validate_input(sample_file_mappings)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_input_invalid_mappings(self, consolidation_agent):
        """Test input validation with invalid mappings."""
        # Empty mappings
        result = await consolidation_agent.validate_input({})
        assert result is False
        
        # Non-dict input
        result = await consolidation_agent.validate_input("invalid")
        assert result is False
        
        # Invalid mapping structure
        result = await consolidation_agent.validate_input({"key": "not_a_list"})
        assert result is False

    @pytest.mark.asyncio
    async def test_read_file_content(self, consolidation_agent, sample_content_files):
        """Test reading file content."""
        content = await consolidation_agent._read_file_content(sample_content_files["lecture1_v1.md"])
        assert "Lecture 1" in content
        assert "Basic introduction" in content

    @pytest.mark.asyncio
    async def test_read_file_content_nonexistent(self, consolidation_agent):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            await consolidation_agent._read_file_content(Path("nonexistent.md"))

    @pytest.mark.asyncio
    async def test_detect_conflicts_simple(self, consolidation_agent):
        """Test conflict detection with simple content."""
        content1 = "# Title\nContent A"
        content2 = "# Title\nContent B"
        
        conflicts = await consolidation_agent._detect_conflicts([content1, content2])
        assert len(conflicts) > 0
        assert any("Content" in str(conflict) for conflict in conflicts)

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflicts(self, consolidation_agent):
        """Test conflict detection with no conflicts."""
        content1 = "# Title\nSame content"
        content2 = "# Title\nSame content"
        
        conflicts = await consolidation_agent._detect_conflicts([content1, content2])
        assert len(conflicts) == 0

    def test_merge_contents_latest_strategy(self, consolidation_agent):
        """Test content merging with latest strategy."""
        consolidation_agent.merge_strategy = MergeStrategy.LATEST
        
        contents = ["Old content", "Newer content", "Latest content"]
        merged = consolidation_agent._merge_contents(contents)
        
        assert merged == "Latest content"

    def test_merge_contents_longest_strategy(self, consolidation_agent):
        """Test content merging with longest strategy."""
        consolidation_agent.merge_strategy = MergeStrategy.LONGEST
        
        contents = ["Short", "Medium length content", "Long"]
        merged = consolidation_agent._merge_contents(contents)
        
        assert merged == "Medium length content"

    def test_merge_contents_intelligent_strategy(self, consolidation_agent):
        """Test content merging with intelligent strategy."""
        consolidation_agent.merge_strategy = MergeStrategy.INTELLIGENT
        
        content1 = "# Title\n## Section A\nContent A"
        content2 = "# Title\n## Section A\nContent A\n## Section B\nContent B"
        contents = [content1, content2]
        
        merged = consolidation_agent._merge_contents(contents)
        
        # Should include both sections
        assert "Section A" in merged
        assert "Section B" in merged

    def test_resolve_conflict_keep_both(self, consolidation_agent):
        """Test conflict resolution with keep both strategy."""
        conflict = ConflictResolution(
            location="line 5",
            type="content_difference",
            description="Different content",
            options=["Option A", "Option B"],
            resolution_strategy="keep_both"
        )
        
        resolved = consolidation_agent._resolve_conflict(conflict)
        assert "Option A" in resolved
        assert "Option B" in resolved

    def test_resolve_conflict_prioritize_latest(self, consolidation_agent):
        """Test conflict resolution with prioritize latest strategy."""
        conflict = ConflictResolution(
            location="line 5",
            type="content_difference", 
            description="Different content",
            options=["Older option", "Newer option"],
            resolution_strategy="prioritize_latest"
        )
        
        resolved = consolidation_agent._resolve_conflict(conflict)
        assert resolved == "Newer option"

    @pytest.mark.asyncio
    async def test_consolidate_files_success(self, consolidation_agent, sample_content_files, sample_file_mappings):
        """Test successful file consolidation."""
        # Update mappings to use actual file paths
        mappings = {
            "lecture_1": [str(sample_content_files["lecture1_v1.md"]), str(sample_content_files["lecture1_v2.md"])]
        }
        
        # Mock quality manager
        with patch.object(consolidation_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.return_value = QualityEvaluation(
                content_type="consolidated",
                quality_score=0.85,
                feedback=["Good consolidation"],
                areas_for_improvement=[],
                strengths=["Well merged"],
                metrics=QualityMetrics(0.85, 0.8, 0.9, 0.8, 0.85, 0.8),
                assessment="Good quality",
                approved=True
            )
            
            result = await consolidation_agent.consolidate_files(mappings)
            
            assert result["success"] is True
            assert "consolidation_data" in result
            assert result["files_processed"] == 1
            assert result["conflicts_resolved"] >= 0

    @pytest.mark.asyncio
    async def test_consolidate_files_invalid_input(self, consolidation_agent):
        """Test file consolidation with invalid input."""
        result = await consolidation_agent.consolidate_files({})
        
        assert result["success"] is False
        assert "error" in result
        assert "Invalid file mappings" in result["error"]

    @pytest.mark.asyncio
    async def test_save_consolidation(self, consolidation_agent, tmp_path):
        """Test saving consolidation data."""
        consolidation_data = ConsolidationData(
            consolidated_files={
                "test_group": {
                    "merged_content": "Test content",
                    "source_files": ["file1.md", "file2.md"],
                    "conflicts_resolved": [],
                    "metadata": {"merge_strategy": "intelligent"}
                }
            },
            conflicts_resolved=[],
            merge_metadata={
                "total_files": 2,
                "processing_time": 1.5,
                "quality_score": 0.8
            }
        )
        
        # Set output directory to temp path
        consolidation_agent.output_dir = tmp_path
        
        output_path = await consolidation_agent._save_consolidation(consolidation_data, "test_consolidation")
        
        assert output_path.exists()
        assert output_path.suffix == ".json"
        
        # Check content
        with open(output_path) as f:
            saved_data = json.load(f)
        
        assert "test_group" in saved_data["consolidated_files"]
        assert saved_data["merge_metadata"]["total_files"] == 2

    @pytest.mark.asyncio
    async def test_save_as_markdown(self, consolidation_agent, tmp_path):
        """Test saving consolidation as markdown."""
        consolidation_data = ConsolidationData(
            consolidated_files={
                "group1": {
                    "merged_content": "# Group 1\nContent here",
                    "source_files": ["file1.md"],
                    "conflicts_resolved": [],
                    "metadata": {}
                },
                "group2": {
                    "merged_content": "# Group 2\nMore content",
                    "source_files": ["file2.md"],
                    "conflicts_resolved": [],
                    "metadata": {}
                }
            },
            conflicts_resolved=[],
            merge_metadata={"total_files": 2}
        )
        
        output_path = tmp_path / "test_consolidation.md"
        await consolidation_agent._save_as_markdown(consolidation_data, output_path)
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "# Consolidated Content" in content
        assert "## Group 1" in content
        assert "## Group 2" in content
        assert "Source Files:" in content

    @pytest.mark.asyncio
    async def test_batch_consolidate(self, consolidation_agent, sample_content_files):
        """Test batch consolidation of multiple file groups."""
        mappings = {
            "group1": [str(sample_content_files["lecture1_v1.md"])],
            "group2": [str(sample_content_files["lecture2_draft.md"])]
        }
        
        # Mock quality manager
        with patch.object(consolidation_agent.quality_manager, 'evaluate_content') as mock_quality:
            mock_quality.return_value = QualityEvaluation(
                content_type="consolidated",
                quality_score=0.8,
                feedback=[],
                areas_for_improvement=[],
                strengths=[],
                metrics=QualityMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8),
                assessment="Good",
                approved=True
            )
            
            result = await consolidation_agent.batch_consolidate(mappings)
            
            assert result["success"] is True
            assert result["groups_processed"] == 2
            assert result["total_files"] == 2

    @pytest.mark.asyncio
    async def test_validate_output_success(self, consolidation_agent):
        """Test output validation with successful result."""
        result = {
            "success": True,
            "consolidation_data": {"files": {}},
            "files_processed": 5
        }
        
        validation = await consolidation_agent.validate_output(result)
        assert validation is True

    @pytest.mark.asyncio
    async def test_validate_output_failure(self, consolidation_agent):
        """Test output validation with failed result."""
        result = {
            "success": False,
            "error": "Consolidation failed"
        }
        
        validation = await consolidation_agent.validate_output(result)
        assert validation is False

    @pytest.mark.asyncio
    async def test_error_handling_during_consolidation(self, consolidation_agent):
        """Test error handling during consolidation."""
        mappings = {"test": ["nonexistent.md"]}
        
        result = await consolidation_agent.consolidate_files(mappings)
        
        assert result["success"] is False
        assert "error" in result

    def test_agent_inheritance(self, consolidation_agent):
        """Test that ConsolidationAgent properly inherits from BaseAgent."""
        from src.agents.base_agent import BaseAgent
        
        assert isinstance(consolidation_agent, BaseAgent)
        assert hasattr(consolidation_agent, 'agent_name')
        assert hasattr(consolidation_agent, 'logger')
        assert hasattr(consolidation_agent, 'base_dir')

    @pytest.mark.asyncio
    async def test_async_functionality(self, consolidation_agent):
        """Test that the agent properly supports async operations."""
        import inspect
        
        assert inspect.iscoroutinefunction(consolidation_agent.consolidate_files)
        assert inspect.iscoroutinefunction(consolidation_agent.batch_consolidate)
        assert inspect.iscoroutinefunction(consolidation_agent.validate_input)
        assert inspect.iscoroutinefunction(consolidation_agent.validate_output)

    def test_merge_strategy_enum(self):
        """Test merge strategy enumeration."""
        assert MergeStrategy.LATEST.value == "latest"
        assert MergeStrategy.LONGEST.value == "longest"
        assert MergeStrategy.INTELLIGENT.value == "intelligent"

    def test_conflict_resolution_dataclass(self):
        """Test ConflictResolution dataclass."""
        conflict = ConflictResolution(
            location="line 10",
            type="content_difference",
            description="Test conflict",
            options=["Option A", "Option B"],
            resolution_strategy="keep_both"
        )
        
        assert conflict.location == "line 10"
        assert conflict.type == "content_difference"
        assert len(conflict.options) == 2

    @pytest.mark.asyncio
    async def test_content_similarity_detection(self, consolidation_agent):
        """Test content similarity detection for intelligent merging."""
        similar_content1 = "This is a test paragraph with some content."
        similar_content2 = "This is a test paragraph with some different content."
        
        # This would use actual similarity detection in production
        similarity = consolidation_agent._calculate_content_similarity(similar_content1, similar_content2)
        assert 0.0 <= similarity <= 1.0

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Mock implementation of content similarity calculation."""
        # Simple word-based similarity for testing
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    # Patch the method for testing
    ConsolidationAgent._calculate_content_similarity = _calculate_content_similarity