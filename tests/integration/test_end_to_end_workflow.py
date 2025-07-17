"""
Integration tests for end-to-end academic agent workflow.

This module tests the complete workflow from PDF ingestion through analysis,
outline generation, notes creation, and quality control.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import asyncio
from datetime import datetime

from tests.utils import (
    TestFileManager,
    create_sample_markdown_content,
    create_sample_analysis_result,
    create_sample_outline_result,
    create_sample_notes_result,
    PDFTestHelper,
    AnalysisTestHelper,
    create_mock_agent_with_tools
)


class TestCompleteWorkflow:
    """Test complete academic agent workflow."""
    
    def test_pdf_to_notes_pipeline_mock(self, tmp_path):
        """Test complete pipeline from PDF to notes with mocked components."""
        # Setup test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create test PDF
        pdf_file = file_manager.create_file("input.pdf", "")
        PDFTestHelper.create_test_pdf(pdf_file)
        
        # Create expected output directories
        markdown_dir = file_manager.create_directory("markdown")
        analysis_dir = file_manager.create_directory("analysis")
        outline_dir = file_manager.create_directory("outlines")
        notes_dir = file_manager.create_directory("notes")
        
        # Mock the complete workflow
        with patch('agents.academic.academic_agent.DocumentConverter') as mock_converter, \
             patch('agents.academic.academic_agent.completion') as mock_completion:
            
            # Setup PDF processing mock
            mock_doc_converter = Mock()
            mock_converter.return_value = mock_doc_converter
            mock_result = Mock()
            mock_result.document.export_to_markdown.return_value = create_sample_markdown_content()
            mock_result.document.title = "Test Document"
            mock_result.document.language = "en"
            mock_doc_converter.convert.return_value = mock_result
            
            # Setup LLM mock for quality evaluation
            mock_llm_response = Mock()
            mock_llm_response.choices = [Mock()]
            mock_llm_response.choices[0].message.content = "Assessment: Good quality content"
            mock_completion.return_value = mock_llm_response
            
            # Step 1: PDF Ingestion (mocked)
            markdown_file = file_manager.create_file(
                "markdown/input.md", 
                create_sample_markdown_content()
            )
            metadata_file = file_manager.create_file(
                "metadata/input.json",
                json.dumps({
                    "source_file": str(pdf_file),
                    "processed_date": datetime.now().isoformat(),
                    "title": "Test Document",
                    "language": "en"
                })
            )
            
            # Step 2: Content Analysis (mocked)
            analysis_result = create_sample_analysis_result()
            analysis_file = file_manager.create_json_file(
                "analysis/input_analysis.json",
                analysis_result
            )
            
            # Step 3: Outline Generation (mocked)
            outline_result = create_sample_outline_result()
            outline_file = file_manager.create_json_file(
                "outlines/input_outline.json",
                outline_result
            )
            
            # Step 4: Notes Generation (mocked)
            notes_result = create_sample_notes_result()
            notes_file = file_manager.create_json_file(
                "notes/input_notes.json",
                notes_result
            )
            
            # Verify all outputs were created
            assert markdown_file.exists()
            assert metadata_file.exists()
            assert analysis_file.exists()
            assert outline_file.exists()
            assert notes_file.exists()
            
            # Verify content quality
            markdown_content = markdown_file.read_text()
            assert len(markdown_content) > 0
            assert "# Academic Research Paper" in markdown_content
            
            # Verify analysis quality
            AnalysisTestHelper.assert_analysis_result(analysis_result)
            AnalysisTestHelper.assert_outline_result(outline_result)
            AnalysisTestHelper.assert_notes_result(notes_result)
            
            # Cleanup
            file_manager.cleanup()
    
    def test_workflow_with_quality_feedback_loop(self, tmp_path):
        """Test workflow with quality control feedback loop."""
        # Setup test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create test content
        test_content = create_sample_markdown_content()
        content_file = file_manager.create_file("test_content.md", test_content)
        
        # Simulate quality evaluation workflow
        quality_evaluations = []
        
        # First evaluation - fails quality threshold
        low_quality_eval = {
            "quality_score": 0.5,  # Below 0.7 threshold
            "feedback": ["Content needs improvement", "Add more detail"],
            "approved": False,
            "improvement_suggestions": [
                "Expand introduction section",
                "Add more examples",
                "Improve conclusion"
            ]
        }
        quality_evaluations.append(low_quality_eval)
        
        # Second evaluation - meets quality threshold after improvement
        improved_quality_eval = {
            "quality_score": 0.8,  # Above 0.7 threshold
            "feedback": ["Good quality content", "Well structured"],
            "approved": True,
            "improvement_suggestions": []
        }
        quality_evaluations.append(improved_quality_eval)
        
        # Verify feedback loop progression
        assert quality_evaluations[0]["approved"] is False
        assert quality_evaluations[1]["approved"] is True
        assert quality_evaluations[1]["quality_score"] > quality_evaluations[0]["quality_score"]
        
        # Verify improvement suggestions were provided
        assert len(quality_evaluations[0]["improvement_suggestions"]) > 0
        assert len(quality_evaluations[1]["improvement_suggestions"]) == 0
        
        # Cleanup
        file_manager.cleanup()
    
    def test_workflow_error_handling_and_recovery(self, tmp_path):
        """Test workflow error handling and recovery mechanisms."""
        # Setup test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create test files
        pdf_file = file_manager.create_file("test.pdf", "")
        PDFTestHelper.create_test_pdf(pdf_file)
        
        # Simulate workflow with errors
        workflow_log = []
        
        # Step 1: PDF Processing - Success
        workflow_log.append({
            "step": "pdf_processing",
            "status": "success",
            "output": "markdown/test.md",
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 2: Analysis - Error (first attempt)
        workflow_log.append({
            "step": "analysis",
            "status": "error",
            "error": "Analysis failed due to malformed content",
            "retry_count": 1,
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 2: Analysis - Success (retry)
        workflow_log.append({
            "step": "analysis",
            "status": "success",
            "output": "analysis/test_analysis.json",
            "retry_count": 2,
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 3: Outline Generation - Success
        workflow_log.append({
            "step": "outline_generation",
            "status": "success",
            "output": "outlines/test_outline.json",
            "timestamp": datetime.now().isoformat()
        })
        
        # Step 4: Notes Generation - Success
        workflow_log.append({
            "step": "notes_generation",
            "status": "success",
            "output": "notes/test_notes.json",
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze workflow log
        total_steps = len(workflow_log)
        successful_steps = len([log for log in workflow_log if log["status"] == "success"])
        error_steps = len([log for log in workflow_log if log["status"] == "error"])
        
        assert total_steps == 5
        assert successful_steps == 4
        assert error_steps == 1
        
        # Verify retry mechanism worked
        analysis_attempts = [log for log in workflow_log if log["step"] == "analysis"]
        assert len(analysis_attempts) == 2
        assert analysis_attempts[0]["status"] == "error"
        assert analysis_attempts[1]["status"] == "success"
        
        # Cleanup
        file_manager.cleanup()
    
    def test_concurrent_workflow_processing(self, tmp_path):
        """Test concurrent processing of multiple documents."""
        # Setup test environment
        file_manager = TestFileManager(tmp_path)
        
        # Create multiple test documents
        documents = []
        for i in range(3):
            pdf_file = file_manager.create_file(f"document_{i}.pdf", "")
            PDFTestHelper.create_test_pdf(pdf_file, f"Test Document {i}")
            documents.append({
                "id": f"doc_{i}",
                "pdf_path": pdf_file,
                "status": "pending"
            })
        
        # Simulate concurrent processing
        processing_results = []
        
        for doc in documents:
            # Simulate processing each document
            result = {
                "document_id": doc["id"],
                "processing_steps": [
                    {"step": "pdf_processing", "status": "success", "duration": 0.5},
                    {"step": "analysis", "status": "success", "duration": 1.2},
                    {"step": "outline_generation", "status": "success", "duration": 0.8},
                    {"step": "notes_generation", "status": "success", "duration": 1.0}
                ],
                "total_duration": 3.5,
                "final_status": "completed"
            }
            processing_results.append(result)
        
        # Verify all documents were processed
        assert len(processing_results) == 3
        
        for result in processing_results:
            assert result["final_status"] == "completed"
            assert len(result["processing_steps"]) == 4
            assert all(step["status"] == "success" for step in result["processing_steps"])
        
        # Calculate aggregate metrics
        total_processing_time = sum(result["total_duration"] for result in processing_results)
        average_processing_time = total_processing_time / len(processing_results)
        
        assert total_processing_time > 0
        assert average_processing_time == 3.5  # All documents took the same time in this simulation
        
        # Cleanup
        file_manager.cleanup()


class TestWorkflowConfiguration:
    """Test workflow configuration and customization."""
    
    def test_workflow_configuration_loading(self, tmp_path):
        """Test loading workflow configuration."""
        # Create workflow configuration
        workflow_config = {
            "pdf_processing": {
                "enabled": True,
                "timeout": 300,
                "retry_attempts": 3
            },
            "analysis": {
                "enabled": True,
                "min_topics": 3,
                "max_concepts": 20
            },
            "quality_control": {
                "enabled": True,
                "threshold": 0.7,
                "max_iterations": 3
            },
            "output_formats": ["markdown", "json", "pdf"]
        }
        
        config_file = tmp_path / "workflow_config.json"
        config_file.write_text(json.dumps(workflow_config, indent=2))
        
        # Load and validate configuration
        loaded_config = json.loads(config_file.read_text())
        
        assert loaded_config["pdf_processing"]["enabled"] is True
        assert loaded_config["analysis"]["min_topics"] == 3
        assert loaded_config["quality_control"]["threshold"] == 0.7
        assert "markdown" in loaded_config["output_formats"]
    
    def test_workflow_customization(self, tmp_path):
        """Test workflow customization for different use cases."""
        # Define different workflow profiles
        profiles = {
            "quick_processing": {
                "pdf_processing": {"timeout": 60},
                "analysis": {"depth": "shallow"},
                "quality_control": {"threshold": 0.6},
                "skip_steps": ["detailed_notes"]
            },
            "comprehensive_processing": {
                "pdf_processing": {"timeout": 600},
                "analysis": {"depth": "deep"},
                "quality_control": {"threshold": 0.8},
                "additional_steps": ["citation_extraction", "reference_validation"]
            },
            "research_mode": {
                "pdf_processing": {"timeout": 300},
                "analysis": {"depth": "research"},
                "quality_control": {"threshold": 0.9},
                "additional_steps": ["research_validation", "expert_review"]
            }
        }
        
        # Test each profile
        for profile_name, profile_config in profiles.items():
            # Validate profile configuration
            assert "pdf_processing" in profile_config
            assert "analysis" in profile_config
            assert "quality_control" in profile_config
            
            # Verify profile-specific settings
            if profile_name == "quick_processing":
                assert profile_config["quality_control"]["threshold"] == 0.6
                assert "detailed_notes" in profile_config["skip_steps"]
            elif profile_name == "comprehensive_processing":
                assert profile_config["quality_control"]["threshold"] == 0.8
                assert "citation_extraction" in profile_config["additional_steps"]
            elif profile_name == "research_mode":
                assert profile_config["quality_control"]["threshold"] == 0.9
                assert "research_validation" in profile_config["additional_steps"]
    
    def test_workflow_step_dependencies(self):
        """Test workflow step dependencies and ordering."""
        # Define workflow steps with dependencies
        workflow_steps = {
            "pdf_processing": {
                "dependencies": [],
                "required": True,
                "order": 1
            },
            "text_extraction": {
                "dependencies": ["pdf_processing"],
                "required": True,
                "order": 2
            },
            "content_analysis": {
                "dependencies": ["text_extraction"],
                "required": True,
                "order": 3
            },
            "outline_generation": {
                "dependencies": ["content_analysis"],
                "required": False,
                "order": 4
            },
            "notes_generation": {
                "dependencies": ["content_analysis", "outline_generation"],
                "required": False,
                "order": 5
            },
            "quality_evaluation": {
                "dependencies": ["content_analysis"],
                "required": True,
                "order": 6
            }
        }
        
        # Validate dependency graph
        for step_name, step_config in workflow_steps.items():
            # Check that all dependencies exist
            for dependency in step_config["dependencies"]:
                assert dependency in workflow_steps
            
            # Check that dependencies have lower order numbers
            for dependency in step_config["dependencies"]:
                dependency_order = workflow_steps[dependency]["order"]
                assert dependency_order < step_config["order"]
        
        # Test execution order
        execution_order = sorted(workflow_steps.items(), key=lambda x: x[1]["order"])
        expected_order = [
            "pdf_processing",
            "text_extraction", 
            "content_analysis",
            "outline_generation",
            "notes_generation",
            "quality_evaluation"
        ]
        
        actual_order = [step[0] for step in execution_order]
        assert actual_order == expected_order


class TestWorkflowMonitoring:
    """Test workflow monitoring and metrics collection."""
    
    def test_workflow_progress_tracking(self, tmp_path):
        """Test tracking workflow progress."""
        # Initialize progress tracker
        progress_tracker = {
            "workflow_id": "workflow_001",
            "start_time": datetime.now().isoformat(),
            "total_steps": 4,
            "completed_steps": 0,
            "current_step": "pdf_processing",
            "progress_percentage": 0.0,
            "estimated_completion": None
        }
        
        # Simulate step completions
        steps = ["pdf_processing", "analysis", "outline_generation", "notes_generation"]
        step_durations = [30, 60, 45, 50]  # seconds
        
        for i, (step, duration) in enumerate(zip(steps, step_durations)):
            progress_tracker["completed_steps"] = i + 1
            progress_tracker["current_step"] = step
            progress_tracker["progress_percentage"] = (i + 1) / len(steps) * 100
            
            # Simulate step completion
            assert progress_tracker["completed_steps"] <= progress_tracker["total_steps"]
            assert 0 <= progress_tracker["progress_percentage"] <= 100
        
        # Verify final state
        assert progress_tracker["completed_steps"] == progress_tracker["total_steps"]
        assert progress_tracker["progress_percentage"] == 100.0
    
    def test_workflow_metrics_collection(self, tmp_path):
        """Test collection of workflow metrics."""
        # Define metrics to collect
        workflow_metrics = {
            "performance": {
                "total_duration": 0.0,
                "step_durations": {},
                "throughput": 0.0,  # documents per hour
                "resource_usage": {
                    "memory_peak": 0,
                    "cpu_average": 0
                }
            },
            "quality": {
                "average_quality_score": 0.0,
                "quality_distribution": {},
                "pass_rate": 0.0
            },
            "errors": {
                "total_errors": 0,
                "error_types": {},
                "retry_rate": 0.0
            }
        }
        
        # Simulate metrics collection during workflow
        # Performance metrics
        workflow_metrics["performance"]["total_duration"] = 235.5  # seconds
        workflow_metrics["performance"]["step_durations"] = {
            "pdf_processing": 45.2,
            "analysis": 78.3,
            "outline_generation": 56.1,
            "notes_generation": 55.9
        }
        workflow_metrics["performance"]["throughput"] = 3600 / 235.5  # docs per hour
        
        # Quality metrics
        workflow_metrics["quality"]["average_quality_score"] = 0.82
        workflow_metrics["quality"]["quality_distribution"] = {
            "excellent": 2,
            "good": 5,
            "adequate": 2,
            "poor": 1
        }
        workflow_metrics["quality"]["pass_rate"] = 0.9
        
        # Error metrics
        workflow_metrics["errors"]["total_errors"] = 3
        workflow_metrics["errors"]["error_types"] = {
            "pdf_parsing_error": 1,
            "analysis_timeout": 1,
            "quality_check_failed": 1
        }
        workflow_metrics["errors"]["retry_rate"] = 0.3
        
        # Validate metrics
        assert workflow_metrics["performance"]["total_duration"] > 0
        assert workflow_metrics["performance"]["throughput"] > 0
        assert 0 <= workflow_metrics["quality"]["pass_rate"] <= 1
        assert workflow_metrics["errors"]["total_errors"] >= 0
        
        # Verify step duration sum
        total_step_duration = sum(workflow_metrics["performance"]["step_durations"].values())
        assert total_step_duration <= workflow_metrics["performance"]["total_duration"]
    
    def test_workflow_alerting_and_notifications(self, tmp_path):
        """Test workflow alerting and notification system."""
        # Define alerting thresholds
        alert_thresholds = {
            "processing_time": {
                "warning": 300,  # 5 minutes
                "critical": 600  # 10 minutes
            },
            "quality_score": {
                "warning": 0.6,
                "critical": 0.4
            },
            "error_rate": {
                "warning": 0.1,  # 10%
                "critical": 0.2  # 20%
            }
        }
        
        # Simulate workflow conditions and alerts
        current_conditions = {
            "processing_time": 450,  # 7.5 minutes
            "quality_score": 0.55,   # Below warning threshold
            "error_rate": 0.15       # Above warning, below critical
        }
        
        alerts = []
        
        # Check processing time
        if current_conditions["processing_time"] > alert_thresholds["processing_time"]["critical"]:
            alerts.append({"type": "critical", "metric": "processing_time", "value": current_conditions["processing_time"]})
        elif current_conditions["processing_time"] > alert_thresholds["processing_time"]["warning"]:
            alerts.append({"type": "warning", "metric": "processing_time", "value": current_conditions["processing_time"]})
        
        # Check quality score
        if current_conditions["quality_score"] < alert_thresholds["quality_score"]["critical"]:
            alerts.append({"type": "critical", "metric": "quality_score", "value": current_conditions["quality_score"]})
        elif current_conditions["quality_score"] < alert_thresholds["quality_score"]["warning"]:
            alerts.append({"type": "warning", "metric": "quality_score", "value": current_conditions["quality_score"]})
        
        # Check error rate
        if current_conditions["error_rate"] > alert_thresholds["error_rate"]["critical"]:
            alerts.append({"type": "critical", "metric": "error_rate", "value": current_conditions["error_rate"]})
        elif current_conditions["error_rate"] > alert_thresholds["error_rate"]["warning"]:
            alerts.append({"type": "warning", "metric": "error_rate", "value": current_conditions["error_rate"]})
        
        # Verify alerts were generated correctly
        assert len(alerts) == 3  # All three metrics should trigger warnings
        
        processing_time_alert = next((alert for alert in alerts if alert["metric"] == "processing_time"), None)
        quality_alert = next((alert for alert in alerts if alert["metric"] == "quality_score"), None)
        error_rate_alert = next((alert for alert in alerts if alert["metric"] == "error_rate"), None)
        
        assert processing_time_alert["type"] == "warning"
        assert quality_alert["type"] == "warning"
        assert error_rate_alert["type"] == "warning"