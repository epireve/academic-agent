"""
Integration tests for the Academic Agent v2 system.

These tests verify the complete functionality of the simplified academic agent system,
including plugin integration, state management, and workflow execution.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.agents.academic_agent import (
    AcademicAgent,
    ContentAnalysisPlugin,
    PDFProcessorPlugin,
    TaskResult,
    analyze_content,
    create_academic_agent,
    process_pdfs,
)
from src.agents.agent_config import (
    AgentConfig,
    AgentConfigManager,
    create_content_analysis_config,
    create_pdf_processing_config,
)
from src.agents.plugin_system import create_plugin_manager
from src.agents.state_manager import create_state_manager, create_task_state
from src.core.logging import get_logger

logger = get_logger(__name__)


class TestAcademicAgent:
    """Test suite for Academic Agent core functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample configuration for testing."""
        config = create_pdf_processing_config("test_agent")
        config.working_directory = temp_dir
        config.output_directory = temp_dir / "output"
        config.temp_directory = temp_dir / "temp"
        config.state_file = temp_dir / "state.json"
        return config
    
    @pytest.fixture
    def agent(self, sample_config):
        """Create test agent instance."""
        agent = AcademicAgent(agent_id="test_agent")
        agent.config = sample_config.to_dict()
        return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "test_agent"
        assert "PDFProcessorPlugin" in agent.plugins
        assert "ContentAnalysisPlugin" in agent.plugins
        assert agent.state.agent_id == "test_agent"
        assert agent.state.status == "idle"
    
    def test_plugin_registration(self, agent):
        """Test plugin registration and management."""
        # Check initial plugins
        initial_plugins = agent.get_available_plugins()
        assert len(initial_plugins) >= 2
        
        # Create custom plugin
        class TestPlugin(ContentAnalysisPlugin):
            def get_plugin_name(self):
                return "TestPlugin"
        
        # Register plugin
        test_plugin = TestPlugin()
        agent.register_plugin("TestPlugin", test_plugin)
        
        # Verify registration
        assert "TestPlugin" in agent.get_available_plugins()
        assert len(agent.get_available_plugins()) == len(initial_plugins) + 1
        
        # Unregister plugin
        agent.unregister_plugin("TestPlugin")
        assert "TestPlugin" not in agent.get_available_plugins()
    
    @pytest.mark.asyncio
    async def test_single_task_execution(self, agent):
        """Test execution of a single task."""
        task_data = {
            "task_id": "test_task",
            "task_type": "content_analysis",
            "plugin": "ContentAnalysisPlugin",
            "input_data": {
                "content": "This is test content for analysis."
            }
        }
        
        result = await agent.execute_task(task_data)
        
        assert result.task_id == "test_task"
        assert result.task_type == "content_analysis"
        assert result.success is True
        assert result.output_data is not None
        assert "word_count" in result.output_data
        assert result.processing_time is not None
        assert result.quality_score is not None
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, agent):
        """Test execution of a complete workflow."""
        workflow_data = {
            "content": "This is sample academic content for workflow testing. " * 20
        }
        
        results = await agent.execute_workflow(workflow_data)
        
        assert len(results) > 0
        assert all(isinstance(r, TaskResult) for r in results)
        
        # Check that at least one task succeeded
        successful_tasks = [r for r in results if r.success]
        assert len(successful_tasks) > 0
        
        # Check content analysis result
        analysis_results = [r for r in results if r.task_type == "content_analysis"]
        assert len(analysis_results) > 0
        
        analysis_result = analysis_results[0]
        assert analysis_result.success
        assert "word_count" in analysis_result.output_data
        assert analysis_result.output_data["word_count"] > 0
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, agent, temp_dir):
        """Test agent state persistence."""
        state_file = temp_dir / "test_state.json"
        
        # Save state
        agent.save_state(state_file)
        assert state_file.exists()
        
        # Modify agent state
        agent.state.status = "test_status"
        agent.metrics["test_metric"] = 123
        
        # Load state
        agent.load_state(state_file)
        
        # State should be restored to original values
        assert agent.state.status == "idle"  # Original status
        assert "test_metric" not in agent.metrics
    
    @pytest.mark.asyncio
    async def test_agent_shutdown(self, agent):
        """Test agent shutdown process."""
        initial_status = agent.state.status
        
        await agent.shutdown()
        
        assert agent.state.status == "shutdown"
    
    def test_agent_status(self, agent):
        """Test agent status reporting."""
        status = agent.get_status()
        
        assert "agent_id" in status
        assert "state" in status
        assert "metrics" in status
        assert "plugins" in status
        assert "config" in status
        
        assert status["agent_id"] == "test_agent"
        assert status["state"]["agent_id"] == "test_agent"


class TestPluginSystem:
    """Test suite for plugin system functionality."""
    
    @pytest.fixture
    def plugin_manager(self):
        """Create plugin manager for testing."""
        return create_plugin_manager(load_builtin=True)
    
    def test_plugin_registry(self, plugin_manager):
        """Test plugin registry functionality."""
        registry = plugin_manager.registry
        
        # Check built-in plugins are loaded
        plugins = registry.list_plugins()
        assert "PDFProcessorPlugin" in plugins
        assert "ContentAnalysisPlugin" in plugins
        
        # Check plugin metadata
        pdf_metadata = registry.get_plugin_metadata("PDFProcessorPlugin")
        assert pdf_metadata is not None
        assert pdf_metadata["name"] == "PDFProcessorPlugin"
        
        analysis_metadata = registry.get_plugin_metadata("ContentAnalysisPlugin")
        assert analysis_metadata is not None
        assert analysis_metadata["name"] == "ContentAnalysisPlugin"
    
    def test_plugin_manager_lifecycle(self, plugin_manager):
        """Test plugin manager lifecycle operations."""
        from src.agents.agent_config import PluginConfig
        
        # Initialize plugin
        config = PluginConfig(
            name="ContentAnalysisPlugin",
            enabled=True,
            config={"test": True}
        )
        
        success = plugin_manager.initialize_plugin("ContentAnalysisPlugin", config)
        assert success
        
        # Check active plugins
        active = plugin_manager.list_active_plugins()
        assert "ContentAnalysisPlugin" in active
        
        # Get plugin instance
        instance = plugin_manager.get_plugin_instance("ContentAnalysisPlugin")
        assert instance is not None
        
        # Check plugin status
        status = plugin_manager.get_plugin_status("ContentAnalysisPlugin")
        assert status["status"] == "active"
        assert status["enabled"] is True
        
        # Shutdown plugin
        plugin_manager.shutdown_plugin("ContentAnalysisPlugin")
        assert "ContentAnalysisPlugin" not in plugin_manager.list_active_plugins()


class TestConfigurationSystem:
    """Test suite for configuration system functionality."""
    
    @pytest.fixture
    def config_manager(self):
        """Create configuration manager for testing."""
        return AgentConfigManager()
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "agent_id": "test_agent",
                "plugins": {
                    "TestPlugin": {
                        "name": "TestPlugin",
                        "enabled": True,
                        "config": {"test": True}
                    }
                }
            }
            yaml.dump(config_data, f)
            yield Path(f.name)
        Path(f.name).unlink()
    
    def test_default_config_creation(self, config_manager):
        """Test default configuration creation."""
        config = config_manager.create_default_config("test_agent")
        
        assert config.agent_id == "test_agent"
        assert len(config.plugins) > 0
        assert "PDFProcessorPlugin" in config.plugins
        assert "ContentAnalysisPlugin" in config.plugins
    
    def test_config_file_operations(self, config_manager, temp_config_file):
        """Test configuration file save/load operations."""
        # Load configuration
        config = config_manager.load_config_from_file(temp_config_file)
        
        assert config.agent_id == "test_agent"
        assert "TestPlugin" in config.plugins
        
        # Modify configuration
        config.agent_id = "modified_agent"
        
        # Save configuration
        new_config_file = temp_config_file.parent / "modified_config.yaml"
        config_manager.save_config_to_file(config, new_config_file)
        
        # Load modified configuration
        loaded_config = config_manager.load_config_from_file(new_config_file)
        assert loaded_config.agent_id == "modified_agent"
        
        # Cleanup
        new_config_file.unlink()
    
    def test_config_validation(self, config_manager):
        """Test configuration validation."""
        # Valid configuration
        valid_config = config_manager.create_default_config("test_agent")
        issues = config_manager.validate_config(valid_config)
        assert len(issues) == 0
        
        # Invalid configuration (modify to create issues)
        invalid_config = config_manager.create_default_config("test_agent")
        invalid_config.quality.quality_threshold = 1.5  # Invalid threshold
        
        issues = config_manager.validate_config(invalid_config)
        assert len(issues) > 0
    
    def test_specialized_configs(self):
        """Test specialized configuration creation."""
        # PDF processing config
        pdf_config = create_pdf_processing_config("pdf_agent")
        assert pdf_config.agent_id == "pdf_agent"
        assert "PDFProcessorPlugin" in pdf_config.plugins
        
        # Content analysis config
        analysis_config = create_content_analysis_config("analysis_agent")
        assert analysis_config.agent_id == "analysis_agent"
        assert "ContentAnalysisPlugin" in analysis_config.plugins


class TestStateManagement:
    """Test suite for state management functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for state files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_state_manager_lifecycle(self, temp_dir):
        """Test state manager lifecycle operations."""
        # Create state manager
        state_manager = await create_state_manager(
            agent_id="test_agent",
            state_dir=temp_dir,
            auto_save=False  # Disable auto-save for testing
        )
        
        assert state_manager.agent_id == "test_agent"
        assert state_manager.state.agent_id == "test_agent"
        
        # Stop state manager
        await state_manager.stop()
    
    @pytest.mark.asyncio
    async def test_task_state_management(self, temp_dir):
        """Test task state management operations."""
        state_manager = await create_state_manager(
            agent_id="test_agent",
            state_dir=temp_dir,
            auto_save=False
        )
        
        # Create test task
        task = create_task_state(
            task_id="test_task",
            task_type="test_type",
            input_data={"test": "data"},
            priority=1
        )
        
        # Add task to state
        state_manager.add_task(task)
        
        # Verify task was added
        retrieved_task = state_manager.get_task("test_task")
        assert retrieved_task is not None
        assert retrieved_task.task_id == "test_task"
        assert retrieved_task.status == "pending"
        
        # Start task
        task.start()
        state_manager.update_task("test_task", task)
        
        # Complete task
        task.complete({"result": "success"})
        state_manager.update_task("test_task", task)
        
        # Verify final state
        final_task = state_manager.get_task("test_task")
        assert final_task.status == "completed"
        assert final_task.output_data["result"] == "success"
        
        await state_manager.stop()
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, temp_dir):
        """Test state persistence and recovery."""
        # Create and populate state manager
        state_manager1 = await create_state_manager(
            agent_id="test_agent",
            state_dir=temp_dir,
            auto_save=False
        )
        
        # Add test data
        task = create_task_state(
            task_id="persistent_task",
            task_type="test_type",
            input_data={"test": "persistence"}
        )
        state_manager1.add_task(task)
        
        # Save state
        await state_manager1.save_state()
        await state_manager1.stop()
        
        # Create new state manager and load state
        state_manager2 = await create_state_manager(
            agent_id="test_agent",
            state_dir=temp_dir,
            auto_save=False
        )
        
        # Verify state was loaded
        retrieved_task = state_manager2.get_task("persistent_task")
        assert retrieved_task is not None
        assert retrieved_task.task_id == "persistent_task"
        assert retrieved_task.input_data["test"] == "persistence"
        
        await state_manager2.stop()


class TestIntegrationScenarios:
    """Test suite for end-to-end integration scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.mark.asyncio
    async def test_complete_pdf_processing_workflow(self, temp_dir):
        """Test complete PDF processing workflow."""
        # Create test PDF file
        test_pdf = temp_dir / "test.pdf"
        with open(test_pdf, "w") as f:
            f.write("This is a test PDF file")
        
        # Create agent
        agent = create_academic_agent(agent_id="integration_test")
        
        # Mock PDF processor to avoid marker dependency
        with patch.object(agent.plugins["PDFProcessorPlugin"], "pdf_processor") as mock_processor:
            mock_processor.process_pdf.return_value = {
                "success": True,
                "content_extracted": "Sample PDF content",
                "metadata": {"title": "Test Document"}
            }
            
            # Process PDF
            results = await process_pdfs(agent, [test_pdf], temp_dir / "output")
            
            assert len(results) >= 1
            pdf_result = results[0]
            assert pdf_result.success
            assert pdf_result.task_type == "pdf_processing"
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_content_analysis_workflow(self, temp_dir):
        """Test content analysis workflow."""
        agent = create_academic_agent(agent_id="analysis_test")
        
        # Test content
        content = """
        # Academic Paper Title
        
        This is the abstract of an academic paper that discusses important
        research findings in the field of computer science.
        
        ## Introduction
        
        The introduction provides background information and motivation
        for the research presented in this paper.
        
        ## Methodology
        
        This section describes the methods used in the research.
        
        ## Results
        
        The results section presents the findings of the study.
        
        ## Conclusion
        
        The conclusion summarizes the key contributions of this work.
        """
        
        # Analyze content
        result = await analyze_content(agent, content)
        
        assert result.success
        assert result.task_type == "content_analysis"
        
        analysis = result.output_data
        assert analysis["word_count"] > 50
        assert analysis["paragraph_count"] > 3
        assert analysis["has_headings"] is True
        assert analysis["quality_score"] > 0
        
        await agent.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_recovery_scenario(self, temp_dir):
        """Test agent recovery after shutdown."""
        # Create agent with state persistence
        config = create_pdf_processing_config("recovery_test")
        config.working_directory = temp_dir
        config.state_file = temp_dir / "recovery_state.json"
        
        agent1 = AcademicAgent(agent_id="recovery_test")
        agent1.config = config.to_dict()
        
        # Simulate some work
        task_data = {
            "task_id": "recovery_task",
            "task_type": "content_analysis",
            "plugin": "ContentAnalysisPlugin",
            "input_data": {"content": "Test content"}
        }
        
        result1 = await agent1.execute_task(task_data)
        assert result1.success
        
        # Save state and shutdown
        agent1.save_state(config.state_file)
        await agent1.shutdown()
        
        # Create new agent and load state
        agent2 = AcademicAgent(agent_id="recovery_test")
        agent2.config = config.to_dict()
        agent2.load_state(config.state_file)
        
        # Verify recovery
        assert agent2.metrics["tasks_completed"] > 0
        
        await agent2.shutdown()


# Utility functions for testing

def create_mock_pdf_result():
    """Create mock PDF processing result."""
    return {
        "success": True,
        "file_path": "test.pdf",
        "content_extracted": "Mock PDF content",
        "metadata": {"title": "Mock Document"},
        "processing_time": 1.0,
    }


def create_sample_academic_content():
    """Create sample academic content for testing."""
    return """
    # Research Paper: Machine Learning Applications
    
    ## Abstract
    
    This paper presents a comprehensive study of machine learning applications
    in various domains, including healthcare, finance, and autonomous systems.
    
    ## Introduction
    
    Machine learning has emerged as a powerful tool for solving complex problems
    across multiple disciplines. This research investigates the effectiveness
    of different ML approaches in real-world scenarios.
    
    ### Research Questions
    
    1. What are the most effective ML algorithms for each domain?
    2. How do performance metrics vary across different applications?
    3. What are the key challenges in deploying ML systems?
    
    ## Methodology
    
    Our study employed a mixed-methods approach combining:
    
    - Quantitative analysis of algorithm performance
    - Qualitative assessment of implementation challenges
    - Case studies from industry partners
    
    ## Results
    
    The results demonstrate significant variations in algorithm effectiveness
    depending on the specific application domain and data characteristics.
    
    ## Conclusion
    
    This research provides valuable insights for practitioners seeking to
    implement machine learning solutions in their respective fields.
    """


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])