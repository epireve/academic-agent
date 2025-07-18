"""
Academic Agent v2 Demo Script

This script demonstrates the capabilities of the simplified academic agent system,
showing how to process PDFs, analyze content, and manage agent workflows.
"""

import asyncio
import json
from pathlib import Path

from src.agents.academic_agent import (
    AcademicAgent,
    analyze_content,
    create_academic_agent,
    process_pdfs,
)
from src.agents.agent_config import (
    AgentConfigManager,
    create_content_analysis_config,
    create_pdf_processing_config,
)
from src.agents.plugin_system import create_plugin_manager
from src.agents.state_manager import create_state_manager, create_task_state
from src.core.logging import get_logger, setup_logging

# Set up logging
setup_logging("INFO")
logger = get_logger(__name__)


async def demo_pdf_processing():
    """Demonstrate PDF processing capabilities."""
    print("\n" + "="*50)
    print("PDF Processing Demo")
    print("="*50)
    
    # Create a PDF processing agent
    config = create_pdf_processing_config(
        agent_id="pdf_demo_agent",
        use_gpu=True,
        max_concurrent=2,
        quality_threshold=0.7,
    )
    
    agent = AcademicAgent(agent_id="pdf_demo_agent")
    agent.config = config.to_dict()
    
    print(f"Created agent: {agent.agent_id}")
    print(f"Available plugins: {agent.get_available_plugins()}")
    
    # Simulate PDF files (you would replace these with real PDF paths)
    pdf_files = [
        Path("examples/sample_document1.pdf"),
        Path("examples/sample_document2.pdf"),
    ]
    
    # Note: For demo purposes, we'll create dummy files
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample PDF files (dummy content)
    for pdf_file in pdf_files:
        pdf_file.parent.mkdir(parents=True, exist_ok=True)
        if not pdf_file.exists():
            # Create a dummy PDF file for demo
            with open(pdf_file, "w") as f:
                f.write("This is a dummy PDF file for demonstration purposes.")
    
    print(f"\nProcessing {len(pdf_files)} PDF files...")
    
    try:
        # Process PDFs
        results = await process_pdfs(agent, pdf_files, output_dir)
        
        # Display results
        print(f"\nProcessing Results:")
        for result in results:
            print(f"  Task ID: {result.task_id}")
            print(f"  Success: {result.success}")
            if result.success:
                print(f"  Processing time: {result.processing_time:.2f}s")
                print(f"  Quality score: {result.quality_score:.2f}")
                print(f"  Output file: {result.output_data.get('output_file', 'N/A')}")
            else:
                print(f"  Error: {result.error_message}")
            print()
        
        # Show agent status
        status = agent.get_status()
        print(f"Agent Status:")
        print(f"  Total tasks: {status['metrics']['tasks_completed'] + status['metrics']['tasks_failed']}")
        print(f"  Success rate: {status['metrics']['tasks_completed'] / max(1, status['metrics']['tasks_completed'] + status['metrics']['tasks_failed']):.2%}")
        
    except Exception as e:
        print(f"Error in PDF processing: {e}")
    
    finally:
        await agent.shutdown()


async def demo_content_analysis():
    """Demonstrate content analysis capabilities."""
    print("\n" + "="*50)
    print("Content Analysis Demo")
    print("="*50)
    
    # Create a content analysis agent
    config = create_content_analysis_config(
        agent_id="analysis_demo_agent",
        advanced_analysis=True,
        quality_threshold=0.8,
    )
    
    agent = AcademicAgent(agent_id="analysis_demo_agent")
    agent.config = config.to_dict()
    
    # Sample academic content
    sample_content = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that provides systems
    the ability to automatically learn and improve from experience without being
    explicitly programmed. Machine learning focuses on the development of computer
    programs that can access data and use it to learn for themselves.
    
    ## Types of Machine Learning
    
    There are several types of machine learning algorithms:
    
    1. **Supervised Learning**: Uses labeled training data to learn a mapping function
       from input variables to output variables.
    
    2. **Unsupervised Learning**: Uses input data without labeled responses to find
       hidden patterns or intrinsic structures in data.
    
    3. **Reinforcement Learning**: Learns optimal actions through trial and error
       interactions with a dynamic environment.
    
    ## Applications
    
    Machine learning has numerous applications across various domains:
    
    - Image recognition and computer vision
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    - Financial analysis
    
    ## Conclusion
    
    As data continues to grow exponentially, machine learning becomes increasingly
    important for extracting insights and making predictions from complex datasets.
    The field continues to evolve with new algorithms and techniques being developed
    regularly.
    """
    
    print("Analyzing sample content...")
    
    try:
        # Analyze content
        result = await analyze_content(agent, sample_content)
        
        if result.success:
            analysis = result.output_data
            print(f"\nAnalysis Results:")
            print(f"  Word count: {analysis.get('word_count', 0)}")
            print(f"  Character count: {analysis.get('character_count', 0)}")
            print(f"  Paragraph count: {analysis.get('paragraph_count', 0)}")
            print(f"  Quality score: {analysis.get('quality_score', 0):.2f}")
            print(f"  Reading time: {analysis.get('estimated_reading_time', 0)} minutes")
            print(f"  Has headings: {'Yes' if analysis.get('has_headings') else 'No'}")
            print(f"  Has lists: {'Yes' if analysis.get('has_lists') else 'No'}")
            print(f"  Processing time: {result.processing_time:.2f}s")
        else:
            print(f"Analysis failed: {result.error_message}")
    
    except Exception as e:
        print(f"Error in content analysis: {e}")
    
    finally:
        await agent.shutdown()


async def demo_workflow_execution():
    """Demonstrate workflow execution capabilities."""
    print("\n" + "="*50)
    print("Workflow Execution Demo")
    print("="*50)
    
    # Create agent
    agent = create_academic_agent(agent_id="workflow_demo_agent")
    
    # Define a workflow
    workflow_data = {
        "workflow_name": "Academic Content Processing",
        "description": "Process academic documents and analyze content",
        "content": "This is sample academic content for workflow demonstration. " * 50,
        "pdf_files": [],  # Would contain PDF file paths in real scenario
    }
    
    print("Executing workflow...")
    
    try:
        # Execute workflow
        results = await agent.execute_workflow(workflow_data)
        
        print(f"\nWorkflow Results:")
        print(f"  Total tasks: {len(results)}")
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        for i, result in enumerate(results, 1):
            print(f"\n  Task {i}:")
            print(f"    ID: {result.task_id}")
            print(f"    Type: {result.task_type}")
            print(f"    Success: {result.success}")
            if result.processing_time:
                print(f"    Processing time: {result.processing_time:.2f}s")
            if result.quality_score:
                print(f"    Quality score: {result.quality_score:.2f}")
    
    except Exception as e:
        print(f"Error in workflow execution: {e}")
    
    finally:
        await agent.shutdown()


async def demo_plugin_system():
    """Demonstrate plugin system capabilities."""
    print("\n" + "="*50)
    print("Plugin System Demo")
    print("="*50)
    
    # Create plugin manager
    plugin_manager = create_plugin_manager(load_builtin=True)
    
    print("Available plugins:")
    plugins = plugin_manager.registry.list_plugins()
    for plugin_name in plugins:
        metadata = plugin_manager.registry.get_plugin_metadata(plugin_name)
        print(f"  - {plugin_name}")
        print(f"    Version: {metadata.get('version', 'unknown')}")
        print(f"    Supported tasks: {metadata.get('supported_tasks', [])}")
        print(f"    Description: {metadata.get('description', 'No description')}")
        print()
    
    # Initialize plugins
    from src.agents.agent_config import PluginConfig
    
    for plugin_name in plugins:
        config = PluginConfig(
            name=plugin_name,
            enabled=True,
            config={"test_mode": True}
        )
        success = plugin_manager.initialize_plugin(plugin_name, config)
        print(f"Initialized {plugin_name}: {'Success' if success else 'Failed'}")
    
    # Show active plugins
    active_plugins = plugin_manager.list_active_plugins()
    print(f"\nActive plugins: {active_plugins}")
    
    # Shutdown plugins
    plugin_manager.shutdown_all_plugins()
    print("All plugins shut down")


async def demo_state_management():
    """Demonstrate state management capabilities."""
    print("\n" + "="*50)
    print("State Management Demo")
    print("="*50)
    
    # Create state manager
    state_manager = await create_state_manager(
        agent_id="state_demo_agent",
        state_dir=Path("temp/state"),
        auto_save=True,
        save_interval=10.0,
    )
    
    print(f"State manager created for agent: {state_manager.agent_id}")
    
    # Create some tasks
    tasks = [
        create_task_state(
            task_id="task_1",
            task_type="pdf_processing",
            input_data={"file": "document1.pdf"},
            priority=1,
        ),
        create_task_state(
            task_id="task_2",
            task_type="content_analysis",
            input_data={"content": "Sample content"},
            priority=2,
            dependencies=["task_1"],
        ),
    ]
    
    # Add tasks to state
    for task in tasks:
        state_manager.add_task(task)
        print(f"Added task: {task.task_id}")
    
    # Simulate task execution
    for task in tasks:
        task.start()
        state_manager.update_task(task.task_id, task)
        print(f"Started task: {task.task_id}")
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        task.complete({"result": f"Processed {task.task_id}"})
        state_manager.update_task(task.task_id, task)
        print(f"Completed task: {task.task_id}")
    
    # Get state information
    state = state_manager.get_state()
    print(f"\nState Summary:")
    print(f"  Agent ID: {state.agent_id}")
    print(f"  Status: {state.status}")
    print(f"  Total tasks: {len(state.task_history)}")
    print(f"  Completed tasks: {len(state.get_completed_tasks())}")
    print(f"  Success rate: {state.metrics.success_rate:.2%}")
    
    # Save state
    await state_manager.save_state()
    print("State saved to disk")
    
    # Get recovery info
    recovery_info = state_manager.get_recovery_info()
    print(f"\nRecovery Info:")
    print(f"  State files: {recovery_info['state_files']}")
    print(f"  Last checkpoint: {recovery_info['last_checkpoint']}")
    
    # Stop state manager
    await state_manager.stop()
    print("State manager stopped")


async def demo_configuration_management():
    """Demonstrate configuration management capabilities."""
    print("\n" + "="*50)
    print("Configuration Management Demo")
    print("="*50)
    
    # Create configuration manager
    config_manager = AgentConfigManager()
    
    # Create different types of configurations
    configs = {
        "pdf_config": create_pdf_processing_config("pdf_agent"),
        "analysis_config": create_content_analysis_config("analysis_agent"),
        "default_config": config_manager.create_default_config("default_agent"),
    }
    
    # Display configurations
    for config_name, config in configs.items():
        print(f"\n{config_name.replace('_', ' ').title()}:")
        print(f"  Agent ID: {config.agent_id}")
        print(f"  Plugins: {list(config.plugins.keys())}")
        print(f"  Quality threshold: {config.quality.quality_threshold}")
        print(f"  Max concurrent tasks: {config.task_planning.max_concurrent_tasks}")
        
        # Validate configuration
        issues = config_manager.validate_config(config)
        if issues:
            print(f"  Validation issues: {issues}")
        else:
            print("  Configuration is valid ✓")
    
    # Save and load configuration
    config_file = Path("temp/demo_config.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    config_manager.save_config_to_file(configs["default_config"], config_file)
    print(f"\nConfiguration saved to: {config_file}")
    
    loaded_config = config_manager.load_config_from_file(config_file)
    print(f"Configuration loaded successfully: {loaded_config.agent_id}")


async def main():
    """Run all demos."""
    print("Academic Agent v2 - Comprehensive Demo")
    print("="*60)
    
    try:
        # Run individual demos
        await demo_pdf_processing()
        await demo_content_analysis()
        await demo_workflow_execution()
        await demo_plugin_system()
        await demo_state_management()
        await demo_configuration_management()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())