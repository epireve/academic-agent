"""
Command Line Interface for Academic Agent v2.

This module provides a user-friendly CLI for interacting with the academic agent system,
including task execution, configuration management, and monitoring capabilities.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml

from ..core.config_manager import ConfigManager
from ..core.logging import get_logger, setup_logging
from .academic_agent import AcademicAgent, analyze_content, create_academic_agent, process_pdfs
from .agent_config import AgentConfigManager, create_content_analysis_config, create_pdf_processing_config
from .plugin_system import create_plugin_manager
from .state_manager import create_state_manager

logger = get_logger(__name__)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--quiet", is_flag=True, help="Suppress output except errors")
@click.pass_context
def cli(ctx, config: Optional[Path], debug: bool, quiet: bool):
    """Academic Agent v2 - Simplified academic content processing system."""
    
    # Set up logging
    log_level = "DEBUG" if debug else "ERROR" if quiet else "INFO"
    setup_logging(log_level)
    
    # Store config in context
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["debug"] = debug
    ctx.obj["quiet"] = quiet


@cli.command()
@click.argument("pdf_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--agent-id", default="pdf_processor", help="Agent identifier")
@click.option("--concurrent", "-j", default=2, help="Number of concurrent tasks")
@click.pass_context
def process_pdf(ctx, pdf_files: List[Path], output: Optional[Path], agent_id: str, concurrent: int):
    """Process PDF files and extract content."""
    
    async def _process():
        try:
            # Create agent configuration
            config = create_pdf_processing_config(
                agent_id=agent_id,
                max_concurrent=concurrent,
            )
            
            # Create agent
            agent = AcademicAgent(agent_id=agent_id)
            agent.config = config.to_dict()
            
            if not ctx.obj["quiet"]:
                click.echo(f"Processing {len(pdf_files)} PDF files...")
            
            # Process PDFs
            results = await process_pdfs(agent, pdf_files, output)
            
            # Display results
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            if not ctx.obj["quiet"]:
                click.echo(f"\nProcessing complete:")
                click.echo(f"  Successful: {successful}")
                click.echo(f"  Failed: {failed}")
                
                if failed > 0:
                    click.echo("\nFailed files:")
                    for result in results:
                        if not result.success:
                            click.echo(f"  - {result.metadata.get('input_file', 'unknown')}: {result.error_message}")
            
            # Save results summary
            if output:
                summary_file = output / "processing_summary.json"
                with open(summary_file, "w") as f:
                    json.dump([r.to_dict() for r in results], f, indent=2)
                
                if not ctx.obj["quiet"]:
                    click.echo(f"\nResults saved to: {summary_file}")
            
            # Shutdown agent
            await agent.shutdown()
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_process())


@cli.command()
@click.argument("content_file", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file")
@click.option("--agent-id", default="content_analyzer", help="Agent identifier")
@click.pass_context
def analyze(ctx, content_file: Path, output: Optional[Path], agent_id: str):
    """Analyze text content."""
    
    async def _analyze():
        try:
            # Read content
            with open(content_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create agent configuration
            config = create_content_analysis_config(agent_id=agent_id)
            
            # Create agent
            agent = AcademicAgent(agent_id=agent_id)
            agent.config = config.to_dict()
            
            if not ctx.obj["quiet"]:
                click.echo(f"Analyzing content from {content_file}...")
            
            # Analyze content
            result = await analyze_content(agent, content)
            
            # Display results
            if result.success:
                analysis = result.output_data
                
                if not ctx.obj["quiet"]:
                    click.echo("\nAnalysis Results:")
                    click.echo(f"  Word count: {analysis.get('word_count', 0)}")
                    click.echo(f"  Character count: {analysis.get('character_count', 0)}")
                    click.echo(f"  Paragraph count: {analysis.get('paragraph_count', 0)}")
                    click.echo(f"  Quality score: {analysis.get('quality_score', 0):.2f}")
                    click.echo(f"  Reading time: {analysis.get('estimated_reading_time', 0)} minutes")
                    
                    if analysis.get('has_headings'):
                        click.echo("  ✓ Contains headings")
                    if analysis.get('has_lists'):
                        click.echo("  ✓ Contains lists")
                
                # Save results
                if output:
                    with open(output, "w") as f:
                        json.dump(result.to_dict(), f, indent=2)
                    
                    if not ctx.obj["quiet"]:
                        click.echo(f"\nResults saved to: {output}")
            else:
                click.echo(f"Analysis failed: {result.error_message}", err=True)
                sys.exit(1)
            
            # Shutdown agent
            await agent.shutdown()
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_analyze())


@cli.command()
@click.option("--agent-id", default="academic_agent", help="Agent identifier")
@click.option("--format", "output_format", default="json", type=click.Choice(["json", "yaml"]), help="Output format")
@click.pass_context
def status(ctx, agent_id: str, output_format: str):
    """Show agent status and metrics."""
    
    async def _show_status():
        try:
            # Create agent
            config_path = ctx.obj.get("config_path")
            agent = create_academic_agent(config_path=config_path, agent_id=agent_id)
            
            # Get status
            status_info = agent.get_status()
            
            # Format output
            if output_format == "yaml":
                output = yaml.dump(status_info, default_flow_style=False)
            else:
                output = json.dumps(status_info, indent=2)
            
            click.echo(output)
            
            # Shutdown agent
            await agent.shutdown()
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_show_status())


@cli.command()
@click.argument("config_file", type=click.Path(path_type=Path))
@click.option("--template", default="default", help="Configuration template to use")
@click.pass_context
def init_config(ctx, config_file: Path, template: str):
    """Initialize a new configuration file."""
    
    try:
        config_manager = AgentConfigManager()
        
        if template == "pdf":
            config = create_pdf_processing_config("academic_agent")
        elif template == "analysis":
            config = create_content_analysis_config("academic_agent")
        else:
            config = config_manager.create_default_config("academic_agent")
        
        config_manager.save_config_to_file(config, config_file)
        
        if not ctx.obj["quiet"]:
            click.echo(f"Configuration file created: {config_file}")
            click.echo(f"Template used: {template}")
        
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_config(ctx, config_file: Path):
    """Validate a configuration file."""
    
    try:
        config_manager = AgentConfigManager()
        config = config_manager.load_config_from_file(config_file)
        
        issues = config_manager.validate_config(config)
        
        if issues:
            click.echo("Configuration validation failed:", err=True)
            for issue in issues:
                click.echo(f"  - {issue}", err=True)
            sys.exit(1)
        else:
            if not ctx.obj["quiet"]:
                click.echo("Configuration is valid ✓")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--plugin-dirs", multiple=True, type=click.Path(exists=True, path_type=Path), help="Plugin directories")
@click.option("--builtin", is_flag=True, default=True, help="Include built-in plugins")
@click.pass_context
def list_plugins(ctx, plugin_dirs: List[Path], builtin: bool):
    """List available plugins."""
    
    try:
        # Create plugin manager
        plugin_manager = create_plugin_manager(
            plugin_dirs=list(plugin_dirs) if plugin_dirs else None,
            load_builtin=builtin,
        )
        
        # Get plugin information
        plugins = plugin_manager.registry.get_all_metadata()
        
        if not plugins:
            click.echo("No plugins found.")
            return
        
        click.echo(f"Found {len(plugins)} plugins:")
        for plugin_name, metadata in plugins.items():
            click.echo(f"\n  {plugin_name}")
            click.echo(f"    Version: {metadata.get('version', 'unknown')}")
            click.echo(f"    Description: {metadata.get('description', 'No description')}")
            
            supported_tasks = metadata.get('supported_tasks', [])
            if supported_tasks:
                click.echo(f"    Supported tasks: {', '.join(supported_tasks)}")
        
    except Exception as e:
        logger.error(f"Failed to list plugins: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("workflow_file", type=click.Path(exists=True, path_type=Path))
@click.option("--agent-id", default="workflow_agent", help="Agent identifier")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.pass_context
def run_workflow(ctx, workflow_file: Path, agent_id: str, output: Optional[Path]):
    """Run a workflow from a configuration file."""
    
    async def _run_workflow():
        try:
            # Load workflow configuration
            with open(workflow_file, "r") as f:
                if workflow_file.suffix.lower() == ".yaml":
                    workflow_data = yaml.safe_load(f)
                else:
                    workflow_data = json.load(f)
            
            # Create agent
            config_path = ctx.obj.get("config_path")
            agent = create_academic_agent(config_path=config_path, agent_id=agent_id)
            
            if not ctx.obj["quiet"]:
                click.echo(f"Running workflow from {workflow_file}...")
            
            # Execute workflow
            results = await agent.execute_workflow(workflow_data)
            
            # Display results
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            if not ctx.obj["quiet"]:
                click.echo(f"\nWorkflow complete:")
                click.echo(f"  Tasks completed: {successful}")
                click.echo(f"  Tasks failed: {failed}")
            
            # Save results
            if output:
                results_file = output / "workflow_results.json"
                with open(results_file, "w") as f:
                    json.dump([r.to_dict() for r in results], f, indent=2)
                
                if not ctx.obj["quiet"]:
                    click.echo(f"\nResults saved to: {results_file}")
            
            # Shutdown agent
            await agent.shutdown()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_run_workflow())


@cli.command()
@click.option("--agent-id", default="academic_agent", help="Agent identifier")
@click.option("--state-dir", type=click.Path(path_type=Path), help="State directory")
@click.pass_context
def monitor(ctx, agent_id: str, state_dir: Optional[Path]):
    """Monitor agent status in real-time."""
    
    async def _monitor():
        try:
            # Create state manager
            state_manager = await create_state_manager(
                agent_id=agent_id,
                state_dir=state_dir,
                auto_save=True,
            )
            
            click.echo(f"Monitoring agent: {agent_id}")
            click.echo("Press Ctrl+C to stop monitoring")
            
            try:
                while True:
                    # Get current state
                    state = state_manager.get_state()
                    
                    # Clear screen and show status
                    click.clear()
                    click.echo(f"Agent: {state.agent_id}")
                    click.echo(f"Status: {state.status}")
                    click.echo(f"Current tasks: {len(state.current_tasks)}")
                    click.echo(f"Total tasks: {len(state.task_history)}")
                    click.echo(f"Completed: {state.metrics.completed_tasks}")
                    click.echo(f"Failed: {state.metrics.failed_tasks}")
                    click.echo(f"Success rate: {state.metrics.success_rate:.2%}")
                    click.echo(f"Uptime: {state.metrics.get_uptime():.0f} seconds")
                    
                    if state.current_tasks:
                        click.echo("\nCurrent tasks:")
                        for task_id in state.current_tasks:
                            task = state.get_task(task_id)
                            if task:
                                click.echo(f"  - {task_id}: {task.task_type} ({task.status})")
                    
                    # Update every 5 seconds
                    await asyncio.sleep(5)
                    
            except KeyboardInterrupt:
                click.echo("\nMonitoring stopped.")
            
            # Stop state manager
            await state_manager.stop()
            
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(_monitor())


if __name__ == "__main__":
    cli()