from typing import Dict, List, Any
import time
import psutil
from datetime import datetime
from .base_agent import BaseAgent, AgentMessage
from .communication_manager import get_communication_manager
import json
import os
import logging


class MainAcademicAgent(BaseAgent):
    """Main Academic Agent for orchestrating the workflow"""

    def __init__(self, config_path: str = None):
        super().__init__("main_academic_agent")
        self.active_processes: Dict[str, Dict] = {}
        self.performance_metrics: Dict[str, List[float]] = {
            "processing_times": [],
            "memory_usage": [],
            "quality_scores": [],
        }

        # Agent states and workflow tracking
        self.workflow_status = {}
        self.current_stage = None
        self.improvement_cycles = 0
        self.max_improvement_cycles = 3

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize communication manager
        self.comm_manager = get_communication_manager(config_path)

        # Register with communication manager
        self.comm_manager.register_agent(self.agent_id, self)

        # Register message handlers
        self._register_message_handlers()

        # Setup feedback loops
        self._setup_feedback_loops()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "quality_threshold": 0.7,
            "improvement_threshold": 0.3,
            "max_improvement_cycles": 3,
            "communication_interval": 30,
            "agent_specialized_prompts": {
                "ingestion_agent": "Focus on accurately converting PDFs to markdown while preserving all content and structure. Pay special attention to academic notation and diagrams.",
                "outline_agent": "Create comprehensive outlines that capture the hierarchical relationships between concepts. Ensure all key topics are included with proper weighting of importance.",
                "notes_agent": "Expand outlines into detailed academic notes that maintain academic rigor while being clear and comprehensive. Include examples and applications where appropriate.",
                "quality_manager": "Rigorously evaluate content quality against academic standards. Look for completeness, accuracy, clarity, and proper citation of sources.",
                "update_agent": "Improve notes without changing original meaning. Focus on enrichment, clarification, and better organization while preserving the academic integrity of the content.",
            },
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    for key, value in loaded_config.items():
                        default_config[key] = value
            except Exception as e:
                self.logger.error(f"Error loading config: {str(e)}")

        return default_config

    def _register_message_handlers(self) -> None:
        """Register handlers for different message types"""
        self.comm_manager.register_message_handler(
            self.agent_id, "status_update", self._handle_status_update
        )
        self.comm_manager.register_message_handler(
            self.agent_id, "quality_report", self._handle_quality_report
        )
        self.comm_manager.register_message_handler(
            self.agent_id, "improvement_complete", self._handle_improvement_complete
        )

    def _setup_feedback_loops(self) -> None:
        """Setup feedback loops between agents"""
        # Quality feedback loops
        self.comm_manager.create_feedback_loop(
            "quality_manager", "notes_agent", "quality", 60 * 5  # Every 5 minutes
        )
        self.comm_manager.create_feedback_loop(
            "quality_manager", "outline_agent", "quality", 60 * 10  # Every 10 minutes
        )

        # Improvement suggestion loops
        self.comm_manager.create_feedback_loop(
            "update_agent",
            "notes_agent",
            "improvement_suggestions",
            60 * 15,  # Every 15 minutes
        )

    def initialize_pipeline(self, config: Dict[str, Any]) -> bool:
        """Initialize the processing pipeline with configuration"""
        try:
            self.logger.info("Initializing processing pipeline")
            self.validate_config(config)

            # Set up processing directories
            self._setup_directories()

            # Initialize performance monitoring
            self._init_monitoring()

            # Initialize workflow status
            self.workflow_status = {
                "ingestion": {"status": "pending", "files": []},
                "outline": {"status": "pending", "files": []},
                "notes": {"status": "pending", "files": []},
                "quality": {"status": "pending", "files": []},
                "improvement": {"status": "pending", "cycles": 0},
            }

            # Share configuration with agents
            self._share_configuration()

            self.logger.info("Pipeline initialized successfully")
            return True

        except Exception as e:
            self.handle_error(e, {"operation": "pipeline_initialization"})
            return False

    def _share_configuration(self) -> None:
        """Share configuration with other agents"""
        for agent_id, prompt in self.config.get(
            "agent_specialized_prompts", {}
        ).items():
            self.send_message(
                recipient=agent_id,
                message_type="configuration",
                content={
                    "specialized_prompt": prompt,
                    "quality_threshold": self.config.get("quality_threshold", 0.7),
                    "improvement_threshold": self.config.get(
                        "improvement_threshold", 0.3
                    ),
                },
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate pipeline configuration"""
        required_fields = ["input_dir", "output_dir", "quality_threshold"]
        if not all(field in config for field in required_fields):
            self.logger.error(
                f"Missing required configuration fields: {required_fields}"
            )
            return False
        return True

    def _setup_directories(self) -> None:
        """Set up required processing directories"""
        directories = [
            "input",
            "processed/markdown",
            "processed/img",
            "processed/notes",
            "processed/quality",
            "processed/outlines",
            "processed/improved_notes",
            "logs",
            "logs/messages",
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _init_monitoring(self) -> None:
        """Initialize performance monitoring"""
        self.start_time = datetime.now()
        self.process = psutil.Process()

    def monitor_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Monitor status and resource usage of an agent"""
        # First check with communication manager
        comm_status = self.comm_manager.get_agent_status(agent_id)
        if comm_status["status"] != "unknown":
            # Merge with local tracking if available
            if agent_id in self.active_processes:
                process_info = self.active_processes[agent_id]
                comm_status.update(
                    {
                        "runtime": time.time() - process_info["start_time"],
                        "memory_usage": process_info["process"].memory_info().rss
                        / 1024
                        / 1024,  # MB
                        "cpu_usage": process_info["process"].cpu_percent(),
                    }
                )
            return comm_status

        # Fall back to local tracking
        if agent_id not in self.active_processes:
            return {"status": "inactive"}

        process_info = self.active_processes[agent_id]
        current_time = time.time()

        status = {
            "status": "active",
            "runtime": current_time - process_info["start_time"],
            "memory_usage": process_info["process"].memory_info().rss
            / 1024
            / 1024,  # MB
            "cpu_usage": process_info["process"].cpu_percent(),
            "last_heartbeat": process_info["last_heartbeat"],
        }

        # Check for timeout
        if status["runtime"] > self.processing_timeout:
            status["status"] = "timeout"
            self.handle_timeout(agent_id)

        return status

    def handle_timeout(self, agent_id: str) -> None:
        """Handle agent timeout"""
        self.logger.warning(f"Agent {agent_id} timed out")
        if agent_id in self.active_processes:
            process_info = self.active_processes[agent_id]
            process_info["process"].terminate()
            del self.active_processes[agent_id]

        # Update workflow status
        for stage, info in self.workflow_status.items():
            if stage.startswith(agent_id.split("_")[0]):
                info["status"] = "timeout"

        # Notify other agents
        self.send_message(
            recipient="all",
            message_type="agent_timeout",
            content={"agent_id": agent_id, "timestamp": datetime.now().isoformat()},
        )

    def coordinate_workflow(self, input_files: List[str]) -> bool:
        """Coordinate the processing workflow"""
        try:
            # Update workflow status with input files
            self.workflow_status["ingestion"]["files"] = input_files

            # Process each stage sequentially with feedback loops
            stages = ["ingestion", "outline", "notes", "quality", "improvement"]

            for stage in stages:
                self.current_stage = stage
                self.logger.info(f"Starting {stage} stage")

                # Skip improvement if improvement cycles already maxed out
                if (
                    stage == "improvement"
                    and self.improvement_cycles >= self.max_improvement_cycles
                ):
                    self.logger.info(
                        f"Maximum improvement cycles reached ({self.improvement_cycles})"
                    )
                    continue

                start_time = time.time()

                # Process stage
                if stage == "improvement":
                    success = self._process_improvement_cycle()
                else:
                    success = self._process_stage(
                        stage, self.workflow_status[stage]["files"]
                    )

                if not success:
                    self.logger.error(f"Failed to process {stage} stage")
                    return False

                # Process communication manager messages
                self.comm_manager.process_messages(10)

                # Process feedback loops
                self.comm_manager.process_feedback_loops()

                # Update metrics
                processing_time = time.time() - start_time
                self.performance_metrics["processing_times"].append(processing_time)
                self.performance_metrics["memory_usage"].append(
                    self.process.memory_info().rss / 1024 / 1024  # MB
                )

                # Update workflow status
                self.workflow_status[stage]["status"] = "completed"

            # Generate final report
            self.generate_workflow_report(self.workflow_status)
            return True

        except Exception as e:
            self.handle_error(e, {"operation": "workflow_coordination"})
            return False

    def _process_stage(self, stage: str, files: List[str]) -> bool:
        """Process a specific workflow stage"""
        try:
            # Determine the agent for this stage
            agent_id = f"{stage}_agent"

            # Prepare message content based on stage
            message_content = {
                "stage": stage,
                "files": files,
                "timestamp": datetime.now().isoformat(),
            }

            # Add stage-specific content
            if stage == "outline":
                # For outline stage, files are markdown files from ingestion
                message_content["markdown_files"] = self.workflow_status[
                    "ingestion"
                ].get("output_files", [])

            elif stage == "notes":
                # For notes stage, include outline path
                message_content["outline_path"] = self.workflow_status["outline"].get(
                    "output_path", ""
                )
                message_content["source_files"] = self.workflow_status["ingestion"].get(
                    "output_files", []
                )

            elif stage == "quality":
                # For quality stage, include paths to notes and outlines
                message_content["notes_path"] = self.workflow_status["notes"].get(
                    "output_path", ""
                )
                message_content["outline_path"] = self.workflow_status["outline"].get(
                    "output_path", ""
                )

            # Send process message to appropriate agent
            success = self.send_message(
                recipient=agent_id,
                message_type="process_start",
                content=message_content,
            )

            if not success:
                self.logger.error(f"Failed to initiate {stage} stage")
                return False

            # Wait for completion
            completed = self._wait_for_stage_completion(agent_id)
            if not completed:
                return False

            # Get output files/paths from agent
            self._update_stage_outputs(stage, agent_id)

            return True

        except Exception as e:
            self.handle_error(e, {"operation": f"stage_processing_{stage}"})
            return False

    def _process_improvement_cycle(self) -> bool:
        """Process an improvement cycle"""
        try:
            if self.improvement_cycles >= self.max_improvement_cycles:
                self.logger.info(
                    f"Maximum improvement cycles reached ({self.improvement_cycles})"
                )
                return True

            self.improvement_cycles += 1
            self.logger.info(f"Starting improvement cycle {self.improvement_cycles}")

            # Get current notes path
            current_notes_path = self.workflow_status["notes"].get("output_path", "")
            if not current_notes_path:
                self.logger.error("No notes path available for improvement")
                return False

            # Get quality feedback
            quality_feedback = self.workflow_status["quality"].get("feedback", {})

            # Send improvement message to update agent
            success = self.send_message(
                recipient="update_agent",
                message_type="improve_notes",
                content={
                    "notes_path": current_notes_path,
                    "feedback": quality_feedback,
                    "cycle": self.improvement_cycles,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            if not success:
                self.logger.error("Failed to initiate improvement cycle")
                return False

            # Wait for improvement completion
            completed = self._wait_for_stage_completion("update_agent")
            if not completed:
                return False

            # Update improved notes path
            self._update_stage_outputs("improvement", "update_agent")

            # Verify improved quality with quality manager
            success = self.send_message(
                recipient="quality_manager",
                message_type="verify_quality",
                content={
                    "content_path": self.workflow_status["improvement"].get(
                        "output_path", ""
                    ),
                    "content_type": "notes",
                },
            )

            if not success:
                self.logger.error("Failed to initiate quality verification")
                return False

            # Wait for quality verification
            completed = self._wait_for_stage_completion("quality_manager")

            return completed

        except Exception as e:
            self.handle_error(e, {"operation": "improvement_cycle"})
            return False

    def _wait_for_stage_completion(self, agent_id: str, timeout: int = 3600) -> bool:
        """Wait for a stage to complete with timeout"""
        start_time = time.time()
        check_interval = 5  # seconds

        while (time.time() - start_time) < timeout:
            # Process any pending messages
            self.comm_manager.process_messages(5)

            # Check agent status
            status = self.monitor_agent_status(agent_id)

            if status["status"] == "completed":
                return True
            elif status["status"] in ["error", "timeout"]:
                self.logger.error(
                    f"Agent {agent_id} failed with status: {status['status']}"
                )
                return False

            # Wait before checking again
            time.sleep(check_interval)

        # Timeout reached
        self.logger.error(f"Timeout waiting for {agent_id} to complete")
        self.handle_timeout(agent_id)
        return False

    def _update_stage_outputs(self, stage: str, agent_id: str) -> None:
        """Update workflow status with stage outputs"""
        # This would normally get output paths from agent messages
        # Simulate for now
        if stage == "ingestion":
            self.workflow_status[stage]["output_files"] = [
                f"processed/markdown/{os.path.basename(f).replace('.pdf', '.md')}"
                for f in self.workflow_status[stage]["files"]
            ]
        elif stage == "outline":
            self.workflow_status[stage]["output_path"] = "processed/outlines/outline.md"
        elif stage == "notes":
            self.workflow_status[stage][
                "output_path"
            ] = "processed/notes/comprehensive_notes.md"
        elif stage == "quality":
            self.workflow_status[stage]["feedback"] = {
                "content_quality": {"score": 0.75}
            }
            self.workflow_status[stage]["quality_score"] = 0.75
        elif stage == "improvement":
            cycle = self.improvement_cycles
            self.workflow_status[stage][
                "output_path"
            ] = f"processed/improved_notes/comprehensive_notes_improved_{cycle}.md"
            self.workflow_status[stage]["cycles"] = cycle

    def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle status update messages from agents"""
        agent_id = message.sender
        status = message.content.get("status")

        if not status:
            return

        # Update agent status
        if agent_id in self.active_processes:
            self.active_processes[agent_id]["status"] = status
            self.active_processes[agent_id]["last_heartbeat"] = time.time()

        # Handle completed status
        if status == "completed":
            stage = agent_id.split("_")[0]
            if stage in self.workflow_status:
                self.workflow_status[stage]["status"] = "completed"

                # Save output paths
                if "output_path" in message.content:
                    self.workflow_status[stage]["output_path"] = message.content[
                        "output_path"
                    ]
                if "output_files" in message.content:
                    self.workflow_status[stage]["output_files"] = message.content[
                        "output_files"
                    ]

    def _handle_quality_report(self, message: AgentMessage) -> None:
        """Handle quality report messages"""
        # Store quality feedback
        self.workflow_status["quality"]["feedback"] = message.content.get(
            "feedback", {}
        )
        self.workflow_status["quality"]["quality_score"] = message.content.get(
            "quality_score", 0.0
        )

        # Add to performance metrics
        quality_score = message.content.get("quality_score", 0.0)
        self.performance_metrics["quality_scores"].append(quality_score)

        # Decide if improvement needed
        if quality_score < self.config.get("quality_threshold", 0.7):
            self.logger.info(
                f"Quality score {quality_score} below threshold, improvement needed"
            )
            # If currently in quality stage, next will be improvement
            if self.current_stage == "quality":
                self.workflow_status["improvement"]["status"] = "pending"
        else:
            self.logger.info(f"Quality score {quality_score} meets threshold")

    def _handle_improvement_complete(self, message: AgentMessage) -> None:
        """Handle improvement completion messages"""
        improvement_metrics = message.content.get("improvement_metrics", {})
        improved_path = message.content.get("improved_path", "")

        # Update workflow status
        self.workflow_status["improvement"]["status"] = "completed"
        self.workflow_status["improvement"]["output_path"] = improved_path
        self.workflow_status["improvement"]["metrics"] = improvement_metrics

        # Update notes path to point to improved version
        if improved_path:
            self.workflow_status["notes"]["output_path"] = improved_path

    def generate_workflow_report(self, workflow_status: Dict) -> None:
        """Generate workflow processing report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "workflow_status": workflow_status,
            "performance_metrics": {
                "average_processing_time": (
                    sum(self.performance_metrics["processing_times"])
                    / len(self.performance_metrics["processing_times"])
                    if self.performance_metrics["processing_times"]
                    else 0
                ),
                "average_memory_usage": (
                    sum(self.performance_metrics["memory_usage"])
                    / len(self.performance_metrics["memory_usage"])
                    if self.performance_metrics["memory_usage"]
                    else 0
                ),
                "average_quality_score": (
                    sum(self.performance_metrics["quality_scores"])
                    / len(self.performance_metrics["quality_scores"])
                    if self.performance_metrics["quality_scores"]
                    else 0
                ),
                "improvement_cycles": self.improvement_cycles,
                "total_runtime": (datetime.now() - self.start_time).total_seconds(),
            },
        }

        # Save report
        report_path = (
            f"logs/workflow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Workflow report generated: {report_path}")

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if not isinstance(input_data, (list, dict)):
            return False
        return True

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if not isinstance(output_data, (list, dict)):
            return False
        return True


def main():
    """Main entry point for the main academic agent"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Main Academic Agent for orchestrating the workflow"
    )

    # Define command line arguments
    parser.add_argument("--input", nargs="+", help="Input PDF files to process")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output-dir", help="Base output directory")

    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("logs/main_agent.log"), logging.StreamHandler()],
    )

    # Create the main agent
    agent = MainAcademicAgent(args.config)

    # Initialize pipeline
    config = {
        "input_dir": "input",
        "output_dir": args.output_dir or "processed",
        "quality_threshold": 0.7,
    }

    if not agent.initialize_pipeline(config):
        logging.error("Failed to initialize pipeline")
        return

    # Start workflow
    if args.input:
        input_files = args.input
    else:
        # Use all PDFs in input directory
        input_files = [
            os.path.join("input", f)
            for f in os.listdir("input")
            if f.lower().endswith(".pdf")
        ]

    if not input_files:
        logging.error("No input files found")
        return

    # Coordinate workflow
    success = agent.coordinate_workflow(input_files)

    if success:
        logging.info("Workflow completed successfully")
    else:
        logging.error("Workflow failed")


if __name__ == "__main__":
    main()
