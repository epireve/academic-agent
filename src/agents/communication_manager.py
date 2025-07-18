#!/usr/bin/env python
"""
Communication Manager - Manages communication between academic agents
Enables feedback loops and information sharing
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from queue import PriorityQueue
from threading import Lock
from .base_agent import AgentMessage, BaseAgent


class CommunicationManager:
    """Manager for inter-agent communication and feedback loops"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize communication manager with config"""
        self.message_queue = PriorityQueue()
        self.agent_registry = {}
        self.message_handlers = {}
        self.feedback_loops = {}
        self.lock = Lock()
        self.message_log = []
        self.max_log_size = 1000

        # Create logs directory
        os.makedirs("logs/messages", exist_ok=True)

        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self._configure_from_dict(config)

    def register_agent(self, agent_id: str, agent_instance: BaseAgent) -> None:
        """Register an agent with the communication manager"""
        with self.lock:
            self.agent_registry[agent_id] = {
                "instance": agent_instance,
                "status": "ready",
                "last_active": datetime.now().isoformat(),
            }

    def register_message_handler(
        self, agent_id: str, message_type: str, handler: Callable
    ) -> None:
        """Register a handler for a specific message type"""
        handler_key = f"{agent_id}:{message_type}"
        with self.lock:
            self.message_handlers[handler_key] = handler

    def send_message(self, message: AgentMessage) -> bool:
        """Send a message to the queue for processing"""
        if not message.validate():
            return False

        with self.lock:
            # Check if recipient exists
            if message.recipient not in self.agent_registry:
                return False

            # Add to queue with priority
            queue_item = (message.priority, time.time(), message)
            self.message_queue.put(queue_item)

            # Update sender status
            if message.sender in self.agent_registry:
                self.agent_registry[message.sender][
                    "last_active"
                ] = datetime.now().isoformat()

            # Log message
            self._log_message(message, "sent")

        return True

    def process_messages(self, max_messages: int = 10) -> int:
        """Process messages in the queue"""
        processed_count = 0

        for _ in range(max_messages):
            if self.message_queue.empty():
                break

            # Get next message
            _, _, message = self.message_queue.get()

            # Process message
            success = self._deliver_message(message)

            # Count as processed even if delivery failed
            processed_count += 1

            # Update recipient status
            with self.lock:
                if message.recipient in self.agent_registry:
                    self.agent_registry[message.recipient][
                        "last_active"
                    ] = datetime.now().isoformat()
                    self.agent_registry[message.recipient]["status"] = "processing"

        return processed_count

    def _deliver_message(self, message: AgentMessage) -> bool:
        """Deliver a message to its recipient"""
        # Get handler if registered
        handler_key = f"{message.recipient}:{message.message_type}"
        handler = self.message_handlers.get(handler_key)

        # Get recipient agent
        recipient = self.agent_registry.get(message.recipient, {}).get("instance")

        if not recipient:
            self._log_message(message, "failed_no_recipient")
            return False

        try:
            # Use specific handler if available
            if handler:
                handler(message)
            else:
                # Use default receive_message method
                recipient.receive_message(message)

            self._log_message(message, "delivered")
            return True

        except Exception as e:
            self._log_message(message, f"failed_error: {str(e)}")
            return False

    def create_feedback_loop(
        self,
        source_agent: str,
        target_agent: str,
        feedback_type: str,
        interval_seconds: int = 60,
    ) -> str:
        """Create a feedback loop between agents"""
        loop_id = f"{source_agent}_{target_agent}_{feedback_type}_{int(time.time())}"

        with self.lock:
            self.feedback_loops[loop_id] = {
                "source": source_agent,
                "target": target_agent,
                "feedback_type": feedback_type,
                "interval": interval_seconds,
                "last_execution": None,
                "active": True,
            }

        return loop_id

    def process_feedback_loops(self) -> int:
        """Process all active feedback loops"""
        now = time.time()
        loops_processed = 0

        with self.lock:
            for loop_id, loop_info in self.feedback_loops.items():
                if not loop_info["active"]:
                    continue

                last_exec = loop_info["last_execution"]
                if last_exec is None or (now - last_exec) >= loop_info["interval"]:
                    self._execute_feedback_loop(loop_id, loop_info)
                    self.feedback_loops[loop_id]["last_execution"] = now
                    loops_processed += 1

        return loops_processed

    def _execute_feedback_loop(self, loop_id: str, loop_info: Dict) -> None:
        """Execute a single feedback loop"""
        source = self.agent_registry.get(loop_info["source"], {}).get("instance")

        if not source:
            return

        # Create feedback message
        feedback_message = AgentMessage(
            sender=loop_info["source"],
            recipient=loop_info["target"],
            message_type=f"feedback_{loop_info['feedback_type']}",
            content={"loop_id": loop_id},
            metadata={"automated": True, "timestamp": datetime.now().isoformat()},
            timestamp=datetime.now(),
            priority=3,
        )

        # Send the message
        self.send_message(feedback_message)

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status information for an agent"""
        with self.lock:
            if agent_id not in self.agent_registry:
                return {"status": "unknown"}

            return self.agent_registry[agent_id].copy()

    def get_all_agent_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all agents"""
        with self.lock:
            return {
                agent_id: info.copy() for agent_id, info in self.agent_registry.items()
            }

    def _log_message(self, message: AgentMessage, status: str) -> None:
        """Log a message and its status"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "sender": message.sender,
            "recipient": message.recipient,
            "type": message.message_type,
            "status": status,
            "priority": message.priority,
        }

        # Add to in-memory log with size limit
        self.message_log.append(log_entry)
        if len(self.message_log) > self.max_log_size:
            self.message_log = self.message_log[-self.max_log_size :]

        # Log to file periodically
        if len(self.message_log) % 100 == 0:
            self._write_logs_to_file()

    def _write_logs_to_file(self) -> None:
        """Write message logs to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"logs/messages/message_log_{timestamp}.json"

        with open(log_path, "w") as f:
            json.dump(self.message_log, f, indent=2)

    def _configure_from_dict(self, config: Dict[str, Any]) -> None:
        """Configure manager from dictionary"""
        if "max_log_size" in config:
            self.max_log_size = config["max_log_size"]

        # Configure feedback loops
        if "feedback_loops" in config:
            for loop_config in config["feedback_loops"]:
                source = loop_config.get("source")
                target = loop_config.get("target")
                feedback_type = loop_config.get("type")
                interval = loop_config.get("interval", 60)

                if source and target and feedback_type:
                    self.create_feedback_loop(source, target, feedback_type, interval)


# Singleton instance
_instance = None


def get_communication_manager(
    config_path: Optional[str] = None,
) -> CommunicationManager:
    """Get singleton instance of communication manager"""
    global _instance
    if _instance is None:
        _instance = CommunicationManager(config_path)
    return _instance