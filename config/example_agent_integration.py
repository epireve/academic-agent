#!/usr/bin/env python3
"""
Example showing how to integrate existing agents with the new configuration system.

This example demonstrates updating the existing academic agents to use the
YAML-based configuration system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add the project paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config_manager import ConfigurationManager
    from integration import ConfiguredBaseAgent
    
    class ConfiguredIngestionAgent(ConfiguredBaseAgent):
        """
        Example of updating the ingestion agent to use the new configuration system.
        """
        
        def __init__(self, config_manager: Optional[ConfigurationManager] = None):
            super().__init__("ingestion_agent", config_manager)
            
            # Initialize ingestion-specific settings
            self.supported_formats = ["pdf", "docx", "txt", "md"]
            self.max_file_size = 100 * 1024 * 1024  # 100MB
            
            # Get processing device from environment settings
            if self.env_settings and hasattr(self.env_settings, 'pdf_processing_device'):
                self.processing_device = self.env_settings.pdf_processing_device
            else:
                self.processing_device = "cpu"
            
            self.logger.info(f"Ingestion agent initialized with device: {self.processing_device}")
        
        def validate_input(self, input_data: Any) -> bool:
            """Validate input file for ingestion."""
            if not isinstance(input_data, dict):
                return False
            
            file_path = input_data.get("file_path")
            if not file_path or not Path(file_path).exists():
                self.logger.error(f"File not found: {file_path}")
                return False
            
            # Check file size
            file_size = Path(file_path).stat().st_size
            if file_size > self.max_file_size:
                self.logger.error(f"File too large: {file_size} bytes")
                return False
            
            # Check file format
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            if file_ext not in self.supported_formats:
                self.logger.error(f"Unsupported format: {file_ext}")
                return False
            
            return True
        
        def process_document(self, file_path: str) -> Dict[str, Any]:
            """Process a document using the configured settings."""
            try:
                # Validate input
                if not self.validate_input({"file_path": file_path}):
                    return {"success": False, "error": "Input validation failed"}
                
                self.logger.info(f"Processing document: {file_path}")
                
                # Use specialized prompt if available
                if self.specialized_prompt:
                    self.logger.debug(f"Using specialized prompt: {self.specialized_prompt[:100]}...")
                
                # Simulate processing with timeout
                import time
                processing_time = min(5, self.processing_timeout)  # Simulate processing
                time.sleep(0.1)  # Brief pause for demo
                
                # Check quality threshold
                quality_score = 0.85  # Simulated quality score
                if quality_score < self.quality_threshold:
                    self.logger.warning(f"Quality score {quality_score} below threshold {self.quality_threshold}")
                    return {"success": False, "error": "Quality threshold not met"}
                
                result = {
                    "success": True,
                    "file_path": file_path,
                    "processing_time": processing_time,
                    "quality_score": quality_score,
                    "device_used": self.processing_device,
                    "output_format": "markdown"
                }
                
                # Send message to outline agent if communication is enabled
                if self.should_communicate_with("outline_agent"):
                    self.send_message(
                        "outline_agent",
                        "document_processed",
                        result
                    )
                
                return result
                
            except Exception as e:
                error_context = {"operation": "process_document", "file_path": file_path}
                if self.handle_error(e, error_context):
                    # Retry logic would go here
                    pass
                
                return {"success": False, "error": str(e)}
        
        def check_quality(self, content: Any) -> float:
            """Check quality of processed content."""
            if not isinstance(content, dict):
                return 0.0
            
            # Simple quality metrics
            quality_score = 0.0
            
            # Check if processing was successful
            if content.get("success"):
                quality_score += 0.5
            
            # Check quality score from processing
            if "quality_score" in content:
                quality_score += content["quality_score"] * 0.5
            
            return min(quality_score, 1.0)
    
    
    class ConfiguredNotesAgent(ConfiguredBaseAgent):
        """
        Example of updating the notes agent to use the new configuration system.
        """
        
        def __init__(self, config_manager: Optional[ConfigurationManager] = None):
            super().__init__("notes_agent", config_manager)
            
            # Initialize notes-specific settings
            self.min_note_length = 100
            self.max_note_length = 50000
            self.citation_required = True
            
            self.logger.info("Notes agent initialized")
        
        def validate_input(self, input_data: Any) -> bool:
            """Validate input for notes generation."""
            if not isinstance(input_data, dict):
                return False
            
            outline = input_data.get("outline")
            if not outline:
                self.logger.error("No outline provided")
                return False
            
            return True
        
        def generate_notes(self, outline: Dict[str, Any]) -> Dict[str, Any]:
            """Generate notes from outline using configured settings."""
            try:
                # Validate input
                if not self.validate_input({"outline": outline}):
                    return {"success": False, "error": "Input validation failed"}
                
                self.logger.info("Generating notes from outline")
                
                # Use specialized prompt if available
                if self.specialized_prompt:
                    self.logger.debug(f"Using specialized prompt: {self.specialized_prompt[:100]}...")
                
                # Simulate note generation
                import time
                time.sleep(0.1)  # Brief pause for demo
                
                # Generate notes (simplified simulation)
                notes = {
                    "title": outline.get("title", "Academic Notes"),
                    "content": f"Generated notes based on outline: {outline.get('structure', 'No structure')}",
                    "word_count": 1500,  # Simulated
                    "citations": ["Reference 1", "Reference 2"] if self.citation_required else [],
                    "quality_indicators": {
                        "clarity": 0.8,
                        "completeness": 0.9,
                        "structure": 0.85
                    }
                }
                
                # Check quality
                quality_score = self.check_quality(notes)
                if quality_score < self.quality_threshold:
                    self.logger.warning(f"Quality score {quality_score} below threshold {self.quality_threshold}")
                    
                    # Send message to quality manager for review
                    if self.should_communicate_with("quality_manager"):
                        self.send_message(
                            "quality_manager",
                            "quality_review_needed",
                            {"notes": notes, "quality_score": quality_score}
                        )
                
                result = {
                    "success": True,
                    "notes": notes,
                    "quality_score": quality_score,
                    "processing_time": 0.1
                }
                
                return result
                
            except Exception as e:
                error_context = {"operation": "generate_notes", "outline": outline}
                if self.handle_error(e, error_context):
                    # Retry logic would go here
                    pass
                
                return {"success": False, "error": str(e)}
        
        def check_quality(self, content: Any) -> float:
            """Check quality of generated notes."""
            if not isinstance(content, dict):
                return 0.0
            
            quality_score = 0.0
            
            # Check word count
            word_count = content.get("word_count", 0)
            if self.min_note_length <= word_count <= self.max_note_length:
                quality_score += 0.3
            
            # Check citations
            citations = content.get("citations", [])
            if self.citation_required and citations:
                quality_score += 0.2
            elif not self.citation_required:
                quality_score += 0.2
            
            # Check quality indicators
            quality_indicators = content.get("quality_indicators", {})
            if quality_indicators:
                avg_quality = sum(quality_indicators.values()) / len(quality_indicators)
                quality_score += avg_quality * 0.5
            
            return min(quality_score, 1.0)
        
        def _process_message(self, message: Dict[str, Any]) -> bool:
            """Process received messages."""
            message_type = message.get("message_type")
            content = message.get("content", {})
            
            if message_type == "quality_feedback":
                # Process quality feedback from quality manager
                feedback = content.get("feedback", {})
                suggestions = feedback.get("suggestions", [])
                
                self.logger.info(f"Received quality feedback with {len(suggestions)} suggestions")
                
                # Apply suggestions (simplified)
                for suggestion in suggestions:
                    self.logger.info(f"Processing suggestion: {suggestion}")
                
                return True
            
            elif message_type == "improvement_request":
                # Process improvement request from update agent
                improvement_areas = content.get("improvement_areas", [])
                
                self.logger.info(f"Received improvement request for {len(improvement_areas)} areas")
                
                # Apply improvements (simplified)
                for area in improvement_areas:
                    self.logger.info(f"Improving area: {area}")
                
                return True
            
            else:
                return super()._process_message(message)
    
    
    def demonstration():
        """Demonstrate the configuration system integration with agents."""
        print("=== Configuration System Integration Demonstration ===\n")
        
        try:
            # Initialize configuration manager
            config_manager = ConfigurationManager("config")
            config = config_manager.load_config("development")
            
            print(f"1. Loaded configuration for environment: {config.environment}")
            print(f"   Debug mode: {config.debug}")
            print(f"   Global quality threshold: {config.quality_threshold}")
            print()
            
            # Create configured agents
            print("2. Creating configured agents...")
            
            ingestion_agent = ConfiguredIngestionAgent(config_manager)
            notes_agent = ConfiguredNotesAgent(config_manager)
            
            print(f"   Ingestion Agent - Timeout: {ingestion_agent.get_processing_timeout()}s")
            print(f"   Ingestion Agent - Quality threshold: {ingestion_agent.get_quality_threshold()}")
            print(f"   Notes Agent - Timeout: {notes_agent.get_processing_timeout()}s")
            print(f"   Notes Agent - Quality threshold: {notes_agent.get_quality_threshold()}")
            print()
            
            # Demonstrate document processing
            print("3. Processing document...")
            
            # Create a mock file path for demonstration
            mock_file_path = "test_document.pdf"
            result = ingestion_agent.process_document(mock_file_path)
            
            print(f"   Processing result: {result.get('success', False)}")
            if result.get('success'):
                print(f"   Quality score: {result.get('quality_score', 0)}")
                print(f"   Device used: {result.get('device_used', 'unknown')}")
            print()
            
            # Demonstrate notes generation
            print("4. Generating notes...")
            
            mock_outline = {
                "title": "Academic Topic",
                "structure": {
                    "introduction": "Overview of topic",
                    "main_content": "Detailed analysis",
                    "conclusion": "Summary and implications"
                }
            }
            
            notes_result = notes_agent.generate_notes(mock_outline)
            
            print(f"   Notes generation result: {notes_result.get('success', False)}")
            if notes_result.get('success'):
                notes = notes_result.get('notes', {})
                print(f"   Notes title: {notes.get('title', 'Unknown')}")
                print(f"   Word count: {notes.get('word_count', 0)}")
                print(f"   Citations: {len(notes.get('citations', []))}")
                print(f"   Quality score: {notes_result.get('quality_score', 0)}")
            print()
            
            # Demonstrate inter-agent communication
            print("5. Testing inter-agent communication...")
            
            # Check if agents can communicate
            can_communicate = ingestion_agent.should_communicate_with("outline_agent")
            print(f"   Ingestion -> Outline communication: {can_communicate}")
            
            can_communicate = notes_agent.should_communicate_with("quality_manager")
            print(f"   Notes -> Quality Manager communication: {can_communicate}")
            
            # Get communication intervals
            interval = notes_agent.get_communication_interval("quality_manager")
            print(f"   Communication interval: {interval}s")
            print()
            
            # Display agent metrics
            print("6. Agent metrics:")
            ingestion_agent.log_metrics()
            notes_agent.log_metrics()
            print()
            
            # Display configuration summaries
            print("7. Configuration summaries:")
            print(f"   Ingestion Agent: {ingestion_agent.get_configuration_summary()}")
            print(f"   Notes Agent: {notes_agent.get_configuration_summary()}")
            print()
            
            print("=== Demonstration completed successfully ===")
            
        except Exception as e:
            print(f"Demonstration failed: {e}")
            import traceback
            traceback.print_exc()
    
    
    if __name__ == "__main__":
        demonstration()

except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure the configuration system is properly set up.")
    sys.exit(1)