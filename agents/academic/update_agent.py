#!/usr/bin/env python
"""
Update Agent - Specialized agent for enriching and improving academic notes
while preserving original meaning
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
from groq import Groq
# Use unified BaseAgent for standardized interface
from ...src.agents.base_agent import BaseAgent, AgentMessage

# Import from unified architecture
from ...src.core.output_manager import get_output_manager, OutputCategory, ContentType


class UpdateAgent(BaseAgent):
    """Agent responsible for enriching and improving academic notes"""

    def __init__(self, groq_api_key: str):
        super().__init__("update_agent")
        self.output_manager = None
        self.groq = Groq(api_key=groq_api_key)
        self.improvement_threshold = 0.3  # Minimum improvement required

        # Specialized prompts for different improvement types
        self.prompts = {
            "enrich": """
            Enrich these academic notes with additional context, examples, and explanations.
            Focus on making the content more comprehensive while preserving the original meaning.
            
            Original notes:
            {content}
            
            Guidelines:
            1. Add relevant examples that illustrate key concepts
            2. Expand explanations of complex ideas
            3. Include additional contextual information
            4. Add relevant connections to related fields or concepts
            5. DO NOT contradict or alter the original meaning
            6. DO NOT remove any original content
            7. Format with proper markdown
            
            Return the enriched notes in markdown format.
            """,
            "clarify": """
            Improve the clarity and readability of these academic notes.
            Focus on making the content more understandable while preserving all original meaning.
            
            Original notes:
            {content}
            
            Guidelines:
            1. Simplify complex language without losing technical accuracy
            2. Improve the logical flow and structure
            3. Add clarifying statements for difficult concepts
            4. Ensure consistent terminology throughout
            5. DO NOT contradict or alter the original meaning
            6. DO NOT remove any original content
            7. Format with proper markdown
            
            Return the clarified notes in markdown format.
            """,
            "structure": """
            Improve the structure and organization of these academic notes.
            Focus on making the content more navigable while preserving all original meaning.
            
            Original notes:
            {content}
            
            Guidelines:
            1. Reorganize content into logical sections and subsections
            2. Add appropriate headings and subheadings
            3. Create consistent formatting for similar elements
            4. Improve the flow between ideas and sections
            5. DO NOT contradict or alter the original meaning
            6. DO NOT remove any original content
            7. Format with proper markdown

            Return the restructured notes in markdown format.
            """,
        }

    async def initialize(self):
        """Initialize agent-specific resources."""
        try:
            # Initialize output manager
            self.output_manager = get_output_manager()
            
            # Setup output directories for improved notes
            improved_dir = self.output_manager.get_output_path(
                OutputCategory.PROCESSED, 
                ContentType.MARKDOWN, 
                subdirectory="improved_notes"
            )
            improved_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup improvement reports directory
            reports_dir = self.output_manager.get_output_path(
                OutputCategory.REPORTS,
                ContentType.JSON,
                subdirectory="improvement_reports"
            )
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"{self.agent_id} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.agent_id}: {e}")
            raise

    async def cleanup(self):
        """Cleanup agent resources."""
        try:
            # Cleanup Groq client (no explicit cleanup needed)
            
            self.logger.info(f"{self.agent_id} cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during {self.agent_id} cleanup: {e}")

    def check_quality(self, content: Any) -> float:
        """Check quality of improved notes."""
        if isinstance(content, dict):
            # Check improvement results
            improvement_data = content
            quality_score = 1.0
            
            # Check improvement metrics
            improvement_metrics = improvement_data.get("improvement_metrics", {})
            if improvement_metrics:
                quality_improvement = improvement_metrics.get("quality_improvement", 0)
                if quality_improvement >= self.improvement_threshold:
                    quality_score = quality_improvement
                else:
                    quality_score -= 0.3  # Penalty for insufficient improvement
            
            # Check verification score
            verification_score = improvement_data.get("verification_score", 0.8)
            quality_score = (quality_score + verification_score) / 2
            
            # Check for successful improvements
            improvements_applied = improvement_data.get("improvements_applied", [])
            if not improvements_applied:
                quality_score -= 0.4
            
            return max(0.0, min(1.0, quality_score))
            
        elif isinstance(content, str):
            # Check content string quality
            quality_score = 1.0
            
            # Basic content checks
            if len(content) < 100:
                quality_score -= 0.3
            if not content.strip():
                quality_score = 0.0
            if not any(line.startswith("#") for line in content.split("\n")):
                quality_score -= 0.2
                
            return max(0.0, min(1.0, quality_score))
        
        return 0.0

    def improve_notes(
        self,
        notes_path: str,
        feedback: Dict[str, Any],
        improvement_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Improve notes based on feedback"""
        try:
            # Load notes
            with open(notes_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Determine improvement types if not specified
            if not improvement_types:
                improvement_types = self._determine_improvement_types(feedback)

            # Apply improvements sequentially
            improved_content = original_content
            improvements_applied = []

            for imp_type in improvement_types:
                if imp_type in self.prompts:
                    result = self._apply_improvement(
                        improved_content, imp_type, feedback
                    )
                    improved_content = result["content"]
                    improvements_applied.append(
                        {"type": imp_type, "changes": result["changes"]}
                    )

            # Save improved notes
            output = self._save_improved_notes(
                notes_path, improved_content, improvements_applied, feedback
            )

            # Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                original_content, improved_content, feedback, improvements_applied
            )

            # Check if improvement meets threshold
            improved_quality = (
                improvement_metrics["quality_improvement"] >= self.improvement_threshold
            )

            # Send feedback to quality manager for verification
            verification_result = self._request_quality_verification(
                output["improved_path"]
            )

            return {
                "success": True,
                "improved_path": output["improved_path"],
                "improvement_metrics": improvement_metrics,
                "improvements_applied": improvements_applied,
                "quality_verified": verification_result["verified"],
                "verification_score": verification_result.get("score", 0.0),
            }

        except Exception as e:
            self.handle_error(e, {"operation": "notes_improvement"})
            return {"success": False, "error": str(e)}

    def _determine_improvement_types(self, feedback: Dict[str, Any]) -> List[str]:
        """Determine appropriate improvement types based on feedback"""
        improvement_types = []

        if feedback.get("content_quality", {}).get("score", 1.0) < 0.8:
            improvement_types.append("enrich")

        if feedback.get("clarity", {}).get("score", 1.0) < 0.8:
            improvement_types.append("clarify")

        if feedback.get("structure", {}).get("score", 1.0) < 0.8:
            improvement_types.append("structure")

        # Default to enrichment if no specific issues identified
        if not improvement_types:
            improvement_types = ["enrich"]

        return improvement_types

    def _apply_improvement(
        self, content: str, improvement_type: str, feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply specific improvement to content"""
        # Get the appropriate prompt
        prompt = self.prompts.get(improvement_type, self.prompts["enrich"])

        # Add specific feedback to prompt if available
        if improvement_type in feedback:
            specific_feedback = f"\nAddress these specific issues:\n"
            for issue, details in feedback[improvement_type].items():
                specific_feedback += f"- {issue}: {details.get('feedback', '')}\n"
            prompt += specific_feedback

        # Call LLM for improvement
        response = self.groq.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": "You are an academic editor specializing in improving educational content while preserving original meaning.",
                },
                {
                    "role": "user",
                    "content": prompt.format(content=content[:12000]),
                },  # Limit content length
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        improved_content = response.choices[0].message.content

        # Identify what changed (simplified approach)
        changes = self._identify_changes(content, improved_content)

        return {"content": improved_content, "changes": changes}

    def _identify_changes(self, original: str, improved: str) -> Dict[str, Any]:
        """Identify major changes between original and improved content"""
        # Calculate basic metrics
        original_words = len(original.split())
        improved_words = len(improved.split())
        word_diff = improved_words - original_words

        # Count paragraphs, headings, and lists
        orig_paragraphs = len([p for p in original.split("\n\n") if p.strip()])
        impr_paragraphs = len([p for p in improved.split("\n\n") if p.strip()])

        orig_headings = len(
            [l for l in original.split("\n") if l.strip().startswith("#")]
        )
        impr_headings = len(
            [l for l in improved.split("\n") if l.strip().startswith("#")]
        )

        orig_lists = len(
            [l for l in original.split("\n") if l.strip().startswith(("-", "*", "1."))]
        )
        impr_lists = len(
            [l for l in improved.split("\n") if l.strip().startswith(("-", "*", "1."))]
        )

        return {
            "word_count_change": word_diff,
            "word_count_percent": (
                (word_diff / original_words) * 100 if original_words > 0 else 0
            ),
            "paragraph_count_change": impr_paragraphs - orig_paragraphs,
            "heading_count_change": impr_headings - orig_headings,
            "list_count_change": impr_lists - orig_lists,
        }

    def _save_improved_notes(
        self,
        original_path: str,
        improved_content: str,
        improvements_applied: List[Dict],
        feedback: Dict,
    ) -> Dict[str, str]:
        """Save improved notes with changelog"""
        # Create paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        improved_dir = os.path.join("processed", "improved_notes")
        os.makedirs(improved_dir, exist_ok=True)

        # Save improved content
        improved_path = os.path.join(
            improved_dir, f"{base_name}_improved_{timestamp}.md"
        )
        with open(improved_path, "w", encoding="utf-8") as f:
            f.write(improved_content)

        # Create and save changelog
        changelog = {
            "original_path": original_path,
            "improved_path": improved_path,
            "timestamp": datetime.now().isoformat(),
            "improvements_applied": improvements_applied,
            "feedback_addressed": feedback,
        }

        changelog_path = os.path.join(
            improved_dir, f"{base_name}_changelog_{timestamp}.json"
        )
        with open(changelog_path, "w", encoding="utf-8") as f:
            json.dump(changelog, f, indent=2)

        return {"improved_path": improved_path, "changelog_path": changelog_path}

    def _calculate_improvement_metrics(
        self, original: str, improved: str, feedback: Dict, improvements: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate metrics for the improvement"""
        # Basic metrics
        changes = self._identify_changes(original, improved)

        # Estimate quality improvement (simplified approach)
        # In a real system, would call quality manager for evaluation
        quality_improvement = 0.0

        for improvement in improvements:
            if improvement["type"] == "enrich" and changes["word_count_change"] > 0:
                quality_improvement += 0.15
            if improvement["type"] == "clarify":
                quality_improvement += 0.1
            if (
                improvement["type"] == "structure"
                and changes["heading_count_change"] > 0
            ):
                quality_improvement += 0.1

        # Cap improvement score
        quality_improvement = min(1.0, quality_improvement)

        return {
            "changes": changes,
            "quality_improvement": quality_improvement,
            "improvements_count": len(improvements),
        }

    def _request_quality_verification(self, improved_path: str) -> Dict[str, Any]:
        """Request quality verification from QualityManagerAgent"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient="quality_manager",
            message_type="verify_improvement",
            content={"path": improved_path},
            metadata={},
            timestamp=datetime.now(),
            priority=1,
        )

        # In a real system, this would wait for a response
        # For now, simulate a response
        self.logger.info(f"Requested quality verification for {improved_path}")

        # Simulated response
        return {"verified": True, "score": 0.85}

    def receive_feedback(self, feedback_message: AgentMessage) -> Dict[str, Any]:
        """Process feedback from other agents"""
        if not feedback_message.validate():
            self.logger.error(f"Received invalid feedback message: {feedback_message}")
            return {"success": False, "error": "Invalid feedback message"}

        self.logger.info(f"Received feedback from {feedback_message.sender}")

        # Process feedback based on type
        if feedback_message.message_type == "quality_feedback":
            # Used for future improvements
            return {
                "success": True,
                "feedback_processed": True,
                "feedback_id": id(feedback_message),
            }

        return {"success": False, "error": "Unknown feedback type"}

    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if not isinstance(input_data, dict):
            return False

        required_fields = ["notes_path", "feedback"]
        return all(field in input_data for field in required_fields)

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        if not isinstance(output_data, dict):
            return False

        required_fields = ["success", "improved_path"]
        return all(field in output_data for field in required_fields)


def main():
    """Main entry point for the update agent"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update Agent for improving academic notes"
    )

    # Define command line arguments
    parser.add_argument("--notes", required=True, help="Path to notes file to improve")
    parser.add_argument("--feedback", help="Path to feedback JSON file")
    parser.add_argument(
        "--improvements",
        nargs="+",
        choices=["enrich", "clarify", "structure"],
        help="Specific improvements to apply",
    )
    parser.add_argument("--api-key", help="Groq API key")

    args = parser.parse_args()

    # Get API key from environment or command line
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print(
            "Error: Groq API key is required. Set GROQ_API_KEY environment variable or use --api-key"
        )
        sys.exit(1)

    # Create the update agent
    agent = UpdateAgent(api_key)

    # Load feedback if provided
    feedback = {}
    if args.feedback:
        try:
            with open(args.feedback, "r") as f:
                feedback = json.load(f)
        except Exception as e:
            print(f"Error loading feedback file: {str(e)}")
            sys.exit(1)

    # Improve the notes
    result = agent.improve_notes(args.notes, feedback, args.improvements)

    if result["success"]:
        print(f"Notes improved and saved to: {result['improved_path']}")
        print(
            f"Quality improvement: {result['improvement_metrics']['quality_improvement']:.2f}"
        )
    else:
        print(f"Error improving notes: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
