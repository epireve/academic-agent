"""
Quality Manager for Academic Agent System.

This module provides quality assessment and control functionality for the academic agent system,
including content evaluation, quality metrics tracking, and feedback generation.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import re
import statistics
from pathlib import Path


@dataclass
class QualityMetrics:
    """Data class for quality metrics."""
    quality_score: float
    completeness: float
    accuracy: float
    clarity: float
    structure: float
    consistency: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityEvaluation:
    """Data class for quality evaluation results."""
    content_type: str
    quality_score: float
    feedback: List[str]
    areas_for_improvement: List[str]
    strengths: List[str]
    metrics: QualityMetrics
    assessment: str
    approved: bool
    timestamp: datetime = field(default_factory=datetime.now)


class QualityManager:
    """
    Manages quality assessment and control for academic content.
    
    This class provides comprehensive quality evaluation capabilities including
    content assessment, metrics tracking, and feedback generation.
    """
    
    def __init__(self, quality_threshold: float = 0.7):
        """
        Initialize the quality manager.
        
        Args:
            quality_threshold: Minimum quality score required (0.0 to 1.0)
        """
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        
        self.quality_threshold = quality_threshold
        self.evaluation_history: List[QualityEvaluation] = []
        self.quality_metrics: Dict[str, List[float]] = {}
        self.logger = self._setup_logger()
        
        # Quality criteria weights
        self.criteria_weights = {
            "completeness": 0.2,
            "accuracy": 0.25,
            "clarity": 0.2,
            "structure": 0.2,
            "consistency": 0.15
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the quality manager."""
        logger = logging.getLogger("QualityManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_content(self, content: Any, content_type: str) -> QualityEvaluation:
        """
        Evaluate the quality of content.
        
        Args:
            content: The content to evaluate (string or dict)
            content_type: Type of content (markdown, analysis, outline, notes)
            
        Returns:
            QualityEvaluation object with detailed assessment
        """
        try:
            self.logger.info(f"Evaluating {content_type} content")
            
            # Convert content to string if it's a dict
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content)
            
            # Basic quality assessment
            quality_score = self._calculate_basic_quality(content_str)
            
            metrics = QualityMetrics(
                quality_score=quality_score,
                completeness=quality_score,
                accuracy=0.8,  # Base accuracy
                clarity=quality_score,
                structure=quality_score,
                consistency=0.8  # Base consistency
            )
            
            # Generate feedback
            feedback = self._generate_feedback(content_type, quality_score)
            areas_for_improvement = self._identify_improvements(content_type, quality_score)
            strengths = self._identify_strengths(content_type, quality_score)
            
            evaluation = QualityEvaluation(
                content_type=content_type,
                quality_score=quality_score,
                feedback=feedback,
                areas_for_improvement=areas_for_improvement,
                strengths=strengths,
                metrics=metrics,
                assessment=self._generate_assessment(quality_score),
                approved=quality_score >= self.quality_threshold
            )
            
            # Store evaluation in history
            self.evaluation_history.append(evaluation)
            
            # Update metrics tracking
            self._update_metrics_tracking(content_type, quality_score)
            
            self.logger.info(f"Quality evaluation completed: {quality_score:.2f}")
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating content: {str(e)}")
            return QualityEvaluation(
                content_type=content_type,
                quality_score=0.0,
                feedback=[f"Evaluation error: {str(e)}"],
                areas_for_improvement=["Fix evaluation error"],
                strengths=[],
                metrics=QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                assessment="Error during evaluation",
                approved=False
            )
    
    def _calculate_basic_quality(self, content: str) -> float:
        """Calculate basic quality score based on content characteristics."""
        if not content or len(content.strip()) == 0:
            return 0.0
        
        score = 0.0
        
        # Length factor
        length = len(content.strip())
        if length < 50:
            score += 0.2
        elif length < 200:
            score += 0.4
        elif length < 1000:
            score += 0.6
        else:
            score += 0.8
        
        # Structure factor - check for headers, lists, etc.
        if re.search(r'^#+\s', content, re.MULTILINE):  # Headers
            score += 0.1
        
        if re.search(r'^[-*+]\s', content, re.MULTILINE):  # Lists
            score += 0.05
        
        if re.search(r'^\d+\.\s', content, re.MULTILINE):  # Numbered lists
            score += 0.05
        
        return min(score, 1.0)
    
    def _generate_feedback(self, content_type: str, quality_score: float) -> List[str]:
        """Generate feedback based on content type and quality score."""
        feedback = []
        
        if quality_score < 0.5:
            feedback.append("Content quality is significantly below expectations")
        elif quality_score < 0.7:
            feedback.append("Content quality needs improvement")
        else:
            feedback.append("Content quality is acceptable")
        
        if content_type == "markdown":
            if quality_score < 0.7:
                feedback.append("Consider improving document structure and formatting")
        elif content_type == "analysis":
            if quality_score < 0.7:
                feedback.append("Analysis could be more comprehensive")
        elif content_type == "outline":
            if quality_score < 0.7:
                feedback.append("Outline structure could be more detailed")
        elif content_type == "notes":
            if quality_score < 0.7:
                feedback.append("Notes could be more comprehensive")
        
        return feedback
    
    def _identify_improvements(self, content_type: str, quality_score: float) -> List[str]:
        """Identify areas for improvement based on content type and quality score."""
        improvements = []
        
        if quality_score < 0.5:
            improvements.append("Significant content revision needed")
        elif quality_score < 0.7:
            improvements.append("Minor content improvements needed")
        
        if content_type == "markdown":
            if quality_score < 0.7:
                improvements.append("Improve document structure")
                improvements.append("Add more detailed content")
        elif content_type == "analysis":
            if quality_score < 0.7:
                improvements.append("Provide more comprehensive analysis")
                improvements.append("Include more key concepts")
        
        return improvements
    
    def _identify_strengths(self, content_type: str, quality_score: float) -> List[str]:
        """Identify strengths based on content type and quality score."""
        strengths = []
        
        if quality_score >= 0.8:
            strengths.append("High quality content")
        elif quality_score >= 0.7:
            strengths.append("Good quality content")
        
        if content_type == "markdown":
            if quality_score >= 0.7:
                strengths.append("Well-structured document")
        elif content_type == "analysis":
            if quality_score >= 0.7:
                strengths.append("Comprehensive analysis")
        
        return strengths
    
    def _generate_assessment(self, quality_score: float) -> str:
        """Generate a text assessment based on quality score."""
        if quality_score >= 0.9:
            return "Excellent quality content with outstanding characteristics"
        elif quality_score >= 0.8:
            return "Good quality content with strong performance"
        elif quality_score >= 0.7:
            return "Adequate quality content meeting basic requirements"
        elif quality_score >= 0.6:
            return "Below average quality content with room for improvement"
        else:
            return "Poor quality content requiring significant improvement"
    
    def _update_metrics_tracking(self, content_type: str, quality_score: float) -> None:
        """Update metrics tracking for content type."""
        if content_type not in self.quality_metrics:
            self.quality_metrics[content_type] = []
        
        self.quality_metrics[content_type].append(quality_score)
        
        # Keep only last 100 scores per content type
        if len(self.quality_metrics[content_type]) > 100:
            self.quality_metrics[content_type] = self.quality_metrics[content_type][-100:]
    
    def get_quality_threshold(self) -> float:
        """Get the current quality threshold."""
        return self.quality_threshold
    
    def set_quality_threshold(self, threshold: float) -> None:
        """
        Set the quality threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
            
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        
        self.quality_threshold = threshold
        self.logger.info(f"Quality threshold updated to {threshold}")
    
    def get_evaluation_history(self) -> List[QualityEvaluation]:
        """Get the history of quality evaluations."""
        return self.evaluation_history.copy()
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get aggregated quality metrics."""
        metrics = {}
        
        for content_type, scores in self.quality_metrics.items():
            if scores:
                metrics[content_type] = {
                    "average_quality_score": statistics.mean(scores),
                    "total_evaluations": len(scores),
                    "pass_rate": sum(1 for score in scores if score >= self.quality_threshold) / len(scores)
                }
        
        # Overall metrics
        all_scores = [score for scores in self.quality_metrics.values() for score in scores]
        if all_scores:
            metrics["overall"] = {
                "average_quality_score": statistics.mean(all_scores),
                "total_evaluations": len(all_scores),
                "pass_rate": sum(1 for score in all_scores if score >= self.quality_threshold) / len(all_scores)
            }
        
        return metrics
    
    def clear_history(self) -> None:
        """Clear evaluation history and metrics."""
        self.evaluation_history.clear()
        self.quality_metrics.clear()
        self.logger.info("Quality evaluation history cleared")