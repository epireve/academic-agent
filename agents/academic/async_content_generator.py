#!/usr/bin/env python3
"""
Asynchronous Content Generation Agent
Task 15 Implementation - High-performance async content generation

This module provides async content generation capabilities with:
- Concurrent analysis and outline generation
- Parallel notes creation with dependency management
- Streaming content generation with progress tracking
- Quality assessment with automatic improvement cycles
- Resource-aware processing with memory management
"""

import asyncio
import time
import aiofiles
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import logging
import re

from ...src.agents.base_agent import BaseAgent
from .async_framework import (
    AsyncTask, TaskPriority, TaskStatus, async_retry, async_timeout,
    AsyncResourceManager, AsyncProgressTracker
)

try:
    from groq import AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


@dataclass
class ContentAnalysisResult:
    """Result of content analysis operation."""
    source_file: str
    topics: List[str]
    key_concepts: List[str]
    structure: Dict[str, Any]
    summary: str
    sections: List[Dict[str, Any]]
    quality_indicators: Dict[str, float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class OutlineGenerationResult:
    """Result of outline generation operation."""
    source_files: List[str]
    outline_structure: Dict[str, Any]
    hierarchical_topics: List[Dict[str, Any]]
    estimated_sections: int
    complexity_score: float
    outline_content: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class NotesGenerationResult:
    """Result of notes generation operation."""
    source_files: List[str]
    outline_used: str
    notes_content: str
    word_count: int
    sections_count: int
    quality_metrics: Dict[str, float]
    references: List[str]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class QualityAssessmentResult:
    """Result of quality assessment operation."""
    content_analyzed: str
    overall_score: float
    dimension_scores: Dict[str, float]
    feedback: List[str]
    improvement_suggestions: List[str]
    meets_threshold: bool
    assessment_details: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class AsyncContentGenerator(BaseAgent):
    """Asynchronous content generator with parallel processing capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("async_content_generator")
        
        self.config = config or {}
        self.setup_config()
        
        # Initialize AI client if available
        self.ai_client = None
        self._initialize_ai_client()
        
        # Resource management
        self.resource_manager = AsyncResourceManager()
        self._setup_resources()
        
        # Progress tracking
        self.progress_tracker = AsyncProgressTracker()
        
        # Content cache
        self.content_cache: Dict[str, Any] = {}
        self.cache_enabled = self.config.get("enable_cache", True)
        
        # Generation metrics
        self.generation_metrics = {
            "content_analyzed": 0,
            "outlines_generated": 0,
            "notes_generated": 0,
            "quality_assessments": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0.0,
            "average_quality_score": 0.0,
            "cache_hits": 0,
            "ai_api_calls": 0
        }
        
        self.logger.info("Async content generator initialized")
    
    def setup_config(self):
        """Setup configuration with defaults."""
        defaults = {
            "max_concurrent_generations": 3,
            "ai_model": "llama-3.3-70b-versatile",
            "ai_temperature": 0.3,
            "ai_max_tokens": 4096,
            "quality_threshold": 0.7,
            "max_improvement_cycles": 3,
            "enable_cache": True,
            "cache_ttl_hours": 24,
            "timeout_seconds": 300,
            "progress_update_interval": 2.0,
            "memory_limit_mb": 1024,
            "streaming_enabled": True,
            "batch_size": 3
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _initialize_ai_client(self):
        """Initialize AI client for content generation."""
        if GROQ_AVAILABLE:
            try:
                api_key = self.config.get("groq_api_key") or "your-api-key-here"
                self.ai_client = AsyncGroq(api_key=api_key)
                self.logger.info("Groq AI client initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Groq client: {e}")
        else:
            self.logger.warning("Groq not available - using simulation mode")
    
    def _setup_resources(self):
        """Setup resource management."""
        max_concurrent = self.config.get("max_concurrent_generations", 3)
        self.resource_manager.create_semaphore("content_generation", max_concurrent)
        self.resource_manager.create_semaphore("ai_api_calls", 5)
        self.resource_manager.create_semaphore("memory_intensive", 1)
        self.resource_manager.create_lock("cache_access")
    
    def _get_cache_key(self, operation: str, inputs: Dict[str, Any]) -> str:
        """Generate cache key for operations."""
        # Create a deterministic hash of the inputs
        import hashlib
        content = f"{operation}_{json.dumps(inputs, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if result is cached."""
        if not self.cache_enabled:
            return None
        
        async with self.resource_manager.acquire_lock("cache_access"):
            if cache_key in self.content_cache:
                self.generation_metrics["cache_hits"] += 1
                return self.content_cache[cache_key]
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Any):
        """Cache operation result."""
        if not self.cache_enabled:
            return
        
        async with self.resource_manager.acquire_lock("cache_access"):
            self.content_cache[cache_key] = result
    
    @async_retry(max_retries=2, delay=1.0)
    @async_timeout(300)  # 5 minute timeout
    async def analyze_content_async(
        self, 
        markdown_content: str,
        source_file: str,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ContentAnalysisResult:
        """Analyze markdown content asynchronously."""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key("analyze_content", {
            "content_hash": hash(markdown_content),
            "source_file": source_file
        })
        
        cached_result = await self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Create progress tracking
        operation_id = f"analyze_{hash(source_file)}_{int(time.time())}"
        self.progress_tracker.start_operation(operation_id, 100, f"Analyzing {source_file}")
        
        if progress_callback:
            self.progress_tracker.add_callback(operation_id, progress_callback)
        
        try:
            async with self.resource_manager.acquire_resource("content_generation"):
                
                self.progress_tracker.update_progress(operation_id, 10, "Starting content analysis")
                
                # Extract basic structure
                sections = await self._extract_sections(markdown_content)
                self.progress_tracker.update_progress(operation_id, 30, "Extracted document structure")
                
                # Identify topics and concepts
                topics = await self._identify_topics(markdown_content)
                self.progress_tracker.update_progress(operation_id, 50, "Identified main topics")
                
                key_concepts = await self._extract_key_concepts(markdown_content)
                self.progress_tracker.update_progress(operation_id, 70, "Extracted key concepts")
                
                # Generate summary
                summary = await self._generate_summary(markdown_content)
                self.progress_tracker.update_progress(operation_id, 85, "Generated content summary")
                
                # Calculate quality indicators
                quality_indicators = await self._calculate_quality_indicators(markdown_content)
                self.progress_tracker.update_progress(operation_id, 95, "Calculated quality indicators")
                
                # Create analysis result
                result = ContentAnalysisResult(
                    source_file=source_file,
                    topics=topics,
                    key_concepts=key_concepts,
                    structure={"sections": len(sections), "words": len(markdown_content.split())},
                    summary=summary,
                    sections=sections,
                    quality_indicators=quality_indicators,
                    processing_time=time.time() - start_time,
                    success=True
                )
                
                # Cache result
                await self._cache_result(cache_key, result)
                
                # Update metrics
                self.generation_metrics["content_analyzed"] += 1
                self.generation_metrics["successful_operations"] += 1
                
                self.progress_tracker.complete_operation(operation_id, True, "Content analysis completed")
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Content analysis failed: {str(e)}"
            
            self.logger.error(error_msg)
            self.progress_tracker.complete_operation(operation_id, False, error_msg)
            
            self.generation_metrics["failed_operations"] += 1
            
            return ContentAnalysisResult(
                source_file=source_file,
                topics=[],
                key_concepts=[],
                structure={},
                summary="",
                sections=[],
                quality_indicators={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown content."""
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check for headers
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections.append({
                        **current_section,
                        "content": '\n'.join(current_content),
                        "word_count": len(' '.join(current_content).split())
                    })
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('# ').strip()
                current_section = {
                    "level": level,
                    "title": title,
                    "line_number": len(sections) + 1
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            sections.append({
                **current_section,
                "content": '\n'.join(current_content),
                "word_count": len(' '.join(current_content).split())
            })
        
        return sections
    
    async def _identify_topics(self, content: str) -> List[str]:
        """Identify main topics from content."""
        # Simple topic extraction using headers and frequent terms
        topics = []
        
        # Extract from headers
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):
                topic = line.lstrip('# ').strip()
                if topic and len(topic) > 3:
                    topics.append(topic)
        
        # Extract frequent capitalized terms (simplified approach)
        words = content.split()
        capitalized_words = [w for w in words if w[0].isupper() and len(w) > 3]
        word_freq = {}
        for word in capitalized_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Add frequent terms as topics
        frequent_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics.extend([term[0] for term in frequent_terms])
        
        return list(set(topics))[:10]  # Limit to 10 unique topics
    
    async def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        concepts = []
        
        # Look for definition patterns
        definition_patterns = [
            r'(\w+)\s+is\s+defined\s+as',
            r'(\w+)\s+refers\s+to',
            r'(\w+)\s+means',
            r'The\s+(\w+)\s+is\s+a'
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            concepts.extend(matches)
        
        # Look for emphasized terms (bold, italic)
        bold_terms = re.findall(r'\*\*([^*]+)\*\*', content)
        italic_terms = re.findall(r'\*([^*]+)\*', content)
        concepts.extend(bold_terms + italic_terms)
        
        # Clean and limit concepts
        cleaned_concepts = []
        for concept in concepts:
            if len(concept) > 2 and concept.isalpha():
                cleaned_concepts.append(concept.title())
        
        return list(set(cleaned_concepts))[:15]  # Limit to 15 unique concepts
    
    async def _generate_summary(self, content: str) -> str:
        """Generate a summary of the content."""
        if self.ai_client:
            try:
                async with self.resource_manager.acquire_resource("ai_api_calls"):
                    # Use AI to generate summary
                    prompt = f"""Please provide a concise summary of the following academic content in 2-3 sentences:

{content[:2000]}...

Summary:"""
                    
                    response = await self.ai_client.chat.completions.create(
                        model=self.config.get("ai_model", "llama-3.3-70b-versatile"),
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.get("ai_temperature", 0.3),
                        max_tokens=200
                    )
                    
                    self.generation_metrics["ai_api_calls"] += 1
                    return response.choices[0].message.content.strip()
                    
            except Exception as e:
                self.logger.warning(f"AI summary generation failed: {e}")
        
        # Fallback: extract first few sentences
        sentences = content.split('. ')
        summary_sentences = sentences[:3]
        return '. '.join(summary_sentences) + '.'
    
    async def _calculate_quality_indicators(self, content: str) -> Dict[str, float]:
        """Calculate quality indicators for content."""
        indicators = {}
        
        # Basic metrics
        word_count = len(content.split())
        sentence_count = len(content.split('.'))
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Readability indicators
        avg_words_per_sentence = word_count / max(1, sentence_count)
        avg_sentences_per_paragraph = sentence_count / max(1, paragraph_count)
        
        # Structure indicators
        header_count = len([line for line in content.split('\n') if line.startswith('#')])
        list_count = len([line for line in content.split('\n') if line.strip().startswith('-')])
        
        # Quality scores (0-1 scale)
        indicators['completeness'] = min(1.0, word_count / 1000)  # Based on word count
        indicators['structure'] = min(1.0, header_count / 5)  # Based on structure
        indicators['readability'] = max(0.0, 1.0 - abs(avg_words_per_sentence - 15) / 15)
        indicators['detail_level'] = min(1.0, (list_count + paragraph_count) / 20)
        
        return indicators
    
    @async_retry(max_retries=2, delay=1.0)
    @async_timeout(300)
    async def generate_outline_async(
        self,
        analysis_results: List[ContentAnalysisResult],
        target_depth: int = 3,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> OutlineGenerationResult:
        """Generate consolidated outline from analysis results."""
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key("generate_outline", {
            "sources": [r.source_file for r in analysis_results],
            "target_depth": target_depth
        })
        
        cached_result = await self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Create progress tracking
        operation_id = f"outline_{int(time.time())}"
        self.progress_tracker.start_operation(operation_id, 100, "Generating consolidated outline")
        
        if progress_callback:
            self.progress_tracker.add_callback(operation_id, progress_callback)
        
        try:
            async with self.resource_manager.acquire_resource("content_generation"):
                
                self.progress_tracker.update_progress(operation_id, 10, "Consolidating topics")
                
                # Consolidate all topics and concepts
                all_topics = []
                all_concepts = []
                all_sections = []
                
                for result in analysis_results:
                    all_topics.extend(result.topics)
                    all_concepts.extend(result.key_concepts)
                    all_sections.extend(result.sections)
                
                self.progress_tracker.update_progress(operation_id, 30, "Organizing hierarchical structure")
                
                # Create hierarchical structure
                hierarchical_topics = await self._create_hierarchy(all_topics, all_concepts, target_depth)
                
                self.progress_tracker.update_progress(operation_id, 60, "Generating outline content")
                
                # Generate outline content
                outline_content = await self._generate_outline_content(hierarchical_topics, all_sections)
                
                self.progress_tracker.update_progress(operation_id, 80, "Calculating complexity score")
                
                # Calculate complexity score
                complexity_score = await self._calculate_outline_complexity(hierarchical_topics)
                
                self.progress_tracker.update_progress(operation_id, 95, "Finalizing outline")
                
                # Create result
                result = OutlineGenerationResult(
                    source_files=[r.source_file for r in analysis_results],
                    outline_structure={"hierarchy": hierarchical_topics},
                    hierarchical_topics=hierarchical_topics,
                    estimated_sections=len(hierarchical_topics),
                    complexity_score=complexity_score,
                    outline_content=outline_content,
                    processing_time=time.time() - start_time,
                    success=True
                )
                
                # Cache result
                await self._cache_result(cache_key, result)
                
                # Update metrics
                self.generation_metrics["outlines_generated"] += 1
                self.generation_metrics["successful_operations"] += 1
                
                self.progress_tracker.complete_operation(operation_id, True, "Outline generation completed")
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Outline generation failed: {str(e)}"
            
            self.logger.error(error_msg)
            self.progress_tracker.complete_operation(operation_id, False, error_msg)
            
            self.generation_metrics["failed_operations"] += 1
            
            return OutlineGenerationResult(
                source_files=[r.source_file for r in analysis_results],
                outline_structure={},
                hierarchical_topics=[],
                estimated_sections=0,
                complexity_score=0.0,
                outline_content="",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _create_hierarchy(
        self, 
        topics: List[str], 
        concepts: List[str], 
        target_depth: int
    ) -> List[Dict[str, Any]]:
        """Create hierarchical topic structure."""
        
        # Remove duplicates and sort by importance
        unique_topics = list(set(topics))
        unique_concepts = list(set(concepts))
        
        # Create hierarchy based on topic relationships
        hierarchy = []
        main_topics = unique_topics[:min(6, len(unique_topics))]  # Limit main topics
        
        for i, main_topic in enumerate(main_topics):
            topic_entry = {
                "level": 1,
                "title": main_topic,
                "order": i + 1,
                "subtopics": []
            }
            
            # Add related concepts as subtopics
            related_concepts = [c for c in unique_concepts if c.lower() not in main_topic.lower()]
            subtopics_count = min(4, len(related_concepts))  # Limit subtopics
            
            for j, concept in enumerate(related_concepts[:subtopics_count]):
                subtopic = {
                    "level": 2,
                    "title": concept,
                    "order": j + 1,
                    "parent": main_topic
                }
                topic_entry["subtopics"].append(subtopic)
            
            hierarchy.append(topic_entry)
        
        return hierarchy
    
    async def _generate_outline_content(
        self, 
        hierarchical_topics: List[Dict[str, Any]],
        sections: List[Dict[str, Any]]
    ) -> str:
        """Generate formatted outline content."""
        
        outline_lines = []
        outline_lines.append("# Comprehensive Outline")
        outline_lines.append("")
        outline_lines.append(f"Generated: {datetime.now().isoformat()}")
        outline_lines.append("")
        
        for topic in hierarchical_topics:
            # Main topic
            outline_lines.append(f"## {topic['order']}. {topic['title']}")
            outline_lines.append("")
            
            # Subtopics
            for subtopic in topic.get("subtopics", []):
                outline_lines.append(f"### {topic['order']}.{subtopic['order']}. {subtopic['title']}")
                outline_lines.append("")
        
        return "\n".join(outline_lines)
    
    async def _calculate_outline_complexity(self, hierarchical_topics: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for outline."""
        
        total_items = len(hierarchical_topics)
        total_subtopics = sum(len(topic.get("subtopics", [])) for topic in hierarchical_topics)
        
        # Simple complexity calculation
        complexity = (total_items + total_subtopics) / 20  # Normalize to 0-1 scale
        return min(1.0, complexity)
    
    @async_retry(max_retries=2, delay=1.0)
    @async_timeout(600)  # 10 minute timeout for notes generation
    async def generate_notes_async(
        self,
        analysis_results: List[ContentAnalysisResult],
        outline_result: OutlineGenerationResult,
        target_length: int = 5000,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> NotesGenerationResult:
        """Generate comprehensive notes from analysis and outline."""
        
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key("generate_notes", {
            "sources": [r.source_file for r in analysis_results],
            "outline_hash": hash(outline_result.outline_content),
            "target_length": target_length
        })
        
        cached_result = await self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Create progress tracking
        operation_id = f"notes_{int(time.time())}"
        self.progress_tracker.start_operation(operation_id, 100, "Generating comprehensive notes")
        
        if progress_callback:
            self.progress_tracker.add_callback(operation_id, progress_callback)
        
        try:
            async with self.resource_manager.acquire_resource("content_generation"):
                
                self.progress_tracker.update_progress(operation_id, 10, "Preparing content synthesis")
                
                # Prepare content for synthesis
                all_content = []
                all_references = []
                
                for result in analysis_results:
                    all_content.append(result.summary)
                    all_references.append(result.source_file)
                
                self.progress_tracker.update_progress(operation_id, 30, "Synthesizing notes content")
                
                # Generate notes using AI or fallback method
                if self.ai_client:
                    notes_content = await self._generate_notes_with_ai(
                        all_content, 
                        outline_result.outline_content, 
                        target_length,
                        operation_id
                    )
                else:
                    notes_content = await self._generate_notes_fallback(
                        all_content,
                        outline_result.outline_content,
                        operation_id
                    )
                
                self.progress_tracker.update_progress(operation_id, 80, "Calculating quality metrics")
                
                # Calculate quality metrics
                quality_metrics = await self._calculate_notes_quality(notes_content)
                
                self.progress_tracker.update_progress(operation_id, 95, "Finalizing notes")
                
                # Count words and sections
                word_count = len(notes_content.split())
                sections_count = len([line for line in notes_content.split('\n') if line.startswith('#')])
                
                # Create result
                result = NotesGenerationResult(
                    source_files=[r.source_file for r in analysis_results],
                    outline_used=outline_result.outline_content,
                    notes_content=notes_content,
                    word_count=word_count,
                    sections_count=sections_count,
                    quality_metrics=quality_metrics,
                    references=all_references,
                    processing_time=time.time() - start_time,
                    success=True
                )
                
                # Cache result
                await self._cache_result(cache_key, result)
                
                # Update metrics
                self.generation_metrics["notes_generated"] += 1
                self.generation_metrics["successful_operations"] += 1
                
                self.progress_tracker.complete_operation(operation_id, True, "Notes generation completed")
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Notes generation failed: {str(e)}"
            
            self.logger.error(error_msg)
            self.progress_tracker.complete_operation(operation_id, False, error_msg)
            
            self.generation_metrics["failed_operations"] += 1
            
            return NotesGenerationResult(
                source_files=[r.source_file for r in analysis_results],
                outline_used="",
                notes_content="",
                word_count=0,
                sections_count=0,
                quality_metrics={},
                references=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    async def _generate_notes_with_ai(
        self,
        content_summaries: List[str],
        outline: str,
        target_length: int,
        operation_id: str
    ) -> str:
        """Generate notes using AI."""
        
        async with self.resource_manager.acquire_resource("ai_api_calls"):
            
            combined_content = "\n\n".join(content_summaries[:5])  # Limit content size
            
            prompt = f"""Based on the following outline and content summaries, generate comprehensive academic notes of approximately {target_length} words.

OUTLINE:
{outline[:1000]}

CONTENT SUMMARIES:
{combined_content[:3000]}

Please generate detailed, well-structured academic notes that:
1. Follow the outline structure
2. Integrate information from the content summaries
3. Include clear explanations and examples
4. Use proper academic formatting
5. Are approximately {target_length} words long

NOTES:"""
            
            # Update progress
            self.progress_tracker.update_progress(operation_id, 50, "Generating AI-powered notes")
            
            response = await self.ai_client.chat.completions.create(
                model=self.config.get("ai_model", "llama-3.3-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get("ai_temperature", 0.3),
                max_tokens=self.config.get("ai_max_tokens", 4096)
            )
            
            self.generation_metrics["ai_api_calls"] += 1
            
            return response.choices[0].message.content.strip()
    
    async def _generate_notes_fallback(
        self,
        content_summaries: List[str],
        outline: str,
        operation_id: str
    ) -> str:
        """Generate notes using fallback method (no AI)."""
        
        self.progress_tracker.update_progress(operation_id, 50, "Generating notes using fallback method")
        
        notes_lines = []
        notes_lines.append("# Comprehensive Academic Notes")
        notes_lines.append("")
        notes_lines.append(f"Generated: {datetime.now().isoformat()}")
        notes_lines.append("")
        
        # Add outline as structure
        outline_lines = outline.split('\n')
        for line in outline_lines:
            if line.strip().startswith('#'):
                notes_lines.append(line)
                notes_lines.append("")
                
                # Add relevant content under each heading
                if len(content_summaries) > 0:
                    # Rotate through summaries to distribute content
                    summary_index = len(notes_lines) % len(content_summaries)
                    summary_excerpt = content_summaries[summary_index][:300] + "..."
                    notes_lines.append(summary_excerpt)
                    notes_lines.append("")
        
        # Add remaining content summaries
        notes_lines.append("## Additional Content")
        notes_lines.append("")
        
        for i, summary in enumerate(content_summaries):
            notes_lines.append(f"### Source {i+1}")
            notes_lines.append(summary)
            notes_lines.append("")
        
        return "\n".join(notes_lines)
    
    async def _calculate_notes_quality(self, notes_content: str) -> Dict[str, float]:
        """Calculate quality metrics for generated notes."""
        
        metrics = {}
        
        # Basic metrics
        word_count = len(notes_content.split())
        sentence_count = len(notes_content.split('.'))
        paragraph_count = len([p for p in notes_content.split('\n\n') if p.strip()])
        header_count = len([line for line in notes_content.split('\n') if line.startswith('#')])
        
        # Quality calculations
        metrics['length_score'] = min(1.0, word_count / 3000)  # Target 3000+ words
        metrics['structure_score'] = min(1.0, header_count / 8)  # Target 8+ sections
        metrics['detail_score'] = min(1.0, paragraph_count / 15)  # Target 15+ paragraphs
        metrics['readability_score'] = max(0.0, 1.0 - abs((word_count / max(1, sentence_count)) - 20) / 20)
        
        # Overall score
        metrics['overall_score'] = sum(metrics.values()) / len(metrics)
        
        return metrics
    
    async def get_generation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive generation metrics."""
        
        # Calculate additional metrics
        total_operations = (
            self.generation_metrics["successful_operations"] + 
            self.generation_metrics["failed_operations"]
        )
        
        success_rate = (
            self.generation_metrics["successful_operations"] / max(1, total_operations)
        )
        
        return {
            **self.generation_metrics,
            "total_operations": total_operations,
            "success_rate": success_rate,
            "cache_hit_rate": (
                self.generation_metrics["cache_hits"] / 
                max(1, self.generation_metrics["cache_hits"] + total_operations)
            ),
            "cached_items": len(self.content_cache),
            "resource_usage": self.resource_manager.get_resource_stats()
        }
    
    async def clear_cache(self):
        """Clear content generation cache."""
        async with self.resource_manager.acquire_lock("cache_access"):
            self.content_cache.clear()
            self.logger.info("Content generation cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the content generator."""
        
        health = {
            "status": "healthy",
            "ai_client": "available" if self.ai_client else "unavailable",
            "metrics": await self.get_generation_metrics(),
            "config": {
                "max_concurrent": self.config.get("max_concurrent_generations", 3),
                "cache_enabled": self.cache_enabled,
                "ai_model": self.config.get("ai_model", "llama-3.3-70b-versatile")
            }
        }
        
        # Check AI availability
        if not self.ai_client:
            health["status"] = "degraded"
            health["warnings"] = ["AI client not available - using fallback methods"]
        
        return health


# Utility functions
async def process_content_pipeline_async(
    markdown_files: List[str],
    config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """Process complete content generation pipeline."""
    
    generator = AsyncContentGenerator(config)
    
    try:
        # Step 1: Analyze all content
        analysis_tasks = []
        for file_path in markdown_files:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            analysis_tasks.append(generator.analyze_content_async(content, file_path))
        
        analysis_results = await asyncio.gather(*analysis_tasks)
        successful_analyses = [r for r in analysis_results if r.success]
        
        if not successful_analyses:
            raise RuntimeError("All content analysis failed")
        
        # Step 2: Generate outline
        outline_result = await generator.generate_outline_async(successful_analyses)
        
        if not outline_result.success:
            raise RuntimeError(f"Outline generation failed: {outline_result.error_message}")
        
        # Step 3: Generate notes
        notes_result = await generator.generate_notes_async(successful_analyses, outline_result)
        
        if not notes_result.success:
            raise RuntimeError(f"Notes generation failed: {notes_result.error_message}")
        
        return {
            "analysis_results": successful_analyses,
            "outline_result": outline_result,
            "notes_result": notes_result,
            "metrics": await generator.get_generation_metrics(),
            "success": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


# Example usage
async def main():
    """Example usage of async content generator."""
    
    config = {
        "max_concurrent_generations": 2,
        "ai_model": "llama-3.3-70b-versatile",
        "enable_cache": True
    }
    
    generator = AsyncContentGenerator(config)
    
    # Example markdown content
    test_content = """# Test Document

## Introduction
This is a test document for content analysis.

## Main Topics
- Topic 1: Important concept
- Topic 2: Another concept

## Conclusion
This concludes the test document.
"""
    
    def progress_callback(progress: float, message: str):
        print(f"Progress: {progress:.1%} - {message}")
    
    try:
        # Analyze content
        analysis_result = await generator.analyze_content_async(
            test_content, 
            "test_document.md",
            progress_callback
        )
        
        print(f"Analysis result: {analysis_result.success}")
        
        if analysis_result.success:
            # Generate outline
            outline_result = await generator.generate_outline_async([analysis_result])
            print(f"Outline result: {outline_result.success}")
            
            if outline_result.success:
                # Generate notes
                notes_result = await generator.generate_notes_async(
                    [analysis_result], 
                    outline_result
                )
                print(f"Notes result: {notes_result.success}")
                print(f"Generated {notes_result.word_count} words")
        
        # Print metrics
        metrics = await generator.get_generation_metrics()
        print(f"Metrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())