#!/usr/bin/env python3
"""
Comprehensive integration tests for multi-agent workflows

Tests complete academic processing pipelines involving multiple agents
working together to process academic content.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Import agents for integration testing
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.academic.main_agent import MainAcademicAgent
from agents.academic.analysis_agent import AnalysisAgent
from agents.academic.notes_agent import NotesAgent
from agents.academic.consolidation_agent import ContentConsolidationAgent
from agents.academic.quality_validation_system import QualityValidationSystem
from agents.academic.study_notes_generator import StudyNotesGeneratorTool
from src.agents.quality_manager import QualityManager


class TestMultiAgentWorkflows:
    """Comprehensive test suite for multi-agent workflows"""
    
    @pytest.fixture
    async def agent_ecosystem(self):
        """Set up a complete agent ecosystem for testing"""
        ecosystem = {
            'main_agent': MainAcademicAgent(),
            'analysis_agent': AnalysisAgent(),
            'notes_agent': NotesAgent(),
            'consolidation_agent': ContentConsolidationAgent(),
            'quality_system': QualityValidationSystem(),
            'quality_manager': QualityManager()
        }
        
        # Initialize all agents
        for agent_name, agent in ecosystem.items():
            if hasattr(agent, 'initialize'):
                await agent.initialize()
        
        yield ecosystem
        
        # Cleanup all agents
        for agent_name, agent in ecosystem.items():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()

    @pytest.fixture
    def sample_academic_files(self):
        """Create sample academic files for workflow testing"""
        files = {}
        
        # Lecture content
        lecture_content = """# Computer Networks - Lecture 5: Network Protocols

## Learning Objectives
- Understand the TCP/IP protocol stack
- Learn about HTTP and HTTPS protocols
- Explore network security basics

## TCP/IP Protocol Stack

The TCP/IP model consists of four layers:

### Application Layer
- HTTP/HTTPS for web traffic
- SMTP for email
- FTP for file transfer

### Transport Layer  
- TCP for reliable communication
- UDP for faster, unreliable communication

### Internet Layer
- IP for routing packets
- ICMP for control messages

### Network Access Layer
- Ethernet for local networks
- WiFi for wireless communication

## HTTP vs HTTPS

HTTP (HyperText Transfer Protocol) is the foundation of web communication.
HTTPS adds SSL/TLS encryption for security.

### Key Differences
- **Security**: HTTPS encrypts data, HTTP does not
- **Port**: HTTP uses port 80, HTTPS uses port 443
- **Performance**: HTTPS has slight overhead due to encryption

## Network Security Basics

Common threats include:
- Man-in-the-middle attacks
- Packet sniffing
- DNS spoofing
- DDoS attacks

Protection mechanisms:
- Encryption (SSL/TLS)
- Firewalls
- Intrusion detection systems
- Access control lists

## Summary

Network protocols form the backbone of internet communication. Understanding
the protocol stack and security considerations is essential for network
administration and application development.
"""
        
        # Textbook chapter
        textbook_content = """# Chapter 7: Database Design Principles

Database design is a critical aspect of information systems development
that determines how data is organized, stored, and accessed.

## Normalization

Normalization is the process of organizing data to reduce redundancy.

### First Normal Form (1NF)
- Eliminate repeating groups
- Each cell contains atomic values
- Each row is unique

### Second Normal Form (2NF)
- Must be in 1NF
- Remove partial dependencies
- Non-key attributes depend on entire primary key

### Third Normal Form (3NF)
- Must be in 2NF
- Remove transitive dependencies
- Non-key attributes depend only on primary key

## Entity-Relationship Modeling

ER diagrams represent:
- **Entities**: Things or objects
- **Attributes**: Properties of entities
- **Relationships**: Associations between entities

### Relationship Types
- One-to-One (1:1)
- One-to-Many (1:M)
- Many-to-Many (M:N)

## ACID Properties

Database transactions must maintain:
- **Atomicity**: All or nothing execution
- **Consistency**: Data integrity maintained
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed changes persist

## Indexing Strategies

Indexes improve query performance:
- **Primary Index**: On primary key
- **Secondary Index**: On non-key attributes
- **Composite Index**: On multiple columns
- **Unique Index**: Enforces uniqueness

Trade-offs:
- Faster queries vs. slower updates
- Storage overhead vs. performance gain

## Conclusion

Effective database design requires understanding normalization, ER modeling,
ACID properties, and indexing strategies. These principles ensure data
integrity, performance, and maintainability.
"""
        
        # Assignment content
        assignment_content = """# Assignment 3: Network Security Analysis

## Objective
Analyze network security vulnerabilities and propose mitigation strategies
for a small business network infrastructure.

## Requirements

### Part A: Vulnerability Assessment (30 points)
1. Identify potential security vulnerabilities in the given network diagram
2. Categorize vulnerabilities by severity (Critical, High, Medium, Low)
3. Explain the potential impact of each vulnerability

### Part B: Risk Analysis (25 points)
1. Calculate risk scores using the formula: Risk = Likelihood × Impact
2. Create a risk matrix showing all identified risks
3. Prioritize risks for remediation

### Part C: Mitigation Plan (30 points)
1. Propose specific security controls for each high-risk vulnerability
2. Estimate implementation costs and timelines
3. Justify your recommendations with industry best practices

### Part D: Implementation Roadmap (15 points)
1. Create a phased implementation plan
2. Identify dependencies between security controls
3. Establish success metrics for each phase

## Deliverables
- Executive summary (1-2 pages)
- Technical report (8-10 pages)
- Network security diagram (updated)
- Implementation timeline (Gantt chart)

## Evaluation Criteria
- Technical accuracy (40%)
- Completeness of analysis (30%)
- Quality of recommendations (20%)
- Presentation and clarity (10%)

## Submission Guidelines
- Due date: March 15, 2024, 11:59 PM
- Submit via course portal as single PDF
- Include all supporting materials in appendices
- Follow APA citation format for references

## Resources
- Course textbook: Chapters 8-12
- NIST Cybersecurity Framework
- ISO 27001 standard
- Industry vulnerability databases

## Academic Integrity
This is an individual assignment. Collaboration is not permitted.
Cite all sources and avoid plagiarism.
"""
        
        # Save files to temporary locations
        for name, content in [
            ('lecture_networks.md', lecture_content),
            ('textbook_database_design.md', textbook_content),
            ('assignment_security_analysis.md', assignment_content)
        ]:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                files[name] = f.name
        
        return files

    @pytest.mark.integration
    async def test_complete_academic_pipeline(self, agent_ecosystem, sample_academic_files):
        """Test complete academic content processing pipeline"""
        main_agent = agent_ecosystem['main_agent']
        analysis_agent = agent_ecosystem['analysis_agent']
        notes_agent = agent_ecosystem['notes_agent']
        consolidation_agent = agent_ecosystem['consolidation_agent']
        quality_system = agent_ecosystem['quality_system']
        
        results = {}
        
        # Step 1: Analyze all content files
        for filename, filepath in sample_academic_files.items():
            analysis_result = analysis_agent.analyze_file(filepath)
            results[f'analysis_{filename}'] = analysis_result
            
            assert analysis_result['success'] == True
            assert 'analysis_result' in analysis_result
            assert 'quality_score' in analysis_result
        
        # Step 2: Generate notes for each file
        for filename, filepath in sample_academic_files.items():
            notes_result = notes_agent.process_file(filepath, output_formats=['markdown', 'json'])
            results[f'notes_{filename}'] = notes_result
            
            assert notes_result['success'] == True
            assert len(notes_result['output_files']) > 0
        
        # Step 3: Consolidate all content
        file_paths = list(sample_academic_files.values())
        consolidation_result = consolidation_agent.consolidate_files(file_paths)
        results['consolidation'] = consolidation_result
        
        assert consolidation_result['success'] == True
        assert 'consolidated_content' in consolidation_result
        
        # Step 4: Quality validation of consolidated content
        if consolidation_result['success']:
            consolidated_content = consolidation_result['consolidated_content']
            quality_result = quality_system.validate_content(consolidated_content)
            results['quality_validation'] = quality_result
            
            assert hasattr(quality_result, 'overall_quality_score')
            assert 0.0 <= quality_result.overall_quality_score <= 1.0
        
        # Verify pipeline integrity
        assert len(results) >= 7  # 3 analyses + 3 notes + 1 consolidation + quality
        
        # Check that all steps completed successfully
        successful_steps = sum(1 for key, result in results.items() 
                              if key.startswith(('analysis_', 'notes_')) and result.get('success', False))
        assert successful_steps >= 6  # All analysis and notes steps should succeed

    @pytest.mark.integration
    async def test_quality_consolidation_workflow(self, agent_ecosystem, sample_academic_files):
        """Test workflow focused on quality assessment and consolidation"""
        consolidation_agent = agent_ecosystem['consolidation_agent']
        quality_system = agent_ecosystem['quality_system']
        quality_manager = agent_ecosystem['quality_manager']
        
        # Step 1: Process files with quality assessment
        file_paths = list(sample_academic_files.values())
        
        # Consolidate with quality checks
        consolidation_result = consolidation_agent.consolidate_files(
            file_paths, 
            enable_quality_assessment=True
        )
        
        assert consolidation_result['success'] == True
        
        # Step 2: Detailed quality validation
        consolidated_content = consolidation_result['consolidated_content']
        detailed_quality = quality_system.validate_content(
            consolidated_content,
            quality_threshold=0.7
        )
        
        assert hasattr(detailed_quality, 'validation_feedback')
        assert len(detailed_quality.validation_feedback) > 0
        
        # Step 3: Quality manager assessment
        manager_assessment = quality_manager.assess_notes_quality(consolidated_content)
        
        assert 0.0 <= manager_assessment <= 1.0
        
        # Verify quality consistency across different validators
        quality_scores = [
            detailed_quality.overall_quality_score,
            manager_assessment
        ]
        
        # Scores should be within reasonable range of each other
        max_score, min_score = max(quality_scores), min(quality_scores)
        assert max_score - min_score <= 0.3  # Allow for some variation

    @pytest.mark.integration
    async def test_concurrent_agent_processing(self, agent_ecosystem, sample_academic_files):
        """Test concurrent processing by multiple agents"""
        analysis_agent = agent_ecosystem['analysis_agent']
        notes_agent = agent_ecosystem['notes_agent']
        
        # Prepare concurrent tasks
        analysis_tasks = []
        notes_tasks = []
        
        for filename, filepath in sample_academic_files.items():
            # Create analysis task
            analysis_task = asyncio.create_task(
                asyncio.to_thread(analysis_agent.analyze_file, filepath)
            )
            analysis_tasks.append(analysis_task)
            
            # Create notes generation task
            notes_task = asyncio.create_task(
                asyncio.to_thread(notes_agent.process_file, filepath)
            )
            notes_tasks.append(notes_task)
        
        # Run all tasks concurrently
        analysis_results = await asyncio.gather(*analysis_tasks)
        notes_results = await asyncio.gather(*notes_tasks)
        
        # Verify all analyses completed successfully
        for result in analysis_results:
            assert result['success'] == True
        
        # Verify all notes generation completed successfully
        for result in notes_results:
            assert result['success'] == True
        
        # Verify concurrent processing didn't cause issues
        assert len(analysis_results) == len(sample_academic_files)
        assert len(notes_results) == len(sample_academic_files)

    @pytest.mark.integration
    async def test_error_recovery_across_agents(self, agent_ecosystem):
        """Test error handling and recovery in multi-agent workflows"""
        main_agent = agent_ecosystem['main_agent']
        analysis_agent = agent_ecosystem['analysis_agent']
        notes_agent = agent_ecosystem['notes_agent']
        
        # Test with non-existent file
        nonexistent_file = "/path/to/nonexistent/file.md"
        
        # Analysis agent should handle gracefully
        analysis_result = analysis_agent.analyze_file(nonexistent_file)
        assert analysis_result['success'] == False
        assert 'error' in analysis_result
        
        # Notes agent should handle gracefully
        notes_result = notes_agent.process_file(nonexistent_file)
        assert notes_result['success'] == False
        assert 'error' in notes_result
        
        # Test with corrupted content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("Corrupted content: \x00\x01\x02")
            corrupted_file = f.name
        
        # Agents should handle corrupted content gracefully
        analysis_result = analysis_agent.analyze_file(corrupted_file)
        notes_result = notes_agent.process_file(corrupted_file)
        
        # Results may succeed or fail, but should not crash
        assert 'success' in analysis_result
        assert 'success' in notes_result

    @pytest.mark.integration
    async def test_state_management_across_agents(self, agent_ecosystem, sample_academic_files):
        """Test state management and data flow between agents"""
        analysis_agent = agent_ecosystem['analysis_agent']
        notes_agent = agent_ecosystem['notes_agent']
        consolidation_agent = agent_ecosystem['consolidation_agent']
        
        # Track intermediate results
        intermediate_results = {}
        
        # Process each file and track state
        for filename, filepath in sample_academic_files.items():
            # Analyze and store result
            analysis_result = analysis_agent.analyze_file(filepath)
            intermediate_results[f'analysis_{filename}'] = analysis_result
            
            # Generate notes using analysis context
            notes_result = notes_agent.process_file(
                filepath,
                context=analysis_result.get('analysis_result')
            )
            intermediate_results[f'notes_{filename}'] = notes_result
        
        # Consolidate using all intermediate results
        consolidation_result = consolidation_agent.consolidate_content(
            content_items=[r.get('analysis_result') for r in intermediate_results.values() 
                          if r.get('success') and 'analysis_result' in r],
            context={'intermediate_results': intermediate_results}
        )
        
        # Verify state consistency
        assert consolidation_result['success'] == True
        
        # Check that consolidation used information from previous steps
        consolidated_content = consolidation_result.get('consolidated_content', {})
        assert isinstance(consolidated_content, dict)

    @pytest.mark.integration 
    async def test_workflow_performance_metrics(self, agent_ecosystem, sample_academic_files):
        """Test performance metrics collection across workflow"""
        import time
        
        analysis_agent = agent_ecosystem['analysis_agent']
        notes_agent = agent_ecosystem['notes_agent']
        quality_system = agent_ecosystem['quality_system']
        
        workflow_metrics = {
            'start_time': time.time(),
            'step_times': {},
            'memory_usage': {},
            'success_rates': {}
        }
        
        # Analysis step
        step_start = time.time()
        analysis_results = []
        for filename, filepath in sample_academic_files.items():
            result = analysis_agent.analyze_file(filepath)
            analysis_results.append(result)
        
        workflow_metrics['step_times']['analysis'] = time.time() - step_start
        workflow_metrics['success_rates']['analysis'] = sum(
            1 for r in analysis_results if r.get('success', False)
        ) / len(analysis_results)
        
        # Notes generation step
        step_start = time.time()
        notes_results = []
        for filename, filepath in sample_academic_files.items():
            result = notes_agent.process_file(filepath)
            notes_results.append(result)
        
        workflow_metrics['step_times']['notes'] = time.time() - step_start
        workflow_metrics['success_rates']['notes'] = sum(
            1 for r in notes_results if r.get('success', False)
        ) / len(notes_results)
        
        # Quality validation step
        step_start = time.time()
        quality_results = []
        for result in analysis_results:
            if result.get('success'):
                quality_result = quality_system.validate_content(
                    result['analysis_result']
                )
                quality_results.append(quality_result)
        
        workflow_metrics['step_times']['quality'] = time.time() - step_start
        workflow_metrics['total_time'] = time.time() - workflow_metrics['start_time']
        
        # Verify performance characteristics
        assert workflow_metrics['total_time'] < 60.0  # Should complete within 1 minute
        assert workflow_metrics['success_rates']['analysis'] >= 0.8  # 80% success rate
        assert workflow_metrics['success_rates']['notes'] >= 0.8  # 80% success rate
        
        # Check that no single step takes too long
        for step, duration in workflow_metrics['step_times'].items():
            assert duration < 30.0  # No step should take more than 30 seconds

    @pytest.mark.integration
    async def test_configuration_integration(self, agent_ecosystem):
        """Test configuration management across agents"""
        # Test that agents can share configuration
        shared_config = {
            'quality_threshold': 0.8,
            'output_format': 'markdown',
            'enable_diagrams': True,
            'ai_model': 'test-model'
        }
        
        # Apply configuration to agents that support it
        configurable_agents = [
            'notes_agent',
            'quality_system'
        ]
        
        for agent_name in configurable_agents:
            agent = agent_ecosystem[agent_name]
            if hasattr(agent, 'update_config'):
                agent.update_config(shared_config)
            elif hasattr(agent, 'config'):
                agent.config.update(shared_config)
        
        # Verify configuration is applied
        notes_agent = agent_ecosystem['notes_agent']
        quality_system = agent_ecosystem['quality_system']
        
        # Test that configuration affects behavior
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Content\n\nThis is test content for configuration testing.")
            test_file = f.name
        
        # Process with configured agents
        notes_result = notes_agent.process_file(test_file)
        
        # Verify configuration was used (agents should succeed with valid config)
        assert notes_result.get('success', False) == True

    @pytest.mark.slow
    @pytest.mark.integration
    async def test_large_scale_workflow(self, agent_ecosystem):
        """Test workflow with larger scale content"""
        analysis_agent = agent_ecosystem['analysis_agent']
        consolidation_agent = agent_ecosystem['consolidation_agent']
        
        # Create multiple large content files
        large_files = []
        for i in range(10):
            content = f"# Large Document {i}\n\n" + "Large content section. " * 1000
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(content)
                large_files.append(f.name)
        
        # Process all files
        analysis_results = []
        for file_path in large_files:
            result = analysis_agent.analyze_file(file_path)
            analysis_results.append(result)
        
        # Verify processing succeeded for most files
        success_count = sum(1 for r in analysis_results if r.get('success', False))
        assert success_count >= 8  # At least 80% should succeed
        
        # Test consolidation of large content
        consolidation_result = consolidation_agent.consolidate_files(large_files[:5])
        
        # Should handle large consolidation
        assert consolidation_result.get('success', False) == True

    def teardown_method(self, method):
        """Clean up after each test"""
        # Clean up temporary files if needed
        pass


if __name__ == "__main__":
    pytest.main([__file__])