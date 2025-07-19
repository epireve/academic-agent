#!/usr/bin/env python
"""
Content Management System Integration Module

This module provides integration between the CMS and existing academic agent systems,
including the main agent, consolidation agent, and quality management systems.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import uuid

from .content_management_system import (
    ContentManagementSystem, CourseInfo, ContentItem, ContentType,
    ProcessingStatus, ProcessingRecord
)
from ...src.agents.base_agent import BaseAgent, AgentMessage
from .consolidation_agent import ContentConsolidationAgent
from ...src.agents.quality_manager import QualityManager
from .main_agent import MainAcademicAgent


class CMSIntegration(BaseAgent):
    """Integration layer between CMS and existing academic agent systems"""
    
    def __init__(self, base_path: str, config: Dict[str, Any] = None):
        super().__init__("cms_integration")
        
        self.base_path = Path(base_path)
        self.config = config or {}
        
        # Initialize CMS
        self.cms = ContentManagementSystem(str(self.base_path), config)
        
        # Initialize existing systems
        self.consolidation_agent = ContentConsolidationAgent()
        self.quality_manager = QualityManager()
        
        # Integration flags
        self.auto_import_enabled = self.config.get('auto_import_enabled', True)
        self.auto_quality_check = self.config.get('auto_quality_check', True)
        self.auto_relationship_detection = self.config.get('auto_relationship_detection', True)
        
        self.logger.info("CMS Integration initialized")
    
    def integrate_with_main_agent(self, main_agent: 'MainAcademicAgent') -> bool:
        """Integrate CMS with the main academic agent"""
        try:
            # Set up message handlers for CMS operations
            main_agent.cms_integration = self
            
            # Register CMS-specific message types
            cms_message_types = [
                "content_import_request",
                "content_search_request", 
                "quality_assessment_request",
                "relationship_detection_request",
                "analytics_request"
            ]
            
            # Add CMS capabilities to main agent
            main_agent.capabilities.extend([
                "content_management",
                "search_and_discovery",
                "relationship_mapping",
                "quality_assessment",
                "content_analytics"
            ])
            
            self.logger.info("Successfully integrated with main academic agent")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with main agent: {str(e)}")
            return False
    
    # Required BaseAgent methods
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        return isinstance(input_data, dict)
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        return isinstance(output_data, dict)
    
    def check_quality(self, content: Any) -> float:
        """Check quality of integration operations"""
        try:
            # Basic health check
            status = self.get_cms_status()
            
            if 'error' in status:
                return 0.1
            
            # Quality based on content processing success
            total_content = status.get('total_content', 0)
            completed_content = status.get('content_by_status', {}).get('completed', 0)
            
            if total_content > 0:
                completion_rate = completed_content / total_content
                return min(0.9, 0.5 + (completion_rate * 0.4))
            else:
                return 0.7  # Default for empty system
                
        except Exception:
            return 0.1
    
    def get_cms_status(self) -> Dict[str, Any]:
        """Get comprehensive CMS status"""
        try:
            # Get basic statistics
            courses = self.cms.list_courses()
            all_content = self.cms.list_content(limit=10000)  # Get all content
            
            # Content statistics by type
            content_by_type = {}
            content_by_status = {}
            
            for item in all_content:
                content_type = item.content_type.value
                status = item.processing_status.value
                
                content_by_type[content_type] = content_by_type.get(content_type, 0) + 1
                content_by_status[status] = content_by_status.get(status, 0) + 1
            
            # Quality statistics
            quality_items = [item for item in all_content if item.quality_score is not None]
            avg_quality = sum(item.quality_score for item in quality_items) / len(quality_items) if quality_items else 0
            
            # Storage statistics
            total_storage = sum(item.file_size for item in all_content)
            
            return {
                'total_courses': len(courses),
                'total_content': len(all_content),
                'content_by_type': content_by_type,
                'content_by_status': content_by_status,
                'average_quality_score': avg_quality,
                'total_storage_bytes': total_storage,
                'total_storage_mb': total_storage / (1024 * 1024),
                'cms_agent_status': self.get_agent_status()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting CMS status: {str(e)}")
            return {
                'error': str(e)
            }
    
    def shutdown(self) -> None:
        """Shutdown the integration system"""
        try:
            self.cms.shutdown()
            self.logger.info("CMS Integration shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during integration shutdown: {str(e)}")
        
        super().shutdown()


def create_integration_config() -> Dict[str, Any]:
    """Create default integration configuration"""
    return {
        'auto_import_enabled': True,
        'auto_quality_check': True,
        'auto_relationship_detection': True,
        'cms': {
            'database': {
                'path': 'cms/content_database.db',
                'backup_enabled': True
            },
            'storage': {
                'base_path': 'cms/storage',
                'versioning_enabled': True
            },
            'processing': {
                'auto_index_content': True,
                'batch_processing_size': 10
            }
        }
    }


def main():
    """Main function for testing integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CMS Integration Testing")
    parser.add_argument("--base-path", required=True, help="Base path for CMS")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    
    args = parser.parse_args()
    
    # Initialize integration
    config = create_integration_config()
    integration = CMSIntegration(args.base_path, config)
    
    if args.test:
        print("Running integration tests...")
        
        # Test status
        status = integration.get_cms_status()
        print(f"CMS Status: {json.dumps(status, indent=2)}")
        
        print("Integration tests completed")
    
    integration.shutdown()


if __name__ == "__main__":
    main()