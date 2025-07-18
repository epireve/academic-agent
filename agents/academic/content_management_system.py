#!/usr/bin/env python
"""
Content Management System for Academic Agent

A comprehensive system for managing course information, processing history,
content relationships, versioning, search capabilities, and analytics.

This CMS integrates with the existing consolidation and quality systems to provide
a complete content lifecycle management solution.
"""

import os
import json
import sqlite3
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import re
import logging
from enum import Enum
import uuid

from .base_agent import BaseAgent, AgentMessage
from .consolidation_agent import ContentConsolidationAgent, ConsolidationResult, FileMapping
from .quality_manager import QualityManager, QualityEvaluation


class ContentType(Enum):
    """Content type enumeration"""
    LECTURE = "lecture"
    TRANSCRIPT = "transcript"
    NOTES = "notes"
    TEXTBOOK = "textbook"
    ASSIGNMENT = "assignment"
    TUTORIAL = "tutorial"
    EXAM = "exam"
    IMAGE = "image"
    DIAGRAM = "diagram"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ChangeType(Enum):
    """Change type enumeration for version control"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"
    MERGE = "merge"
    SPLIT = "split"


@dataclass
class CourseInfo:
    """Course information data structure"""
    course_id: str
    course_name: str
    course_code: str
    academic_year: str
    semester: str
    instructor: str
    department: str
    description: Optional[str] = None
    credits: Optional[int] = None
    prerequisites: List[str] = field(default_factory=list)
    learning_outcomes: List[str] = field(default_factory=list)
    assessment_methods: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)


@dataclass
class ContentItem:
    """Content item data structure"""
    content_id: str
    title: str
    content_type: ContentType
    course_id: str
    file_path: str
    original_filename: str
    file_size: int
    file_hash: str
    mime_type: str
    week_number: Optional[int] = None
    chapter_number: Optional[int] = None
    section_number: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    description: Optional[str] = None
    author: Optional[str] = None
    source_url: Optional[str] = None
    language: str = "en"
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    quality_score: Optional[float] = None
    created_date: datetime = field(default_factory=datetime.now)
    modified_date: datetime = field(default_factory=datetime.now)
    accessed_date: Optional[datetime] = None
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentRelationship:
    """Content relationship data structure"""
    relationship_id: str
    source_content_id: str
    target_content_id: str
    relationship_type: str
    strength: float
    description: Optional[str] = None
    auto_detected: bool = False
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingRecord:
    """Processing history record"""
    record_id: str
    content_id: str
    operation: str
    agent_id: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    quality_score: Optional[float] = None
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentVersion:
    """Content version data structure"""
    version_id: str
    content_id: str
    version_number: int
    change_type: ChangeType
    file_path: str
    file_hash: str
    change_description: str
    author: str
    parent_version_id: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchIndex:
    """Search index entry"""
    content_id: str
    term: str
    frequency: int
    position: int
    context: str
    content_type: ContentType
    week_number: Optional[int] = None
    chapter_number: Optional[int] = None


class ContentManagementSystem(BaseAgent):
    """Comprehensive Content Management System for Academic Content"""

    def __init__(self, base_path: str, config: Dict[str, Any] = None):
        super().__init__("content_management_system")
        
        self.base_path = Path(base_path)
        self.config = config or {}
        
        # Initialize paths
        self.db_path = self.base_path / "cms" / "content_database.db"
        self.storage_path = self.base_path / "cms" / "storage"
        self.versions_path = self.base_path / "cms" / "versions"
        self.index_path = self.base_path / "cms" / "search_index"
        self.reports_path = self.base_path / "cms" / "reports"
        
        # Create directories
        for path in [self.db_path.parent, self.storage_path, self.versions_path, 
                     self.index_path, self.reports_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Initialize components
        self.consolidation_agent = ContentConsolidationAgent()
        self.quality_manager = QualityManager()
        
        # Search index
        self._search_index: Dict[str, List[SearchIndex]] = defaultdict(list)
        self._load_search_index()
        
        # Content cache
        self._content_cache: Dict[str, ContentItem] = {}
        self._relationship_cache: Dict[str, List[ContentRelationship]] = defaultdict(list)
        
        self.logger.info(f"Content Management System initialized at {self.base_path}")

    def _init_database(self) -> None:
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Course information table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id TEXT PRIMARY KEY,
                course_name TEXT NOT NULL,
                course_code TEXT NOT NULL,
                academic_year TEXT NOT NULL,
                semester TEXT NOT NULL,
                instructor TEXT NOT NULL,
                department TEXT NOT NULL,
                description TEXT,
                credits INTEGER,
                prerequisites TEXT, -- JSON array
                learning_outcomes TEXT, -- JSON array
                assessment_methods TEXT, -- JSON array
                created_date TEXT NOT NULL,
                modified_date TEXT NOT NULL
            )
        """)
        
        # Content items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_items (
                content_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content_type TEXT NOT NULL,
                course_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                week_number INTEGER,
                chapter_number INTEGER,
                section_number INTEGER,
                tags TEXT, -- JSON array
                keywords TEXT, -- JSON array
                description TEXT,
                author TEXT,
                source_url TEXT,
                language TEXT DEFAULT 'en',
                processing_status TEXT NOT NULL,
                quality_score REAL,
                created_date TEXT NOT NULL,
                modified_date TEXT NOT NULL,
                accessed_date TEXT,
                access_count INTEGER DEFAULT 0,
                metadata TEXT, -- JSON object
                FOREIGN KEY (course_id) REFERENCES courses (course_id)
            )
        """)
        
        # Content relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_content_id TEXT NOT NULL,
                target_content_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL NOT NULL,
                description TEXT,
                auto_detected INTEGER DEFAULT 0,
                created_date TEXT NOT NULL,
                metadata TEXT, -- JSON object
                FOREIGN KEY (source_content_id) REFERENCES content_items (content_id),
                FOREIGN KEY (target_content_id) REFERENCES content_items (content_id)
            )
        """)
        
        # Processing history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_history (
                record_id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                input_data TEXT NOT NULL, -- JSON object
                output_data TEXT NOT NULL, -- JSON object
                processing_time REAL NOT NULL,
                success INTEGER NOT NULL,
                error_message TEXT,
                quality_score REAL,
                created_date TEXT NOT NULL,
                metadata TEXT, -- JSON object
                FOREIGN KEY (content_id) REFERENCES content_items (content_id)
            )
        """)
        
        # Content versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_versions (
                version_id TEXT PRIMARY KEY,
                content_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                change_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                change_description TEXT NOT NULL,
                author TEXT NOT NULL,
                parent_version_id TEXT,
                created_date TEXT NOT NULL,
                metadata TEXT, -- JSON object
                FOREIGN KEY (content_id) REFERENCES content_items (content_id),
                FOREIGN KEY (parent_version_id) REFERENCES content_versions (version_id)
            )
        """)
        
        # Search index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_index (
                content_id TEXT NOT NULL,
                term TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                position INTEGER NOT NULL,
                context TEXT NOT NULL,
                content_type TEXT NOT NULL,
                week_number INTEGER,
                chapter_number INTEGER,
                PRIMARY KEY (content_id, term, position),
                FOREIGN KEY (content_id) REFERENCES content_items (content_id)
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_course ON content_items(course_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON content_items(content_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_week ON content_items(week_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_source ON content_relationships(source_content_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_target ON content_relationships(target_content_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_content ON processing_history(content_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_content ON content_versions(content_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_term ON search_index(term)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_search_content ON search_index(content_id)")
        
        conn.commit()
        conn.close()
        
        self.logger.info("Database initialized successfully")

    # Course Management Methods
    def create_course(self, course_info: CourseInfo) -> bool:
        """Create a new course in the system"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO courses (
                    course_id, course_name, course_code, academic_year, semester,
                    instructor, department, description, credits, prerequisites,
                    learning_outcomes, assessment_methods, created_date, modified_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                course_info.course_id,
                course_info.course_name,
                course_info.course_code,
                course_info.academic_year,
                course_info.semester,
                course_info.instructor,
                course_info.department,
                course_info.description,
                course_info.credits,
                json.dumps(course_info.prerequisites),
                json.dumps(course_info.learning_outcomes),
                json.dumps(course_info.assessment_methods),
                course_info.created_date.isoformat(),
                course_info.modified_date.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Course created: {course_info.course_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating course {course_info.course_id}: {str(e)}")
            return False

    def get_course(self, course_id: str) -> Optional[CourseInfo]:
        """Retrieve course information"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM courses WHERE course_id = ?", (course_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return CourseInfo(
                    course_id=row[0],
                    course_name=row[1],
                    course_code=row[2],
                    academic_year=row[3],
                    semester=row[4],
                    instructor=row[5],
                    department=row[6],
                    description=row[7],
                    credits=row[8],
                    prerequisites=json.loads(row[9]) if row[9] else [],
                    learning_outcomes=json.loads(row[10]) if row[10] else [],
                    assessment_methods=json.loads(row[11]) if row[11] else [],
                    created_date=datetime.fromisoformat(row[12]),
                    modified_date=datetime.fromisoformat(row[13])
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving course {course_id}: {str(e)}")
            return None

    def list_courses(self) -> List[CourseInfo]:
        """List all courses in the system"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM courses ORDER BY course_name")
            rows = cursor.fetchall()
            conn.close()
            
            courses = []
            for row in rows:
                course = CourseInfo(
                    course_id=row[0],
                    course_name=row[1],
                    course_code=row[2],
                    academic_year=row[3],
                    semester=row[4],
                    instructor=row[5],
                    department=row[6],
                    description=row[7],
                    credits=row[8],
                    prerequisites=json.loads(row[9]) if row[9] else [],
                    learning_outcomes=json.loads(row[10]) if row[10] else [],
                    assessment_methods=json.loads(row[11]) if row[11] else [],
                    created_date=datetime.fromisoformat(row[12]),
                    modified_date=datetime.fromisoformat(row[13])
                )
                courses.append(course)
            
            return courses
            
        except Exception as e:
            self.logger.error(f"Error listing courses: {str(e)}")
            return []

    # Content Management Methods
    def add_content(self, content_item: ContentItem, file_data: bytes = None) -> bool:
        """Add new content to the system"""
        try:
            # Store file if provided
            if file_data:
                storage_dir = self.storage_path / content_item.course_id / content_item.content_type.value
                storage_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = storage_dir / f"{content_item.content_id}_{content_item.original_filename}"
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                
                content_item.file_path = str(file_path)
                content_item.file_size = len(file_data)
                content_item.file_hash = hashlib.sha256(file_data).hexdigest()
            
            # Store in database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO content_items (
                    content_id, title, content_type, course_id, file_path,
                    original_filename, file_size, file_hash, mime_type,
                    week_number, chapter_number, section_number, tags, keywords,
                    description, author, source_url, language, processing_status,
                    quality_score, created_date, modified_date, accessed_date,
                    access_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content_item.content_id,
                content_item.title,
                content_item.content_type.value,
                content_item.course_id,
                content_item.file_path,
                content_item.original_filename,
                content_item.file_size,
                content_item.file_hash,
                content_item.mime_type,
                content_item.week_number,
                content_item.chapter_number,
                content_item.section_number,
                json.dumps(content_item.tags),
                json.dumps(content_item.keywords),
                content_item.description,
                content_item.author,
                content_item.source_url,
                content_item.language,
                content_item.processing_status.value,
                content_item.quality_score,
                content_item.created_date.isoformat(),
                content_item.modified_date.isoformat(),
                content_item.accessed_date.isoformat() if content_item.accessed_date else None,
                content_item.access_count,
                json.dumps(content_item.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            self._content_cache[content_item.content_id] = content_item
            
            # Create initial version
            if file_data:
                self.create_version(
                    content_item.content_id,
                    ChangeType.CREATE,
                    "Initial version",
                    content_item.author or "system"
                )
            
            self.logger.info(f"Content added: {content_item.content_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding content {content_item.content_id}: {str(e)}")
            return False

    def get_content(self, content_id: str) -> Optional[ContentItem]:
        """Retrieve content item by ID"""
        # Check cache first
        if content_id in self._content_cache:
            content_item = self._content_cache[content_id]
            # Update access tracking
            content_item.access_count += 1
            content_item.accessed_date = datetime.now()
            self._update_content_access(content_id, content_item.access_count, content_item.accessed_date)
            return content_item
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM content_items WHERE content_id = ?", (content_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                content_item = ContentItem(
                    content_id=row[0],
                    title=row[1],
                    content_type=ContentType(row[2]),
                    course_id=row[3],
                    file_path=row[4],
                    original_filename=row[5],
                    file_size=row[6],
                    file_hash=row[7],
                    mime_type=row[8],
                    week_number=row[9],
                    chapter_number=row[10],
                    section_number=row[11],
                    tags=json.loads(row[12]) if row[12] else [],
                    keywords=json.loads(row[13]) if row[13] else [],
                    description=row[14],
                    author=row[15],
                    source_url=row[16],
                    language=row[17],
                    processing_status=ProcessingStatus(row[18]),
                    quality_score=row[19],
                    created_date=datetime.fromisoformat(row[20]),
                    modified_date=datetime.fromisoformat(row[21]),
                    accessed_date=datetime.fromisoformat(row[22]) if row[22] else None,
                    access_count=row[23],
                    metadata=json.loads(row[24]) if row[24] else {}
                )
                
                # Update cache and access tracking
                self._content_cache[content_id] = content_item
                content_item.access_count += 1
                content_item.accessed_date = datetime.now()
                self._update_content_access(content_id, content_item.access_count, content_item.accessed_date)
                
                return content_item
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving content {content_id}: {str(e)}")
            return None

    def _update_content_access(self, content_id: str, access_count: int, accessed_date: datetime) -> None:
        """Update content access tracking"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE content_items 
                SET access_count = ?, accessed_date = ? 
                WHERE content_id = ?
            """, (access_count, accessed_date.isoformat(), content_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating access tracking for {content_id}: {str(e)}")

    def search_content(self, query: str, course_id: str = None, content_type: ContentType = None,
                      week_number: int = None, limit: int = 50) -> List[Tuple[ContentItem, float]]:
        """Search content with relevance scoring"""
        try:
            # Normalize query
            query_terms = [term.lower().strip() for term in query.split() if term.strip()]
            if not query_terms:
                return []
            
            # Build search query
            conditions = []
            params = []
            
            # Add filters
            if course_id:
                conditions.append("ci.course_id = ?")
                params.append(course_id)
            
            if content_type:
                conditions.append("ci.content_type = ?")
                params.append(content_type.value)
            
            if week_number is not None:
                conditions.append("ci.week_number = ?")
                params.append(week_number)
            
            # Search in multiple fields
            search_conditions = []
            for term in query_terms:
                term_condition = "(ci.title LIKE ? OR ci.description LIKE ? OR ci.keywords LIKE ? OR si.term LIKE ?)"
                search_conditions.append(term_condition)
                params.extend([f"%{term}%", f"%{term}%", f"%{term}%", f"%{term}%"])
            
            if search_conditions:
                conditions.append(f"({' OR '.join(search_conditions)})")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Main search query with relevance scoring
            cursor.execute(f"""
                SELECT DISTINCT ci.*, 
                       COALESCE(si.frequency, 0) as search_relevance,
                       CASE 
                           WHEN ci.title LIKE ? THEN 10 
                           WHEN ci.description LIKE ? THEN 5 
                           ELSE 1 
                       END as title_boost
                FROM content_items ci
                LEFT JOIN search_index si ON ci.content_id = si.content_id
                WHERE {where_clause}
                ORDER BY (search_relevance + title_boost) DESC, ci.modified_date DESC
                LIMIT ?
            """, [f"%{' '.join(query_terms)}%", f"%{' '.join(query_terms)}%"] + params + [limit])
            
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                content_item = ContentItem(
                    content_id=row[0],
                    title=row[1],
                    content_type=ContentType(row[2]),
                    course_id=row[3],
                    file_path=row[4],
                    original_filename=row[5],
                    file_size=row[6],
                    file_hash=row[7],
                    mime_type=row[8],
                    week_number=row[9],
                    chapter_number=row[10],
                    section_number=row[11],
                    tags=json.loads(row[12]) if row[12] else [],
                    keywords=json.loads(row[13]) if row[13] else [],
                    description=row[14],
                    author=row[15],
                    source_url=row[16],
                    language=row[17],
                    processing_status=ProcessingStatus(row[18]),
                    quality_score=row[19],
                    created_date=datetime.fromisoformat(row[20]),
                    modified_date=datetime.fromisoformat(row[21]),
                    accessed_date=datetime.fromisoformat(row[22]) if row[22] else None,
                    access_count=row[23],
                    metadata=json.loads(row[24]) if row[24] else {}
                )
                
                # Calculate relevance score
                relevance_score = float(row[25]) + float(row[26])  # search_relevance + title_boost
                
                results.append((content_item, relevance_score))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching content: {str(e)}")
            return []

    def list_content(self, course_id: str = None, content_type: ContentType = None,
                    week_number: int = None, limit: int = 100) -> List[ContentItem]:
        """List content with optional filters"""
        try:
            conditions = []
            params = []
            
            if course_id:
                conditions.append("course_id = ?")
                params.append(course_id)
            
            if content_type:
                conditions.append("content_type = ?")
                params.append(content_type.value)
            
            if week_number is not None:
                conditions.append("week_number = ?")
                params.append(week_number)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT * FROM content_items 
                WHERE {where_clause}
                ORDER BY week_number, chapter_number, section_number, modified_date DESC
                LIMIT ?
            """, params + [limit])
            
            rows = cursor.fetchall()
            conn.close()
            
            content_items = []
            for row in rows:
                content_item = ContentItem(
                    content_id=row[0],
                    title=row[1],
                    content_type=ContentType(row[2]),
                    course_id=row[3],
                    file_path=row[4],
                    original_filename=row[5],
                    file_size=row[6],
                    file_hash=row[7],
                    mime_type=row[8],
                    week_number=row[9],
                    chapter_number=row[10],
                    section_number=row[11],
                    tags=json.loads(row[12]) if row[12] else [],
                    keywords=json.loads(row[13]) if row[13] else [],
                    description=row[14],
                    author=row[15],
                    source_url=row[16],
                    language=row[17],
                    processing_status=ProcessingStatus(row[18]),
                    quality_score=row[19],
                    created_date=datetime.fromisoformat(row[20]),
                    modified_date=datetime.fromisoformat(row[21]),
                    accessed_date=datetime.fromisoformat(row[22]) if row[22] else None,
                    access_count=row[23],
                    metadata=json.loads(row[24]) if row[24] else {}
                )
                content_items.append(content_item)
            
            return content_items
            
        except Exception as e:
            self.logger.error(f"Error listing content: {str(e)}")
            return []

    # Content Relationship Methods
    def add_relationship(self, relationship: ContentRelationship) -> bool:
        """Add content relationship"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO content_relationships (
                    relationship_id, source_content_id, target_content_id,
                    relationship_type, strength, description, auto_detected,
                    created_date, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relationship.relationship_id,
                relationship.source_content_id,
                relationship.target_content_id,
                relationship.relationship_type,
                relationship.strength,
                relationship.description,
                int(relationship.auto_detected),
                relationship.created_date.isoformat(),
                json.dumps(relationship.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            self._relationship_cache[relationship.source_content_id].append(relationship)
            
            self.logger.info(f"Relationship added: {relationship.relationship_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding relationship {relationship.relationship_id}: {str(e)}")
            return False

    def get_relationships(self, content_id: str, relationship_type: str = None) -> List[ContentRelationship]:
        """Get relationships for a content item"""
        try:
            # Check cache first
            cached_relationships = self._relationship_cache.get(content_id, [])
            if cached_relationships and not relationship_type:
                return cached_relationships
            
            conditions = ["(source_content_id = ? OR target_content_id = ?)"]
            params = [content_id, content_id]
            
            if relationship_type:
                conditions.append("relationship_type = ?")
                params.append(relationship_type)
            
            where_clause = " AND ".join(conditions)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT * FROM content_relationships 
                WHERE {where_clause}
                ORDER BY strength DESC, created_date DESC
            """, params)
            
            rows = cursor.fetchall()
            conn.close()
            
            relationships = []
            for row in rows:
                relationship = ContentRelationship(
                    relationship_id=row[0],
                    source_content_id=row[1],
                    target_content_id=row[2],
                    relationship_type=row[3],
                    strength=row[4],
                    description=row[5],
                    auto_detected=bool(row[6]),
                    created_date=datetime.fromisoformat(row[7]),
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                relationships.append(relationship)
            
            # Update cache
            if not relationship_type:
                self._relationship_cache[content_id] = relationships
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error getting relationships for {content_id}: {str(e)}")
            return []

    def detect_relationships(self, content_id: str, auto_create: bool = True) -> List[ContentRelationship]:
        """Automatically detect relationships based on content analysis"""
        try:
            content_item = self.get_content(content_id)
            if not content_item:
                return []
            
            # Get all content from the same course
            course_content = self.list_content(course_id=content_item.course_id)
            
            detected_relationships = []
            
            for other_content in course_content:
                if other_content.content_id == content_id:
                    continue
                
                # Calculate relationship strength
                strength = self._calculate_relationship_strength(content_item, other_content)
                
                if strength > 0.3:  # Threshold for significant relationship
                    relationship_type = self._determine_relationship_type(content_item, other_content)
                    
                    relationship = ContentRelationship(
                        relationship_id=str(uuid.uuid4()),
                        source_content_id=content_id,
                        target_content_id=other_content.content_id,
                        relationship_type=relationship_type,
                        strength=strength,
                        description=f"Auto-detected {relationship_type} relationship",
                        auto_detected=True,
                        metadata={
                            "detection_method": "content_analysis",
                            "common_keywords": self._find_common_keywords(content_item, other_content)
                        }
                    )
                    
                    detected_relationships.append(relationship)
                    
                    if auto_create:
                        self.add_relationship(relationship)
            
            return detected_relationships
            
        except Exception as e:
            self.logger.error(f"Error detecting relationships for {content_id}: {str(e)}")
            return []

    def _calculate_relationship_strength(self, content1: ContentItem, content2: ContentItem) -> float:
        """Calculate relationship strength between two content items"""
        strength = 0.0
        
        # Week proximity
        if content1.week_number and content2.week_number:
            week_diff = abs(content1.week_number - content2.week_number)
            if week_diff == 0:
                strength += 0.3
            elif week_diff == 1:
                strength += 0.2
            elif week_diff <= 3:
                strength += 0.1
        
        # Content type relationships
        type_relationships = {
            (ContentType.LECTURE, ContentType.TRANSCRIPT): 0.8,
            (ContentType.LECTURE, ContentType.NOTES): 0.6,
            (ContentType.TEXTBOOK, ContentType.NOTES): 0.5,
            (ContentType.ASSIGNMENT, ContentType.TUTORIAL): 0.7
        }
        
        type_pair = (content1.content_type, content2.content_type)
        if type_pair in type_relationships:
            strength += type_relationships[type_pair]
        elif type_pair[::-1] in type_relationships:
            strength += type_relationships[type_pair[::-1]]
        
        # Keyword overlap
        common_keywords = set(content1.keywords).intersection(set(content2.keywords))
        if common_keywords:
            keyword_strength = min(0.4, len(common_keywords) * 0.1)
            strength += keyword_strength
        
        # Tag overlap
        common_tags = set(content1.tags).intersection(set(content2.tags))
        if common_tags:
            tag_strength = min(0.3, len(common_tags) * 0.1)
            strength += tag_strength
        
        return min(1.0, strength)

    def _determine_relationship_type(self, content1: ContentItem, content2: ContentItem) -> str:
        """Determine the type of relationship between content items"""
        # Same week content
        if content1.week_number == content2.week_number:
            if content1.content_type == ContentType.LECTURE and content2.content_type == ContentType.TRANSCRIPT:
                return "transcript_of"
            elif content1.content_type == ContentType.LECTURE and content2.content_type == ContentType.NOTES:
                return "notes_for"
            elif content1.content_type == ContentType.TEXTBOOK and content2.content_type == ContentType.LECTURE:
                return "supports"
            else:
                return "related_to"
        
        # Sequential content
        if (content1.week_number and content2.week_number and 
            abs(content1.week_number - content2.week_number) == 1):
            return "follows" if content1.week_number < content2.week_number else "precedes"
        
        # Default relationship
        return "related_to"

    def _find_common_keywords(self, content1: ContentItem, content2: ContentItem) -> List[str]:
        """Find common keywords between content items"""
        return list(set(content1.keywords).intersection(set(content2.keywords)))

    # Processing History Methods
    def record_processing(self, record: ProcessingRecord) -> bool:
        """Record processing history"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_history (
                    record_id, content_id, operation, agent_id, input_data,
                    output_data, processing_time, success, error_message,
                    quality_score, created_date, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.content_id,
                record.operation,
                record.agent_id,
                json.dumps(record.input_data),
                json.dumps(record.output_data),
                record.processing_time,
                int(record.success),
                record.error_message,
                record.quality_score,
                record.created_date.isoformat(),
                json.dumps(record.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Processing record created: {record.record_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording processing {record.record_id}: {str(e)}")
            return False

    def get_processing_history(self, content_id: str = None, agent_id: str = None,
                             operation: str = None, limit: int = 100) -> List[ProcessingRecord]:
        """Get processing history with optional filters"""
        try:
            conditions = []
            params = []
            
            if content_id:
                conditions.append("content_id = ?")
                params.append(content_id)
            
            if agent_id:
                conditions.append("agent_id = ?")
                params.append(agent_id)
            
            if operation:
                conditions.append("operation = ?")
                params.append(operation)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT * FROM processing_history 
                WHERE {where_clause}
                ORDER BY created_date DESC
                LIMIT ?
            """, params + [limit])
            
            rows = cursor.fetchall()
            conn.close()
            
            records = []
            for row in rows:
                record = ProcessingRecord(
                    record_id=row[0],
                    content_id=row[1],
                    operation=row[2],
                    agent_id=row[3],
                    input_data=json.loads(row[4]),
                    output_data=json.loads(row[5]),
                    processing_time=row[6],
                    success=bool(row[7]),
                    error_message=row[8],
                    quality_score=row[9],
                    created_date=datetime.fromisoformat(row[10]),
                    metadata=json.loads(row[11]) if row[11] else {}
                )
                records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"Error getting processing history: {str(e)}")
            return []

    # Version Control Methods
    def create_version(self, content_id: str, change_type: ChangeType, 
                      change_description: str, author: str) -> Optional[str]:
        """Create a new version of content"""
        try:
            content_item = self.get_content(content_id)
            if not content_item:
                return None
            
            # Get current version number
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT MAX(version_number) FROM content_versions 
                WHERE content_id = ?
            """, (content_id,))
            
            result = cursor.fetchone()
            current_version = result[0] if result[0] else 0
            new_version_number = current_version + 1
            
            # Create version record
            version_id = str(uuid.uuid4())
            
            # Copy current file to versions directory
            if os.path.exists(content_item.file_path):
                version_dir = self.versions_path / content_id
                version_dir.mkdir(parents=True, exist_ok=True)
                
                version_file_path = version_dir / f"v{new_version_number}_{content_item.original_filename}"
                shutil.copy2(content_item.file_path, version_file_path)
            else:
                version_file_path = content_item.file_path
            
            cursor.execute("""
                INSERT INTO content_versions (
                    version_id, content_id, version_number, change_type,
                    file_path, file_hash, change_description, author,
                    parent_version_id, created_date, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_id,
                content_id,
                new_version_number,
                change_type.value,
                str(version_file_path),
                content_item.file_hash,
                change_description,
                author,
                None,  # Could link to parent version
                datetime.now().isoformat(),
                json.dumps({})
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Version created: {version_id} for content {content_id}")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error creating version for {content_id}: {str(e)}")
            return None

    def get_versions(self, content_id: str) -> List[ContentVersion]:
        """Get all versions of content"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM content_versions 
                WHERE content_id = ?
                ORDER BY version_number DESC
            """, (content_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            versions = []
            for row in rows:
                version = ContentVersion(
                    version_id=row[0],
                    content_id=row[1],
                    version_number=row[2],
                    change_type=ChangeType(row[3]),
                    file_path=row[4],
                    file_hash=row[5],
                    change_description=row[6],
                    author=row[7],
                    parent_version_id=row[8],
                    created_date=datetime.fromisoformat(row[9]),
                    metadata=json.loads(row[10]) if row[10] else {}
                )
                versions.append(version)
            
            return versions
            
        except Exception as e:
            self.logger.error(f"Error getting versions for {content_id}: {str(e)}")
            return []

    # Search Index Methods
    def _load_search_index(self) -> None:
        """Load search index from database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM search_index")
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                index_entry = SearchIndex(
                    content_id=row[0],
                    term=row[1],
                    frequency=row[2],
                    position=row[3],
                    context=row[4],
                    content_type=ContentType(row[5]),
                    week_number=row[6],
                    chapter_number=row[7]
                )
                self._search_index[row[1]].append(index_entry)
            
        except Exception as e:
            self.logger.error(f"Error loading search index: {str(e)}")

    def build_search_index(self, content_id: str = None) -> bool:
        """Build or rebuild search index"""
        try:
            # Get content to index
            if content_id:
                content_items = [self.get_content(content_id)]
                content_items = [item for item in content_items if item]
            else:
                content_items = self.list_content()
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            for content_item in content_items:
                # Clear existing index entries for this content
                cursor.execute("DELETE FROM search_index WHERE content_id = ?", (content_item.content_id,))
                
                # Extract text for indexing
                text_content = self._extract_text_for_indexing(content_item)
                
                # Tokenize and index
                term_frequencies = self._tokenize_and_count(text_content)
                
                for term, frequency in term_frequencies.items():
                    # Find term positions
                    positions = self._find_term_positions(text_content, term)
                    
                    for position in positions:
                        context = self._extract_context(text_content, position, term)
                        
                        cursor.execute("""
                            INSERT INTO search_index (
                                content_id, term, frequency, position, context,
                                content_type, week_number, chapter_number
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            content_item.content_id,
                            term,
                            frequency,
                            position,
                            context,
                            content_item.content_type.value,
                            content_item.week_number,
                            content_item.chapter_number
                        ))
            
            conn.commit()
            conn.close()
            
            # Reload index
            self._search_index.clear()
            self._load_search_index()
            
            self.logger.info(f"Search index built for {len(content_items)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building search index: {str(e)}")
            return False

    def _extract_text_for_indexing(self, content_item: ContentItem) -> str:
        """Extract text content for search indexing"""
        text_parts = []
        
        # Add metadata text
        text_parts.append(content_item.title)
        if content_item.description:
            text_parts.append(content_item.description)
        text_parts.extend(content_item.keywords)
        text_parts.extend(content_item.tags)
        
        # Read file content if it's a text-based file
        if content_item.file_path and os.path.exists(content_item.file_path):
            try:
                if content_item.mime_type.startswith('text/') or content_item.file_path.endswith('.md'):
                    with open(content_item.file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                        text_parts.append(file_content)
            except Exception as e:
                self.logger.warning(f"Could not read file for indexing {content_item.file_path}: {str(e)}")
        
        return ' '.join(text_parts)

    def _tokenize_and_count(self, text: str) -> Dict[str, int]:
        """Tokenize text and count term frequencies"""
        # Simple tokenization - could be enhanced with NLP libraries
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them'
        }
        
        filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Count frequencies
        term_frequencies = {}
        for word in filtered_words:
            term_frequencies[word] = term_frequencies.get(word, 0) + 1
        
        return term_frequencies

    def _find_term_positions(self, text: str, term: str) -> List[int]:
        """Find all positions of a term in text"""
        positions = []
        text_lower = text.lower()
        term_lower = term.lower()
        start = 0
        
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions

    def _extract_context(self, text: str, position: int, term: str, context_size: int = 50) -> str:
        """Extract context around a term position"""
        start = max(0, position - context_size)
        end = min(len(text), position + len(term) + context_size)
        return text[start:end].strip()

    # Analytics Methods
    def generate_analytics_report(self, course_id: str = None, 
                                start_date: datetime = None, 
                                end_date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_date": datetime.now().isoformat(),
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "course_id": course_id,
                "content_statistics": self._get_content_statistics(course_id),
                "processing_statistics": self._get_processing_statistics(course_id, start_date, end_date),
                "quality_statistics": self._get_quality_statistics(course_id),
                "usage_statistics": self._get_usage_statistics(course_id, start_date, end_date),
                "relationship_statistics": self._get_relationship_statistics(course_id),
                "storage_statistics": self._get_storage_statistics(course_id)
            }
            
            # Save report
            report_file = self.reports_path / f"analytics_report_{report['report_id']}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Analytics report generated: {report['report_id']}")
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating analytics report: {str(e)}")
            return {}

    def _get_content_statistics(self, course_id: str = None) -> Dict[str, Any]:
        """Get content statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Base query conditions
            where_clause = "WHERE 1=1"
            params = []
            
            if course_id:
                where_clause += " AND course_id = ?"
                params.append(course_id)
            
            # Total content count
            cursor.execute(f"SELECT COUNT(*) FROM content_items {where_clause}", params)
            total_content = cursor.fetchone()[0]
            
            # Content by type
            cursor.execute(f"""
                SELECT content_type, COUNT(*) 
                FROM content_items {where_clause}
                GROUP BY content_type
            """, params)
            content_by_type = dict(cursor.fetchall())
            
            # Content by week
            cursor.execute(f"""
                SELECT week_number, COUNT(*) 
                FROM content_items {where_clause} AND week_number IS NOT NULL
                GROUP BY week_number
                ORDER BY week_number
            """, params)
            content_by_week = dict(cursor.fetchall())
            
            # Processing status distribution
            cursor.execute(f"""
                SELECT processing_status, COUNT(*) 
                FROM content_items {where_clause}
                GROUP BY processing_status
            """, params)
            status_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_content": total_content,
                "content_by_type": content_by_type,
                "content_by_week": content_by_week,
                "status_distribution": status_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Error getting content statistics: {str(e)}")
            return {}

    def _get_processing_statistics(self, course_id: str = None, 
                                 start_date: datetime = None, 
                                 end_date: datetime = None) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build where clause
            conditions = []
            params = []
            
            if course_id:
                conditions.append("ci.course_id = ?")
                params.append(course_id)
            
            if start_date:
                conditions.append("ph.created_date >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("ph.created_date <= ?")
                params.append(end_date.isoformat())
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Total processing operations
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM processing_history ph
                JOIN content_items ci ON ph.content_id = ci.content_id
                WHERE {where_clause}
            """, params)
            total_operations = cursor.fetchone()[0]
            
            # Operations by agent
            cursor.execute(f"""
                SELECT ph.agent_id, COUNT(*) 
                FROM processing_history ph
                JOIN content_items ci ON ph.content_id = ci.content_id
                WHERE {where_clause}
                GROUP BY ph.agent_id
            """, params)
            operations_by_agent = dict(cursor.fetchall())
            
            # Success rate
            cursor.execute(f"""
                SELECT 
                    SUM(CASE WHEN ph.success = 1 THEN 1 ELSE 0 END) as successful,
                    COUNT(*) as total
                FROM processing_history ph
                JOIN content_items ci ON ph.content_id = ci.content_id
                WHERE {where_clause}
            """, params)
            result = cursor.fetchone()
            success_rate = (result[0] / result[1]) if result[1] > 0 else 0
            
            # Average processing time
            cursor.execute(f"""
                SELECT AVG(ph.processing_time) 
                FROM processing_history ph
                JOIN content_items ci ON ph.content_id = ci.content_id
                WHERE {where_clause} AND ph.success = 1
            """, params)
            avg_processing_time = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_operations": total_operations,
                "operations_by_agent": operations_by_agent,
                "success_rate": success_rate,
                "average_processing_time": avg_processing_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting processing statistics: {str(e)}")
            return {}

    def _get_quality_statistics(self, course_id: str = None) -> Dict[str, Any]:
        """Get quality statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            where_clause = "WHERE quality_score IS NOT NULL"
            params = []
            
            if course_id:
                where_clause += " AND course_id = ?"
                params.append(course_id)
            
            # Average quality score
            cursor.execute(f"""
                SELECT AVG(quality_score) 
                FROM content_items {where_clause}
            """, params)
            avg_quality = cursor.fetchone()[0] or 0
            
            # Quality distribution
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN quality_score >= 0.8 THEN 'high'
                        WHEN quality_score >= 0.6 THEN 'medium'
                        ELSE 'low'
                    END as quality_level,
                    COUNT(*)
                FROM content_items {where_clause}
                GROUP BY quality_level
            """, params)
            quality_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "average_quality_score": avg_quality,
                "quality_distribution": quality_distribution
            }
            
        except Exception as e:
            self.logger.error(f"Error getting quality statistics: {str(e)}")
            return {}

    def _get_usage_statistics(self, course_id: str = None, 
                            start_date: datetime = None, 
                            end_date: datetime = None) -> Dict[str, Any]:
        """Get usage statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            where_clause = "WHERE 1=1"
            params = []
            
            if course_id:
                where_clause += " AND course_id = ?"
                params.append(course_id)
            
            # Total access count
            cursor.execute(f"""
                SELECT SUM(access_count) 
                FROM content_items {where_clause}
            """, params)
            total_accesses = cursor.fetchone()[0] or 0
            
            # Most accessed content
            cursor.execute(f"""
                SELECT content_id, title, access_count 
                FROM content_items {where_clause}
                ORDER BY access_count DESC
                LIMIT 10
            """, params)
            most_accessed = [{
                "content_id": row[0],
                "title": row[1],
                "access_count": row[2]
            } for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "total_accesses": total_accesses,
                "most_accessed_content": most_accessed
            }
            
        except Exception as e:
            self.logger.error(f"Error getting usage statistics: {str(e)}")
            return {}

    def _get_relationship_statistics(self, course_id: str = None) -> Dict[str, Any]:
        """Get relationship statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            where_clause = "WHERE 1=1"
            params = []
            
            if course_id:
                where_clause += " AND (ci1.course_id = ? OR ci2.course_id = ?)"
                params.extend([course_id, course_id])
            
            # Total relationships
            cursor.execute(f"""
                SELECT COUNT(*) 
                FROM content_relationships cr
                JOIN content_items ci1 ON cr.source_content_id = ci1.content_id
                JOIN content_items ci2 ON cr.target_content_id = ci2.content_id
                {where_clause}
            """, params)
            total_relationships = cursor.fetchone()[0]
            
            # Relationships by type
            cursor.execute(f"""
                SELECT cr.relationship_type, COUNT(*) 
                FROM content_relationships cr
                JOIN content_items ci1 ON cr.source_content_id = ci1.content_id
                JOIN content_items ci2 ON cr.target_content_id = ci2.content_id
                {where_clause}
                GROUP BY cr.relationship_type
            """, params)
            relationships_by_type = dict(cursor.fetchall())
            
            # Auto-detected vs manual
            cursor.execute(f"""
                SELECT 
                    CASE WHEN cr.auto_detected = 1 THEN 'auto' ELSE 'manual' END as detection_type,
                    COUNT(*)
                FROM content_relationships cr
                JOIN content_items ci1 ON cr.source_content_id = ci1.content_id
                JOIN content_items ci2 ON cr.target_content_id = ci2.content_id
                {where_clause}
                GROUP BY detection_type
            """, params)
            detection_stats = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_relationships": total_relationships,
                "relationships_by_type": relationships_by_type,
                "detection_statistics": detection_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting relationship statistics: {str(e)}")
            return {}

    def _get_storage_statistics(self, course_id: str = None) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            where_clause = "WHERE 1=1"
            params = []
            
            if course_id:
                where_clause += " AND course_id = ?"
                params.append(course_id)
            
            # Total storage used
            cursor.execute(f"""
                SELECT SUM(file_size) 
                FROM content_items {where_clause}
            """, params)
            total_storage = cursor.fetchone()[0] or 0
            
            # Storage by content type
            cursor.execute(f"""
                SELECT content_type, SUM(file_size) 
                FROM content_items {where_clause}
                GROUP BY content_type
            """, params)
            storage_by_type = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_storage_bytes": total_storage,
                "total_storage_mb": total_storage / (1024 * 1024),
                "storage_by_type": storage_by_type
            }
            
        except Exception as e:
            self.logger.error(f"Error getting storage statistics: {str(e)}")
            return {}

    # Integration with existing systems
    def consolidate_content(self, search_paths: List[str], course_id: str) -> ConsolidationResult:
        """Integrate with consolidation agent to import content"""
        try:
            # Use consolidation agent to process content
            output_path = self.storage_path / course_id / "consolidated"
            output_path.mkdir(parents=True, exist_ok=True)
            
            result = self.consolidation_agent.consolidate_workflow(search_paths, str(output_path))
            
            if result.success:
                # Import consolidated content into CMS
                for mapping in result.processed_files:
                    # Create content item from mapping
                    content_id = str(uuid.uuid4())
                    
                    # Determine content type
                    content_type = ContentType.UNKNOWN
                    if mapping.content_type:
                        try:
                            content_type = ContentType(mapping.content_type)
                        except ValueError:
                            content_type = ContentType.UNKNOWN
                    
                    # Read file to get size and hash
                    file_size = 0
                    file_hash = ""
                    if os.path.exists(mapping.source_path):
                        with open(mapping.source_path, 'rb') as f:
                            file_data = f.read()
                            file_size = len(file_data)
                            file_hash = hashlib.sha256(file_data).hexdigest()
                    
                    content_item = ContentItem(
                        content_id=content_id,
                        title=os.path.splitext(os.path.basename(mapping.source_path))[0],
                        content_type=content_type,
                        course_id=course_id,
                        file_path=mapping.source_path,
                        original_filename=os.path.basename(mapping.source_path),
                        file_size=file_size,
                        file_hash=file_hash,
                        mime_type="text/markdown" if mapping.source_path.endswith('.md') else "application/octet-stream",
                        week_number=mapping.week_number,
                        processing_status=ProcessingStatus.COMPLETED,
                        metadata=mapping.metadata or {}
                    )
                    
                    # Add to CMS
                    self.add_content(content_item)
                    
                    # Build search index for this content
                    self.build_search_index(content_id)
                    
                    # Detect relationships
                    self.detect_relationships(content_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error consolidating content: {str(e)}")
            return ConsolidationResult(
                success=False,
                processed_files=[],
                skipped_files=[],
                errors=[{"error": str(e)}],
                consolidation_report={},
                unified_structure={}
            )

    def quality_assessment(self, content_id: str) -> Optional[QualityEvaluation]:
        """Integrate with quality manager to assess content"""
        try:
            content_item = self.get_content(content_id)
            if not content_item:
                return None
            
            # Read content for quality assessment
            content_text = ""
            if os.path.exists(content_item.file_path):
                try:
                    with open(content_item.file_path, 'r', encoding='utf-8') as f:
                        content_text = f.read()
                except:
                    content_text = content_item.title + " " + (content_item.description or "")
            
            # Perform quality assessment
            evaluation = self.quality_manager.evaluate_content(content_text, content_item.content_type.value)
            
            # Update content item with quality score
            content_item.quality_score = evaluation.quality_score
            self._update_content_quality(content_id, evaluation.quality_score)
            
            # Record processing history
            processing_record = ProcessingRecord(
                record_id=str(uuid.uuid4()),
                content_id=content_id,
                operation="quality_assessment",
                agent_id="quality_manager",
                input_data={"content_length": len(content_text)},
                output_data={
                    "quality_score": evaluation.quality_score,
                    "approved": evaluation.approved,
                    "feedback_count": len(evaluation.feedback)
                },
                processing_time=0.0,  # Would need to measure actual time
                success=True,
                quality_score=evaluation.quality_score
            )
            
            self.record_processing(processing_record)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error performing quality assessment for {content_id}: {str(e)}")
            return None

    def _update_content_quality(self, content_id: str, quality_score: float) -> None:
        """Update content quality score in database"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE content_items 
                SET quality_score = ?, modified_date = ? 
                WHERE content_id = ?
            """, (quality_score, datetime.now().isoformat(), content_id))
            
            conn.commit()
            conn.close()
            
            # Update cache
            if content_id in self._content_cache:
                self._content_cache[content_id].quality_score = quality_score
                self._content_cache[content_id].modified_date = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating content quality for {content_id}: {str(e)}")

    # Validation methods (required by BaseAgent)
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data"""
        if isinstance(input_data, dict):
            return True
        return False

    def validate_output(self, output_data: Any) -> bool:
        """Validate output data"""
        return True

    def check_quality(self, content: Any) -> float:
        """Check quality of CMS operations"""
        # Return a basic quality score based on system state
        try:
            # Check database connectivity
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM content_items")
            content_count = cursor.fetchone()[0]
            conn.close()
            
            # Basic quality assessment
            if content_count > 0:
                return 0.9
            else:
                return 0.5
                
        except Exception:
            return 0.1

    def shutdown(self) -> None:
        """Gracefully shutdown the CMS"""
        try:
            # Clear caches
            self._content_cache.clear()
            self._relationship_cache.clear()
            self._search_index.clear()
            
            # Close any open connections
            self.logger.info("Content Management System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during CMS shutdown: {str(e)}")
        
        super().shutdown()


def main():
    """Main entry point for testing the CMS"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Content Management System")
    parser.add_argument("--base-path", required=True, help="Base path for CMS")
    parser.add_argument("--init", action="store_true", help="Initialize CMS")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    
    args = parser.parse_args()
    
    # Initialize CMS
    cms = ContentManagementSystem(args.base_path)
    
    if args.init:
        print("CMS initialized successfully")
    
    if args.test:
        # Run basic tests
        print("Running basic tests...")
        
        # Test course creation
        course = CourseInfo(
            course_id="test_course_1",
            course_name="Test Course",
            course_code="TEST101",
            academic_year="2023",
            semester="Fall",
            instructor="Test Instructor",
            department="Computer Science"
        )
        
        success = cms.create_course(course)
        print(f"Course creation: {'SUCCESS' if success else 'FAILED'}")
        
        # Test course retrieval
        retrieved_course = cms.get_course("test_course_1")
        print(f"Course retrieval: {'SUCCESS' if retrieved_course else 'FAILED'}")
        
        # Generate analytics report
        report = cms.generate_analytics_report()
        print(f"Analytics report: {'SUCCESS' if report else 'FAILED'}")
        
        print("Basic tests completed")
    
    cms.shutdown()


if __name__ == "__main__":
    main()