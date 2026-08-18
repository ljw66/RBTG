"""
Document Metadata Database Management

- Uses SQLite to store document metadata for precise querying and filtering
- Works together with the Milvus vector database to enable a hybrid storage solution
"""

import sqlite3
import json
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class DocMetadataDB:
    """Document Metadata Database Manager (SQLite)"""

    def __init__(self, db_path: str = None, project_root: str = None):
        """
        :param db_path: Path to the database file (relative or absolute)
                        - None: Use the default path (db/ folder under the project root)
                        - Relative path: Relative to the current working directory
                        - Absolute path: Full path, e.g. "/path/to/doc_metadata.db"
        :param project_root: Project root directory path (optional; defaults to current working directory)
        """
        if db_path is None:
            # If project root is not provided, use the current working directory
            if project_root is None:
                project_root = os.getcwd()

            # Create db folder if it does not exist
            db_dir = os.path.join(project_root, 'db')
            os.makedirs(db_dir, exist_ok=True)

            # Default database file path
            db_path = os.path.join(db_dir, 'doc_metadata.db')

        # Convert relative path to absolute path
        if not os.path.isabs(db_path):
            self.db_path = os.path.abspath(db_path)
        else:
            self.db_path = db_path

        # Ensure the database directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SQLite database file path: {self.db_path}")
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Document table: stores basic document information and metadata
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,  -- Unique document identifier (can match Milvus ID)
                milvus_id TEXT,  -- Vector ID in Milvus
                collection_name TEXT NOT NULL,  -- Milvus collection name

                -- Document content (optional; large content can be stored by reference instead)
                content TEXT,
                content_hash TEXT,  -- Content hash for deduplication

                -- Metadata fields
                source TEXT,  -- Document source, e.g. "Requirements Specification"
                title TEXT,
                title_no TEXT,
                level INTEGER,

                -- Chunking information
                chunk_index INTEGER,
                total_chunks INTEGER,
                is_chunked BOOLEAN DEFAULT 0,

                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Create indexes to improve query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON documents(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection ON documents(collection_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON documents(title)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_milvus_id ON documents(milvus_id)')

            conn.commit()
            self.logger.info(f"Database tables created successfully: {self.db_path}")

    def insert_document(self, doc_id: str, milvus_id: str, collection_name: str,
                        content: str, metadata: Dict[str, Any]) -> bool:
        """
        Insert document metadata

        :param doc_id: Unique document identifier
        :param milvus_id: Vector ID in Milvus
        :param collection_name: Milvus collection name
        :param content: Document content
        :param metadata: Metadata dictionary
        :return: Whether insertion succeeded
        """
        try:
            import hashlib
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                INSERT OR REPLACE INTO documents (
                    doc_id, milvus_id, collection_name, content, content_hash,
                    source, title, title_no, level,
                    chunk_index, total_chunks, is_chunked, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    doc_id,
                    milvus_id,
                    collection_name,
                    content,
                    content_hash,
                    metadata.get('source', ''),
                    metadata.get('title', ''),
                    metadata.get('title_no', ''),
                    metadata.get('level', 1),
                    metadata.get('chunk_index'),
                    metadata.get('total_chunks'),
                    1 if metadata.get('is_chunked', False) else 0
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Failed to insert document metadata: {e}")
            return False

    def query_by_source(self, source_value: str, collection_name: Optional[str] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query documents by the source field

        :param source_value: Value of the source field
        :param collection_name: Collection name (optional filter)
        :param limit: Maximum number of returned results
        :return: List of documents
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Return results as dictionaries
                cursor = conn.cursor()

                if collection_name:
                    cursor.execute('''
                    SELECT * FROM documents
                    WHERE source = ? AND collection_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''', (source_value, collection_name, limit))
                else:
                    cursor.execute('''
                    SELECT * FROM documents
                    WHERE source = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''', (source_value, limit))

                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                self.logger.info(f"Found {len(results)} records with source = '{source_value}'")
                return results
        except Exception as e:
            self.logger.error(f"Failed to query document metadata: {e}")
            return []

    def query_by_title(self, title: str, collection_name: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Query documents by title"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if collection_name:
                    cursor.execute('''
                    SELECT * FROM documents
                    WHERE title LIKE ? AND collection_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''', (f'%{title}%', collection_name, limit))
                else:
                    cursor.execute('''
                    SELECT * FROM documents
                    WHERE title LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    ''', (f'%{title}%', limit))

                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to query documents: {e}")
            return []

    def query_by_metadata(self, filters: Dict[str, Any],
                          collection_name: Optional[str] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query documents by multiple metadata fields

        :param filters: Filter conditions, e.g. {"source": "Requirements Spec", "level": 2}
        :param collection_name: Collection name (optional)
        :param limit: Maximum number of returned results
        :return: List of documents
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Build WHERE clause
                conditions = []
                params = []

                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    elif isinstance(value, (int, float)):
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    elif isinstance(value, list):
                        # Support IN query
                        placeholders = ','.join(['?'] * len(value))
                        conditions.append(f"{key} IN ({placeholders})")
                        params.extend(value)

                if collection_name:
                    conditions.append("collection_name = ?")
                    params.append(collection_name)

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                params.append(limit)

                query = f'''
                SELECT * FROM documents
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ?
                '''

                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to query documents: {e}")
            return []

    def get_document_by_milvus_id(self, milvus_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata by Milvus ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM documents WHERE milvus_id = ?', (milvus_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Failed to query document: {e}")
            return None

    def delete_by_collection(self, collection_name: str) -> int:
        """Delete all document metadata from the specified collection"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM documents WHERE collection_name = ?', (collection_name,))
                conn.commit()
                deleted_count = cursor.rowcount
                self.logger.info(f"Deleted {deleted_count} document metadata records")
                return deleted_count
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return 0

    def get_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if collection_name:
                    cursor.execute('''
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(DISTINCT source) as unique_sources,
                        COUNT(DISTINCT title) as unique_titles
                    FROM documents
                    WHERE collection_name = ?
                    ''', (collection_name,))
                else:
                    cursor.execute('''
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(DISTINCT source) as unique_sources,
                        COUNT(DISTINCT title) as unique_titles
                    FROM documents
                    ''')

                row = cursor.fetchone()
                return {
                    'total_docs': row[0],
                    'unique_sources': row[1],
                    'unique_titles': row[2]
                }
        except Exception as e:
            self.logger.error(f"Failed to retrieve statistics: {e}")
            return {}


# Test function
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create database instance
    db = DocMetadataDB("test_doc_metadata.db")

    # Insert test data
    db.insert_document(
        doc_id="doc_001",
        milvus_id="milvus_001",
        collection_name="doc_1211",
        content="This is test document content",
        metadata={
            "source": "Requirements Specification",
            "title": "Functional Requirements",
            "title_no": "1.1",
            "level": 2
        }
    )

    # Query test
    results = db.query_by_source("Requirements Specification", collection_name="doc_1211")
    print(f"Query results: {len(results)} records")
    for result in results:
        print(f"  - {result['title']}: {result['content'][:50]}...")

    # Get statistics
    stats = db.get_statistics("doc_1211")
    print(f"Statistics: {stats}")
