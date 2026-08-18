"""
文档元数据数据库管理
使用 SQLite 存储文档的元数据信息，用于精确查询和过滤
与 Milvus 向量数据库配合使用，实现混合存储方案
"""

import sqlite3
import json
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class DocMetadataDB:
    """文档元数据数据库管理器（SQLite）"""
    
    def __init__(self, db_path: str = None, project_root: str = None):
        """
        :param db_path: 数据库文件路径（可以是相对路径或绝对路径）
                        - None: 使用默认路径（项目根目录下的 db/ 文件夹）
                        - 相对路径：相对于当前工作目录
                        - 绝对路径：完整路径，如 "/path/to/doc_metadata.db"
        :param project_root: 项目根目录路径（可选，如果不提供会使用当前工作目录）
        """
        if db_path is None:
            # 如果没有提供项目根目录，使用当前工作目录
            if project_root is None:
                project_root = os.getcwd()
            
            # 创建 db 文件夹（如果不存在）
            db_dir = os.path.join(project_root, 'db')
            os.makedirs(db_dir, exist_ok=True)
            
            # 默认数据库文件路径
            db_path = os.path.join(db_dir, 'doc_metadata.db')
        
        # 如果是相对路径，转换为绝对路径
        if not os.path.isabs(db_path):
            self.db_path = os.path.abspath(db_path)
        else:
            self.db_path = db_path
        
        # 确保数据库文件所在目录存在
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SQLite 数据库文件路径: {self.db_path}")
        self._create_tables()
    
    def _create_tables(self):
        """创建数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 文档表：存储文档的基本信息和元数据
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,  -- 文档唯一标识（可以是 Milvus 中的 ID）
                milvus_id TEXT,  -- Milvus 中的向量 ID
                collection_name TEXT NOT NULL,  -- Milvus 集合名称
                
                -- 文档内容（可选，如果内容很大可以只存引用）
                content TEXT,
                content_hash TEXT,  -- 内容哈希，用于去重
                
                -- 元数据字段
                source TEXT,  -- 文档来源，如"需求规格说明书"
                title TEXT,
                title_no TEXT,
                level INTEGER,
                
                -- 分块信息
                chunk_index INTEGER,
                total_chunks INTEGER,
                is_chunked BOOLEAN DEFAULT 0,
                
                -- 时间戳
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON documents(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_collection ON documents(collection_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_title ON documents(title)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_milvus_id ON documents(milvus_id)')
            
            conn.commit()
            self.logger.info(f"数据库表创建完成: {self.db_path}")
    
    def insert_document(self, doc_id: str, milvus_id: str, collection_name: str,
                       content: str, metadata: Dict[str, Any]) -> bool:
        """
        插入文档元数据
        
        :param doc_id: 文档唯一标识
        :param milvus_id: Milvus 中的向量 ID
        :param collection_name: Milvus 集合名称
        :param content: 文档内容
        :param metadata: 元数据字典
        :return: 是否插入成功
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
            self.logger.error(f"插入文档元数据失败: {e}")
            return False
    
    def query_by_source(self, source_value: str, collection_name: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据 source 字段查询文档
        
        :param source_value: source 字段的值
        :param collection_name: 集合名称（可选，用于过滤）
        :param limit: 返回结果的最大数量
        :return: 文档列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # 返回字典格式
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
                self.logger.info(f"找到 {len(results)} 条 source 为 '{source_value}' 的数据")
                return results
        except Exception as e:
            self.logger.error(f"查询文档元数据失败: {e}")
            return []
    
    def query_by_title(self, title: str, collection_name: Optional[str] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """根据标题查询文档"""
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
            self.logger.error(f"查询文档失败: {e}")
            return []
    
    def query_by_metadata(self, filters: Dict[str, Any], 
                         collection_name: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据多个元数据字段查询文档
        
        :param filters: 过滤条件字典，如 {"source": "需求规格说明书", "level": 2}
        :param collection_name: 集合名称（可选）
        :param limit: 返回结果的最大数量
        :return: 文档列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 构建 WHERE 子句
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
                        # 支持 IN 查询
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
            self.logger.error(f"查询文档失败: {e}")
            return []
    
    def get_document_by_milvus_id(self, milvus_id: str) -> Optional[Dict[str, Any]]:
        """根据 Milvus ID 获取文档元数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM documents WHERE milvus_id = ?', (milvus_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"查询文档失败: {e}")
            return None
    
    def delete_by_collection(self, collection_name: str) -> int:
        """删除指定集合的所有文档元数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM documents WHERE collection_name = ?', (collection_name,))
                conn.commit()
                deleted_count = cursor.rowcount
                self.logger.info(f"删除了 {deleted_count} 条文档元数据")
                return deleted_count
        except Exception as e:
            self.logger.error(f"删除文档失败: {e}")
            return 0
    
    def get_statistics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
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
            self.logger.error(f"获取统计信息失败: {e}")
            return {}


# 测试函数
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据库实例
    db = DocMetadataDB("test_doc_metadata.db")
    
    # 插入测试数据
    db.insert_document(
        doc_id="doc_001",
        milvus_id="milvus_001",
        collection_name="doc_1211",
        content="这是测试文档内容",
        metadata={
            "source": "需求规格说明书",
            "title": "功能需求",
            "title_no": "1.1",
            "level": 2
        }
    )
    
    # 查询测试
    results = db.query_by_source("需求规格说明书", collection_name="doc_1211")
    print(f"查询结果: {len(results)} 条")
    for result in results:
        print(f"  - {result['title']}: {result['content'][:50]}...")
    
    # 获取统计信息
    stats = db.get_statistics("doc_1211")
    print(f"统计信息: {stats}")
