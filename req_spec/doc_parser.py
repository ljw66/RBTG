"""
文档解析模块
负责将各种格式的文档转换为 LangChain Document 对象，并提供分块处理功能
"""

import logging
from collections import Counter
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DictToDocumentConverter:
    """
    将文档字典转换为 LangChain Document 对象
    
    输入格式：
    {
        "章节标题": {
            "content": "章节内容",
            "title_no": "1.1",
            "level": 2,
            "source": "需求规格说明书"  # 可选
        },
        ...
    }
    """
    
    def __init__(self, default_source: str = "其他来源"):
        """
        初始化转换器
        
        :param default_source: 默认文档来源，当 doc_dict 中未指定 source 时使用
        """
        self.logger = logging.getLogger(__name__)
        self.default_source = default_source
    
    def convert(self, doc_dict: Dict[str, Dict[str, Any]]) -> List[Document]:
        """
        将文档字典转换为 Document 对象列表
        
        :param doc_dict: 文档字典
        :return: Document 对象列表
        """
        if not doc_dict:
            self.logger.warning("输入的文档字典为空")
            return []
        
        documents = []
        for title, info in doc_dict.items():
            content = info.get('content', '')
            title_no = info.get('title_no', '')
            level = info.get('level', 1)
            source = info.get('source', self.default_source)
            
            if content:  # 只处理有内容的条目
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "title_no": title_no,
                        "level": level,
                        "source": source
                    }
                )
                documents.append(doc)
        
        # 统计各来源的文档数量
        sources = [doc.metadata.get('source', '未知') for doc in documents]
        source_counts = Counter(sources)
        
        self.logger.info(f"共转换 {len(documents)} 个文档为 LangChain Document 对象")
        self.logger.info(f"文档来源统计: {dict(source_counts)}")
        
        return documents


class DocumentChunker:
    """
    文档分块处理器
    对长文本进行智能分块，保留上下文信息
    """
    
    def __init__(
        self,
        chunk_size: int = 6000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        初始化分块处理器
        
        :param chunk_size: 每个分块最大字符数（默认 6000，适配 bge-m3 的 8192 tokens 限制）
        :param chunk_overlap: 分块之间的重叠字符数（默认 200）
        :param separators: 分块分隔符优先级列表
        """
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "，", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行分块处理
        
        :param documents: 原始 Document 对象列表
        :return: 分块后的 Document 对象列表
        """
        if not documents:
            self.logger.warning("输入的文档列表为空")
            return []
        
        self.logger.info(f"开始对 {len(documents)} 个文档进行分块处理...")
        
        chunked_documents = []
        chunked_count = 0
        
        for doc in documents:
            # 如果文档较短，直接使用
            if len(doc.page_content) <= self.chunk_size:
                chunked_documents.append(doc)
            else:
                # 对长文档进行分块
                chunks = self.text_splitter.split_documents([doc])
                chunked_count += 1
                
                # 为每个分块保留原始 metadata，并添加分块索引
                for idx, chunk in enumerate(chunks):
                    chunk.metadata = doc.metadata.copy()
                    chunk.metadata['chunk_index'] = idx
                    chunk.metadata['total_chunks'] = len(chunks)
                    chunk.metadata['is_chunked'] = True
                chunked_documents.extend(chunks)
                
                self.logger.info(
                    f"  文档 '{doc.metadata.get('title', 'Unknown')}' 被分成 {len(chunks)} 个分块"
                )
        
        self.logger.info(
            f"分块完成：原始文档 {len(documents)} 个，"
            f"其中 {chunked_count} 个被分块，共生成 {len(chunked_documents)} 个分块"
        )
        
        return chunked_documents


class DocParser:
    """
    文档解析器（便捷封装类）
    组合 DictToDocumentConverter 和 DocumentChunker 的功能
    """
    
    def __init__(
        self,
        default_source: str = "其他来源",
        chunk_size: int = 6000,
        chunk_overlap: int = 200
    ):
        """
        初始化文档解析器
        
        :param default_source: 默认文档来源
        :param chunk_size: 分块大小
        :param chunk_overlap: 分块重叠
        """
        self.logger = logging.getLogger(__name__)
        self.converter = DictToDocumentConverter(default_source=default_source)
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def parse(self, doc_dict: Dict[str, Dict[str, Any]], enable_chunking: bool = True) -> List[Document]:
        """
        解析文档字典并返回 Document 对象列表
        
        :param doc_dict: 文档字典
        :param enable_chunking: 是否启用分块（默认启用）
        :return: Document 对象列表
        """
        # 1. 转换为 Document 对象
        documents = self.converter.convert(doc_dict)
        
        if not documents:
            return []
        
        # 2. 分块处理（如果启用）
        if enable_chunking:
            documents = self.chunker.chunk_documents(documents)
        
        return documents
