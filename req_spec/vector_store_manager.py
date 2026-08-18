"""
向量存储管理模块
负责文档的向量化存储（Milvus）、元数据存储（SQLite）和数据查询
"""

import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_milvus import Milvus

from req_spec.doc_metadata_db import DocMetadataDB
from config.settings import MILVUS_VECTOR_STORE_CONFIG


class VectorStoreManager:
    """
    向量存储管理器
    统一管理 Milvus 向量存储和 SQLite 元数据存储
    """
    
    def __init__(
        self,
        project_id: Union[str, int],
        embedding_model,
        metadata_db: DocMetadataDB,
        milvus_config: Optional[Dict] = None
    ):
        """
        初始化向量存储管理器
        
        :param project_id: 项目ID，用于生成 collection_name
        :param embedding_model: 嵌入模型（如 DashScopeEmbeddings）
        :param metadata_db: 元数据数据库实例
        :param milvus_config: Milvus 配置（可选，默认从 settings 读取）
        """
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.embedding_model = embedding_model
        self.metadata_db = metadata_db
        self.milvus_config = milvus_config or MILVUS_VECTOR_STORE_CONFIG
        
        # 使用项目ID生成唯一的 collection_name
        self.collection_name = f"proj_{self.project_id}"
        
        # 向量存储实例（延迟初始化）
        self._vector_store = None
    
    def _init_vector_store(self) -> Milvus:
        """
        初始化 Milvus 向量存储
        """
        if self._vector_store is None:
            self._vector_store = Milvus(
                embedding_function=self.embedding_model,
                connection_args={
                    "uri": self.milvus_config["uri"],
                    "db_name": self.milvus_config["db_name"],
                },
                collection_name=self.collection_name,
                index_params={
                    "index_type": self.milvus_config["index_type"],
                    "metric_type": self.milvus_config["metric_type"],
                },
                drop_old=self.milvus_config["drop_old"],
            )
        return self._vector_store
    
    @property
    def vector_store(self) -> Milvus:
        """获取向量存储实例"""
        return self._init_vector_store()
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量存储和元数据数据库
        
        :param documents: Document 对象列表
        :return: 文档 ID 列表
        """
        if not documents:
            self.logger.warning("没有文档需要添加")
            return []
        
        # 1. 添加到 Milvus 向量存储
        self.logger.info(f"正在添加 {len(documents)} 个文档到 Milvus...")
        vector_store = self._init_vector_store()
        ids = vector_store.add_documents(documents=documents)
        
        # 确保 ids 是列表格式
        if ids is None:
            ids = []
        if not isinstance(ids, list):
            ids = [ids] if ids else []
        
        self.logger.info(f"✓ 文档添加到 Milvus 完成，共 {len(ids)} 个文档")
        
        # 2. 保存元数据到 SQLite
        if ids:
            self.logger.info("正在保存文档元数据到 SQLite...")
            saved_count = 0
            for doc, doc_id in zip(documents, ids):
                # 生成唯一文档 ID
                unique_doc_id = f"{self.collection_name}_{doc_id}"
                
                # 确保 metadata 中有 source 字段
                doc_metadata = doc.metadata.copy()
                if 'source' not in doc_metadata:
                    doc_metadata['source'] = "其他来源"
                
                # 保存到元数据数据库
                success = self.metadata_db.insert_document(
                    doc_id=unique_doc_id,
                    milvus_id=str(doc_id),
                    collection_name=self.collection_name,
                    content=doc.page_content,
                    metadata=doc_metadata
                )
                if success:
                    saved_count += 1
            
            self.logger.info(f"✓ 元数据保存完成，成功保存 {saved_count}/{len(ids)} 条")
        else:
            self.logger.warning("未获取到文档 IDs，跳过元数据保存")
        
        return ids
    
    def clear_collection(self) -> bool:
        """
        清空当前 collection 的向量数据
        
        :return: 是否成功
        """
        try:
            from pymilvus import utility, connections
            
            # 连接到 Milvus
            connections.connect(
                alias="default",
                uri=self.milvus_config["uri"],
                db_name=self.milvus_config["db_name"]
            )
            
            # 检查 collection 是否存在
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.logger.info(f"✓ 已删除 Milvus collection: {self.collection_name}")
            else:
                self.logger.info(f"Milvus collection '{self.collection_name}' 不存在，无需删除")
            
            # 重置向量存储实例
            self._vector_store = None
            return True
        except Exception as e:
            self.logger.error(f"清空 Milvus collection 失败: {e}")
            return False
    
    def query_by_source(
        self,
        source_value: str = "需求规格说明书",
        limit: int = 100,
        return_format: str = "dict"
    ) -> Union[Dict[str, Dict], List[Document]]:
        """
        查询所有 source 字段为指定值的数据
        
        注意：此方法使用 SQLite 元数据数据库进行精确查询，而不是向量数据库
        向量数据库（Milvus）更适合用于相似性搜索，而精确匹配查询使用传统数据库更高效
        
        :param source_value: source 字段的值，默认为"需求规格说明书"
        :param limit: 返回结果的最大数量，默认为100
        :param return_format: 返回格式，可选值：
                            - "dict": 返回简单字典，格式为 {title: {content: ..., title_no: ...}}
                            - "document": 返回 Document 对象列表（包含完整元数据）
        :return: 根据 return_format 返回不同格式
        """
        try:
            # 使用 SQLite 元数据数据库进行精确查询
            results = self.metadata_db.query_by_source(
                source_value=source_value,
                collection_name=self.collection_name,
                limit=limit
            )
            
            if return_format == "dict":
                # 返回简单字典格式：{title: {content: ..., title_no: ...}}
                result_dict = {}
                for result in results:
                    title = result.get("title", "")
                    content = result.get("content", "")
                    title_no = result.get("title_no", "")
                    
                    if title:  # 只添加有标题的条目
                        if title in result_dict:
                            # 如果标题已存在，将内容合并
                            result_dict[title]["content"] += f"\n\n{content}"
                        else:
                            result_dict[title] = {
                                "content": content,
                                "title_no": title_no
                            }
                
                self.logger.info(f"找到 {len(result_dict)} 条 source 为 '{source_value}' 的数据（字典格式）")
                return result_dict
            
            else:
                # 返回 Document 格式
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "title_no": result.get("title_no", ""),
                            "level": result.get("level", 1),
                            "chunk_index": result.get("chunk_index"),
                            "total_chunks": result.get("total_chunks"),
                            "is_chunked": bool(result.get("is_chunked", False)),
                            "doc_id": result.get("doc_id"),
                            "milvus_id": result.get("milvus_id")
                        }
                    )
                    documents.append(doc)
                
                self.logger.info(f"找到 {len(documents)} 条 source 为 '{source_value}' 的数据（Document 格式）")
                return documents
                
        except Exception as e:
            self.logger.error(f"查询数据时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {} if return_format == "dict" else []
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        相似性搜索（从 Milvus 向量库检索）
        
        :param query: 查询文本
        :param k: 返回结果数量
        :param filter_dict: 过滤条件
        :return: Document 对象列表
        """
        vector_store = self._init_vector_store()
        if filter_dict:
            return vector_store.similarity_search(query, k=k, filter=filter_dict)
        return vector_store.similarity_search(query, k=k)
