"""
vector_store_manager 模块单元测试
"""
import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
from unittest.mock import Mock, MagicMock

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_vector_store_manager_init():
    """测试 VectorStoreManager 初始化"""
    print("=" * 50)
    print("测试 VectorStoreManager 初始化")
    print("=" * 50)
    
    from req_spec.vector_store_manager import VectorStoreManager
    
    # 创建 mock 对象
    mock_embedding_model = Mock()
    mock_metadata_db = Mock()
    
    # 初始化 VectorStoreManager
    manager = VectorStoreManager(
        project_id=123,
        embedding_model=mock_embedding_model,
        metadata_db=mock_metadata_db
    )
    
    # 验证
    assert manager.project_id == 123
    assert manager.collection_name == "proj_123"
    assert manager.embedding_model == mock_embedding_model
    assert manager.metadata_db == mock_metadata_db
    
    print("  collection_name:", manager.collection_name)
    print("  通过!")


def test_query_by_source_dict_format():
    """测试 query_by_source 方法（字典格式）"""
    print("\n测试 query_by_source（字典格式）...")
    
    from req_spec.vector_store_manager import VectorStoreManager
    
    # 创建 mock 对象
    mock_embedding_model = Mock()
    mock_metadata_db = Mock()
    
    # 模拟查询结果
    mock_metadata_db.query_by_source.return_value = [
        {"title": "功能概述", "content": "软件功能描述...", "title_no": "1.1"},
        {"title": "性能要求", "content": "响应时间要求...", "title_no": "2.1"},
    ]
    
    manager = VectorStoreManager(
        project_id=456,
        embedding_model=mock_embedding_model,
        metadata_db=mock_metadata_db
    )
    
    # 测试查询
    result = manager.query_by_source(source_value="需求规格说明书", return_format="dict")
    
    # 验证
    assert isinstance(result, dict)
    assert len(result) == 2
    assert "功能概述" in result
    assert result["功能概述"]["content"] == "软件功能描述..."
    assert result["功能概述"]["title_no"] == "1.1"
    
    print(f"  查询到 {len(result)} 条数据")
    print("  通过!")


def test_query_by_source_document_format():
    """测试 query_by_source 方法（Document 格式）"""
    print("\n测试 query_by_source（Document 格式）...")
    
    from req_spec.vector_store_manager import VectorStoreManager
    
    # 创建 mock 对象
    mock_embedding_model = Mock()
    mock_metadata_db = Mock()
    
    # 模拟查询结果
    mock_metadata_db.query_by_source.return_value = [
        {"title": "接口设计", "content": "RS422接口...", "title_no": "3.1", "source": "接口协议"},
    ]
    
    manager = VectorStoreManager(
        project_id=789,
        embedding_model=mock_embedding_model,
        metadata_db=mock_metadata_db
    )
    
    # 测试查询
    result = manager.query_by_source(source_value="接口协议", return_format="document")
    
    # 验证
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].page_content == "RS422接口..."
    assert result[0].metadata["title"] == "接口设计"
    
    print(f"  查询到 {len(result)} 个 Document 对象")
    print("  通过!")


def test_add_documents_mock():
    """测试 add_documents 方法（使用 mock）"""
    print("\n测试 add_documents（mock 模式）...")
    
    from req_spec.vector_store_manager import VectorStoreManager
    from langchain_core.documents import Document
    
    # 创建 mock 对象
    mock_embedding_model = Mock()
    mock_metadata_db = Mock()
    mock_metadata_db.insert_document.return_value = True
    
    manager = VectorStoreManager(
        project_id=999,
        embedding_model=mock_embedding_model,
        metadata_db=mock_metadata_db
    )
    
    # Mock Milvus
    mock_vector_store = Mock()
    mock_vector_store.add_documents.return_value = ["id_1", "id_2"]
    manager._vector_store = mock_vector_store
    
    # 测试文档
    test_docs = [
        Document(page_content="测试内容1", metadata={"title": "标题1", "source": "来源1"}),
        Document(page_content="测试内容2", metadata={"title": "标题2", "source": "来源2"}),
    ]
    
    # 添加文档
    ids = manager.add_documents(test_docs)
    
    # 验证
    assert len(ids) == 2
    assert mock_vector_store.add_documents.called
    assert mock_metadata_db.insert_document.call_count == 2
    
    print(f"  添加了 {len(ids)} 个文档")
    print("  通过!")


if __name__ == "__main__":
    test_vector_store_manager_init()
    test_query_by_source_dict_format()
    test_query_by_source_document_format()
    test_add_documents_mock()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
