"""
doc_parser 模块单元测试
"""
import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from req_spec.doc_parser import DocParser, DictToDocumentConverter, DocumentChunker
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_doc_parser():
    """测试 DocParser 完整流程"""
    print("=" * 50)
    print("测试 DocParser 模块")
    print("=" * 50)
    
    # 测试数据
    test_doc_dict = {
        "功能概述": {
            "content": "本软件实现数据采集和处理功能。",
            "title_no": "1.1",
            "level": 2,
            "source": "需求规格说明书"
        },
        "性能要求": {
            "content": "响应时间不超过100ms。" * 100,  # 长文本测试分块
            "title_no": "2.1",
            "level": 2,
            "source": "任务书"
        },
        "接口设计": {
            "content": "支持RS422接口通信。",
            "title_no": "3.1",
            "level": 2
            # 无 source，测试默认值
        }
    }
    
    # 测试 DocParser（使用较小的 chunk_size 以触发分块）
    parser = DocParser(default_source="其他来源", chunk_size=500, chunk_overlap=50)
    docs = parser.parse(test_doc_dict, enable_chunking=True)
    
    print(f"\n结果: 共生成 {len(docs)} 个 Document 对象")
    print("-" * 50)
    for i, doc in enumerate(docs):
        meta = doc.metadata
        chunk_info = ""
        if meta.get("is_chunked"):
            chunk_info = f" [chunk {meta['chunk_index']+1}/{meta['total_chunks']}]"
        print(f"  [{i+1}] title={meta['title']}, source={meta['source']}, len={len(doc.page_content)}{chunk_info}")
    
    # 验证
    assert len(docs) >= 3, "应该至少有3个文档"
    assert docs[0].metadata["source"] == "需求规格说明书", "source 应该正确设置"
    assert docs[-1].metadata["source"] == "其他来源", "默认 source 应该是其他来源"
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)


def test_converter_only():
    """单独测试 DictToDocumentConverter"""
    print("\n测试 DictToDocumentConverter...")
    converter = DictToDocumentConverter(default_source="测试来源")
    docs = converter.convert({
        "测试章节": {"content": "测试内容", "title_no": "1", "level": 1}
    })
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "测试来源"
    print("  通过!")


def test_chunker_only():
    """单独测试 DocumentChunker"""
    print("\n测试 DocumentChunker...")
    from langchain_core.documents import Document
    
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    long_doc = Document(
        page_content="这是一段很长的测试文本。" * 50,
        metadata={"title": "长文档", "source": "测试"}
    )
    chunks = chunker.chunk_documents([long_doc])
    assert len(chunks) > 1, "长文档应该被分块"
    assert all(c.metadata.get("is_chunked") for c in chunks), "分块应该有 is_chunked 标记"
    print(f"  长文档被分成 {len(chunks)} 个分块，通过!")


if __name__ == "__main__":
    test_converter_only()
    test_chunker_only()
    test_doc_parser()
