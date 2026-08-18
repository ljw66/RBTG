"""
requirement_extractor 模块集成测试（使用真实数据）
"""
import sys
import os
import logging

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_with_real_data():
    """使用真实数据测试 RequirementExtractor"""
    print("=" * 60)
    print("RequirementExtractor 集成测试（真实数据）")
    print("=" * 60)
    
    # 1. 导入必要模块
    from langchain.chat_models import init_chat_model
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_milvus import Milvus
    from config.settings import (
        LLM_MODEL, LLM_API_KEY, LLM_BASE_URL,
        EMBEDDING_MODEL, EMBEDDING_API_KEY,
        MILVUS_VECTOR_STORE_CONFIG
    )
    from req_spec.requirement_extractor import RequirementExtractor
    from req_spec.doc_parser import DocParser
    
    # 2. 初始化模型
    print("\n[1] 初始化模型...")
    model = init_chat_model(
        model=LLM_MODEL,
        model_provider="openai",
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )
    embedding_model = DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=EMBEDDING_API_KEY
    )
    print("  [OK] LLM 和 Embedding 模型初始化完成")
    
    # 3. 准备测试数据（简化的需求规格说明书）
    print("\n[2] 准备测试数据...")
    test_doc_dict = {
        "系统概述": {
            "content": "本软件是飞行器数据管理系统，负责数据采集、处理和传输功能。软件级别为C级，采用C语言开发。",
            "title_no": "1",
            "level": 1,
            "source": "需求规格说明书"
        },
        "数据采集功能": {
            "content": "软件通过RS422接口采集传感器数据，采样频率100Hz，数据包大小256字节。采集的数据包括温度、压力、速度等参数。",
            "title_no": "2.1",
            "level": 2,
            "source": "需求规格说明书"
        },
        "数据处理功能": {
            "content": "对采集的数据进行滤波、校准和格式转换处理。滤波算法采用滑动平均法，窗口大小为10。",
            "title_no": "2.2",
            "level": 2,
            "source": "需求规格说明书"
        },
        "性能要求": {
            "content": "系统响应时间不超过50ms，CPU占用率不超过60%，内存占用不超过2MB。",
            "title_no": "3.1",
            "level": 2,
            "source": "需求规格说明书"
        }
    }
    print(f"  [OK] 测试数据包含 {len(test_doc_dict)} 个章节")
    
    # 4. 解析文档并创建向量存储
    print("\n[3] 创建向量存储...")
    doc_parser = DocParser(default_source="需求规格说明书")
    documents = doc_parser.parse(test_doc_dict, enable_chunking=True)
    
    # 使用临时 collection
    test_collection = "test_req_extractor_integration"
    vector_store = Milvus(
        embedding_function=embedding_model,
        connection_args={
            "uri": MILVUS_VECTOR_STORE_CONFIG["uri"],
            "db_name": MILVUS_VECTOR_STORE_CONFIG["db_name"],
        },
        collection_name=test_collection,
        index_params={
            "index_type": MILVUS_VECTOR_STORE_CONFIG["index_type"],
            "metric_type": MILVUS_VECTOR_STORE_CONFIG["metric_type"],
        },
        drop_old=True,  # 每次测试清空
    )
    
    # 添加文档到向量存储
    ids = vector_store.add_documents(documents)
    print(f"  [OK] 添加了 {len(ids)} 个文档到向量存储")
    
    # 5. 创建 RequirementExtractor 并测试
    print("\n[4] 初始化 RequirementExtractor...")
    extractor = RequirementExtractor(
        model=model,
        embedding_model=embedding_model,
        max_iterations=1,  # 快速测试
        max_workers=5
    )
    print("  [OK] RequirementExtractor 初始化完成")
    
    # 构建 req_dict（只包含需求规格说明书的内容）
    req_dict = {title: {"content": info["content"], "title_no": info["title_no"]} 
                for title, info in test_doc_dict.items()}
    
    # 6. 测试功能需求提取
    print("\n[5] 测试功能需求提取...")
    print("-" * 40)
    
    workflow_results = extractor.extract_functional_requirements(
        req_dict=req_dict,
        vector_store=vector_store,
        target_title="功能",  # 搜索包含"功能"的章节
        output_dir="./test_output",
        project_id="integration_test"
    )
    
    print(f"\n功能需求提取结果：")
    print(f"  - 提取章节数: {len(workflow_results)}")
    for item in workflow_results:
        chapter = item.get("chapter_title", "")
        result = item.get("result", {})
        has_result = bool(result.get("final_result"))
        print(f"  - {chapter}: {'有结果' if has_result else '无结果'}")
    
    # 7. 测试其他需求提取（只测试部分类型，加快速度）
    print("\n[6] 测试其他需求提取（仅测试 software_name）...")
    print("-" * 40)
    
    other_results = extractor.extract_other_requirements(
        req_dict=req_dict,
        vector_store=vector_store,
        req_types=["software_name"],  # 只测试一个类型
        output_dir="./test_output",
        project_id="integration_test"
    )
    
    print(f"\n其他需求提取结果：")
    for req_type, results in other_results.items():
        print(f"  - {req_type}: {len(results)} 个关键词处理完成")
    
    print("\n" + "=" * 60)
    print("集成测试完成!")
    print("=" * 60)
    
    return workflow_results, other_results


if __name__ == "__main__":
    test_with_real_data()
