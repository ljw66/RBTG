"""
requirement_extractor 模块单元测试
"""
import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging
from unittest.mock import Mock, MagicMock, patch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_requirement_extractor_init():
    """测试 RequirementExtractor 初始化"""
    print("=" * 50)
    print("测试 RequirementExtractor 初始化")
    print("=" * 50)
    
    from req_spec.requirement_extractor import RequirementExtractor
    
    mock_model = Mock()
    mock_embedding_model = Mock()
    
    extractor = RequirementExtractor(
        model=mock_model,
        embedding_model=mock_embedding_model,
        max_iterations=2,
        max_workers=10
    )
    
    assert extractor.model == mock_model
    assert extractor.embedding_model == mock_embedding_model
    assert extractor.max_iterations == 2
    assert extractor.max_workers == 10
    
    print("  max_iterations:", extractor.max_iterations)
    print("  max_workers:", extractor.max_workers)
    print("  通过!")


def test_other_req_types_list():
    """测试 OTHER_REQ_TYPES 列表"""
    print("\n测试 OTHER_REQ_TYPES 列表...")
    
    from req_spec.requirement_extractor import RequirementExtractor
    
    # 验证列表包含预期的需求类型
    expected_types = [
        "software_name", "subsystem_relation", "cpu_storage",
        "perf_req_items", "interface_req_items", "reliable_req_items"
    ]
    
    for req_type in expected_types:
        assert req_type in RequirementExtractor.OTHER_REQ_TYPES, f"{req_type} 应该在列表中"
    
    print(f"  共有 {len(RequirementExtractor.OTHER_REQ_TYPES)} 个需求类型")
    print("  通过!")


def test_extract_functional_requirements_no_match():
    """测试功能需求提取（无匹配章节）"""
    print("\n测试功能需求提取（无匹配章节）...")
    
    from req_spec.requirement_extractor import RequirementExtractor
    
    mock_model = Mock()
    mock_embedding_model = Mock()
    mock_vector_store = Mock()
    
    extractor = RequirementExtractor(
        model=mock_model,
        embedding_model=mock_embedding_model
    )
    
    # Mock LabelSearchAgent 返回空匹配
    with patch('req_spec.requirement_extractor.LabelSearchAgent') as MockAgent:
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.matched_chapters = []  # 无匹配
        mock_agent_instance.analyze.return_value = mock_result
        MockAgent.return_value = mock_agent_instance
        
        req_dict = {
            "系统概述": {"content": "系统概述内容", "title_no": "1"}
        }
        
        results = extractor.extract_functional_requirements(
            req_dict=req_dict,
            vector_store=mock_vector_store
        )
        
        assert results == []
        print("  无匹配时返回空列表")
        print("  通过!")


def test_extract_functional_requirements_with_match():
    """测试功能需求提取（有匹配章节）"""
    print("\n测试功能需求提取（有匹配章节）...")
    
    from req_spec.requirement_extractor import RequirementExtractor
    
    mock_model = Mock()
    mock_embedding_model = Mock()
    mock_vector_store = Mock()
    
    extractor = RequirementExtractor(
        model=mock_model,
        embedding_model=mock_embedding_model,
        max_workers=2
    )
    
    # Mock LabelSearchAgent 和 KeywordSearchWorkflow
    with patch('req_spec.requirement_extractor.LabelSearchAgent') as MockAgent, \
         patch('req_spec.requirement_extractor.KeywordSearchWorkflow') as MockWorkflow:
        
        # 设置 LabelSearchAgent mock
        mock_agent_instance = Mock()
        mock_result = Mock()
        mock_result.matched_chapters = ["功能A", "功能B"]
        mock_agent_instance.analyze.return_value = mock_result
        MockAgent.return_value = mock_agent_instance
        
        # 设置 KeywordSearchWorkflow mock
        mock_workflow_instance = Mock()
        mock_workflow_instance.run.return_value = {
            "matched_chapter": "功能A",
            "evaluation": {"sufficient": True},
            "final_result": "测试结果"
        }
        MockWorkflow.return_value = mock_workflow_instance
        
        req_dict = {
            "功能A": {"content": "功能A内容", "title_no": "2.1"},
            "功能B": {"content": "功能B内容", "title_no": "2.2"}
        }
        
        results = extractor.extract_functional_requirements(
            req_dict=req_dict,
            vector_store=mock_vector_store
        )
        
        assert len(results) == 2
        assert results[0]["chapter_title"] == "功能A"
        assert results[1]["chapter_title"] == "功能B"
        
        print(f"  提取了 {len(results)} 个章节的结果")
        print("  通过!")


def test_extract_all():
    """测试 extract_all 方法"""
    print("\n测试 extract_all 方法...")
    
    from req_spec.requirement_extractor import RequirementExtractor
    
    mock_model = Mock()
    mock_embedding_model = Mock()
    mock_vector_store = Mock()
    
    extractor = RequirementExtractor(
        model=mock_model,
        embedding_model=mock_embedding_model
    )
    
    # Mock 两个提取方法
    with patch.object(extractor, 'extract_functional_requirements') as mock_func, \
         patch.object(extractor, 'extract_other_requirements') as mock_other:
        
        mock_func.return_value = [{"chapter_title": "功能1", "result": {}}]
        mock_other.return_value = {"perf_req_items": [{"keyword": "性能", "result": {}}]}
        
        results = extractor.extract_all(
            req_dict={},
            vector_store=mock_vector_store,
            project_id="test_123"
        )
        
        assert "workflow_results" in results
        assert "other_req_results" in results
        assert len(results["workflow_results"]) == 1
        assert "perf_req_items" in results["other_req_results"]
        
        print("  workflow_results:", len(results["workflow_results"]))
        print("  other_req_results keys:", list(results["other_req_results"].keys()))
        print("  通过!")


if __name__ == "__main__":
    test_requirement_extractor_init()
    test_other_req_types_list()
    test_extract_functional_requirements_no_match()
    test_extract_functional_requirements_with_match()
    test_extract_all()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
