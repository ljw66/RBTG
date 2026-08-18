"""
fill_data_builder 模块单元测试
"""
import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_fill_data_builder_init():
    """测试 FillDataBuilder 初始化"""
    print("=" * 50)
    print("测试 FillDataBuilder 初始化")
    print("=" * 50)
    
    from req_spec.fill_data_builder import FillDataBuilder
    
    builder = FillDataBuilder()
    assert builder is not None
    print("  [OK] 初始化成功")


def test_build_basic_structure():
    """测试基本结构构建"""
    print("\n测试基本结构构建...")
    
    from req_spec.fill_data_builder import FillDataBuilder
    
    builder = FillDataBuilder()
    
    # 最简单的输入
    fill_data = builder.build(
        req_infos=[],
        req_item_dict={},
        output_dict={},
        project_id="test_123",
        create_by="test_user",
        user_id="user_001"
    )
    
    # 验证基本结构
    assert "chapterTrList" in fill_data
    assert fill_data["chapterTrList"]["titleNo"] == "4"
    assert fill_data["chapterTrList"]["title"] == "测试类型说明"
    assert "children" in fill_data["chapterTrList"]
    
    # 应该有 12 个测试类型章节
    assert len(fill_data["chapterTrList"]["children"]) == 12
    
    # 验证元数据
    assert fill_data["projectId"] == "test_123"
    assert fill_data["createBy"] == "test_user"
    assert fill_data["userId"] == "user_001"
    
    print(f"  章节数: {len(fill_data['chapterTrList']['children'])}")
    print("  [OK] 通过!")


def test_build_with_functional_requirements():
    """测试包含功能需求的构建"""
    print("\n测试包含功能需求的构建...")
    
    from req_spec.fill_data_builder import FillDataBuilder
    
    builder = FillDataBuilder()
    
    req_infos = [
        {"title": "数据采集功能"},
        {"title": "数据处理功能"}
    ]
    
    req_item_dict = {
        "数据采集功能": [
            {"funId": "TR-001", "handling": "采集数据", "name": "数据采集功能"}
        ],
        "数据处理功能": [
            {"funId": "TR-002", "handling": "处理数据", "name": "数据处理功能"}
        ]
    }
    
    fill_data = builder.build(
        req_infos=req_infos,
        req_item_dict=req_item_dict,
        output_dict={},
        project_id="test_456",
        create_by="user",
        user_id="uid"
    )
    
    # 验证功能测试章节
    func_test = fill_data["chapterTrList"]["children"][0]
    assert func_test["titleNo"] == "4.1"
    assert func_test["title"] == "功能测试"
    assert len(func_test["children"]) == 2  # 两个功能需求
    
    print(f"  功能测试子章节数: {len(func_test['children'])}")
    print("  [OK] 通过!")


def test_build_with_performance_requirements():
    """测试包含性能需求的构建"""
    print("\n测试包含性能需求的构建...")
    
    from req_spec.fill_data_builder import FillDataBuilder
    
    builder = FillDataBuilder()
    
    output_dict = {
        "perf_req_items": {
            "perf_req_summary": [
                {"perf_req_id": "PERF-001", "perf_req_desc": "响应时间<50ms", "other_desc": "性能项1"}
            ]
        }
    }
    
    fill_data = builder.build(
        req_infos=[],
        req_item_dict={},
        output_dict=output_dict,
        project_id="test_789",
        create_by="user",
        user_id="uid"
    )
    
    # 验证性能测试章节
    perf_test = fill_data["chapterTrList"]["children"][1]  # 4.2 是第二个
    assert perf_test["titleNo"] == "4.2"
    assert perf_test["title"] == "性能测试"
    assert len(perf_test["children"][0]["funItem"]) == 1
    
    print("  性能测试包含 1 个需求项")
    print("  [OK] 通过!")


def test_chapter_numbering():
    """测试章节编号"""
    print("\n测试章节编号...")
    
    from req_spec.fill_data_builder import FillDataBuilder
    
    builder = FillDataBuilder()
    
    fill_data = builder.build(
        req_infos=[],
        req_item_dict={},
        output_dict={},
        project_id="test",
        create_by="user",
        user_id="uid"
    )
    
    children = fill_data["chapterTrList"]["children"]
    
    # 验证章节编号
    expected_numbers = ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10", "4.11", "4.12"]
    actual_numbers = [c["titleNo"] for c in children]
    
    assert actual_numbers == expected_numbers, f"Expected {expected_numbers}, got {actual_numbers}"
    
    print(f"  章节编号正确: {actual_numbers}")
    print("  [OK] 通过!")


if __name__ == "__main__":
    test_fill_data_builder_init()
    test_build_basic_structure()
    test_build_with_functional_requirements()
    test_build_with_performance_requirements()
    test_chapter_numbering()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
