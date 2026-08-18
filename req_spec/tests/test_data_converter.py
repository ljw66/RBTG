"""
data_converter 模块单元测试
"""
import sys
import os

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_data_converter_init():
    """测试 DataConverter 初始化"""
    print("=" * 50)
    print("测试 DataConverter 初始化")
    print("=" * 50)
    
    from req_spec.data_converter import DataConverter
    
    converter = DataConverter()
    assert converter is not None
    print("  [OK] 初始化成功")


def test_convert_functional_requirements():
    """测试功能需求转换"""
    print("\n测试功能需求转换...")
    
    from req_spec.data_converter import DataConverter
    
    converter = DataConverter()
    
    # 模拟 workflow_results
    workflow_results = [
        {
            "chapter_title": "数据采集功能",
            "result": {
                "final_result": '{"fun_items": [{"fun_id": "TR-001", "handling": "采集传感器数据"}], "fun_req_summary": "数据采集功能概述"}'
            }
        },
        {
            "chapter_title": "数据处理功能",
            "result": {
                "final_result": '{"fun_items": [{"fun_id": "TR-002", "handling": "滤波处理"}, {"fun_id": "TR-003", "handling": "格式转换"}], "fun_req_summary": "数据处理功能概述"}'
            }
        }
    ]
    
    req_infos, req_item_dict = converter.convert_functional_requirements(workflow_results)
    
    # 验证
    assert len(req_infos) == 2
    assert len(req_item_dict) == 2
    assert "数据采集功能" in req_item_dict
    assert len(req_item_dict["数据处理功能"]) == 2
    
    print(f"  req_infos: {len(req_infos)} 个章节")
    print(f"  req_item_dict: {len(req_item_dict)} 个章节")
    print("  [OK] 通过!")


def test_convert_other_requirements():
    """测试其他需求转换"""
    print("\n测试其他需求转换...")
    
    from req_spec.data_converter import DataConverter
    
    converter = DataConverter()
    
    # 模拟 other_req_results
    other_req_results = {
        "software_name": [
            {
                "keyword": "软件名称",
                "result": {
                    "final_result": '{"software_name": "测试软件", "software_id": "SW-001"}'
                }
            }
        ],
        "perf_req_items": [
            {
                "keyword": "性能",
                "result": {
                    "final_result": '{"perf_req_items": [{"name": "响应时间", "value": "50ms"}]}'
                }
            }
        ]
    }
    
    output_dict = converter.convert_other_requirements(other_req_results)
    
    # 验证
    assert "software_name" in output_dict
    assert "perf_req_items" in output_dict
    assert output_dict["software_name"]["software_name"] == "测试软件"
    
    print(f"  output_dict: {len(output_dict)} 种需求类型")
    print("  [OK] 通过!")


def test_convert_all():
    """测试 convert_all 方法"""
    print("\n测试 convert_all...")
    
    from req_spec.data_converter import DataConverter
    
    converter = DataConverter()
    
    workflow_results = [
        {
            "chapter_title": "功能A",
            "result": {
                "final_result": '{"fun_items": [{"fun_id": "TR-001", "handling": "处理A"}], "fun_req_summary": "概述"}'
            }
        }
    ]
    
    other_req_results = {
        "software_name": [
            {
                "keyword": "软件",
                "result": {
                    "final_result": '{"software_name": "测试软件"}'
                }
            }
        ]
    }
    
    results = converter.convert_all(
        workflow_results=workflow_results,
        other_req_results=other_req_results
    )
    
    # 验证
    assert "req_infos" in results
    assert "req_item_dict" in results
    assert "output_dict" in results
    assert len(results["req_infos"]) == 1
    assert "req_infos" in results["output_dict"]  # 功能需求也在 output_dict 中
    
    print(f"  req_infos: {len(results['req_infos'])} 个章节")
    print(f"  output_dict: {len(results['output_dict'])} 种类型")
    print("  [OK] 通过!")


def test_empty_input():
    """测试空输入"""
    print("\n测试空输入...")
    
    from req_spec.data_converter import DataConverter
    
    converter = DataConverter()
    
    # 空输入
    req_infos, req_item_dict = converter.convert_functional_requirements([])
    assert req_infos == []
    assert req_item_dict == {}
    
    output_dict = converter.convert_other_requirements({})
    assert output_dict == {}
    
    print("  空输入处理正确")
    print("  [OK] 通过!")


if __name__ == "__main__":
    test_data_converter_init()
    test_convert_functional_requirements()
    test_convert_other_requirements()
    test_convert_all()
    test_empty_input()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
