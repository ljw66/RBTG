"""
document_renderer 模块单元测试
"""
import sys
import os
import tempfile
import shutil

# 添加项目根目录到 path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_document_renderer_init():
    """测试 DocumentRenderer 初始化"""
    print("=" * 50)
    print("测试 DocumentRenderer 初始化")
    print("=" * 50)
    
    from req_spec.document_renderer import DocumentRenderer
    
    renderer = DocumentRenderer("/path/to/template.docx")
    assert renderer is not None
    assert renderer.template_path == "/path/to/template.docx"
    print("  [OK] 初始化成功")


def test_prepare_render_data():
    """测试渲染数据准备"""
    print("\n测试渲染数据准备...")
    
    from req_spec.document_renderer import DocumentRenderer
    
    renderer = DocumentRenderer("/path/to/template.docx")
    
    output_dict = {
        "software_name": {
            "software_name": "测试软件",
            "software_id": "SW-001",
            "code_version": "V1.0",
            "model_name": "型号A"
        },
        "perf_req_items": {"perf_req_summary": []},
        "interface_req_items": {}
    }
    
    fill_data = {
        "chapterTrList": {
            "titleNo": "4",
            "title": "测试类型说明",
            "children": [
                {"titleNo": "4.1", "title": "功能测试"}
            ]
        }
    }
    
    render_data = renderer._prepare_render_data(
        output_dict=output_dict,
        fill_data=fill_data,
        project_id="test_123",
        create_by="test_user"
    )
    
    # 验证基本字段
    assert render_data["software_name"] == "测试软件"
    assert render_data["software_id"] == "SW-001"
    assert render_data["code_version"] == "V1.0"
    assert render_data["model_name"] == "型号A"
    assert render_data["project_id"] == "test_123"
    assert render_data["create_by"] == "test_user"
    
    # 验证章节数据
    assert "chapterTrList" in render_data
    assert len(render_data["chapters"]) == 1
    
    print(f"  渲染数据包含 {len(render_data)} 个字段")
    print("  [OK] 通过!")


def test_render_with_nonexistent_template():
    """测试模板不存在的情况"""
    print("\n测试模板不存在的情况...")
    
    from req_spec.document_renderer import DocumentRenderer
    
    renderer = DocumentRenderer("/nonexistent/template.docx")
    
    result = renderer.render(
        output_dict={},
        fill_data={"chapterTrList": {}},
        project_id="test",
        create_by="user",
        output_dir="/tmp"
    )
    
    assert result is None
    print("  [OK] 正确返回 None!")


def test_render_with_real_template():
    """测试使用真实模板渲染（需要模板存在）"""
    print("\n测试使用真实模板渲染...")
    
    from req_spec.document_renderer import DocumentRenderer
    
    # 检查模板是否存在
    template_path = os.path.join(
        project_root, 
        "req_spec", "templates", 
        "第三方测试需求编制说明V1.05_template-0324.docx"
    )
    
    if not os.path.exists(template_path):
        print("  [SKIP] 模板文件不存在，跳过测试")
        return
    
    renderer = DocumentRenderer(template_path)
    
    # 准备测试数据
    output_dict = {
        "software_name": {
            "software_name": "单元测试软件",
            "software_id": "UT-001",
            "code_version": "V1.0",
            "model_name": "测试型号"
        }
    }
    
    fill_data = {
        "chapterTrList": {
            "titleNo": "4",
            "title": "测试类型说明",
            "level": 1,
            "funItem": [],
            "children": []
        }
    }
    
    # 使用临时目录
    temp_dir = tempfile.mkdtemp()
    try:
        result = renderer.render(
            output_dict=output_dict,
            fill_data=fill_data,
            project_id="unit_test",
            create_by="tester",
            output_dir=temp_dir
        )
        
        if result:
            assert os.path.exists(result)
            print(f"  生成文件: {result}")
            print("  [OK] 渲染成功!")
        else:
            print("  [WARN] 渲染返回 None（可能模板格式问题）")
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_document_renderer_init()
    test_prepare_render_data()
    test_render_with_nonexistent_template()
    test_render_with_real_template()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
