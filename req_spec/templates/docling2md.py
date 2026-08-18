"""
将 Word 文档转换为 Markdown 格式
使用 docling 库
"""
import os
from docling.document_converter import DocumentConverter

def convert_docx_to_md(docx_path: str, output_path: str = None):
    """
    将 Word 文档转换为 Markdown
    
    :param docx_path: Word 文档路径
    :param output_path: 输出 Markdown 文件路径（可选，默认与源文件同名）
    """
    if not os.path.exists(docx_path):
        print(f"文件不存在: {docx_path}")
        return
    
    print(f"正在转换: {docx_path}")
    
    # 使用 docling 转换
    converter = DocumentConverter()
    result = converter.convert(docx_path)
    
    # 导出为 Markdown
    markdown_content = result.document.export_to_markdown()
    
    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(docx_path)[0]
        output_path = f"{base_name}.md"
    
    # 保存 Markdown 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"转换完成: {output_path}")
    return output_path


if __name__ == "__main__":
    # 转换模板文档
    template_dir = os.path.dirname(os.path.abspath(__file__))
    docx_file = os.path.join(template_dir, "第三方测试需求编制说明V1.05_template-0324.docx")
    
    convert_docx_to_md(docx_file)
