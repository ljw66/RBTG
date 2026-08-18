"""
文档渲染模块
负责将数据渲染到 Word 文档模板
"""

import datetime
import logging
import os
from typing import Dict, Any, Optional

from docxtpl import DocxTemplate


class DocumentRenderer:
    """
    文档渲染器
    将数据渲染到 Word 文档模板
    """
    
    def __init__(self, template_path: str):
        """
        初始化文档渲染器
        
        :param template_path: 模板文件路径
        """
        self.template_path = template_path
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def render(
        self,
        output_dict: Dict[str, Any],
        fill_data: Dict[str, Any],
        project_id: str,
        create_by: str,
        output_dir: str
    ) -> Optional[str]:
        """
        渲染 Word 文档
        
        :param output_dict: 包含各种需求数据的字典
        :param fill_data: 包含章节结构的字典
        :param project_id: 项目ID
        :param create_by: 创建者
        :param output_dir: 输出目录
        :return: 生成的文档路径，失败返回 None
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info("开始渲染 Word 文档...")
        self.logger.info(f"{'='*60}")
        
        try:
            # 检查模板文件是否存在
            if not os.path.exists(self.template_path):
                self.logger.error(f"模板文件不存在: {self.template_path}")
                return None
            
            # 加载模板
            self.logger.info(f"加载模板: {self.template_path}")
            doc = DocxTemplate(self.template_path)
            
            # 准备渲染数据
            render_data = self._prepare_render_data(
                output_dict, fill_data, project_id, create_by
            )
            
            # 渲染模板
            self.logger.info("正在渲染模板...")
            doc.render(render_data, autoescape=True)
            
            # 保存渲染后的文档
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成输出文件名
            output_filename = f"测试需求_{project_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
            output_path = os.path.join(output_dir, output_filename)
            
            doc.save(output_path)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Word 文档已生成: {output_path}")
            self.logger.info(f"{'='*60}\n")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"渲染文档时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _prepare_render_data(
        self,
        output_dict: Dict[str, Any],
        fill_data: Dict[str, Any],
        project_id: str,
        create_by: str
    ) -> Dict[str, Any]:
        """
        准备渲染数据，将 output_dict 和 fill_data 合并为模板需要的格式
        
        模板中常用的占位符包括：
        - {{model_name}} - 型号名称
        - {{software_name}} - 软件名称
        - {{software_id}} - 软件标识号
        - {{code_version}} - 代码版本号
        - {{doc_version}} - 文档版本号
        - {{p chapters}} - 章节内容
        """
        render_data = {}
        
        # 基本信息字段（现在都平铺在 output_dict 顶层）
        render_data["software_name"] = output_dict.get("software_name", "")
        render_data["software_id"] = output_dict.get("software_id", "")
        render_data["code_version"] = output_dict.get("code_version", "")
        render_data["model_name"] = output_dict.get("model_name", "")
        render_data["doc_version"] = output_dict.get("doc_version", "V1.0")
        render_data["model_subsystem_name"] = output_dict.get("model_subsystem_name", "")
        render_data["req_spec_file_id"] = output_dict.get("req_spec_file_id", "")
        render_data["entrusting_addr"] = output_dict.get("entrusting_addr", "")
        
        # software_level 相关字段
        render_data["software_level"] = output_dict.get("software_level", "")
        render_data["programming_language"] = output_dict.get("programming_language", "")
        render_data["dev_unit"] = output_dict.get("dev_unit", "")
        render_data["subsystem_function"] = output_dict.get("subsystem_function", "")
        render_data["subsystem_function_qd"] = output_dict.get("subsystem_function_qd", "")
        
        # cpu_storage 相关字段
        render_data["cpu_desc"] = output_dict.get("cpu_desc", "")
        render_data["use_of_storage_io"] = output_dict.get("use_of_storage_io", "")
        
        # 其他平铺字段
        render_data["subsystem_relation"] = output_dict.get("subsystem_relation", "")
        render_data["dev_platform"] = output_dict.get("dev_platform", "")
        
        # 项目信息
        render_data["project_id"] = project_id
        render_data["create_by"] = create_by
        render_data["create_date"] = datetime.datetime.now().strftime("%Y年%m月%d日")
        
        # 章节结构数据（用于生成测试需求表格）
        render_data["chapterTrList"] = fill_data.get("chapterTrList", {})
        render_data["chapters"] = fill_data.get("chapterTrList", {}).get("children", [])
        
        # 各类需求数据（现在都是数组，计数用 len() 计算）
        perf_items = output_dict.get("perf_req_items", [])
        interface_items = output_dict.get("interface_req_items", [])
        reliable_items = output_dict.get("reliable_req_items", [])
        margin_items = output_dict.get("margin_req_items", [])
        boundary_items = output_dict.get("boundary_req_items", [])
        recover_items = output_dict.get("recover_req_items", [])
        safety_items = output_dict.get("safety_critical_req_items", [])
        
        render_data["perf_req_items"] = perf_items
        render_data["interface_req_items"] = interface_items
        render_data["reliable_req_items"] = reliable_items
        render_data["margin_req_items"] = margin_items
        render_data["boundary_req_items"] = boundary_items
        render_data["recover_req_items"] = recover_items
        render_data["safety_critical_req_items"] = safety_items
        
        # 计数字段（用代码计算，不依赖模型输出）
        render_data["perf_req_count"] = str(len(perf_items))
        render_data["interface_req_count"] = str(len(interface_items))
        render_data["reliable_req_count"] = str(len(reliable_items))
        render_data["margin_req_count"] = str(len(margin_items))
        render_data["boundary_req_count"] = str(len(boundary_items))
        render_data["recover_req_count"] = str(len(recover_items))
        render_data["safety_critical_req_count"] = str(len(safety_items))
        
        # 接口相关数据（现在都是数组）
        render_data["gpio_interface"] = output_dict.get("gpio_interface", [])
        render_data["dog_interface"] = output_dict.get("dog_interface", [])
        render_data["other_hardware_interface"] = output_dict.get("other_hardware_interface", [])
        render_data["use_of_interrupt"] = output_dict.get("use_of_interrupt", [])
        
        # 功能需求数据
        render_data["req_infos"] = output_dict.get("req_infos", [])
        render_data["func_req_items"] = output_dict.get("func_req_items", "")
        
        # 记录渲染数据的字段（简化日志）
        self.logger.info(f"准备渲染数据，包含 {len(render_data)} 个字段")
        
        return render_data
