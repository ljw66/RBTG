"""
fill_data 构建模块
负责生成文档模板所需的 fill_data 结构
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional


class FillDataBuilder:
    """
    fill_data 构建器
    生成用于文档模板渲染的数据结构
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _add_dict(
        self,
        titleNo: str,
        title: str,
        level: int,
        children_titleNo: str,
        children_title: str,
        children_level: int,
        input_dict: List[Dict],
        func_id_keyword: str,
        item_handling_keyword: str,
        item_name_keyword: str
    ) -> Dict:
        """
        构建标准的章节结构
        
        :param titleNo: 章节编号（如 "4.1"）
        :param title: 章节标题（如 "功能测试"）
        :param level: 章节级别
        :param children_titleNo: 子章节编号
        :param children_title: 子章节标题
        :param children_level: 子章节级别
        :param input_dict: 需求项列表
        :param func_id_keyword: 需求ID字段名
        :param item_handling_keyword: 需求描述字段名
        :param item_name_keyword: 需求名称字段名
        :return: 章节结构字典
        """
        _dict = {
            "titleNo": titleNo,
            "title": title,
            "level": level,
            "funItem": [],
            "children": []
        }
        _children_dict = {
            "titleNo": children_titleNo,
            "title": children_title,
            "level": children_level,
            "children": None,
            "funItem": []
        }
        for item in input_dict:
            item_dict = {
                "funId": item.get(func_id_keyword, ""),
                "handling": item.get(item_handling_keyword, ""),
                "name": item.get(item_name_keyword, ""),
                "priority": None,
                "sufficiency": None,
                "testMethod": None,
                "passCriteria": None
            }
            _children_dict['funItem'].append(item_dict)
        _dict["children"].append(_children_dict)
        return _dict
    
    def _add_empty_dict(
        self,
        titleNo: str,
        title: str,
        level: int,
        children_titleNo: str,
        children_title: str,
        children_level: int
    ) -> Dict:
        """
        构建空的章节结构（用于暂未生成的需求类型）
        """
        _dict = {
            "titleNo": titleNo,
            "title": title,
            "level": level,
            "funItem": [],
            "children": []
        }
        _children_dict = {
            "titleNo": children_titleNo,
            "title": children_title,
            "level": children_level,
            "children": None,
            "funItem": []
        }
        _dict["children"].append(_children_dict)
        return _dict
    
    def build(
        self,
        req_infos: List[Dict],
        req_item_dict: Dict[str, List[Dict]],
        output_dict: Dict[str, Any],
        project_id: str,
        create_by: str,
        user_id: str,
        doc_type: int = 1,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        构建完整的 fill_data 结构
        
        :param req_infos: 功能需求信息列表
        :param req_item_dict: 功能需求项字典
        :param output_dict: 其他需求数据字典
        :param project_id: 项目ID
        :param create_by: 创建者
        :param user_id: 用户ID
        :param doc_type: 文档类型
        :param output_dir: 输出目录（可选）
        :return: fill_data 结构
        """
        # 1. 构建基础结构
        fill_data = {
            "chapterTrList": {
                "titleNo": "4",
                "title": "测试类型说明",
                "level": 1,
                "funItem": [],
                "children": []
            }
        }
        
        # 2. 添加功能测试章节（4.1）
        self._add_functional_test(fill_data, req_infos, req_item_dict)
        
        # 3. 添加性能测试（4.2）
        self._add_performance_test(fill_data, output_dict)
        
        # 4. 添加接口测试（4.3）
        self._add_interface_test(fill_data, output_dict)
        
        # 5. 添加可靠性安全测试（4.4）
        self._add_reliability_test(fill_data, output_dict)
        
        # 6. 添加余量测试（4.5）
        self._add_margin_test(fill_data, output_dict)
        
        # 7. 添加边界测试（4.6）
        self._add_boundary_test(fill_data, output_dict)
        
        # 8. 添加数据处理测试（4.7）- 暂为空
        fill_data["chapterTrList"]["children"].append(
            self._add_empty_dict("4.7", "数据处理测试", 2, "4.7.1", "数据处理测试需求项", 3))
        self.logger.info("  数据处理测试章节已添加（空）")
        
        # 9. 添加恢复性测试（4.8）
        self._add_recovery_test(fill_data, output_dict)
        
        # 10. 添加强度测试（4.9）- 暂为空
        fill_data["chapterTrList"]["children"].append(
            self._add_empty_dict("4.9", "强度测试", 2, "4.9.1", "强度测试需求项", 3))
        self.logger.info("  强度测试章节已添加（空）")
        
        # 11. 添加代码审查（4.10）
        self._add_code_review(fill_data, output_dict)
        
        # 12. 添加静态分析（4.11）
        self._add_static_analysis(fill_data)
        
        # 13. 添加逻辑测试（4.12）
        self._add_logical_test(fill_data)
        
        # 14. 添加元数据
        fill_data["docType"] = doc_type
        fill_data["projectId"] = project_id
        fill_data["createBy"] = create_by
        fill_data["userId"] = user_id
        fill_data["softwareId"] = output_dict.get("software_id", "")
        fill_data["codeVersion"] = output_dict.get("code_version", "")
        
        # 15. 保存到文件（如果提供了输出目录）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filldata_path = os.path.join(output_dir, f"fill_data_{project_id}.json")
            self._save_json(fill_data, filldata_path)
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"fill_data 已生成并保存到: {filldata_path}")
            self.logger.info(f"  - 共包含 {len(fill_data['chapterTrList']['children'])} 个测试类型章节")
            self.logger.info(f"{'='*60}\n")
        
        return fill_data
    
    def _add_functional_test(self, fill_data: Dict, req_infos: List[Dict], req_item_dict: Dict):
        """添加功能测试章节（4.1）"""
        func_test_node = {
            "titleNo": "4.1",
            "title": "功能测试",
            "level": 2,
            "funItem": [],
            "children": []
        }
        
        if req_infos:
            for i, req_info in enumerate(req_infos):
                chapter_title = req_info.get("title", "")
                child_node = {
                    "titleNo": f"4.1.{i+1}",
                    "title": chapter_title,
                    "level": 3,
                    "funItem": req_item_dict.get(chapter_title, []),
                    "children": None
                }
                func_test_node["children"].append(child_node)
        
        if not func_test_node["children"]:
            func_test_node["children"] = None
        
        fill_data["chapterTrList"]["children"].append(func_test_node)
        self.logger.info(f"功能测试章节已添加，包含 {len(req_infos) if req_infos else 0} 个子章节")
    
    def _add_performance_test(self, fill_data: Dict, output_dict: Dict):
        """添加性能测试（4.2）"""
        perf_items = output_dict.get("perf_req_items", [])
        if perf_items:
            fill_data["chapterTrList"]["children"].append(
                self._add_dict("4.2", "性能测试", 2, "4.2.1", "性能需求项", 3,
                             perf_items, "perf_req_id", "perf_req_desc", "other_desc"))
            self.logger.info(f"性能测试章节已添加，包含 {len(perf_items)} 个需求项")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.2", "性能测试", 2, "4.2.1", "性能需求项", 3))
            self.logger.info("  性能测试章节已添加（空）")
    
    def _add_interface_test(self, fill_data: Dict, output_dict: Dict):
        """添加接口测试（4.3）"""
        interface_items = output_dict.get("interface_req_items", [])
        if interface_items:
            fill_data["chapterTrList"]["children"].append(
                self._add_dict("4.3", "接口测试", 2, "4.3.1", "接口需求项", 3,
                             interface_items, "interface_id", "interface_req_desc", "interface_name"))
            self.logger.info(f"接口测试章节已添加，包含 {len(interface_items)} 个需求项")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.3", "接口测试", 2, "4.3.1", "接口需求项", 3))
            self.logger.info("  接口测试章节已添加（空）")
    
    def _add_reliability_test(self, fill_data: Dict, output_dict: Dict):
        """添加可靠性安全测试（4.4）"""
        reliable_items = output_dict.get("reliable_req_items", [])
        if reliable_items:
            fill_data["chapterTrList"]["children"].append(
                self._add_dict("4.4", "可靠性安全测试", 2, "4.4.1", "可靠性安全测试需求项", 3,
                             reliable_items, "reliable_req_id", "reliable_req_desc", "other_desc"))
            self.logger.info(f"可靠性安全测试章节已添加，包含 {len(reliable_items)} 个需求项")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.4", "可靠性安全测试", 2, "4.4.1", "可靠性安全测试需求项", 3))
            self.logger.info("  可靠性安全测试章节已添加（空）")
    
    def _add_margin_test(self, fill_data: Dict, output_dict: Dict):
        """添加余量测试（4.5）"""
        margin_items = output_dict.get("margin_req_items", [])
        if margin_items:
            fill_data["chapterTrList"]["children"].append(
                self._add_dict("4.5", "余量测试", 2, "4.5.1", "余量测试需求项", 3,
                             margin_items, "margin_req_id", "margin_req_desc", "other_desc"))
            self.logger.info(f"余量测试章节已添加，包含 {len(margin_items)} 个需求项")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.5", "余量测试", 2, "4.5.1", "余量测试需求项", 3))
            self.logger.info("  余量测试章节已添加（空）")
    
    def _add_boundary_test(self, fill_data: Dict, output_dict: Dict):
        """添加边界测试（4.6）"""
        if "boundary_req_items" in output_dict and output_dict.get("boundary_req_items"):
            boundary_items = output_dict["boundary_req_items"]
            if isinstance(boundary_items, list):
                fill_data["chapterTrList"]["children"].append(
                    self._add_dict("4.6", "边界测试", 2, "4.6.1", "边界测试需求项", 3,
                                 boundary_items, "boundary_req_id", "boundary_req_desc", "other_desc"))
                self.logger.info(f"边界测试章节已添加，包含 {len(boundary_items)} 个需求项")
            else:
                fill_data["chapterTrList"]["children"].append(
                    self._add_empty_dict("4.6", "边界测试", 2, "4.6.1", "边界测试需求项", 3))
                self.logger.info("  边界测试章节已添加（空）")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.6", "边界测试", 2, "4.6.1", "边界测试需求项", 3))
            self.logger.info("  边界测试章节已添加（空）")
    
    def _add_recovery_test(self, fill_data: Dict, output_dict: Dict):
        """添加恢复性测试（4.8）"""
        recover_items = output_dict.get("recover_req_items", [])
        if recover_items:
            fill_data["chapterTrList"]["children"].append(
                self._add_dict("4.8", "恢复性测试", 2, "4.8.1", "恢复性测试需求项", 3,
                             recover_items, "recover_req_id", "recover_req_desc", "other_desc"))
            self.logger.info(f"恢复性测试章节已添加，包含 {len(recover_items)} 个需求项")
        else:
            fill_data["chapterTrList"]["children"].append(
                self._add_empty_dict("4.8", "恢复性测试", 2, "4.8.1", "恢复性测试需求项", 3))
            self.logger.info("  恢复性测试章节已添加（空）")
    
    def _add_code_review(self, fill_data: Dict, output_dict: Dict):
        """添加代码审查（4.10）"""
        software_name = output_dict.get("software_name", "被测软件")
        code_review_items = [{
            "code_review_req_id": "TR-DS-ALL",
            "code_review_req_desc": f"本次代码审查范围为被测软件的全部源代码，审查的依据为{software_name}需求规格说明书。要求代码审查人员对代码进行阅读、理解和分析，检查代码和设计的一致性、确认代码是否正确实现了需求中的功能。审查过程中还要求参考代码审查检查单的审查项，检查代码执行标准的情况、代码逻辑表达的正确性、代码结构的合理性以及代码的可读性。代码审查检查单内容见附录B，该检查单在代码审查前完成与委托方的确认；此外，代码审查还需要包括针对中断访问冲突、中断保护等进行专项分析。",
            "other_desc": "代码审查需求项"
        }]
        fill_data["chapterTrList"]["children"].append(
            self._add_dict("4.10", "代码审查", 2, "4.10.1", "代码审查需求项", 3,
                         code_review_items, "code_review_req_id", "code_review_req_desc", "other_desc"))
        self.logger.info("代码审查章节已添加")
    
    def _add_static_analysis(self, fill_data: Dict):
        """添加静态分析（4.11）"""
        static_analysis_items = [{
            "static_analysis_req_id": "TR-JF-ALL",
            "static_analysis_req_desc": "对全部源代码进行控制流、数据流、接口及表达式分析，确认代码实现是否存在相关问题；",
            "other_desc": "静态分析需求项"
        }]
        fill_data["chapterTrList"]["children"].append(
            self._add_dict("4.11", "静态分析", 2, "4.11.1", "静态分析需求项", 3,
                         static_analysis_items, "static_analysis_req_id", "static_analysis_req_desc", "other_desc"))
        self.logger.info("静态分析章节已添加")
    
    def _add_logical_test(self, fill_data: Dict):
        """添加逻辑测试（4.12）"""
        logical_testing_items = [{
            "logical_testing_req_id": "TR-LJ-ALL",
            "logical_testing_req_desc": "测试用例执行过程中使用覆盖率工具统计语句和分支覆盖率；",
            "other_desc": "逻辑测试需求项"
        }]
        fill_data["chapterTrList"]["children"].append(
            self._add_dict("4.12", "逻辑测试", 2, "4.12.1", "逻辑测试需求项", 3,
                         logical_testing_items, "logical_testing_req_id", "logical_testing_req_desc", "other_desc"))
        self.logger.info("逻辑测试章节已添加")
    
    def _save_json(self, data: Any, filepath: str):
        """保存 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
