"""
数据转换模块
负责将需求提取结果转换为旧代码期望的格式
"""

import json
import logging
import os
from typing import List, Dict, Any, Tuple, Optional


class DataConverter:
    """
    数据转换器
    将 workflow_results 和 other_req_results 转换为 req_infos, req_item_dict, output_dict
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
    
    def convert_functional_requirements(
        self, 
        workflow_results: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        转换功能需求结果
        
        :param workflow_results: 功能需求工作流结果列表
        :return: (req_infos, req_item_dict)
        """
        req_infos = []
        req_item_dict = {}
        
        for workflow_item in workflow_results:
            chapter_title = workflow_item.get("chapter_title", "")
            result = workflow_item.get("result", {})
            final_result_str = result.get("final_result", "")
            
            if not final_result_str:
                self.logger.warning(f"章节 '{chapter_title}' 没有生成结果，跳过")
                continue
            
            try:
                # 解析 final_result JSON 字符串
                final_result_dict = json.loads(final_result_str)
                
                # 提取 fun_items
                fun_items = final_result_dict.get("fun_items", [])
                if not fun_items:
                    self.logger.warning(f"章节 '{chapter_title}' 没有 fun_items，跳过")
                    continue
                
                # 构建 req_infos 格式
                req_infos.append({
                    "title": chapter_title,
                    "response": {
                        "fun_items": fun_items,
                        "fun_req_summary": final_result_dict.get("fun_req_summary", ""),
                        "input_stream_desc": final_result_dict.get("input_stream_desc", []),
                        "output_stream_desc": final_result_dict.get("output_stream_desc", [])
                    }
                })
                
                # 构建 req_item_dict 格式（添加额外字段）
                processed_fun_items = []
                for fun_item in fun_items:
                    processed_item = {
                        "funId": fun_item.get("fun_id", ""),
                        "handling": fun_item.get("handling", ""),
                        "name": chapter_title,
                        "priority": None,
                        "sufficiency": None,
                        "testMethod": None,
                        "passCriteria": None
                    }
                    processed_fun_items.append(processed_item)
                
                req_item_dict[chapter_title] = processed_fun_items
                
            except json.JSONDecodeError as e:
                self.logger.error(f"解析章节 '{chapter_title}' 的 final_result 失败: {e}")
                continue
            except Exception as e:
                self.logger.error(f"处理章节 '{chapter_title}' 时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                continue
        
        self.logger.info(f"功能需求转换完成: {len(req_infos)} 个章节, {sum(len(items) for items in req_item_dict.values())} 个功能项")
        
        return req_infos, req_item_dict
    
    def convert_other_requirements(
        self, 
        other_req_results: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        转换其他需求结果
        
        :param other_req_results: 其他需求提取结果 {req_type: [keyword_results]}
        :return: output_dict
        """
        output_dict = {}
        
        if not other_req_results:
            return output_dict
        
        for req_type, keyword_results in other_req_results.items():
            if not keyword_results:
                continue
            
            self.logger.info(f"处理需求类型: {req_type}")
            
            # 合并所有关键词的结果
            merged_items = []
            merged_count = ""
            is_processed = False
            
            for keyword_item in keyword_results:
                keyword = keyword_item.get("keyword", "")
                result = keyword_item.get("result", {})
                final_result_str = result.get("final_result", "")
                
                if not final_result_str:
                    continue
                
                try:
                    final_result_dict = json.loads(final_result_str)
                    
                    # 根据不同的需求类型提取数据
                    if req_type == "perf_req_items":
                        items = final_result_dict.get("perf_req_items", [])
                        merged_items.extend(items)
                    elif req_type == "interface_req_items":
                        items = final_result_dict.get("interface_req_summary", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("interface_req_count", "")
                        if count:
                            merged_count = count
                    elif req_type == "reliable_req_items":
                        items = final_result_dict.get("reliable_req_summary", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("reliable_req_count", "")
                        if count:
                            merged_count = count
                    elif req_type == "margin_req_items":
                        items = final_result_dict.get("margin_req_summary", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("margin_req_count", "")
                        if count:
                            merged_count = count
                    elif req_type == "boundary_req_items":
                        items = final_result_dict.get("boundary_req_items", [])
                        merged_items.extend(items)
                    elif req_type == "safety_critical_req_items":
                        items = final_result_dict.get("all_funcs", [])
                        merged_items.extend(items)
                        count = final_result_dict.get("safety_critical_req_count", "")
                        if count:
                            merged_count = count
                    elif req_type == "recover_req_items":
                        items = final_result_dict.get("recover_req_summary", [])
                        merged_items.extend(items)
                    elif req_type in ["software_name", "use_of_interrupt", "subsystem_relation", 
                                     "cpu_storage", "software_level", "dev_platform"]:
                        # 基本信息类型 - 平铺到顶层（不再嵌套）
                        if req_type == "use_of_interrupt":
                            # use_of_interrupt 有列表，直接取列表
                            new_list = final_result_dict.get("interrupt_list", [])
                            if "use_of_interrupt" not in output_dict:
                                output_dict["use_of_interrupt"] = new_list
                            else:
                                existing = output_dict["use_of_interrupt"] if isinstance(output_dict["use_of_interrupt"], list) else []
                                output_dict["use_of_interrupt"] = existing + new_list
                        else:
                            # 其他类型：将所有字段平铺到 output_dict 顶层
                            for key, value in final_result_dict.items():
                                if key not in output_dict:
                                    output_dict[key] = value
                                else:
                                    self.logger.info(f"字段 {key} 已存在，使用第一个结果")
                        is_processed = True
                    elif req_type in ["gpio_interface", "dog_interface", "other_hardware_interface"]:
                        # 接口类型 - 直接存储数组
                        list_key = {
                            "gpio_interface": "gpio_interfaces",
                            "dog_interface": "dog_interfaces",
                            "other_hardware_interface": "other_hardware_interfaces"
                        }[req_type]
                        new_list = final_result_dict.get(list_key, [])
                        
                        if req_type not in output_dict:
                            output_dict[req_type] = new_list
                        else:
                            # 合并接口列表
                            existing = output_dict[req_type] if isinstance(output_dict[req_type], list) else []
                            output_dict[req_type] = existing + new_list
                        is_processed = True
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"解析 {req_type} 关键词 '{keyword}' 的 final_result 失败: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"处理 {req_type} 关键词 '{keyword}' 时出错: {e}")
                    continue
            
            # 对于需要合并的类型，直接存储数组（不再嵌套，计数由代码计算）
            if not is_processed and merged_items:
                # 所有需求项类型都直接存储为数组
                output_dict[req_type] = merged_items
        
        self.logger.info(f"其他需求转换完成: {len(output_dict)} 种需求类型")
        self._log_output_dict_stats(output_dict)
        
        return output_dict
    
    def _log_output_dict_stats(self, output_dict: Dict[str, Any]):
        """记录 output_dict 统计信息"""
        for req_type, data in output_dict.items():
            if isinstance(data, list):
                # 需求项数组
                self.logger.info(f"  - {req_type}: {len(data)} 项")
            elif isinstance(data, str):
                # 简单字符串类型
                self.logger.info(f"  - {req_type}: 已保存（字符串）")
            elif isinstance(data, dict):
                # 对象类型（如 software_name, cpu_storage 等）
                self.logger.info(f"  - {req_type}: 已保存（对象）")
            elif isinstance(data, list):
                self.logger.info(f"  - {req_type}: {len(data)} 项")
            else:
                self.logger.info(f"  - {req_type}: 已保存")
    
    def convert_all(
        self,
        workflow_results: List[Dict],
        other_req_results: Dict[str, List[Dict]],
        output_dir: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        转换所有需求结果
        
        :param workflow_results: 功能需求工作流结果
        :param other_req_results: 其他需求提取结果
        :param output_dir: 输出目录（可选，用于保存中间文件）
        :param project_id: 项目ID
        :return: 包含所有转换结果的字典
        """
        # 1. 转换功能需求
        req_infos, req_item_dict = self.convert_functional_requirements(workflow_results)
        
        # 2. 转换其他需求
        output_dict = self.convert_other_requirements(other_req_results)
        
        # 3. 合并到 output_dict
        output_dict['req_infos'] = req_infos
        output_dict['func_req_items'] = "{{p chapters}}"  # 占位符
        
        # 4. 保存中间文件（如果提供了输出目录）
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存 req_item_dict
            if req_item_dict:
                self._save_json(req_item_dict, os.path.join(output_dir, "func_req_item.json"))
                self.logger.info(f"功能需求项字典已保存，共 {len(req_item_dict)} 个章节")
            
            # 保存完整的 output_dict
            complete_output_file = os.path.join(output_dir, f"complete_output_dict_{project_id or 'unknown'}.json")
            self._save_json(output_dict, complete_output_file)
            self.logger.info(f"完整输出字典已保存到: {complete_output_file}")
        
        # 5. 输出转换结果摘要
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"转换结果摘要:")
        self.logger.info(f"  - req_infos: {len(req_infos)} 个章节")
        self.logger.info(f"  - req_item_dict: {len(req_item_dict)} 个章节")
        self.logger.info(f"  - output_dict: {len(output_dict)} 种需求类型（包含 req_infos）")
        self.logger.info(f"{'='*60}\n")
        
        return {
            "req_infos": req_infos,
            "req_item_dict": req_item_dict,
            "output_dict": output_dict
        }
    
    def _save_json(self, data: Any, filepath: str):
        """保存 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
