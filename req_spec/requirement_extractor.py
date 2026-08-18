"""
需求提取模块
负责从文档中提取功能需求和其他非功能需求
"""

import json
import logging
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

from req_spec.functional_requirements_agent import LabelSearchAgent
from req_spec.keyword_search_workflow import KeywordSearchWorkflow


class RequirementExtractor:
    """
    需求提取器
    统一处理功能需求和其他非功能需求的提取
    """
    
    # 其他需求类型列表
    OTHER_REQ_TYPES = [
        # 基本信息类
        "software_name",              # 软件名称等信息
        "subsystem_relation",         # 系统概述
        "cpu_storage",                # CPU和存储器信息
        "software_level",             # 软件级别等信息
        "dev_platform",               # 开发平台信息
        
        # 接口信息类
        "gpio_interface",             # GPIO接口信息
        "dog_interface",              # 看门狗接口信息
        "other_hardware_interface",   # 其他硬件接口信息
        "use_of_interrupt",           # 中断使用情况
        
        # 需求项类
        "perf_req_items",             # 性能需求
        "interface_req_items",        # 接口需求
        "reliable_req_items",         # 可靠性安全性需求
        "margin_req_items",           # 余量需求
        "boundary_req_items",         # 边界需求
        "safety_critical_req_items",  # 安全关键功能需求
        "recover_req_items",          # 恢复性需求
    ]
    
    def __init__(
        self,
        model,
        embedding_model,
        max_iterations: int = 1,
        max_workers: int = 30
    ):
        """
        初始化需求提取器
        
        :param model: LLM 模型实例
        :param embedding_model: 嵌入模型实例
        :param max_iterations: 工作流最大迭代次数
        :param max_workers: 并行处理最大线程数
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.model = model
        self.embedding_model = embedding_model
        self.max_iterations = max_iterations
        self.max_workers = max_workers
    
    def extract_functional_requirements(
        self,
        req_dict: Dict[str, Dict],
        vector_store,
        target_title: str = "功能需求",
        output_dir: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[Dict]:
        """
        提取功能需求
        
        :param req_dict: 需求文档字典 {title: {content: "...", title_no: "..."}}
        :param vector_store: 向量存储实例
        :param target_title: 目标标题（默认"功能需求"）
        :param output_dir: 输出目录（可选）
        :param project_id: 项目ID（用于保存文件）
        :return: 工作流结果列表
        """
        self.logger.info(f"开始提取功能需求，目标标题: {target_title}")
        
        # 1. 使用 LabelSearchAgent 搜索功能需求相关章节
        table_of_contents = list(req_dict.keys())
        chapters_dict = {title: info.get('content', '') for title, info in req_dict.items()}
        
        agent = LabelSearchAgent(chapters_dict=chapters_dict, model=self.model)
        result = agent.analyze(table_of_contents=table_of_contents, label=target_title)
        matched_chapters = result.matched_chapters
        self.logger.info(f"LabelSearchAgent 返回的章节: {matched_chapters}")
        
        # 过滤掉不在 req_dict 中的章节（防止 LLM 幻觉）
        valid_chapters = [ch for ch in matched_chapters if ch in req_dict]
        invalid_chapters = [ch for ch in matched_chapters if ch not in req_dict]
        if invalid_chapters:
            self.logger.warning(f"以下章节不存在于文档中，已忽略: {invalid_chapters}")
        matched_chapters = valid_chapters
        self.logger.info(f"有效的匹配章节: {matched_chapters}")
        
        if not matched_chapters:
            self.logger.warning(f"未找到与'{target_title}'相关的有效章节")
            return []
        
        # 2. 并行处理每个匹配的章节
        generation_requirement = "生成详细的功能需求规格说明，包括功能描述、输入输出、处理流程、异常处理等完整信息"
        
        def process_chapter(chapter_title):
            """处理单个章节"""
            try:
                self.logger.info(f"正在处理章节: {chapter_title}")
                
                chapter_workflow = KeywordSearchWorkflow(
                    model=self.model,
                    embeddings=self.embedding_model,
                    main_doc_dict=req_dict,
                    global_vector_store=vector_store,
                    generation_requirement=generation_requirement,
                    max_iterations=self.max_iterations
                )
                result = chapter_workflow.run(
                    keyword=chapter_title,
                    generation_requirement=generation_requirement
                )
                
                self.logger.info(f"章节 '{chapter_title}' 处理完成")
                self.logger.info(f"  - 是否找到匹配章节: {result.get('matched_chapter') is not None}")
                self.logger.info(f"  - 内容是否满足要求: {result.get('evaluation', {}).get('sufficient', False)}")
                
                final_result = result.get('final_result', '')
                if final_result:
                    self.logger.info(f"    ✓ 生成结果长度: {len(final_result)} 字符")
                
                return {
                    "chapter_title": chapter_title,
                    "result": result
                }
            except Exception as e:
                self.logger.error(f"处理章节 '{chapter_title}' 时出错: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {
                    "chapter_title": chapter_title,
                    "result": {"error": str(e)}
                }
        
        # 3. 使用线程池并行处理
        workflow_results = []
        actual_workers = min(self.max_workers, len(matched_chapters))
        self.logger.info(f"使用 {actual_workers} 个线程并行处理 {len(matched_chapters)} 个章节")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(process_chapter, chapter_title): chapter_title 
                      for chapter_title in matched_chapters}
            
            results_dict = {}
            for future in concurrent.futures.as_completed(futures):
                chapter_title = futures[future]
                try:
                    result = future.result()
                    results_dict[chapter_title] = result
                except Exception as e:
                    self.logger.error(f"章节 '{chapter_title}' 处理失败: {e}")
                    results_dict[chapter_title] = {
                        "chapter_title": chapter_title,
                        "result": {"error": str(e)}
                    }
            
            # 按照原始顺序排列结果
            for chapter_title in matched_chapters:
                if chapter_title in results_dict:
                    workflow_results.append(results_dict[chapter_title])
        
        # 4. 保存结果（如果提供了输出目录）
        if workflow_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"workflow_results_{project_id or 'unknown'}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✓ 功能需求结果已保存到: {output_file}")
        
        # 5. 统计信息
        total_chapters = len(workflow_results)
        matched_count = sum(1 for r in workflow_results if r['result'].get('matched_chapter') is not None)
        sufficient_count = sum(1 for r in workflow_results if r['result'].get('evaluation', {}).get('sufficient', False))
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"功能需求提取完成:")
        self.logger.info(f"  - 总章节数: {total_chapters}")
        self.logger.info(f"  - 找到匹配章节: {matched_count}/{total_chapters}")
        self.logger.info(f"  - 内容满足要求: {sufficient_count}/{total_chapters}")
        self.logger.info(f"{'='*60}")
        
        return workflow_results
    
    def extract_other_requirements(
        self,
        req_dict: Dict[str, Dict],
        vector_store,
        req_types: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        提取其他非功能需求（性能需求、接口需求等）
        
        :param req_dict: 需求文档字典
        :param vector_store: 向量存储实例
        :param req_types: 需要处理的需求类型列表（默认使用 OTHER_REQ_TYPES）
        :param output_dir: 输出目录（可选）
        :param project_id: 项目ID
        :return: 各需求类型的结果字典 {req_type: [results]}
        """
        self.logger.info("开始提取其他非功能需求信息...")
        
        # 导入提示词配置
        try:
            from req_spec.prompts.other_prompt import prompt_registry, get_output_model
        except ImportError as e:
            self.logger.warning(f"无法导入提示词字典: {e}，跳过其他需求提取")
            return {}
        
        prompts = prompt_registry
        req_types = req_types or self.OTHER_REQ_TYPES
        other_req_results = {}
        
        for req_type in req_types:
            if req_type not in prompts:
                self.logger.warning(f"  跳过 {req_type}：未在 prompts 中找到")
                continue
            
            self.logger.info(f"处理需求类型: {req_type}")
            prompt_config = prompts[req_type]
            
            # 获取输出模型类
            output_model_class = get_output_model(req_type)
            if output_model_class is None:
                self.logger.warning(f"  {req_type} 没有对应的输出模型，跳过")
                continue
            
            # 提取关键词
            doc_content = prompt_config.get("doc_content", "").strip()
            if not doc_content:
                self.logger.warning(f"  {req_type} 的 doc_content 为空，跳过")
                continue
            
            keywords = [k.strip() for k in doc_content.split(',') if k.strip()]
            if not keywords:
                self.logger.warning(f"  {req_type} 没有有效关键词，跳过")
                continue
            
            # 构建 prompt_template
            prompt_template = {
                "system_prompt": prompt_config.get("system_prompt", ""),
                "user_prompt": prompt_config.get("user_prompt", ""),
                "extract_content": "",
            }
            
            # 处理单个关键词的函数
            def process_keyword(keyword, req_type=req_type, prompt_template=prompt_template, 
                              output_model_class=output_model_class):
                try:
                    self.logger.info(f"  处理关键词: {keyword}")
                    
                    keyword_workflow = KeywordSearchWorkflow(
                        model=self.model,
                        embeddings=self.embedding_model,
                        main_doc_dict=req_dict,
                        global_vector_store=vector_store,
                        generation_requirement=f"从文档中提取{req_type}相关信息",
                        max_iterations=self.max_iterations,
                        prompt_template=prompt_template,
                        output_model=output_model_class,
                        skip_chapter_match=True
                    )
                    
                    result = keyword_workflow.run(
                        keyword=keyword,
                        generation_requirement=f"从文档中提取{req_type}相关信息"
                    )
                    
                    final_result = result.get('final_result', '')
                    if final_result:
                        self.logger.info(f"    ✓ 生成结果长度: {len(final_result)} 字符")
                    
                    return {
                        "keyword": keyword,
                        "result": result
                    }
                except Exception as e:
                    self.logger.error(f"处理关键词 '{keyword}' 时出错: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    return {
                        "keyword": keyword,
                        "result": {"error": str(e)}
                    }
            
            # 并行处理关键词
            req_type_results = []
            actual_workers = min(self.max_workers, len(keywords))
            self.logger.info(f"  使用 {actual_workers} 个线程并行处理 {len(keywords)} 个关键词")
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                futures = {executor.submit(process_keyword, keyword): keyword 
                          for keyword in keywords}
                
                results_dict = {}
                for future in concurrent.futures.as_completed(futures):
                    keyword = futures[future]
                    try:
                        result = future.result()
                        results_dict[keyword] = result
                    except Exception as e:
                        self.logger.error(f"关键词 '{keyword}' 处理失败: {e}")
                        results_dict[keyword] = {
                            "keyword": keyword,
                            "result": {"error": str(e)}
                        }
                
                # 按照原始顺序排列结果
                for keyword in keywords:
                    if keyword in results_dict:
                        req_type_results.append(results_dict[keyword])
            
            other_req_results[req_type] = req_type_results
            self.logger.info(f"  ✓ {req_type} 处理完成，共处理 {len(keywords)} 个关键词")
        
        # 保存结果
        if other_req_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"other_req_results_{project_id or 'unknown'}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(other_req_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✓ 其他需求类型结果已保存到: {output_file}")
        
        return other_req_results
    
    def extract_all(
        self,
        req_dict: Dict[str, Dict],
        vector_store,
        output_dir: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        提取所有需求（功能需求 + 其他需求）
        
        :param req_dict: 需求文档字典
        :param vector_store: 向量存储实例
        :param output_dir: 输出目录
        :param project_id: 项目ID
        :return: 包含所有结果的字典
        """
        # 提取功能需求
        workflow_results = self.extract_functional_requirements(
            req_dict=req_dict,
            vector_store=vector_store,
            output_dir=output_dir,
            project_id=project_id
        )
        
        # 提取其他需求
        other_req_results = self.extract_other_requirements(
            req_dict=req_dict,
            vector_store=vector_store,
            output_dir=output_dir,
            project_id=project_id
        )
        
        return {
            "workflow_results": workflow_results,
            "other_req_results": other_req_results
        }
