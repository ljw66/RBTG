"""
基于关键词的智能检索工作流
使用 LangGraph 实现多轮检索和评估流程

工作流程：
1. 用户提供关键词 → 查找目录中是否有匹配的章节
2. 如果有匹配 → 提取章节内容
3. 大模型评估内容是否满足生成要求
4. 如果不满足 → 大模型提取需要检索的关键词
5. 执行语义检索 → 获取更多内容
6. 合并内容并再次评估
7. 满足要求 → 生成最终内容
"""

import sys
import os
import logging
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 导入测试需求项生成提示词
from req_spec.prompts.req_prompt import req_prompts, TestRequirementOutput


# ==================== 状态定义 ====================

class KeywordSearchState(MessagesState):
    """关键词检索工作流的状态"""
    keyword: str  # 用户提供的关键词
    generation_requirement: str  # 生成要求提示词
    matched_chapter: Optional[Dict[str, Any]]  # 匹配到的章节信息 {title: str, content: str, ...}
    chapter_content: str  # 章节内容
    evaluation: Dict[str, Any]  # 评估结果 {sufficient: bool, missing_aspects: List[str], reason: str}
    missing_keywords: List[str]  # 需要检索的关键词列表
    retrieved_content: List[Dict[str, Any]]  # 语义检索到的内容
    all_content: str  # 合并后的所有内容（章节内容 + 检索内容）
    final_result: str  # 最终生成的内容
    max_iterations: int  # 最大迭代次数（防止无限循环）
    current_iteration: int  # 当前迭代次数


# ==================== Pydantic 模型定义 ====================

class ContentEvaluation(BaseModel):
    """内容评估结果"""
    sufficient: bool = Field(description="内容是否满足生成要求")
    missing_aspects: List[str] = Field(description="缺失的关键信息点或关键词列表")
    reason: str = Field(description="评估理由")


class KeywordExtraction(BaseModel):
    """关键词提取结果"""
    keywords: List[str] = Field(description="需要检索的关键词列表，用于补充缺失的信息")


# ==================== 辅助函数 ====================

def create_vector_store_from_dict(doc_dict: Dict[str, Dict[str, Any]], embeddings: Embeddings) -> InMemoryVectorStore:
    """
    从文档字典创建 LangChain 向量存储
    
    :param doc_dict: 文档字典，格式为 {标题: {content: "文本内容", title_no: "1.1", level: 2}}
    :param embeddings: LangChain 嵌入模型
    :return: InMemoryVectorStore 对象
    """
    documents = []
    
    for title, info in doc_dict.items():
        content = info.get('content', '')
        title_no = info.get('title_no', '')
        level = info.get('level', 1)
        
        # 只处理有内容的条目
        if content:
            # 截断文本以适配embedding模型的输入长度限制
            truncated_content = content[:5000] if len(content) > 5000 else content
            
            doc = Document(
                page_content=truncated_content,
                metadata={
                    "title": title,
                    "level": level,
                    "title_no": title_no,
                    "path": title,
                    "original_length": len(content),
                    "truncated": len(content) > 5000
                }
            )
            documents.append(doc)
    
    # 创建向量存储
    vector_store = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vector_store


# ==================== 工作流类 ====================

class KeywordSearchWorkflow:
    """基于关键词的智能检索工作流（通用版本，支持可配置的 prompt 和输出模型）"""
    
    def __init__(
        self,
        model,
        embeddings: Embeddings,
        main_doc_dict: Dict[str, Dict[str, Any]],
        global_vector_store: Optional[VectorStore] = None,
        generation_requirement: str = "",
        max_iterations: int = 3,
        prompt_template: Optional[Dict[str, str]] = None,
        output_model: Optional[type] = None,
        skip_chapter_match: bool = False
    ):
        """
        初始化工作流
        
        :param model: LangChain 聊天模型
        :param embeddings: LangChain 嵌入模型
        :param main_doc_dict: 主线文档字典（用于章节匹配），格式为 {标题: {content: "文本内容", title_no: "1.1", level: 2}}
        :param global_vector_store: 全局向量存储（用于语义检索，包含所有文档）。如果为None，会从main_doc_dict创建
        :param generation_requirement: 生成要求提示词
        :param max_iterations: 最大迭代次数
        :param prompt_template: 可选的 prompt 模板字典，格式为 {"system_prompt": "...", "user_prompt": "...", "extract_content": "..."}
                               如果不提供，默认使用 req_prompt.py 中的提示词
        :param output_model: 可选的 Pydantic 输出模型类，如果不提供，默认使用 TestRequirementOutput
        :param skip_chapter_match: 是否跳过章节匹配（对于没有明确章节的需求类型，设置为 True）
        """
        self.main_doc_dict = main_doc_dict  # 主线文档（用于章节匹配）
        self.generation_requirement = generation_requirement
        self.max_iterations = max_iterations
        self.skip_chapter_match = skip_chapter_match
        self.logger = logging.getLogger(__name__)
        
        # 设置 prompt 模板（如果未提供，使用默认的 req_prompt.py）
        if prompt_template is None:
            from req_spec.prompts.req_prompt import req_prompts
            self.prompt_template = req_prompts
        else:
            self.prompt_template = prompt_template
        
        # 设置输出模型（如果未提供，使用默认的 TestRequirementOutput）
        if output_model is None:
            from req_spec.prompts.req_prompt import TestRequirementOutput
            self.output_model = TestRequirementOutput
        else:
            self.output_model = output_model
        
        # 创建或使用全局向量存储（用于语义检索）
        if global_vector_store is None:
            self.logger.warning("未提供全局向量库，将从主线文档创建 InMemoryVectorStore（仅包含主线文档）")
            self.global_vector_store = create_vector_store_from_dict(main_doc_dict, embeddings)
        else:
            self.global_vector_store = global_vector_store
        
        # 创建检索器（从全局向量库检索，包含所有文档）
        self.retriever = self.global_vector_store.as_retriever(search_kwargs={"k": 10})  # 增加检索数量，后续会按优先级排序
        
        # 初始化 LLM
        self.llm = model
        
        # 构建工作流图
        self.graph = self._build_graph()
    
    # ==================== 节点函数 ====================
    
    def chapter_match_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """节点1: 在目录中查找匹配的章节（只在主线文档中查找）"""
        keyword = state.get("keyword", "")
        
        # 如果设置了跳过章节匹配，直接返回空匹配
        if self.skip_chapter_match:
            self.logger.info(f"节点1: 跳过章节匹配（skip_chapter_match=True）")
            return {
                "matched_chapter": None,
                "chapter_content": ""
            }
        
        self.logger.info(f"节点1: 查找章节 - 关键词: '{keyword}'")
        self.logger.info(f"  主线文档字典中的章节数量: {len(self.main_doc_dict)}")
        
        # 打印所有可用的章节标题（用于调试）
        if self.main_doc_dict:
            all_titles = list(self.main_doc_dict.keys())
            self.logger.info(f"  所有章节标题: {all_titles}")
        else:
            self.logger.warning(f"  主线文档字典为空！")
            return {
                "matched_chapter": None,
                "chapter_content": ""
            }
        
        matched_chapter = None
        chapter_content = ""
        
        # 首先尝试精确匹配（标题完全相等）
        if keyword in self.main_doc_dict:
            info = self.main_doc_dict[keyword]
            matched_chapter = {
                "title": keyword,
                "content": info.get("content", ""),
                "title_no": info.get("title_no", ""),
                "level": info.get("level", 1),
                "source": "需求规格说明书"
            }
            chapter_content = info.get("content", "")
            self.logger.info(f"  ✓ 精确匹配找到章节: {keyword} (来源: 需求规格说明书)")
        else:
            # 如果精确匹配失败，尝试模糊匹配（标题包含关键词）
            keyword_lower = keyword.lower().strip()
            self.logger.info(f"  精确匹配失败，尝试模糊匹配，关键词（小写）: '{keyword_lower}'")
            
            for title, info in self.main_doc_dict.items():
                title_lower = title.lower().strip()
                # 检查标题是否包含关键词（不区分大小写）
                if keyword_lower in title_lower or title_lower in keyword_lower:
                    matched_chapter = {
                        "title": title,
                        "content": info.get("content", ""),
                        "title_no": info.get("title_no", ""),
                        "level": info.get("level", 1),
                        "source": "需求规格说明书"
                    }
                    chapter_content = info.get("content", "")
                    self.logger.info(f"  ✓ 模糊匹配找到章节: '{title}' (来源: 需求规格说明书)")
                    self.logger.info(f"    关键词: '{keyword}' -> 匹配标题: '{title}'")
                    break
        
        if not matched_chapter:
            self.logger.warning(f"  ✗ 未找到匹配的章节，关键词: '{keyword}'")
            self.logger.info(f"  可用的章节标题示例（前10个）: {list(self.main_doc_dict.keys())[:10]}")
            # 尝试找到最相似的标题
            keyword_lower = keyword.lower().strip()
            similar_titles = []
            for title in self.main_doc_dict.keys():
                title_lower = title.lower().strip()
                # 计算相似度（简单的字符重叠）
                if keyword_lower and title_lower:
                    common_chars = set(keyword_lower) & set(title_lower)
                    similarity = len(common_chars) / max(len(keyword_lower), len(title_lower))
                    if similarity > 0.3:  # 相似度超过30%
                        similar_titles.append((title, similarity))
            if similar_titles:
                similar_titles.sort(key=lambda x: x[1], reverse=True)
                self.logger.info(f"  最相似的标题（前3个）: {[t[0] for t in similar_titles[:3]]}")
        
        return {
            "matched_chapter": matched_chapter,
            "chapter_content": chapter_content
        }
    
    def evaluate_content_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """节点2: 评估章节内容是否满足生成要求"""
        self.logger.info(f"节点2: 评估内容 (迭代 {state.get('current_iteration', 0)+1}/{state.get('max_iterations', self.max_iterations)})")
        
        # 优先使用合并后的内容（all_content），如果没有则使用章节内容
        all_content = state.get("all_content", "")
        chapter_content = state.get("chapter_content", "")
        generation_requirement = state.get("generation_requirement", "")
        
        # 确定要评估的内容
        content_to_evaluate = all_content if all_content else chapter_content
        
        if not content_to_evaluate:
            # 如果没有任何内容，直接返回不满足
            return {
                "evaluation": {
                    "sufficient": False,
                    "missing_aspects": ["需要基本信息"],
                    "reason": "未找到匹配的章节内容"
                },
                "current_iteration": state.get("current_iteration", 0) + 1  # 更新迭代次数
            }
        
        # 构建评估提示词
        system_prompt = """
        你是一位专业的嵌入式软件测试工程师，专门负责评估功能需求内容是否足够生成测试需求项（TR项）。
        
        你的任务是：
        1. 判断当前收集到的内容是否足够生成完整的测试需求项（TR项）
        2. 测试需求项需要将功能需求拆解为"输入流-处理-输出流"的结构，包含以下关键信息：
           
           【功能概述（fun_req_summary）】
           - 对该小节的功能概述，清晰说明该功能是什么，解决什么问题
           
           【输入流说明（input_stream_desc）】
           - 明确功能的输入流，包括输入参数、数据格式、输入来源等
           - 需要补足细节信息，如文档中提到"规定的存储空间"需要找到具体的地址范围
           
           【处理项（fun_items）】
           - 详细描述功能的处理逻辑、执行步骤、业务流程、逻辑判断
           - 每个处理子项需要单独生成一个测试需求项（TR项）
           - 处理逻辑要完整，不能有缺项
           - 需要补足细节信息，如具体的参数值、地址范围、时间周期等
           
           【输出流说明（output_stream_desc）】
           - 明确功能的输出流，包括输出结果、数据格式、输出目标等
           - 需要补足细节信息，如具体的输出格式、协议类型等
        
        3. 文档使用原则：
           - 需求规格说明书是主要文档，优先使用
           - 任务书、接口协议等是参考文档，用于补充主要文档中缺失的信息
           - 如果主要文档内容不足，才使用参考文档进行补充
        
        4. 评估标准：
           - 内容充分：能够基于现有内容生成完整的测试需求项，包含输入流、处理逻辑、输出流，且细节信息完整
           - 内容不足：缺少关键信息（如输入流、处理逻辑、输出流中的任一项），或缺少必要的细节信息（如地址范围、参数值等），无法生成完整测试需求项
        
        5. 如果内容不足，请明确指出缺少哪些具体方面的信息，并列出用于检索的关键词。
        
        请以 JSON 格式输出评估结果。
        """
        
        user_prompt = f"""
        【任务背景】
        这是一个测试需求项（TR项）生成任务。系统正在从多个文档（需求规格说明书、任务书、接口协议等）中收集信息，
        目标是将功能需求拆解为"输入流-处理-输出流"的结构，生成完整的测试需求项。
        
        【生成要求】
        {generation_requirement}
        
        【当前收集到的内容】
        以下是从文档中收集到的内容：
        - 章节内容：从需求规格说明书中通过标题匹配找到的章节内容（这是主要文档，优先使用）
        - 检索内容：从全局向量库中通过语义检索得到的补充内容（可能来自需求规格说明书、任务书、接口协议等所有文档）
        
        {content_to_evaluate}
        
        【评估任务】
        请评估上述内容是否充足，能够生成完整的测试需求项，包括：
        1. 功能概述：是否能够提取或总结出功能概述
        2. 输入流说明：是否能够明确输入流的信息（如果文档中没有明确说明，需要判断是否可以合理推断）
        3. 处理逻辑：是否能够提取或总结出完整的处理逻辑，包括所有必要的步骤和细节信息
        4. 输出流说明：是否能够明确输出流的信息（如果文档中没有明确说明，需要判断是否可以合理推断）
        5. 细节信息：是否包含必要的细节信息（如地址范围、参数值、时间周期等），如果文档中提到但未明确，需要判断是否可以从其他文档中补充
        
        【输出要求】
        请以 JSON 格式输出评估结果：
        - 如果内容充分（sufficient: true）：missing_aspects 为空列表 []，reason 说明内容已满足要求，可以生成完整的测试需求项
        - 如果内容不足（sufficient: false）：missing_aspects 列出缺失的关键信息点或用于检索的关键词，reason 详细说明缺少哪些方面的信息（如缺少输入流、处理逻辑不完整、缺少细节信息等）
        
        输出格式：
        {{
            "sufficient": true/false,
            "missing_aspects": ["关键词1", "关键词2", ...],
            "reason": "详细的评估理由，说明内容是否充分，如果不足则说明缺少哪些方面的信息（如缺少输入流、处理逻辑不完整、缺少输出流、缺少细节信息等）"
        }}
        """
        
        try:
            # 使用结构化输出
            response = (
                self.llm
                .with_structured_output(ContentEvaluation)
                .invoke([{"role": "user", "content": system_prompt + "\n\n" + user_prompt}])
            )
            
            evaluation = {
                "sufficient": response.sufficient,
                "missing_aspects": response.missing_aspects,
                "reason": response.reason
            }
            
            self.logger.info(f"  评估结果: {'满足' if evaluation['sufficient'] else '不满足'}")
            if not evaluation['sufficient']:
                self.logger.info(f"  缺失方面: {evaluation['missing_aspects']}")
            
            # 更新迭代次数并返回
            current_iteration = state.get("current_iteration", 0) + 1
            
            # 保留 matched_chapter 和 chapter_content，确保状态完整传递
            return {
                "evaluation": evaluation,
                "current_iteration": current_iteration,
                # 保留 matched_chapter 和 chapter_content，确保状态完整传递
                "matched_chapter": state.get("matched_chapter"),
                "chapter_content": state.get("chapter_content", "")
            }
            
        except Exception as e:
            self.logger.error(f"内容评估出错：{e}")
            current_iteration = state.get("current_iteration", 0) + 1
            return {
                "evaluation": {
                    "sufficient": False,
                    "missing_aspects": ["评估出错"],
                    "reason": f"Error: {str(e)}"
                },
                "current_iteration": current_iteration,
                # 保留 matched_chapter 和 chapter_content
                "matched_chapter": state.get("matched_chapter"),
                "chapter_content": state.get("chapter_content", "")
            }
    
    def extract_keywords_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """节点3: 从评估结果中提取需要检索的关键词"""
        self.logger.info("节点3: 提取检索关键词")
        
        evaluation = state.get("evaluation", {})
        missing_aspects = evaluation.get("missing_aspects", [])
        generation_requirement = state.get("generation_requirement", "")
        keyword = state.get("keyword", "")  # 原始关键词（章节名）
        chapter_content = state.get("chapter_content", "")  # 当前已有的章节内容
        
        if not missing_aspects:
            return {"missing_keywords": []}
        
        # 构建关键词提取提示词
        system_prompt = """
        你是一位信息检索专家，负责根据缺失的信息点和当前上下文，提取适合向量语义检索的关键词。
        
        向量语义检索的特点：
        1. 使用语义相似度匹配，能够理解同义词和相关概念
        2. 关键词应该具体、明确，能够准确表达要查找的信息
        3. 关键词应该符合技术文档中的常见表述方式
        4. 可以提取短语或短句，而不仅仅是单个词
        
        请以 JSON 格式输出结果。
        """
        
        user_prompt = f"""
        【任务背景】
        这是一个测试需求项生成任务，需要从多个文档（需求规格说明书、任务书、接口协议等）中检索信息。
        
        【生成要求】
        {generation_requirement}
        
        【当前章节】
        正在处理的章节：{keyword}
        
        【当前已有内容】
        {chapter_content[:500] if chapter_content else '暂无章节内容'}
        
        【缺失的信息点】
        {', '.join(missing_aspects)}
        
        【提取要求】
        请根据缺失的信息点和当前上下文，提取适合向量语义检索的关键词。
        
        关键词提取原则：
        1. **具体明确**：关键词应该具体表达要查找的信息，而不是过于抽象的概念
           - 好：输入流、输出流、处理流程、异常处理、存储地址范围、接口协议
           - 差：信息、内容、数据
        
        2. **符合文档表述**：关键词应该符合技术文档中的常见表述方式
           - 好：APID判断、遥测包、数据池、存储空间地址
           - 差：判断、包、池、地址
        
        3. **语义完整**：可以提取短语或短句，保持语义完整性
           - 好：输入流说明、输出流格式、处理逻辑步骤、异常处理机制
           - 差：输入、输出、处理、异常
        
        4. **结合上下文，具体化关键词**：这是最重要的原则！
           - 必须结合当前章节的主题，提取与章节具体相关但缺失的关键词
           - 不能提取泛泛的关键词，必须具体到当前章节的功能
           - 示例：
             * 如果当前章节是"数据查询功能"，缺失"输入流说明"
               ✅ 好：["数据查询输入流", "数据查询输入参数", "查询接口输入"]
               ❌ 差：["输入流", "输入参数", "输入数据格式"]（太泛泛，会检索到所有功能的输入流）
             
             * 如果当前章节是"用户登录功能"，缺失"处理逻辑细节"
               ✅ 好：["用户登录处理流程", "登录验证步骤", "身份认证逻辑"]
               ❌ 差：["处理流程", "处理步骤", "业务逻辑"]（太泛泛）
             
             * 如果当前章节是"遥测数据处理"，缺失"存储地址范围"
               ✅ 好：["遥测数据存储地址", "遥测数据池地址范围", "遥测存储空间"]
               ❌ 差：["存储空间地址", "地址范围", "内存地址"]（太泛泛）
        
        5. **数量控制**：提取3-5个关键词，优先提取最重要的、最可能找到的信息
        
        【输出格式】
        请以 JSON 格式输出，格式如下：
        {{
            "keywords": ["关键词1", "关键词2", ...]
        }}
        
        【重要提醒】
        所有关键词都必须结合当前章节"{keyword}"的主题，不能提取泛泛的关键词！
        例如：如果当前章节是"数据查询功能"，所有关键词都应该包含"数据查询"或"查询"等与章节相关的词汇。
        """
        
        try:
            response = (
                self.llm
                .with_structured_output(KeywordExtraction)
                .invoke([{"role": "user", "content": system_prompt + "\n\n" + user_prompt}])
            )
            
            keywords = response.keywords
            self.logger.info(f"  提取的关键词: {keywords}")
            
            return {"missing_keywords": keywords}
            
        except Exception as e:
            self.logger.error(f"关键词提取出错：{e}")
            # 如果提取失败，直接使用缺失的信息点作为关键词
            return {"missing_keywords": missing_aspects[:5]}
    
    def semantic_search_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """节点4: 执行语义检索（从全局向量库检索，包含所有文档）"""
        self.logger.info("节点4: 执行语义检索（全局向量库）")
        
        missing_keywords = state.get("missing_keywords", [])
        keyword = state.get("keyword", "")  # 如果没有关键词，使用原始关键词
        
        # 如果没有缺失关键词，使用原始关键词进行检索
        if not missing_keywords:
            if keyword:
                query = keyword
                self.logger.info(f"  使用原始关键词检索: {query}")
            else:
                self.logger.warning("  没有检索关键词，返回空结果")
                return {"retrieved_content": []}
        else:
            # 合并关键词构建查询
            query = " ".join(missing_keywords)
            self.logger.info(f"  检索查询: {query}")
        
        # 截断查询文本（如果需要）
        truncated_query = query[:5000] if len(query) > 5000 else query
        
        # 执行向量检索（从全局向量库检索，包含所有文档）
        retrieved_docs = self.retriever.invoke(truncated_query)
        
        # 处理检索结果，标注来源并按优先级排序
        retrieved_content = []
        for doc in retrieved_docs:
            # 从 metadata 中获取来源，如果没有则默认为"其他来源"
            source = doc.metadata.get("source", "其他来源")
            
            # 获取相似度分数（如果可用）
            score = 0.0
            if hasattr(doc, 'score'):
                score = doc.score
            elif hasattr(doc, 'metadata') and 'score' in doc.metadata:
                score = doc.metadata.get('score', 0.0)
            
            retrieved_content.append({
                "content": doc.page_content,
                "title": doc.metadata.get("title", ""),
                "title_no": doc.metadata.get("title_no", ""),
                "score": score,
                "source": source  # 标注来源
            })
        
        # 按优先级排序：需求规格说明书优先，其他文档其次，相同来源内按相似度分数排序
        def sort_key(item):
            source = item.get("source", "其他来源")
            score = item.get("score", 0.0)
            # 需求规格说明书优先级最高（返回较小的值）
            if source == "需求规格说明书":
                priority = 0
            else:
                priority = 1
            # 相同优先级内，按相似度分数降序排序（分数越高越好）
            return (priority, -score)
        
        retrieved_content.sort(key=sort_key)
        
        # 统计各来源的数量
        source_counts = {}
        for item in retrieved_content:
            source = item.get("source", "其他来源")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        self.logger.info(f"  检索到 {len(retrieved_content)} 条结果")
        self.logger.info(f"  来源统计: {source_counts}")
        
        return {"retrieved_content": retrieved_content}
    
    def merge_content_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """节点5: 合并章节内容和检索内容"""
        self.logger.info("节点5: 合并内容")
        
        chapter_content = state.get("chapter_content", "")
        retrieved_content = state.get("retrieved_content", [])
        
        # 合并所有内容
        all_parts = []
        
        if chapter_content:
            all_parts.append(f"=== 章节内容 ===\n{chapter_content}")
        
        if retrieved_content:
            all_parts.append("\n=== 检索到的补充内容 ===")
            # 按来源分组显示
            current_source = None
            for item in retrieved_content:
                source = item.get('source', '未知来源')
                title = item.get('title', '未知标题')
                title_no = item.get('title_no', '')
                
                # 如果来源改变，添加来源标题
                if source != current_source:
                    all_parts.append(f"\n【来源: {source}】")
                    current_source = source
                
                # 添加内容（包含标题编号）
                title_with_no = f"{title_no} {title}" if title_no else title
                all_parts.append(f"\n{title_with_no}\n{item.get('content', '')}")
        
        all_content = "\n\n".join(all_parts)
        
        self.logger.info(f"  合并后的内容长度: {len(all_content)} 字符")
        
        return {"all_content": all_content}
    
    def generate_final_result_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """节点6: 生成最终内容（使用 req_prompt.py 中的提示词生成测试需求项）"""
        self.logger.info("节点6: 生成最终内容（测试需求项）")
        
        all_content = state.get("all_content", "")
        chapter_content = state.get("chapter_content", "")
        retrieved_content = state.get("retrieved_content", [])
        keyword = state.get("keyword", "")
        
        # 如果 all_content 为空但 chapter_content 不为空，使用 chapter_content
        if not all_content and chapter_content:
            self.logger.info("  all_content 为空，使用 chapter_content 作为内容")
            all_content = f"=== 章节内容 ===\n{chapter_content}"
        
        if not all_content:
            self.logger.warning("  没有可用内容，无法生成最终结果")
            # 返回一个空的但有效的 JSON 结构
            empty_result = {
                "fun_req_summary": "",
                "input_stream_desc": [],
                "output_stream_desc": [],
                "fun_items": []
            }
            import json
            return {"final_result": json.dumps(empty_result, ensure_ascii=False)}
        
        # 格式化文档内容，标注主要文档和参考文档
        # 章节内容（从需求规格说明书中匹配的）是主要文档，相似度得分标记为1
        # 检索内容（从向量库检索的）是参考文档，相似度得分标记为<1
        formatted_doc_content = []
        
        # 添加主要文档（章节内容）
        if chapter_content:
            formatted_doc_content.append(f"```主要文档（相似度得分=1）```\n{chapter_content}")
        
        # 添加参考文档（检索内容）
        if retrieved_content:
            formatted_doc_content.append("\n```参考文档（相似度得分<1）```")
            current_source = None
            for item in retrieved_content:
                source = item.get('source', '未知来源')
                title = item.get('title', '未知标题')
                title_no = item.get('title_no', '')
                content = item.get('content', '')
                score = item.get('score', 0.0)
                
                # 如果来源改变，添加来源标题
                if source != current_source:
                    formatted_doc_content.append(f"\n【来源: {source}】")
                    current_source = source
                
                # 添加内容（包含标题编号和相似度分数）
                title_with_no = f"{title_no} {title}" if title_no else title
                formatted_doc_content.append(f"\n{title_with_no}（相似度得分: {score:.4f}）\n{content}")
        
        doc_content_str = "\n\n".join(formatted_doc_content)
        
        # 限制 doc_content 的长度，避免输入过长导致输出被截断
        # 根据经验，保留前 15000 字符通常足够（可以根据实际情况调整）
        max_doc_content_length = 15000
        if len(doc_content_str) > max_doc_content_length:
            self.logger.warning(f"  文档内容过长（{len(doc_content_str)} 字符），截断至 {max_doc_content_length} 字符")
            doc_content_str = doc_content_str[:max_doc_content_length] + "\n\n[注：文档内容已截断，仅保留前部分内容]"
        
        # 使用配置的 prompt 模板生成内容
        system_prompt = self.prompt_template.get("system_prompt", "")
        user_prompt_template = self.prompt_template.get("user_prompt", "")
        extract_content = self.prompt_template.get("extract_content", "")
        
        # 格式化用户提示词
        try:
            user_prompt = user_prompt_template.format(
                doc_content=doc_content_str,
                extract_content=extract_content
            )
        except KeyError as e:
            self.logger.error(f"格式化 user_prompt 时缺少占位符: {e}")
            # 尝试只使用 doc_content
            user_prompt = user_prompt_template.format(doc_content=doc_content_str)
        
        # 使用结构化输出生成内容（使用配置的输出模型）
        # 添加错误处理和重试机制
        max_retries = 3
        response = None
        final_result = ""
        
        for attempt in range(max_retries):
            try:
                response = (
                    self.llm
                    .with_structured_output(self.output_model)
                    .invoke([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ])
                )
                
                # 将 Pydantic 对象转换为字典，再转换为 JSON 字符串
                final_result_dict = response.model_dump()
                import json
                final_result = json.dumps(final_result_dict, ensure_ascii=False, indent=4)
                
                self.logger.info(f"  生成完成，输出结构: {list(final_result_dict.keys())}")
                # 根据不同的输出模型，记录不同的字段
                if 'fun_items' in final_result_dict:
                    self.logger.info(f"  生成的处理项数量: {len(final_result_dict['fun_items'])}")
                elif 'perf_req_items' in final_result_dict:
                    self.logger.info(f"  生成的性能需求项数量: {len(final_result_dict['perf_req_items'])}")
                elif 'interface_req_summary' in final_result_dict:
                    self.logger.info(f"  生成的接口需求项数量: {len(final_result_dict['interface_req_summary'])}")
                
                break  # 成功生成，退出重试循环
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"  生成内容时出错（尝试 {attempt + 1}/{max_retries}）: {error_msg}")
                
                # 如果是 JSON 解析错误，可能是内容太长，尝试截断 doc_content
                if "json_invalid" in error_msg or "EOF" in error_msg or "parsing" in error_msg.lower():
                    if attempt < max_retries - 1:
                        # 截断 doc_content，只保留前 80% 的内容
                        doc_content_str = doc_content_str[:int(len(doc_content_str) * 0.8)]
                        # 重新格式化 user_prompt
                        try:
                            user_prompt = user_prompt_template.format(
                                doc_content=doc_content_str,
                                extract_content=extract_content
                            )
                        except KeyError:
                            user_prompt = user_prompt_template.format(doc_content=doc_content_str)
                        self.logger.info(f"  尝试截断文档内容后重试（保留前 80%）")
                        continue
                
                # 最后一次尝试失败，返回错误信息
                if attempt == max_retries - 1:
                    self.logger.error(f"  生成内容失败，已达到最大重试次数: {error_msg}")
                    final_result = json.dumps({
                        "error": f"生成内容失败: {error_msg}",
                        "error_type": type(e).__name__
                    }, ensure_ascii=False, indent=4)
                    break
        
        # 保留所有状态字段，确保最终状态完整
        return {
            "final_result": final_result,
            # 保留其他重要状态字段
            "matched_chapter": state.get("matched_chapter"),
            "chapter_content": state.get("chapter_content", ""),
            "current_iteration": state.get("current_iteration", 0),
            "evaluation": state.get("evaluation", {}),
            "all_content": state.get("all_content", "")
        }
    
    # ==================== 条件路由函数 ====================
    
    def route_after_chapter_match(self, state: KeywordSearchState) -> Literal["evaluate", "semantic_search"]:
        """路由1: 章节匹配后的路由"""
        matched_chapter = state.get("matched_chapter")
        if matched_chapter:
            return "evaluate"  # 有匹配，进入评估
        else:
            return "semantic_search"  # 无匹配，直接进行语义检索
    
    def route_after_evaluate(self, state: KeywordSearchState) -> Literal["generate", "extract_keywords", "end"]:
        """路由2: 评估后的路由"""
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", self.max_iterations)
        
        # 检查是否超过最大迭代次数
        if current_iteration >= max_iterations:
            return "generate"  # 达到最大迭代次数，也生成最终内容
        
        evaluation = state.get("evaluation", {})
        if evaluation.get("sufficient", False):
            self.logger.info("  内容已满足要求，生成最终内容")
            return "generate"  # 满足要求，生成最终内容
        else:
            return "extract_keywords"  # 不满足，提取关键词继续检索
    
    def route_after_extract(self, state: KeywordSearchState) -> Literal["semantic_search", "generate"]:
        """路由3: 提取关键词后的路由"""
        missing_keywords = state.get("missing_keywords", [])
        if missing_keywords:
            return "semantic_search"  # 有关键词，进行检索
        else:
            self.logger.info("  无关键词需要检索，生成最终内容")
            return "generate"  # 无关键词，生成最终内容
    
    # ==================== 构建图 ====================
    
    def _build_graph(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(KeywordSearchState)
        
        # 添加节点
        workflow.add_node("chapter_match", self.chapter_match_node)
        workflow.add_node("evaluate", self.evaluate_content_node)
        workflow.add_node("extract_keywords", self.extract_keywords_node)
        workflow.add_node("semantic_search", self.semantic_search_node)
        workflow.add_node("merge_content", self.merge_content_node)
        workflow.add_node("generate", self.generate_final_result_node)
        
        # 设置入口
        workflow.add_edge(START, "chapter_match")
        
        # 条件边1: 章节匹配后
        workflow.add_conditional_edges(
            "chapter_match",
            self.route_after_chapter_match,
            {
                "evaluate": "evaluate",
                "semantic_search": "semantic_search"
            }
        )
        
        # 条件边2: 评估后
        workflow.add_conditional_edges(
            "evaluate",
            self.route_after_evaluate,
            {
                "generate": "generate",  # 满足要求或达到最大迭代次数，生成最终内容
                "extract_keywords": "extract_keywords"  # 不满足，继续检索
            }
        )
        
        # 生成最终内容后结束
        workflow.add_edge("generate", END)
        
        # 条件边3: 提取关键词后
        workflow.add_conditional_edges(
            "extract_keywords",
            self.route_after_extract,
            {
                "semantic_search": "semantic_search",  # 有关键词，进行检索
                "generate": "generate"  # 无关键词，生成最终内容
            }
        )
        
        # 语义检索后合并内容
        workflow.add_edge("semantic_search", "merge_content")
        
        # 合并内容后重新评估（循环）
        workflow.add_edge("merge_content", "evaluate")
        
        # 编译图
        return workflow.compile()
    
    # ==================== 运行方法 ====================
    
    def run(self, keyword: str, generation_requirement: str = "") -> Dict[str, Any]:
        """
        运行工作流
        
        :param keyword: 用户提供的关键词
        :param generation_requirement: 生成要求提示词（如果为空，使用初始化时的）
        :return: 最终结果字典
        """
        if not generation_requirement:
            generation_requirement = self.generation_requirement
        
        # 初始化状态
        initial_state = {
            "messages": [],
            "keyword": keyword,
            "generation_requirement": generation_requirement,
            "matched_chapter": None,
            "chapter_content": "",
            "evaluation": {},
            "missing_keywords": [],
            "retrieved_content": [],
            "all_content": "",
            "final_result": "",
            "max_iterations": self.max_iterations,
            "current_iteration": 0
        }
        
        # 运行工作流，设置递归限制防止无限循环
        # 使用 invoke 获取完整最终状态，而不是 stream 的增量更新
        config = {"recursion_limit": self.max_iterations * 10}  # 设置递归限制为最大迭代次数的10倍
        
        # 使用 invoke 获取完整最终状态
        final_state = self.graph.invoke(initial_state, config=config)
        
        return final_state


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 示例：创建文档字典
    example_doc_dict = {
        "功能需求": {
            "content": "系统应支持用户登录、数据查询、报表生成等功能。",
            "title_no": "3.1",
            "level": 2
        },
        "性能需求": {
            "content": "系统响应时间应小于2秒，支持并发用户数不少于1000。",
            "title_no": "3.2",
            "level": 2
        },
        "安全需求": {
            "content": "系统应实现用户身份认证、数据加密、访问控制等安全机制。",
            "title_no": "3.3",
            "level": 2
        }
    }
    from langchain.chat_models import init_chat_model
    from langchain_community.embeddings import DashScopeEmbeddings
    
    model = init_chat_model(
        model_provider="openai",
        model="qwen-plus",
        temperature=0,
        api_key="sk-a5ad92221a5945e2952bbd23dfffe2a0",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key="sk-a5ad92221a5945e2952bbd23dfffe2a0"
    )
    
    # 创建工作流
    # 注意：示例中使用 InMemoryVectorStore，实际使用时应该传入全局向量库（包含所有文档）
    workflow = KeywordSearchWorkflow(
        model=model,
        embeddings=embeddings,
        main_doc_dict=example_doc_dict,  # 主线文档（用于章节匹配）
        global_vector_store=None,  # 如果为None，会从main_doc_dict创建（仅用于示例）
        generation_requirement="生成详细的功能需求规格说明，包括功能描述、输入输出、处理流程等",
        max_iterations=3
    )
    
    # 运行工作流
    keyword = "功能需求"
    result = workflow.run(keyword)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("工作流执行完成")
    print("=" * 60)
    print(f"关键词: {keyword}")
    print(f"是否找到匹配章节: {result.get('matched_chapter') is not None}")
    print(f"评估结果: {result.get('evaluation', {}).get('sufficient', False)}")
    print(f"迭代次数: {result.get('current_iteration', 0)}")
    print(f"\n最终生成内容:\n{result.get('final_result', '')}")

