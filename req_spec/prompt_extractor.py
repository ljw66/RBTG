"""
提示词提取和检索模块
适配新的工作流和向量库，用于提取非功能需求等信息
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document


class PromptExtractor:
    """
    提取提示词相关信息，适配新的向量库和工作流
    """
    
    def __init__(self, vector_store: VectorStore, model=None):
        """
        初始化提示词提取器
        
        :param vector_store: 全局向量库（包含所有文档）
        :param model: LLM模型（可选，用于某些高级功能）
        """
        self.logger = logging.getLogger(__name__)
        self.vector_store = vector_store
        self.model = model
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    @staticmethod
    def parse_prompt_elements(prompts_dict: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        从提示词字典提取元素
        
        :param prompts_dict: 提示词字典，格式如 {key: {system_prompt: "...", user_prompt: "...", doc_content: "...", extract_content: "..."}}
        :return: 提取后的元素字典
        """
        results = {}
        for key, value in prompts_dict.items():
            if isinstance(value, dict) and any(field in value for field in ['system_prompt', 'user_prompt', 'extract_content', 'doc_content']):
                # 提取关键字段，如果不存在则使用空字符串作为默认值
                element_data = {
                    "system_prompt": value.get('system_prompt', ''),
                    "user_prompt": value.get('user_prompt', ''),
                    "extract_content": value.get('extract_content', ''),
                    "doc_content": value.get('doc_content', ''),
                    "output_model": value.get('output_model', '')  # 保留输出模型信息
                }
                results[key] = element_data
        return results
    
    def format_user_prompt(self, element_data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        将user_prompt中的占位符替换为实际检索到的内容
        
        :param element_data: 提示词元素数据
        :return: (格式化后的user_prompt, retrieval_dict)
        """
        # 从doc_content中提取关键词列表
        doc_content = element_data.get('doc_content', '').strip()
        if not doc_content:
            self.logger.warning(f"doc_content为空，跳过检索")
            return element_data['user_prompt'], {"label": "[]", "retrieval_nodes": []}
        
        label_list = [item.strip() for item in doc_content.split(',') if item.strip()]
        
        # 存储检索到的所有内容
        retrieval_contents = []
        all_contents = []
        
        # 对每个标签进行语义检索
        for label in label_list:
            self.logger.info(f"  检索关键词: {label}")
            try:
                # 使用向量库进行语义检索
                retrieved_docs = self.retriever.invoke(label)
                
                # 处理检索结果
                for doc in retrieved_docs:
                    # 从metadata中获取来源和标题
                    source = doc.metadata.get("source", "其他来源")
                    title = doc.metadata.get("title", "未知标题")
                    title_no = doc.metadata.get("title_no", "")
                    score = 0.0
                    
                    # 获取相似度分数（如果可用）
                    if hasattr(doc, 'score'):
                        score = doc.score
                    elif hasattr(doc, 'metadata') and 'score' in doc.metadata:
                        score = doc.metadata.get('score', 0.0)
                    
                    # 构建检索结果
                    result = {
                        "title": title,
                        "title_no": title_no,
                        "content": doc.page_content,
                        "score": score,
                        "source": source
                    }
                    
                    # 避免重复内容
                    content_key = f"{title}:{doc.page_content[:100]}"
                    if content_key not in [f"{r['title']}:{r['content'][:100]}" for r in retrieval_contents]:
                        retrieval_contents.append(result)
                        # 格式化内容用于提示词
                        title_with_no = f"{title_no} {title}" if title_no else title
                        all_contents.append(
                            f"{title_with_no}\n相似度得分：{score:.4f}\n来源：{source}\n\n{doc.page_content}\n\n"
                        )
                
            except Exception as e:
                self.logger.error(f"检索关键词 '{label}' 时出错: {e}")
                continue
        
        # 构建检索字典
        retrieval_dict = {
            "label": str(label_list),
            "retrieval_nodes": retrieval_contents,
        }
        
        # 合并所有找到的内容
        tips = "请注意以下文档中，相似度得分为1的文档是你要使用的```主要文档```，相似度得分<1的文档为检索得到的```参考文档```，生成内容时请根据```主要文档```来生成，出现上下文内容不足时才使用参考文档进行补充。\n"
        search_content = tips + "\n\n".join(all_contents)
        
        # 格式化用户提示词
        try:
            formatted_user_prompt = element_data['user_prompt'].format(
                extract_content=element_data.get('extract_content', ''),
                doc_content=search_content,
            )
            return formatted_user_prompt, retrieval_dict
        except Exception as e:
            self.logger.error(f"格式化user_prompt时出错: {e}")
            return element_data['user_prompt'], retrieval_dict
    
    def parse_prompt_dict(self, prompts: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        主函数：解析提示词字典，为每个提示词检索相关内容并格式化
        
        :param prompts: 提示词字典
        :return: 包含检索信息的提示词字典
        """
        # 解析prompts字典
        data_dict = self.parse_prompt_elements(prompts)
        prompts_dict = {}
        
        self.logger.info(f"开始处理 {len(data_dict)} 个提示词类型")
        
        for key, data in data_dict.items():
            self.logger.info(f"处理提示词类型: {key}")
            # 系统提示词是固定的，无需拼接
            system_prompt = data['system_prompt']
            # 用户提示词需要拼接检索内容
            user_prompt, retrieval_dict = self.format_user_prompt(data)
            
            prompts_dict[key] = {
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'retrieval_dict': retrieval_dict,
                'output_model': data.get('output_model', '')  # 保留输出模型信息
            }
            
            self.logger.info(f"  ✓ {key} 处理完成，检索到 {len(retrieval_dict['retrieval_nodes'])} 条结果")
        
        return prompts_dict


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 这里需要实际的向量库和提示词字典
    # vector_store = ...
    # from service.doc_service.doc_tools.prompt0324 import prompts
    # extractor = PromptExtractor(vector_store)
    # prompts_dict = extractor.parse_prompt_dict(prompts)
    pass
