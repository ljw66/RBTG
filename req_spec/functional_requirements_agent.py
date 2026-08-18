"""
通用标签搜索 Agent

该脚本使用 LangChain Agent 来根据指定的标签（label）搜索文档目录中相关的章节。
Agent 可以根据需要查看章节内容来辅助判断。
"""

import os
import getpass
from dataclasses import dataclass
from typing import List, Optional, Dict

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from pydantic import BaseModel, Field


def truncate_text(text: str, max_length: int = 8000) -> str:
    """
    截断文本以避免超过模型输入限制
    
    Args:
        text: 原始文本
        max_length: 最大字符长度，默认8000（为模型输入留出余量）
    
    Returns:
        截断后的文本
    """
    if not text:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n\n[注意：内容已截断，原始长度超过限制]"


# ============================================
# 响应格式定义
# ============================================
class LabelSearchResponse(BaseModel):
    """标签搜索结果"""
    matched_chapters: List[str] = Field(description="与指定标签相关的目录列表")
    reasoning: str = Field(description="判断理由")


# ============================================
# 通用标签搜索 Agent 类
# ============================================

class LabelSearchAgent:
    """
    通用标签搜索 Agent 类
    
    用于根据指定的标签（label）搜索文档目录中相关的章节。
    Agent 可以根据需要查看章节内容来辅助判断。
    
    示例:
        agent = LabelSearchAgent(chapters_dict=chapters_dict)
        result = agent.analyze(table_of_contents, label="功能需求")
        result = agent.analyze(table_of_contents, label="性能需求")
        result = agent.analyze(table_of_contents, label="安全需求")
    """
    
    def __init__(
        self,
        chapters_dict: Dict[str, str],
        model=None,
    ):
        """
        初始化标签搜索 Agent
        
        Args:
            chapters_dict: 章节字典，键为目录标题，值为章节内容
            model
        """
        self.chapters_dict = chapters_dict
        self.model = model
        self.agent = None
    
    
    def _get_agent(self, label: str = "功能需求"):
        """
        创建 Agent
        Args:
            label: 标签，用于指定要查找的目录类型（如"功能需求"、"性能需求"、"安全需求"等）
        """
        @tool
        def get_chapter_content(chapter_title: str) -> str:
            """
            根据目录标题查询对应的章节内容。
            
            Args:
                chapter_title: 目录标题，例如 "2. 用户登录功能"
            
            Returns:
                章节的详细内容，如果标题不存在则返回提示信息
            """
            if chapter_title in self.chapters_dict:
                content = self.chapters_dict[chapter_title]
                # 截断内容以避免超过模型输入限制
                truncated_content = truncate_text(content)
                return f"章节标题：{chapter_title}\n\n章节内容：\n{truncated_content}"
            else:
                available_titles = "\n".join([f"  - {title}" for title in self.chapters_dict.keys()])
                return f"未找到标题 '{chapter_title}'。\n\n可用的章节标题：\n{available_titles}"
            
        # 根据标签动态生成系统提示词
        system_prompt = f"""你是一个文档分析专家，擅长判断文档中的目录是否与"{label}"相关。

你的任务：
1. 分析给定的目录列表，判断哪些目录与"{label}"相关
2. 如果不确定某个目录是否与"{label}"相关，可以使用 get_chapter_content 工具查看具体内容
3. 最终返回一个列表，包含所有与"{label}"相关的目录标题
4. 简要说明你的判断理由

请仔细分析每个目录，必要时查看章节内容以确保判断准确。"""
            
        # 创建 Agent
        self._agent = create_agent(
            self.model,
            tools=[get_chapter_content],
            system_prompt=system_prompt,
            response_format=LabelSearchResponse
        )
    
        return self._agent
    
    def analyze(
        self,
        table_of_contents: List[str],
        label: str = "功能需求",
        show_reasoning: bool = True,
        verbose: bool = True
    ) -> LabelSearchResponse:
        """
        分析目录列表，判断哪些与指定标签相关
        
        Args:
            table_of_contents: 目录列表
            label: 标签，用于指定要查找的目录类型（如"功能需求"、"性能需求"、"安全需求"等，默认为"功能需求"）
            show_reasoning: 是否显示判断理由
            verbose: 是否显示详细输出
        
        Returns:
            LabelSearchResponse: 包含相关目录列表和判断理由
        """
        agent = self._get_agent(label=label)
        
        # 构建查询：先将目录列表转换为多行字符串
        contents_text = "\n".join(table_of_contents)
        query = f"""请分析以下目录列表，判断哪些目录与"{label}"相关：

{contents_text}

如果不确定某个目录是否与"{label}"相关，请使用工具查看该目录的具体内容来帮助判断。
最终请返回所有与"{label}"相关的目录标题列表，并说明判断理由。"""
        
        # 运行 Agent 并显示结果
        if verbose:
            print(f"正在分析目录（标签：{label}）...")
            print("=" * 60)
        
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        
        result = response['structured_response']
        if verbose:
            print("\n判断结果：")
            print("=" * 60)
            print(f"与\"{label}\"相关的目录（共 {len(result.matched_chapters)} 个）：")
            for i, chapter in enumerate(result.matched_chapters, 1):
                print(f"  {i}. {chapter}")
            
            if show_reasoning:
                print(f"\n判断理由：")
                print(result.reasoning)
        
        return result
    
    def analyze_streaming(
        self,
        table_of_contents: List[str],
        label: str = "功能需求",
    ):
        """
        使用流式模式分析，可以看到 Agent 的推理过程
        
        Args:
            table_of_contents: 目录列表
            label: 标签，用于指定要查找的目录类型（如"功能需求"、"性能需求"、"安全需求"等，默认为"功能需求"）
            verbose: 是否显示详细输出
        """
        agent = self._get_agent(label=label)
        
        # 构建查询：先将目录列表转换为多行字符串
        contents_text = "\n".join(table_of_contents)
        query = f"""请分析以下目录列表，判断哪些目录与"{label}"相关：

{contents_text}

如果不确定某个目录是否与"{label}"相关，请使用工具查看该目录的具体内容来帮助判断。
最终请返回所有与"{label}"相关的目录标题列表，并说明判断理由。"""
        

        print(f"查看 Agent 的推理过程（标签：{label}）：")
        print("=" * 60)
        
        # 使用 LangChain 的流式输出
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            # 每个 chunk 包含该时刻的完整状态
            latest_message = chunk["messages"][-1]

            if latest_message.content:
                print(f"Agent: {latest_message.content}")
            elif latest_message.tool_calls:
                print(f"调用工具: {[tc['name'] for tc in latest_message.tool_calls]}")
    
# ============================================
# 示例数据
# ============================================

# 目录列表
TABLE_OF_CONTENTS = [
    "1. 项目概述",
    "2. 用户登录功能",
    "3. 系统架构设计",
    "4. 数据存储方案",
    "5. 订单管理功能",
    "6. 性能优化策略",
    "7. 支付功能",
    "8. 技术选型说明",
    "9. 消息推送功能",
    "10. 部署方案"
]

# 章节字典：目录标题 -> 章节内容
CHAPTERS_DICT = {
    "1. 项目概述": "本项目是一个电商平台，旨在为用户提供便捷的购物体验。项目采用微服务架构，支持高并发访问。",
    "2. 用户登录功能": "用户可以通过手机号、邮箱或第三方账号（微信、QQ）登录系统。支持密码登录、验证码登录、免密登录等多种方式。登录后可以查看个人信息、订单历史等。",
    "3. 系统架构设计": "系统采用前后端分离架构，前端使用React框架，后端使用Spring Boot微服务。数据库采用MySQL主从复制，缓存使用Redis。",
    "4. 数据存储方案": "数据存储采用MySQL作为主数据库，MongoDB存储非结构化数据，Redis用于缓存热点数据。数据备份策略为每日全量备份，每小时增量备份。",
    "5. 订单管理功能": "用户可以创建订单、查看订单列表、取消订单、申请退款。订单状态包括：待支付、已支付、已发货、已完成、已取消。支持订单搜索和筛选。",
    "6. 性能优化策略": "通过CDN加速静态资源，使用Redis缓存减少数据库查询，采用数据库读写分离，使用消息队列异步处理任务。",
    "7. 支付功能": "支持支付宝、微信支付、银联支付等多种支付方式。支付流程包括：选择支付方式、确认订单、调用支付接口、处理支付结果、更新订单状态。",
    "8. 技术选型说明": "前端框架选择React，后端框架选择Spring Boot，数据库选择MySQL，缓存选择Redis，消息队列选择RabbitMQ。",
    "9. 消息推送功能": "系统可以向用户推送订单状态变更、促销活动、系统通知等消息。支持站内消息、短信、邮件、APP推送等多种推送方式。",
    "10. 部署方案": "采用Docker容器化部署，使用Kubernetes进行容器编排。部署环境包括开发环境、测试环境、预生产环境、生产环境。"
}

# ============================================
# 主函数
# ============================================

def main():
    """主函数：运行示例"""
    print("=" * 60)
    print("目录检索 Agent")
    print("=" * 60)
    
    # 显示输入数据
    print("\n目录列表：")
    for toc in TABLE_OF_CONTENTS:
        print(f"  - {toc}")
    print(f"\n章节字典包含 {len(CHAPTERS_DICT)} 个章节\n")
    model = init_chat_model(
        model_provider="openai",
        model="qwen-plus",
        temperature=0,
        api_key="sk-a5ad92221a5945e2952bbd23dfffe2a0",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 创建 Agent 实例
    agent = LabelSearchAgent(
        chapters_dict=CHAPTERS_DICT,
        model=model
    )
    
    # 方法1：标准分析（默认标签"功能需求"）
    print("\n" + "=" * 60)
    print("方法1：标准分析（标签：功能需求）")
    print("=" * 60)
    result = agent.analyze(TABLE_OF_CONTENTS, label="功能需求")
    print("=" * 60)
    print(f"matched_chapters: {result.matched_chapters}")
    
    # # 方法2：自定义标签示例
    # print("\n" + "=" * 60)
    # print("方法2：自定义标签（标签：支付）")
    # print("=" * 60)
    # result = agent.analyze(TABLE_OF_CONTENTS, label="支付")
    
    # # 方法3：搜索性能相关章节
    # print("\n" + "=" * 60)
    # print("方法3：搜索性能相关章节（标签：性能）")
    # print("=" * 60)
    # result = agent.analyze(TABLE_OF_CONTENTS, label="性能")
    
    # 方法4：流式分析（查看推理过程）
    # print("\n" + "=" * 60)
    # print("方法4：流式分析（查看推理过程）")
    # print("=" * 60)
    # agent.analyze_streaming(TABLE_OF_CONTENTS, label="功能需求")


if __name__ == "__main__":
    main()
