"""
Generic Label Search Agent

This script uses a LangChain Agent to search for relevant chapters in a document
directory based on a specified label.
The agent can inspect chapter contents when necessary to assist decision-making.
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
    Truncate text to avoid exceeding the model input limit.

    Args:
        text: Original text
        max_length: Maximum number of characters (default: 8000, leaving margin for model input)

    Returns:
        Truncated text
    """
    if not text:
        return text
    if len(text) <= max_length:
        return text
    return text[:max_length] + "\n\n[Note: Content has been truncated because the original length exceeds the limit]"


# ============================================
# Response format definition
# ============================================
class LabelSearchResponse(BaseModel):
    """Label search results"""
    matched_chapters: List[str] = Field(description="List of chapter titles related to the specified label")
    reasoning: str = Field(description="Reasoning for the selection")


# ============================================
# Generic Label Search Agent Class
# ============================================

class LabelSearchAgent:
    """
    Generic Label Search Agent Class

    Used to search for relevant chapters in a document directory based on a given label.
    The agent may inspect chapter contents when necessary to assist decision-making.

    Example:
        agent = LabelSearchAgent(chapters_dict=chapters_dict)
        result = agent.analyze(table_of_contents, label="Functional Requirements")
        result = agent.analyze(table_of_contents, label="Performance Requirements")
        result = agent.analyze(table_of_contents, label="Security Requirements")
    """

    def __init__(
        self,
        chapters_dict: Dict[str, str],
        model=None,
    ):
        """
        Initialize the label search agent.

        Args:
            chapters_dict: Chapter dictionary, where keys are titles and values are chapter contents
            model: LLM model instance
        """
        self.chapters_dict = chapters_dict
        self.model = model
        self.agent = None

    def _get_agent(self, label: str = "Functional Requirements"):
        """
        Create an agent.

        Args:
            label: Label specifying the type of chapters to search for
                   (e.g., "Functional Requirements", "Performance Requirements", "Security Requirements")
        """

        @tool
        def get_chapter_content(chapter_title: str) -> str:
            """
            Retrieve chapter content based on the chapter title.

            Args:
                chapter_title: Chapter title, e.g., "2. User Login Feature"

            Returns:
                Detailed chapter content, or an error message if the title does not exist
            """
            if chapter_title in self.chapters_dict:
                content = self.chapters_dict[chapter_title]
                # Truncate content to avoid exceeding the model input limit
                truncated_content = truncate_text(content)
                return f"Chapter Title: {chapter_title}\n\nChapter Content:\n{truncated_content}"
            else:
                available_titles = "\n".join([f"  - {title}" for title in self.chapters_dict.keys()])
                return f"Title '{chapter_title}' not found.\n\nAvailable chapter titles:\n{available_titles}"

        # Dynamically generate the system prompt based on the label
        system_prompt = f"""You are a document analysis expert skilled at determining whether chapters in a document are related to "{label}".

Your tasks:
1. Analyze the given table of contents and identify which chapters are related to "{label}"
2. If uncertain, you may use the get_chapter_content tool to inspect chapter contents
3. Return a list containing all chapter titles related to "{label}"
4. Briefly explain your reasoning

Please carefully analyze each chapter title, and inspect contents when necessary to ensure accurate judgment.
"""

        # Create the agent
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
        label: str = "Functional Requirements",
        show_reasoning: bool = True,
        verbose: bool = True
    ) -> LabelSearchResponse:
        """
        Analyze the table of contents and determine which chapters are related to the specified label.

        Args:
            table_of_contents: List of chapter titles
            label: Label specifying the type of chapters to search for
                   (default: "Functional Requirements")
            show_reasoning: Whether to display reasoning
            verbose: Whether to display detailed output

        Returns:
            LabelSearchResponse: Related chapter titles and reasoning
        """
        agent = self._get_agent(label=label)

        # Build the query: convert TOC list into multiline text
        contents_text = "\n".join(table_of_contents)
        query = f"""Please analyze the following table of contents and determine which chapters are related to "{label}":

{contents_text}

If you are uncertain whether a chapter is related to "{label}", please use the tool to inspect its content.
Finally, return a list of all chapter titles related to "{label}" and explain your reasoning.
"""

        # Run the agent and display results
        if verbose:
            print(f"Analyzing table of contents (Label: {label})...")
            print("=" * 60)

        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )

        result = response['structured_response']
        if verbose:
            print("\nAnalysis Result:")
            print("=" * 60)
            print(f"Chapters related to \"{label}\" (Total: {len(result.matched_chapters)}):")
            for i, chapter in enumerate(result.matched_chapters, 1):
                print(f"  {i}. {chapter}")

            if show_reasoning:
                print("\nReasoning:")
                print(result.reasoning)

        return result

    def analyze_streaming(
        self,
        table_of_contents: List[str],
        label: str = "Functional Requirements",
    ):
        """
        Analyze using streaming mode to observe the agent's reasoning process.

        Args:
            table_of_contents: List of chapter titles
            label: Label specifying the type of chapters to search for
        """
        agent = self._get_agent(label=label)

        # Build query
        contents_text = "\n".join(table_of_contents)
        query = f"""Please analyze the following table of contents and determine which chapters are related to "{label}":

{contents_text}

If you are uncertain whether a chapter is related to "{label}", please use the tool to inspect its content.
Finally, return a list of all chapter titles related to "{label}" and explain your reasoning.
"""

        print(f"Viewing agent reasoning process (Label: {label}):")
        print("=" * 60)

        # Streaming output
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            latest_message = chunk["messages"][-1]

            if latest_message.content:
                print(f"Agent: {latest_message.content}")
            elif latest_message.tool_calls:
                print(f"Tool called: {[tc['name'] for tc in latest_message.tool_calls]}")


# ============================================
# Example Data
# ============================================

# Table of contents list
TABLE_OF_CONTENTS = [
    "1. Project Overview",
    "2. User Login Feature",
    "3. System Architecture Design",
    "4. Data Storage Solution",
    "5. Order Management Feature",
    "6. Performance Optimization Strategy",
    "7. Payment Feature",
    "8. Technology Selection Notes",
    "9. Notification Push Feature",
    "10. Deployment Plan"
]

# Chapter dictionary: title -> content
CHAPTERS_DICT = {
    "1. Project Overview": "This project is an e-commerce platform designed to provide users with a convenient shopping experience. The system adopts a microservice architecture and supports high concurrency.",
    "2. User Login Feature": "Users can log in via phone number, email, or third-party accounts (WeChat, QQ). Multiple login methods are supported, including password login, verification code login, and password-free login.",
    "3. System Architecture Design": "The system adopts a frontend-backend separation architecture. The frontend uses React, and the backend uses Spring Boot microservices.",
    "4. Data Storage Solution": "MySQL is used as the primary database, MongoDB stores unstructured data, and Redis caches hot data. Backup strategy includes daily full backups and hourly incremental backups.",
    "5. Order Management Feature": "Users can create orders, view order lists, cancel orders, and request refunds. Order statuses include pending payment, paid, shipped, completed, and canceled.",
    "6. Performance Optimization Strategy": "Static resources are accelerated via CDN, Redis caching reduces database queries, read-write separation is applied, and message queues handle tasks asynchronously.",
    "7. Payment Feature": "Supports Alipay, WeChat Pay, UnionPay, and other payment methods. The payment process includes selecting payment, confirming orders, calling APIs, processing results, and updating status.",
    "8. Technology Selection Notes": "Frontend framework: React. Backend framework: Spring Boot. Database: MySQL. Cache: Redis. Message queue: RabbitMQ.",
    "9. Notification Push Feature": "The system can push order status updates, promotions, and system notifications via in-app messages, SMS, email, and app push.",
    "10. Deployment Plan": "Deployment uses Docker containerization and Kubernetes orchestration. Environments include development, testing, staging, and production."
}

# ============================================
# Main Function
# ============================================

def main():
    """Main function: run example"""
    print("=" * 60)
    print("Directory Retrieval Agent")
    print("=" * 60)

    # Display input data
    print("\nTable of Contents:")
    for toc in TABLE_OF_CONTENTS:
        print(f"  - {toc}")
    print(f"\nChapter dictionary contains {len(CHAPTERS_DICT)} chapters\n")

    model = init_chat_model(
        model_provider="openai",
        model="qwen-plus",
        temperature=0,
        api_key="sk-a5ad92221a5945e2952bbd23dfffe2a0",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # Create agent instance
    agent = LabelSearchAgent(
        chapters_dict=CHAPTERS_DICT,
        model=model
    )

    # Method 1: Standard analysis
    print("\n" + "=" * 60)
    print("Method 1: Standard Analysis (Label: Functional Requirements)")
    print("=" * 60)
    result = agent.analyze(TABLE_OF_CONTENTS, label="Functional Requirements")
    print("=" * 60)
    print(f"matched_chapters: {result.matched_chapters}")


if __name__ == "__main__":
    main()
