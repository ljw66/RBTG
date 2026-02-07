"""
Keyword-Based Intelligent Retrieval Workflow
Multi-round retrieval and evaluation pipeline implemented with LangGraph

Workflow Overview:
1. User provides a keyword → check whether a matching chapter exists in the table of contents
2. If matched → extract the chapter content
3. The LLM evaluates whether the collected content is sufficient for generation
4. If insufficient → the LLM extracts additional keywords for further retrieval
5. Perform semantic retrieval → obtain more relevant content
6. Merge retrieved content and re-evaluate
7. If sufficient → generate the final output
"""

import sys
import os
import logging
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field

# Add project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import prompt templates for test requirement generation
from req_spec.prompts.req_prompt import req_prompts, TestRequirementOutput


# ==================== State Definition ====================

class KeywordSearchState(MessagesState):
    """State definition for the keyword-based retrieval workflow"""
    keyword: str  # Keyword provided by the user
    generation_requirement: str  # Prompt describing the generation requirement
    matched_chapter: Optional[Dict[str, Any]]  # Matched chapter info {title: str, content: str, ...}
    chapter_content: str  # Extracted chapter content
    evaluation: Dict[str, Any]  # Evaluation result {sufficient: bool, missing_aspects: List[str], reason: str}
    missing_keywords: List[str]  # Keywords to be retrieved
    retrieved_content: List[Dict[str, Any]]  # Content retrieved via semantic search
    all_content: str  # Merged content (chapter content + retrieved content)
    final_result: str  # Final generated output
    max_iterations: int  # Maximum number of iterations (to prevent infinite loops)
    current_iteration: int  # Current iteration counter


# ==================== Pydantic Models ====================

class ContentEvaluation(BaseModel):
    """Content evaluation result"""
    sufficient: bool = Field(description="Whether the content is sufficient for generation")
    missing_aspects: List[str] = Field(description="Missing key information points or keywords")
    reason: str = Field(description="Evaluation rationale")


class KeywordExtraction(BaseModel):
    """Keyword extraction result"""
    keywords: List[str] = Field(description="Keywords to retrieve for supplementing missing information")


# ==================== Helper Functions ====================

def create_vector_store_from_dict(doc_dict: Dict[str, Dict[str, Any]], embeddings: Embeddings) -> InMemoryVectorStore:
    """
    Create a LangChain vector store from a document dictionary.

    :param doc_dict: Document dictionary formatted as
                     {title: {content: "text", title_no: "1.1", level: 2}}
    :param embeddings: LangChain embedding model
    :return: InMemoryVectorStore instance
    """
    documents = []

    for title, info in doc_dict.items():
        content = info.get('content', '')
        title_no = info.get('title_no', '')
        level = info.get('level', 1)

        # Only process entries with non-empty content
        if content:
            # Truncate long text to fit embedding model input length constraints
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

    # Build vector store
    vector_store = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vector_store


# ==================== Workflow Class ====================

class KeywordSearchWorkflow:
    """
    Keyword-based intelligent retrieval workflow (generic version),
    supporting configurable prompts and output schemas.
    """

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
        Initialize the workflow.

        :param model: LangChain chat model
        :param embeddings: LangChain embedding model
        :param main_doc_dict: Main document dictionary for chapter matching,
                              formatted as {title: {content: "...", title_no: "1.1", level: 2}}
        :param global_vector_store: Global vector store for semantic retrieval across all documents.
                                    If None, it will be built from main_doc_dict.
        :param generation_requirement: Prompt describing the generation requirement
        :param max_iterations: Maximum number of retrieval-evaluation iterations
        :param prompt_template: Optional prompt template dict, formatted as:
                                {"system_prompt": "...", "user_prompt": "...", "extract_content": "..."}
                                If not provided, defaults to req_prompt.py
        :param output_model: Optional Pydantic output schema. Defaults to TestRequirementOutput.
        :param skip_chapter_match: Whether to skip chapter matching (useful when no explicit chapter exists)
        """
        self.main_doc_dict = main_doc_dict
        self.generation_requirement = generation_requirement
        self.max_iterations = max_iterations
        self.skip_chapter_match = skip_chapter_match
        self.logger = logging.getLogger(__name__)

        # Configure prompt template (default: req_prompt.py)
        if prompt_template is None:
            from req_spec.prompts.req_prompt import req_prompts
            self.prompt_template = req_prompts
        else:
            self.prompt_template = prompt_template

        # Configure output schema (default: TestRequirementOutput)
        if output_model is None:
            from req_spec.prompts.req_prompt import TestRequirementOutput
            self.output_model = TestRequirementOutput
        else:
            self.output_model = output_model

        # Build or reuse the global vector store for semantic retrieval
        if global_vector_store is None:
            self.logger.warning(
                "No global vector store provided. Building an InMemoryVectorStore "
                "from the main document dictionary (only containing main documents)."
            )
            self.global_vector_store = create_vector_store_from_dict(main_doc_dict, embeddings)
        else:
            self.global_vector_store = global_vector_store

        # Create retriever (retrieve top-10 candidates, later sorted by priority)
        self.retriever = self.global_vector_store.as_retriever(search_kwargs={"k": 10})

        # Initialize LLM
        self.llm = model

        # Build workflow graph
        self.graph = self._build_graph()

    # ==================== Node Functions ====================

    def chapter_match_node(self, state: KeywordSearchState) -> Dict[str, Any]:
        """Node 1: Match the keyword against chapter titles (only within the main document)."""
        keyword = state.get("keyword", "")

        # If chapter matching is skipped, return empty match
        if self.skip_chapter_match:
            self.logger.info("Node 1: Skipping chapter matching (skip_chapter_match=True)")
            return {
                "matched_chapter": None,
                "chapter_content": ""
            }

        self.logger.info(f"Node 1: Chapter matching - keyword: '{keyword}'")
        self.logger.info(f"  Number of chapters in main document: {len(self.main_doc_dict)}")

        # Print all available chapter titles for debugging
        if self.main_doc_dict:
            all_titles = list(self.main_doc_dict.keys())
            self.logger.info(f"  Available chapter titles: {all_titles}")
        else:
            self.logger.warning("  Main document dictionary is empty!")
            return {
                "matched_chapter": None,
                "chapter_content": ""
            }

        matched_chapter = None
        chapter_content = ""

        # Exact match (title equals keyword)
        if keyword in self.main_doc_dict:
            info = self.main_doc_dict[keyword]
            matched_chapter = {
                "title": keyword,
                "content": info.get("content", ""),
                "title_no": info.get("title_no", ""),
                "level": info.get("level", 1),
                "source": "Software Requirements Specification"
            }
            chapter_content = info.get("content", "")
            self.logger.info(f"  ✓ Exact match found: {keyword} (Source: SRS)")
        else:
            # Fuzzy match (keyword contained in title)
            keyword_lower = keyword.lower().strip()
            self.logger.info(
                f"  Exact match failed. Trying fuzzy match with keyword: '{keyword_lower}'"
            )

            for title, info in self.main_doc_dict.items():
                title_lower = title.lower().strip()
                if keyword_lower in title_lower or title_lower in keyword_lower:
                    matched_chapter = {
                        "title": title,
                        "content": info.get("content", ""),
                        "title_no": info.get("title_no", ""),
                        "level": info.get("level", 1),
                        "source": "Software Requirements Specification"
                    }
                    chapter_content = info.get("content", "")
                    self.logger.info(f"  ✓ Fuzzy match found: '{title}' (Source: SRS)")
                    self.logger.info(f"    Keyword: '{keyword}' → Matched title: '{title}'")
                    break

        if not matched_chapter:
            self.logger.warning(f"  ✗ No matching chapter found for keyword: '{keyword}'")
            self.logger.info(
                f"  Example available titles (top-10): {list(self.main_doc_dict.keys())[:10]}"
            )

        return {
            "matched_chapter": matched_chapter,
            "chapter_content": chapter_content
        }

