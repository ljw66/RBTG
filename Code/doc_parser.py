"""
Document Parsing Module
Responsible for converting documents in various formats into LangChain Document objects,
and providing chunking functionality.
"""

import logging
from collections import Counter
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DictToDocumentConverter:
    """
    Convert a document dictionary into LangChain Document objects

    Input format:
    {
        "Chapter Title": {
            "content": "Chapter content",
            "title_no": "1.1",
            "level": 2,
            "source": "Requirement Specification"  # optional
        },
        ...
    }
    """

    def __init__(self, default_source: str = "Other Sources"):
        """
        Initialize the converter

        :param default_source: Default document source, used when source is not specified in doc_dict
        """
        self.logger = logging.getLogger(__name__)
        self.default_source = default_source

    def convert(self, doc_dict: Dict[str, Dict[str, Any]]) -> List[Document]:
        """
        Convert a document dictionary into a list of Document objects

        :param doc_dict: Document dictionary
        :return: List of LangChain Document objects
        """
        if not doc_dict:
            self.logger.warning("Input document dictionary is empty")
            return []

        documents = []
        for title, info in doc_dict.items():
            content = info.get('content', '')
            title_no = info.get('title_no', '')
            level = info.get('level', 1)
            source = info.get('source', self.default_source)

            if content:  # Only process entries with content
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": title,
                        "title_no": title_no,
                        "level": level,
                        "source": source
                    }
                )
                documents.append(doc)

        # Count documents by source
        sources = [doc.metadata.get('source', 'Unknown') for doc in documents]
        source_counts = Counter(sources)

        self.logger.info(
            f"Converted {len(documents)} documents into LangChain Document objects"
        )
        self.logger.info(
            f"Document source statistics: {dict(source_counts)}"
        )

        return documents


class DocumentChunker:
    """
    Document Chunking Processor
    Intelligently splits long texts into chunks while preserving contextual information
    """

    def __init__(
        self,
        chunk_size: int = 6000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the chunking processor

        :param chunk_size: Maximum number of characters per chunk
                           (default: 6000, adapted to the 8192-token limit of bge-m3)
        :param chunk_overlap: Number of overlapping characters between chunks (default: 200)
        :param separators: Priority list of chunk separators
        """
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", ",", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of Document objects

        :param documents: Original list of Document objects
        :return: Chunked list of Document objects
        """
        if not documents:
            self.logger.warning("Input document list is empty")
            return []

        self.logger.info(
            f"Starting chunking process for {len(documents)} documents..."
        )

        chunked_documents = []
        chunked_count = 0

        for doc in documents:
            # If the document is short enough, keep it as is
            if len(doc.page_content) <= self.chunk_size:
                chunked_documents.append(doc)
            else:
                # Split long documents into chunks
                chunks = self.text_splitter.split_documents([doc])
                chunked_count += 1

                # Preserve original metadata and add chunk indexing information
                for idx, chunk in enumerate(chunks):
                    chunk.metadata = doc.metadata.copy()
                    chunk.metadata['chunk_index'] = idx
                    chunk.metadata['total_chunks'] = len(chunks)
                    chunk.metadata['is_chunked'] = True

                chunked_documents.extend(chunks)

                self.logger.info(
                    f"  Document '{doc.metadata.get('title', 'Unknown')}' "
                    f"was split into {len(chunks)} chunks"
                )

        self.logger.info(
            f"Chunking completed: {len(documents)} original documents, "
            f"{chunked_count} were chunked, generating {len(chunked_documents)} total chunks"
        )

        return chunked_documents


class DocParser:
    """
    Document Parser (Convenience Wrapper Class)
    Combines the functionality of DictToDocumentConverter and DocumentChunker
    """

    def __init__(
        self,
        default_source: str = "Other Sources",
        chunk_size: int = 6000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the document parser

        :param default_source: Default document source
        :param chunk_size: Chunk size
        :param chunk_overlap: Chunk overlap
        """
        self.logger = logging.getLogger(__name__)
        self.converter = DictToDocumentConverter(default_source=default_source)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def parse(
        self,
        doc_dict: Dict[str, Dict[str, Any]],
        enable_chunking: bool = True
    ) -> List[Document]:
        """
        Parse a document dictionary and return a list of Document objects

        :param doc_dict: Document dictionary
        :param enable_chunking: Whether to enable chunking (enabled by default)
        :return: List of Document objects
        """
        # 1. Convert into Document objects
        documents = self.converter.convert(doc_dict)

        if not documents:
            return []

        # 2. Apply chunking (if enabled)
        if enable_chunking:
            documents = self.chunker.chunk_documents(documents)

        return documents
