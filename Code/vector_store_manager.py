"""
Vector Store Management Module
Responsible for vectorized document storage (Milvus), metadata storage (SQLite), and data querying
"""

import logging
from typing import List, Dict, Any, Optional, Union

from langchain_core.documents import Document
from langchain_milvus import Milvus

from req_spec.doc_metadata_db_1 import DocMetadataDB
from config.settings import MILVUS_VECTOR_STORE_CONFIG


class VectorStoreManager:
    """
    Vector Store Manager
    Unified management of Milvus vector storage and SQLite metadata storage
    """

    def __init__(
            self,
            project_id: Union[str, int],
            embedding_model,
            metadata_db: DocMetadataDB,
            milvus_config: Optional[Dict] = None
    ):
        """
        Initialize the vector store manager

        :param project_id: Project ID, used to generate collection_name
        :param embedding_model: Embedding model (e.g., DashScopeEmbeddings)
        :param metadata_db: Metadata database instance
        :param milvus_config: Milvus configuration (optional, default from settings)
        """
        self.logger = logging.getLogger(__name__)
        self.project_id = project_id
        self.embedding_model = embedding_model
        self.metadata_db = metadata_db
        self.milvus_config = milvus_config or MILVUS_VECTOR_STORE_CONFIG

        # Generate unique collection_name using project ID
        self.collection_name = f"proj_{self.project_id}"

        # Vector store instance (lazy initialization)
        self._vector_store = None

    def _init_vector_store(self) -> Milvus:
        """
        Initialize Milvus vector store
        """
        if self._vector_store is None:
            self._vector_store = Milvus(
                embedding_function=self.embedding_model,
                connection_args={
                    "uri": self.milvus_config["uri"],
                    "db_name": self.milvus_config["db_name"],
                },
                collection_name=self.collection_name,
                index_params={
                    "index_type": self.milvus_config["index_type"],
                    "metric_type": self.milvus_config["metric_type"],
                },
                drop_old=self.milvus_config["drop_old"],
            )
        return self._vector_store

    @property
    def vector_store(self) -> Milvus:
        """Get the vector store instance"""
        return self._init_vector_store()

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to vector store and metadata database

        :param documents: List of Document objects
        :return: List of document IDs
        """
        if not documents:
            self.logger.warning("No documents to add")
            return []

        # 1. Add to Milvus vector store
        self.logger.info(f"Adding {len(documents)} documents to Milvus...")
        vector_store = self._init_vector_store()
        ids = vector_store.add_documents(documents=documents)

        # Ensure ids is a list
        if ids is None:
            ids = []
        if not isinstance(ids, list):
            ids = [ids] if ids else []

        self.logger.info(f"✓ Documents added to Milvus, total {len(ids)} documents")

        # 2. Save metadata to SQLite
        if ids:
            self.logger.info("Saving document metadata to SQLite...")
            saved_count = 0
            for doc, doc_id in zip(documents, ids):
                # Generate unique document ID
                unique_doc_id = f"{self.collection_name}_{doc_id}"

                # Ensure 'source' field exists in metadata
                doc_metadata = doc.metadata.copy()
                if 'source' not in doc_metadata:
                    doc_metadata['source'] = "Other sources"

                # Save to metadata database
                success = self.metadata_db.insert_document(
                    doc_id=unique_doc_id,
                    milvus_id=str(doc_id),
                    collection_name=self.collection_name,
                    content=doc.page_content,
                    metadata=doc_metadata
                )
                if success:
                    saved_count += 1

            self.logger.info(f"✓ Metadata saved, successfully saved {saved_count}/{len(ids)} entries")
        else:
            self.logger.warning("No document IDs retrieved, skipping metadata saving")

        return ids

    def clear_collection(self) -> bool:
        """
        Clear all vector data in the current collection

        :return: Success status
        """
        try:
            from pymilvus import utility, connections

            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=self.milvus_config["uri"],
                db_name=self.milvus_config["db_name"]
            )

            # Check if collection exists
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.logger.info(f"✓ Milvus collection deleted: {self.collection_name}")
            else:
                self.logger.info(f"Milvus collection '{self.collection_name}' does not exist, no need to delete")

            # Reset vector store instance
            self._vector_store = None
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear Milvus collection: {e}")
            return False

    def query_by_source(
            self,
            source_value: str = "Requirement Specification",
            limit: int = 100,
            return_format: str = "dict"
    ) -> Union[Dict[str, Dict], List[Document]]:
        """
        Query all data with a specific 'source' value

        Note: This method uses SQLite metadata database for exact matching, not the vector database.
        Milvus is better for similarity search; exact match queries are more efficient using traditional databases.

        :param source_value: Value of the 'source' field, default is "Requirement Specification"
        :param limit: Maximum number of results to return, default is 100
        :param return_format: Return format, options:
                            - "dict": simple dict, format {title: {content: ..., title_no: ...}}
                            - "document": list of Document objects (including full metadata)
        :return: Results in the requested format
        """
        try:
            # Perform exact query using SQLite metadata database
            results = self.metadata_db.query_by_source(
                source_value=source_value,
                collection_name=self.collection_name,
                limit=limit
            )

            if return_format == "dict":
                # Return simple dict: {title: {content: ..., title_no: ...}}
                result_dict = {}
                for result in results:
                    title = result.get("title", "")
                    content = result.get("content", "")
                    title_no = result.get("title_no", "")

                    if title:  # Add only entries with a title
                        if title in result_dict:
                            # Merge content if title already exists
                            result_dict[title]["content"] += f"\n\n{content}"
                        else:
                            result_dict[title] = {
                                "content": content,
                                "title_no": title_no
                            }

                self.logger.info(f"Found {len(result_dict)} entries with source '{source_value}' (dict format)")
                return result_dict

            else:
                # Return Document objects
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "title_no": result.get("title_no", ""),
                            "level": result.get("level", 1),
                            "chunk_index": result.get("chunk_index"),
                            "total_chunks": result.get("total_chunks"),
                            "is_chunked": bool(result.get("is_chunked", False)),
                            "doc_id": result.get("doc_id"),
                            "milvus_id": result.get("milvus_id")
                        }
                    )
                    documents.append(doc)

                self.logger.info(f"Found {len(documents)} entries with source '{source_value}' (Document format)")
                return documents

        except Exception as e:
            self.logger.error(f"Error querying data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {} if return_format == "dict" else []

    def similarity_search(
            self,
            query: str,
            k: int = 4,
            filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Similarity search (retrieve from Milvus vector store)

        :param query: Query text
        :param k: Number of results to return
        :param filter_dict: Filter conditions
        :return: List of Document objects
        """
        vector_store = self._init_vector_store()
        if filter_dict:
            return vector_store.similarity_search(query, k=k, filter=filter_dict)
        return vector_store.similarity_search(query, k=k)
