from typing import Any

from langchain.schema import Document

from RAG.domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from RAG.infrastructure.chunk_repository.faiss_chunk_repository import FaissChunkRepository

type Context = list[Document]


class ContextService:
    """
    ContextService is responsible for managing the context of the conversation.
    It provides methods to add, retrieve, and clear context.
    """

    def __init__(
        self,
        notes_repository=FaissAndBM25EnsembleRetriever(),
        photos_repository=FaissChunkRepository()
    ):
        self._notes_repository = notes_repository
        self._photos_repository = photos_repository

    def retrieve_textual_context(self, query: str) -> Context:
        """
        Retrieve context based on the query.
        Returns a list of documents that match the query.
        """
        return self._notes_repository.query(query)

    def retrieve_photo_context(self, query: str) -> Context:
        """
        Returns the top_k photo contexts based on the query.
        """
        return self._photos_repository.query(query)

    @staticmethod
    def get_context_photo_ids(photos_context: Context) -> list[str]:
        """
        Extracts photo IDs from the context.
        """
        return [doc.metadata["image_id"] for doc in photos_context if "image_id" in doc.metadata]
