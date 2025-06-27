from typing import Any

from langchain.schema import Document

from RAG.domain.chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from RAG.domain.port.loader import DocumentLoader
from RAG.infrastructure.chunk_repository.bm25_chunk_repository import (
    BM25ChunkRepository,
)

type Context = list[Document]


class ContextService:
    """
    ContextService is responsible for managing the context of the conversation.
    It provides methods to add, retrieve, and clear context.
    """

    def __init__(
        self,
        notes_loader: DocumentLoader,  # TODO add loader
        photos_loader: DocumentLoader,  # TODO add loader
        notes_repository=FaissAndBM25EnsembleRetriever(),
        photos_repository=BM25ChunkRepository(),
    ):
        self._notes_loader = notes_loader
        self._photos_loader = photos_loader
        self._notes_repository = notes_repository
        self._photos_repository = photos_repository

    def ingest_notes(self, source: Any) -> None:
        """Load notes, and add to the datastore."""
        notes = self._notes_loader.load(source)
        self._notes_repository.add_batch(notes)

    def ingest_photos(self, source: Any) -> None:
        """Load documents, made from photos, and add to the datastore."""
        photos = self._photos_loader.load(source)
        self._photos_repository.add_batch(photos)

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
