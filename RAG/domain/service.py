from typing import Any, List

from langchain.schema import Document

from .port import DocumentLoader, Generator
from .chunk_repo_ensemble import FaissAndBM25EnsembleRetriever
from .port.generator import RussianPhi4Generator
from ..infrastructure.chunk_repository.bm25_chunk_repository import BM25ChunkRepository


class RAGService:
    """
    Service for managing RAG documents.
    """

    def __init__(
        self,
        notes_loader: DocumentLoader,  # TODO add loader
        photos_loader: DocumentLoader,  # TODO add loader
        notes_repository=FaissAndBM25EnsembleRetriever(),
        photos_repository=BM25ChunkRepository(),
        generator: Generator = RussianPhi4Generator(),
    ) -> None:
        self._notes_loader = notes_loader
        self._photos_loader = photos_loader

        self._notes_repository = notes_repository
        self._photos_repository = photos_repository

        self._generator = generator

    def ingest_notes(self, source: Any) -> None:
        """Load notes, and add to the datastore."""
        notes = self._notes_loader.load(source)
        self._notes_repository.add_batch(notes)

    def ingest_photos(self, source: Any) -> None:
        """Load documents, maded from photos, and add to the datastore."""
        photos = self._photos_loader.load(source)
        self._photos_repository.add_batch(photos)

    def ask(self, query: str) -> tuple[str, str] | tuple[str, None]:
        """Retrieve top_k chunks and generate a response."""
        # TODO add chat context
        notes_context: List[Document] | None = self._notes_repository.query(query)
        photos_context: List[Document] | None = self._photos_repository.query(query, 1)
        if photos_context is not None:
            id = photos_context[0].metadata["id"]
        else:
            id = None

        return self._generator.generate(query, notes_context), id
