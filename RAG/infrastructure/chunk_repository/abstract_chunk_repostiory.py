from abc import ABC, abstractmethod

from domain.chunk import Document, DocumentMetadata
from domain.port import ChunkRepository
from infrastructure.chunk_repository.chunker import Chunker
from infrastructure.chunk_repository.embedder import AbstractEmbedder, Embedding


class AbstractChunkRepository(ChunkRepository, ABC):
    """
    Abstract base class for chunk repositories.
    This class defines the interface for chunk repositories.
    """

    def __init__(self, chunker: Chunker, embedder: AbstractEmbedder) -> None:
        self._chunker = chunker
        self._embedder = embedder

    @abstractmethod
    def _add(self, document_meta: DocumentMetadata, chunks: list[Embedding]) -> None:
        """Add chunks to the store. Implementation defines handling."""
        pass

    def add(self, document: Document) -> None:
        """
        Add a document to the repository.
        This method chunks the document and then adds the chunks to the repository.
        """
        chunks = self._chunker.chunk(document)
        embeddings = self._embedder.embed_batch(chunks)
        self._add(document.metadata, embeddings)

    def add_batch(self, documents: list[Document]) -> None:
        """
        Adds a batch of documents to the current collection.
        """
        [self.add(document) for document in documents]
