from abc import ABC, abstractmethod

from domain.chunk import Document, Chunk


class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document into a list of Chunk instances."""
        pass
