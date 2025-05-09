from abc import ABC, abstractmethod

from ..chunk import Chunk, Document


class ChunkRepository(ABC):
    """
    Abstract base class for storing and querying chunks.
    """

    @abstractmethod
    def add(self, document: Document) -> None:
        """Add chunks to the store. Implementation defines handling."""
        pass

    @abstractmethod
    def query(self, key: any, top_k: int) -> list[Chunk]:
        """Return top_k chunks matching the query key."""
        pass
