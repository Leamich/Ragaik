from abc import ABC, abstractmethod
from typing import Any, List
from ..chunk import Chunk

class ChunkRepository(ABC):
    """
    Abstract base class for storing and querying chunks.
    """
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to the store. Implementation defines handling."""
        pass

    @abstractmethod
    def query(self, key: Any, top_k: int) -> List[Chunk]:
        """Return top_k chunks matching the query key."""
        pass
