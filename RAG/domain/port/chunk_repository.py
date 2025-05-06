from abc import ABC, abstractmethod
from typing import Any, List

from .vector_storing_strategy import VectorStoringStrategy
from ..chunk import Chunk


class ChunkRepository(ABC):
    """
    Abstract base class for storing and querying chunks.
    """

    @abstractmethod
    def add(self, chunks: List[Chunk], strategies: list[VectorStoringStrategy]) -> None:
        """Add chunks to the store. Implementation defines handling."""
        pass

    @abstractmethod
    def query(self, key: any, top_k: int, strategy: list[VectorStoringStrategy]) -> list[list[Chunk]]:
        """Return top_k chunks matching the query key."""
        pass
