from abc import ABC, abstractmethod
from typing import List
from .chunk import Chunk

class Retriever(ABC):
    """
    Abstract port class for retrieving relevant chunks.
    """
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """Return a list of chunks most relevant to the query."""
        pass
