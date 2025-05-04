from abc import ABC, abstractmethod
from typing import List
from .chunk import Chunk

class Generator(ABC):
    """
    Abstract port class for generating answers from contexts.
    """
    @abstractmethod
    def generate(self, query: str, contexts: List[Chunk]) -> str:
        """Generate a response given a query and list of chunk contexts."""
        pass
