from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
    """
    Abstract base class for generating embeddings from texts.
    """
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return corresponding embeddings."""
        pass