from abc import ABC, abstractmethod

from torch import FloatTensor

from ..chunk import Chunk


class Embedder(ABC):
    """
    Abstract port class for generating embeddings from texts.
    Supposed to embed one Document, split into chunks.
    """

    @abstractmethod
    def embed(self, chunk: Chunk) -> tuple[FloatTensor]:
        """Embed a list of texts and return corresponding embeddings."""
        pass
