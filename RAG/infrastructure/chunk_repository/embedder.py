from abc import ABC, abstractmethod

from torch import FloatTensor

from domain.chunk import Chunk

type Embedding = tuple[FloatTensor]


class AbstractEmbedder(ABC):
    """
    Abstract port class for generating embeddings from texts.
    Supposed to embed one Document, split into chunks.
    """

    @abstractmethod
    def embed(self, chunk: Chunk) -> Embedding:
        """Embed a list of texts and return corresponding embeddings."""
        pass

    def embed_batch(self, chunks: list[Chunk]) -> list[Embedding]:
        """Embed a list of texts and return corresponding embeddings."""
        return list(map(self.embed, chunks))
