from abc import ABC, abstractmethod

from ..chunk import Chunk


class VectorStoringStrategy(ABC):
    """Strategy for storing chunks."""

    @abstractmethod
    def store_vector(self, chunk: Chunk) -> any:
        pass

    @abstractmethod
    def query_vector(self, key: any, top_k: int) -> any:
        pass
