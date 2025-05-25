from typing import List
from abc import ABC, abstractmethod

from langchain.schema import Document


class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """

    @abstractmethod
    def chunk(self, document: Document) -> List[str]:
        """Split a Document into a list of Chunk instances."""
        pass
