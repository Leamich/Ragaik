from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain.schema import Document

class Chunk(ABC):
    """
    Abstract base class for a document chunk.
    Provides content and metadata properties.
    """
    @property
    @abstractmethod
    def content(self) -> str:
        """Text content of the chunk."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Metadata associated with the chunk."""
        pass

class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into a list of Chunk instances."""
        pass