from pathlib import Path
from abc import ABC, abstractmethod
from typing import Generic, TypeVar



from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

RetriverType = TypeVar('RetriverType', bound=BaseRetriever)
class ChunkRepository(ABC, Generic[RetriverType]):
    """
    Abstract base class for storing and querying chunks.
    """
    retriever: RetriverType

    @abstractmethod
    async def add_batch(self, documents: list[Document]) -> None:
        """Add batch of documents to the store."""
        pass

    
    @abstractmethod
    def store(self, path: Path) -> None:
        """Store instance into a disk."""
        pass