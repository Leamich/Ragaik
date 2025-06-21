from typing import Any
from abc import ABC, abstractmethod

from langchain.schema import Document


class ChunkRepository(ABC):
    """
    Abstract base class for storing and querying chunks.
    """

    @abstractmethod
    def add(self, document: Document) -> None:
        """Add document to the store. Implementation defines handling."""
        pass

    @abstractmethod
    def add_batch(self, documents: list[Document]) -> None:
        """Add batch of documents to the store. Implementation defines handling."""
        pass


    @abstractmethod
    def get_retriever(self) -> Any:
        pass

    @abstractmethod
    def is_init(self)-> bool:
        pass
