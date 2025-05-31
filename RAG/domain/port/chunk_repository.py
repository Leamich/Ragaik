from typing import List, Any
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
    
    def add_batch(self, documents: List[Document]) -> None:
        """
        Adds a batch of documents to the current collection.
        """
        [self.add(document) for document in documents]

    @abstractmethod
    def get_retriever(self) -> Any:
        pass

    @abstractmethod
    def is_init(self)-> bool: 
        pass
