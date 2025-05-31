from typing import List
from abc import ABC, abstractmethod

from langchain.schema import Document


class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """

    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        """Split a Document into a list of Chunk instances."""
        pass
    
    def chunk_many(self, documents: list[Document]) -> List[Document]:
        """Split many Documents into a list of Chunk instances."""
        res: List[Document] = []
        for document in documents:
            new_chunks = self.chunk(document)
            res += new_chunks
        
        return res
