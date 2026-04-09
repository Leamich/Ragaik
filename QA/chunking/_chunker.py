import asyncio
from abc import ABC, abstractmethod

from langchain_core.documents import Document

class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split Documents into a list of Chunk instances."""
        pass
    
    async def achunk(self, documents: list[Document]) -> list[Document]:
        """Split many Documents into a list of Chunk instances."""
        return await asyncio.to_thread(self.chunk, documents)