from abc import ABC, abstractmethod

from langchain_core.documents import Document

# TODO: specify batch_id and image_id
class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        """Split a Document into a list of Chunk instances."""
        pass
    
    async def achunk_many(self, documents: list[Document]) -> list[Document]:
        """Split many Documents into a list of Chunk instances."""
        res: list[Document] = [] #TODO: add async
        for document in documents:
            new_chunks = self.chunk(document)
            res += new_chunks
        
        return res
    
    def chunk_many(self, documents: list[Document]) -> list[Document]:
        """Split many Documents into a list of Chunk instances."""
        res: list[Document] = [] #TODO: add async
        for document in documents:
            new_chunks = self.chunk(document)
            res += new_chunks
        
        return res
