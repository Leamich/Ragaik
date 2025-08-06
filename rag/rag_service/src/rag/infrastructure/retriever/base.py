"""Base protocol for document storing and retrieving."""
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class DocumentStore(ABC):
    """Abstract interface for a document store."""

    @abstractmethod
    async def ainvoke(self, query: str, top_k) -> list[Document]:
        """Retrieve documents based on a query.

        Args:
            query (str): The search query.
            top_k (int): The number of top documents to return.

        Returns:
            list[Document]: A list of retrieved documents.
        """

    @abstractmethod
    async def aadd_document(self, document: Document) -> None:
        """Add a document to the store.

        Args:
            document (Document): The document to add.
        """

    @abstractmethod
    async def aadd_documents(self, documents: list[Document]) -> None:
        """Add multiple documents to the store.

        Args:
            documents (list[Document]): The list of documents to add.
        """
