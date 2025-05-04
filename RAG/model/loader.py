from abc import ABC, abstractmethod
from typing import Any, List
from langchain.schema import Document

class DocumentLoader(ABC):
    """
    Abstract base class for loading source documents.
    """
    @abstractmethod
    def load(self, source: Any) -> List[Document]:
        """Load documents from the given source and return as a list."""
        pass