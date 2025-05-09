from abc import ABC, abstractmethod
from typing import Any

from ..chunk import Document


class DocumentLoader(ABC):
    """
    DocumentLoader port class for loading documents from various sources.
    """

    @abstractmethod
    def load(self, source: Any) -> list[Document]:
        """Load documents from the given source and return as a list."""
        pass
