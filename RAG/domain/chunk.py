from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain.schema import Document

from dataclasses import dataclass

# This is more of a value class as it identity is fully described by its content and metadata.
# We don't need an abstract class as specifics would be implemented in infrastructure layer (in chunker class)
@dataclass
class Chunk:
    """
    Chunk value class representing a part of a Document.
    """
    content: str
    metedata: Dict[str, Any]


class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into a list of Chunk instances."""
        pass
