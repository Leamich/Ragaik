from dataclasses import dataclass
from typing import Any

type DocumentMetadata = dict[str, Any]


@dataclass
class Document:
    """
    Document value class representing a document with content and metadata.
    """
    content: str
    metadata: DocumentMetadata


# This is more of a value class as it identity is fully described by its content and metadata.
# We don't need an abstract class as specifics would be implemented in infrastructure layer (in chunker class)
@dataclass
class Chunk:
    """
    Chunk value class representing a part, no more than 512 tokens, of a Document.
    """
    content: str
    metadata: DocumentMetadata
