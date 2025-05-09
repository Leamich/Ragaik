from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Dict, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# This is more of a value class as it identity is fully described by its content and metadata.
# We don't need an abstract class as specifics would be implemented in infrastructure layer (in chunker class)
@dataclass
class Chunk:
    """
    Chunk value class representing a part, no more than 512 tokens, of a Document.
    """
    content: str
    metadata: Dict[str, Any]


class Chunker(ABC):
    """
    Abstract base class for splitting a Document into chunks.
    """
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a Document into a list of Chunk instances."""
        pass


class RecursiveChunker(Chunker):
    """
    Recursive realization of chunker via BAAI/bge-base-en-v1.5 model.
    Chunk size = 512 tokens.
    """
    def __init__(self) -> None:
        self._CHUNK_SIZE = 512
        self._text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5 model'),
            chunk_size=self._CHUNK_SIZE,
            chunk_overlap=int(self._CHUNK_SIZE / 10),
            add_start_index=True,
            strip_whitespace=True,
        )

    def chunk(self, document: Document) -> List[Chunk]:
        chunk_as_documents: List[Document] = self._text_splitter.split_documents([document])
        return [Chunk(chunk.page_content, chunk.metedata) for chunk in chunk_as_documents]



