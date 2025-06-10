from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from ...domain.port import ChunkRepository
from ..token_chunker import TokenChunker
from .chunker import Chunker


class BM25ChunkRepository(ChunkRepository):
    """
    BM25 realization of ChunkRepository with optional initialization from a list of documents.
    """

    def __init__(
        self, documents: list[Document] | None = None, chunker: Chunker = TokenChunker()
    ):
        self._chunker = chunker

        if documents is None:
            self._chunks: list[Document] = []
            self._retriever = None

        else:
            self._chunks: list[Document] = self._chunker.chunk_many(documents)
            self._retriever = BM25Retriever.from_documents(self._chunks)

    def add(self, document: Document) -> None:
        self._chunks += self._chunker.chunk(document)
        self._retriever = BM25Retriever.from_documents(self._chunks)

    def get_retriever(self):
        return self._retriever

    def is_init(self) -> bool:
        return self._retriever is not None

    def add_batch(self, documents: list[Document]) -> None:
        self._chunks += self._chunker.chunk_many(documents)
        self._retriever = BM25Retriever.from_documents(self._chunks)
