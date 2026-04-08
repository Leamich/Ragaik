import pickle
import asyncio
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from ...domain.port.chunk_repository import ChunkRepository
from ..token_chunker import TokenChunker
from ...domain.port.chunker import Chunker
from ...config import BM25_CACHE_FILE


class BM25ChunkRepository(ChunkRepository[BM25Retriever]):
    """
    BM25 realization of ChunkRepository with optional initialization from a list of documents.
    """

    def __init__(
        self,
        filename: Path | None = Path(BM25_CACHE_FILE),
        documents: list[Document] | None = None,
        chunker: Chunker = TokenChunker(),
        top_k: int = 5
    ) -> None:
        self._chunker = chunker
        if filename is not None and filename.exists():
            with open(filename, "rb") as f:
                self.retriever: BM25Retriever = pickle.load(f)

        elif documents is not None:
            self._chunks = self._chunker.chunk_many(documents)
            self.retriever = BM25Retriever.from_documents(self._chunks)
        
        else:
            raise ValueError(
                "Either filename or documents must be provided. If you've passed filename, it's not valid")
        self.retriever.k = top_k
        

    def store(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.retriever, f)

    async def add_batch(self, documents: list[Document]) -> None:
        new_chunks = await self._chunker.achunk_many(documents)
        top_k = self.retriever.k
        self.retriever = await asyncio.to_thread(BM25Retriever.from_documents, self.retriever.docs + new_chunks)
        self.retriever.k = top_k


