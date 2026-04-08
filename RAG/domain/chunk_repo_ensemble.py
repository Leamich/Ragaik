from typing import List
import asyncio

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from ..infrastructure.chunk_repository.bm25_chunk_repository import BM25ChunkRepository
from ..infrastructure.chunk_repository.faiss_chunk_repository import (
    FaissChunkRepository,
)
from .port.chunk_repository import ChunkRepository


class FaissAndBM25EnsembleRetriever(ChunkRepository[EnsembleRetriever]):
    """
    A class to manage a collection of chunk repositories.
    Supports lazy initialization if repositories are empty.
    """

    def __init__(
        self,
        first_repo: ChunkRepository = FaissChunkRepository(),
        second_repo: ChunkRepository = BM25ChunkRepository(),
        first_weight: float = 0.7,
        second_weight: float = 0.3,
    ) -> None:
        """
        Initialize the ensemble with two chunk repositories.
        """
        self._first_repo = first_repo
        self._second_repo = second_repo

        self.retriever = EnsembleRetriever(
            retrievers=[
                self._first_repo.retriever,
                self._second_repo.retriever,
            ],
            weights=[first_weight, second_weight],
            id_key="chunk_id",
        )

    async def add_batch(self, documents: List[Document]) -> None:
        await asyncio.gather(self._first_repo.add_batch(documents),
                             self._second_repo.add_batch(documents))

    async def query(self, query: str) -> list[Document]:
        """
        Query both repositories and return the results.
        """
        return await self.retriever.ainvoke(query)
