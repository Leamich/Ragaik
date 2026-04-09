from typing import List
import asyncio
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever as LangchainEnsembleRetriever
from langchain_core.documents import Document

from ._bm25_chunk_repository import BM25ChunkRepository
from ._faiss_chunk_repository import FaissChunkRepository
from ._chunk_repository import ChunkRepository
from ..schema import Context



class EnsembleRetriever(ChunkRepository[LangchainEnsembleRetriever]):
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

        self.retriever = LangchainEnsembleRetriever(
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

    async def query(self, query: str) -> Context:
        """
        Query both repositories and return the results.
        """
        return set(await self.retriever.ainvoke(query))
    
    async def store(self, path: Path) -> None:
        await asyncio.gather(
            asyncio.to_thread(self._first_repo.store, path / "retriever1"),
            asyncio.to_thread(self._second_repo.store, path / "retriever2"),
        )
