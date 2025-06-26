from typing import List

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document

from .port.chunk_repository import ChunkRepository
from ..infrastructure.chunk_repository.bm25_chunk_repository import BM25ChunkRepository
from ..infrastructure.chunk_repository.faiss_chunk_repository import (
    FaissChunkRepository,
)

class FaissAndBM25EnsembleRetriever:
    """
    A class to manage a collection of chunk repositories.
    Supports lazy initialization if repositories are empty.
    """

    def __init__(
        self,
        faiss_repo: ChunkRepository = FaissChunkRepository(),
        bm_repo: ChunkRepository = BM25ChunkRepository(),
        faiss_weight: float = 0.7,
        bm_weight: float = 0.3,
    ) -> None:
        """
        Initialize the ensemble with two chunk repositories.
        """
        self._faiss_weight = faiss_weight
        self._bm_weight = bm_weight

        self._faiss_repo = faiss_repo
        self._bm_repo = bm_repo
        self._ensemble = None

        self._try_init_ensemble()

    def _try_init_ensemble(self):
        if self._faiss_repo.is_init() and self._bm_repo.is_init():
            self._ensemble = EnsembleRetriever(
                retrievers=[
                    self._faiss_repo.get_retriever(),
                    self._bm_repo.get_retriever(),
                ],
                weights=[self._faiss_weight, self._bm_weight],
            )

    def add(self, document: Document) -> None:
        """
        Add a document to both repositories.
        Initialize ensemble retriever if necessary.
        """
        self._faiss_repo.add(document)
        self._bm_repo.add(document)
        self._try_init_ensemble()

    def add_batch(self, documents: List[Document]) -> None:
        """
        Add a document list to both repositories.
        Initialize ensemble retriever if necessary.
        """
        self._faiss_repo.add_batch(documents)
        self._bm_repo.add_batch(documents)
        self._try_init_ensemble()

    def query(self, query: str) -> list[Document]:
        """
        Query both repositories and return the results.
        Raises error if ensemble is not initialized.
        """ 
        return self._ensemble.invoke(query) if self._ensemble is not None else []
