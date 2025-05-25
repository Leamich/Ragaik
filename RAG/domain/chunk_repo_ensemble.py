from langchain.schema import Document
from langchain.retrievers.ensemble import EnsembleRetriever

from .port import ChunkRepository
from .. infrastructure.chunk_repository.faiss_chunk_repository import FaissChunkRepository
from .. infrastructure.chunk_repository.bm25_chunk_repository import BM25ChunkRepository


class FaissAndBM25EnsembleRetriever:
    """
    A class to manage a collection of chunk repositories.
    """

    def __init__(self) -> None:
        """
        Initialize the ensemble with two chunk repositories.
        """
        self._faiss_repo: ChunkRepository = FaissChunkRepository()
        self._bm_repo: ChunkRepository =  BM25ChunkRepository()
        self._init_from_repos()

    def _init_from_repos(self):
        self._ensemble = EnsembleRetriever(
            retrievers=[self._faiss_repo.get_retriever(), self._bm_repo.get_retriever()],
            weights=[0.7, 0.3],
            k=5
        )

    def add(self, document: Document) -> None:
        """
        Add a document to both repositories.
        """
        self._faiss_repo.add(document)
        self._bm_repo.add(document)
        self._init_from_repos()

    def query(self, query: any) -> list[str]:
        """
        Query both repositories and return the results.
        """
        return self._ensemble.invoke(query)
