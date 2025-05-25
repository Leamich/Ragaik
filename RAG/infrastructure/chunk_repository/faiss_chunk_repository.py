from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


from ...domain.port import ChunkRepository
from langchain.schema import Document
from .mathbert_embedder import MathBERTEmbedder
from ..chunker import RecursiveChunker

# TODO add docstore from langchain_community.docstore
# TODO add normal initialization, not Document('0')


class FaissChunkRepository(ChunkRepository):
    """
    Faiss realization of ChunkRepository.
    """

    def __init__(self, strategy: DistanceStrategy = DistanceStrategy.COSINE, top_k: int = 5):
        self._embedder = MathBERTEmbedder()
        self._chunker = RecursiveChunker()

        self._vectorstore = FAISS.from_documents([Document('0')], embedding=self._embedder, distance_strategy=strategy)

        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": top_k})

    def add(self, document: Document) -> None:
        """Add document to the store."""
        chunks = self._chunker.chunk(document)
        self._vectorstore.add_documents(chunks)

    def get_retriever(self):
        return self._retriever

