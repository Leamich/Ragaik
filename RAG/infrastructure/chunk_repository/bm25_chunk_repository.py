from typing import List

from langchain_community.retrievers import BM25Retriever


from ...domain.port import ChunkRepository
from langchain.schema import Document
from ..chunker import RecursiveChunker

# TODO add normal initialization, not Document('0')



class BM25ChunkRepository(ChunkRepository):
    """
    BM25 realization of ChunkRepository
    """

    def __init__(self, top_k: int = 5):
        self._chunker = RecursiveChunker()
        self._top_k = top_k
        self._texts: List[Document] = [Document('0')]
        self._retriever = BM25Retriever.from_documents(self._texts)
        self._retriever.k = self._top_k

    def add(self, document: Document) -> None:
        new_chunks = self._chunker.chunk(document)
        self._texts += new_chunks
        self._retriever = BM25Retriever.from_documents(self._texts)
        self._retriever.k = self._top_k

    def get_retriever(self):
        return self._retriever
    