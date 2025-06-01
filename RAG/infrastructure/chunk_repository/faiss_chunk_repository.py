from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from ...domain.port import ChunkRepository
from ..token_chunker import TokenChunker
from .chunker import Chunker


class FaissChunkRepository(ChunkRepository):
    """
    Faiss realization of ChunkRepository, allows initialization with a list of documents.
    """

    def __init__(
        self,
        documents: List[Document] = None,
        strategy: DistanceStrategy = DistanceStrategy.COSINE,
        embedder = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"), 
        chunker: Chunker = TokenChunker()
    ):
        self._embedder = embedder
        self._chunker = chunker
        self._strategy = strategy

        if documents is None:
            self._vectorstore = None
            self._retriever = None

        else:
            self._init_from_documents(documents)
            

    def _init_from_documents(self, documents: List[Document]) -> None:
        self._vectorstore = FAISS.from_documents(documents, embedding=self._embedder, distance_strategy=self._strategy)
        self._retriever = self._vectorstore.as_retriever()


    def add(self, document: Document) -> None:
        """Add document to the store."""
        if self._vectorstore is not None:
            chunks = self._chunker.chunk(document)
            self._vectorstore.add_documents(chunks)
        else:
            self._init_from_documents([document])

    def get_retriever(self):
        return self._retriever
    
    def is_init(self) -> bool:
        return self._retriever is not None

    def add_batch(self, documents: List[Document]) -> None:
        """
        Adds a batch of documents to the current collection.
        """
        [self.add(document) for document in documents]
