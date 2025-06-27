from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from ...domain.port.chunk_repository import ChunkRepository
from ..token_chunker import TokenChunker
from .chunker import Chunker


class FaissChunkRepository(ChunkRepository):
    """
    Faiss realization of ChunkRepository, allows initialization with a list of documents.
    """

    def __init__(
        self,
        filename: Path | None = None,
        documents: list[Document] | None = None,
        strategy: DistanceStrategy = DistanceStrategy.COSINE,
        embedder=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large"),
        chunker: Chunker = TokenChunker(),
    ):
        self._embedder = embedder
        self._chunker = chunker
        self._strategy = strategy

        if documents is not None:
            self._init_from_documents(documents)
        elif filename is not None and filename.exists():
            self._init_from_file(filename)
        else:
            self._vectorstore = None
            self._retriever = None

    def _init_from_file(self, filename: Path) -> None:
        """
        Initializes the vector store from a file.
        """
        self._vectorstore = FAISS.load_local(
            str(filename),
            embeddings=self._embedder,
            distance_strategy=self._strategy,
            allow_dangerous_deserialization=True,
        )
        self._retriever = self._vectorstore.as_retriever()

    def _init_from_documents(self, documents: list[Document]) -> None:
        chunks: list[Document] = self._chunker.chunk_many(documents)
        self._vectorstore = FAISS.from_documents(
            chunks, embedding=self._embedder, distance_strategy=self._strategy
        )
        self._retriever = self._vectorstore.as_retriever()

    def store(self, path: Path) -> None:
        if self._vectorstore is not None:
            self._vectorstore.save_local(path)

    def add(self, document: Document) -> None:
        """Add document to the store."""
        if self._vectorstore is not None:
            chunks = self._chunker.chunk(document)
            if chunks:
                self._vectorstore.add_documents(chunks)
        else:
            self._init_from_documents([document])

    def get_retriever(self):
        return self._retriever

    def is_init(self) -> bool:
        return self._retriever is not None

    def add_batch(self, documents: list[Document]) -> None:
        """
        Adds a batch of documents to the current collection.
        """
        if self._vectorstore is not None:
            chunks = self._chunker.chunk_many(documents)
            if chunks:
                self._vectorstore.add_documents(chunks)
        else:
            self._init_from_documents(documents)
