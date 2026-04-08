import asyncio
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from .chunk_repository import ChunkRepository
from ..infrastructure.token_chunker import TokenChunker
from ..domain.port.chunker import Chunker

from ..config import FAISS_CACHE_DIR


class FaissChunkRepository(ChunkRepository[VectorStoreRetriever]):
    """
    Faiss realization of ChunkRepository, allows initialization with a list of documents.
    """

    def __init__(
        self,
        filename: Path | None = Path(FAISS_CACHE_DIR),
        documents: list[Document] | None = None,
        strategy: DistanceStrategy = DistanceStrategy.COSINE,
        embedder=HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large"),
        chunker: Chunker = TokenChunker(),
        top_k: int = 5
    ) -> None:  
        # TODO: maybe there is no needness to pass documents
        self._chunker = chunker

        if filename is not None and filename.exists():
            self._vectorstore = FAISS.load_local(
                str(filename),
                embeddings=embedder,
                distance_strategy=strategy,
                allow_dangerous_deserialization=True,
            )
        elif documents is not None:
            chunks: list[Document] = self._chunker.achunk_many(
                documents)  # type: ignore
            self._vectorstore = FAISS.from_documents(
                chunks, embedding=embedder, distance_strategy=strategy
            )
        else:
            raise ValueError(
                "Either filename or documents must be provided. If you've passed filename, it's not valid")
        self.retriever = self._vectorstore.as_retriever(search_kwargs={"k": top_k})



    async def store(self, path: Path) -> None:
        def func() -> None:
            self._vectorstore.save_local(str(path))
        asyncio.create_task(asyncio.to_thread(func))

    async def add_batch(self, documents: list[Document]) -> None:
        chunks = await self._chunker.achunk_many(documents)
        asyncio.create_task(self._vectorstore.aadd_documents(chunks))
    
